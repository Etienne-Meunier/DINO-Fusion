"""
Task 4 : candidate remedies tested on the DPS guidance step (lbd = 1), same states and t as p2.

Variants of the step dx at x_t (all monotonicity-only, no reference profile) :
    V0 base      : -grad L_cell, L_cell = NegativeGradDensityHuber (cell-wise hinge)
    V1 dcfree    : V0 minus its ocean mean per channel (no uniform push per level)
    V2 profile   : -grad L_prof, same Huber hinge on the sample's OWN horizontal-mean (volume-weighted) density profile
    V3 split     : V1 + V2 (anomalies from the cell hinge, level means only from the own-profile hinge)
    V4 mass      : V0 projected so that the first-order change of the total mass sum V rho(x0_hat) is zero
    V5 lowonly   : cell hinge with stop-gradient on the upper cell (only densify the lower cell)
    V6 topfree   : cell hinge with stop-gradient on rho[0] only (surface level never pushed directly)
Reported per (set, t, variant) :
    T0, S0  : ocean mean of dx on strided channel T level 0 / S level 0 (absolute, lbd = 1), sign share over states
    rho_k   : first-order horizontal-mean change of rho(x0_hat) by DC(dx) / sqrt(abar), levels 0, 1, 2, 4, 8, 13, 16
    mass    : first-order change of the volume-weighted column-mean rho by dx (kg/m3, lbd = 1)
    cos     : cosine of dx with -grad L_cell (how much the step still decreases the cell loss)
    dLcell  : first-order decrease of L_cell along dx, divided by the decrease along V0 (effectiveness)
Output : p3_remedies.csv, p3_summary.txt
"""
from common import *
import pandas as pd
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from p1_loss_gradient import rho_TS, level_mean
from p2_dps_gradient import dc_part, TS, N

LEVELS = [0, 1, 2, 4, 8, 13, 16]


def hinge(g, tau=1e-4) :
    h = torch.where(g < -tau, -g - tau / 2, g**2 / (2 * tau))
    return torch.where(g < 0, h, torch.zeros_like(g))


def make_losses(obs) :
    """Cell-wise loss variants and the own-profile loss, rho (b, 36, i, j) -> (b,)."""
    base = NegativeGradDensityHuber(observer=obs)
    valid, w = base.valid, base.w
    vol = obs.volume
    lv = vol.sum((-2, -1))
    pw = lv[1:] * (lv[:-1] > 0) * (lv[1:] > 0); pw = pw / pw.sum()

    def cell(rho, mode='base') :
        up, lo = rho[:, :-1], rho[:, 1:]
        if mode == 'lowonly' : up = up.detach()
        if mode == 'topfree' : up = torch.cat([up[:, :1].detach(), up[:, 1:]], 1)
        g = torch.where(valid, lo - up, torch.zeros_like(lo))
        return (w * hinge(g)).sum((1, 2, 3))

    def prof(rho) :
        r = (torch.nan_to_num(rho) * vol).sum((-2, -1)) / lv.clamp(min=1)
        g = r[:, 1:] - r[:, :-1]
        return (pw * hinge(g)).sum(1)
    return cell, prof, base


if __name__ == '__main__' :
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    pipe = DDPMPipeline_Tensor.from_pretrained(MODEL).to(dev)
    unet, sch = pipe.unet, pipe.scheduler
    obs = get_observer(dev)
    cell, prof, base = make_losses(obs)
    oc = ocean_mask_norm(obs).to(dev).float()
    vol = obs.volume
    sets = {'DATA' : load_data(N, seed=1).to(dev), 'NONE' : load_none()[:N].to(dev)}
    noise = torch.randn(N, *sets['DATA'].shape[1:], generator=torch.Generator().manual_seed(0)).to(dev)
    rows = []
    for name, x0 in sets.items() :
        for t in TS :
            abar = sch.alphas_cumprod[t].to(dev)
            tt = torch.tensor([t], device=dev)
            tweedie = lambda z : ((z - (1 - abar).sqrt() * unet(z, tt).sample) / abar.sqrt()).clamp(-1, 1)
            xt = (abar.sqrt() * x0 + (1 - abar).sqrt() * noise).requires_grad_(True)
            x0h = tweedie(xt)
            rho = rho_TS(x0h, obs)[0]
            grads = {}
            for k, Lf in (('cell', lambda : cell(rho)), ('prof', lambda : prof(rho)), ('lowonly', lambda : cell(rho, 'lowonly')),
                          ('topfree', lambda : cell(rho, 'topfree')),
                          ('mass', lambda : (torch.nan_to_num(rho) * vol).sum((1, 2, 3)) / vol.sum())) :
                grads[k] = torch.autograd.grad(Lf().sum(), xt, retain_graph=True)[0].detach() * oc
            assert torch.allclose(cell(rho), base(x0h), rtol=1e-4)
            d0 = -grads['cell']
            a = grads['mass']
            proj = lambda d : d - ((a * d).sum((1, 2, 3), keepdim=True) / (a * a).sum((1, 2, 3), keepdim=True)) * a
            var = {'V0 base' : d0, 'V1 dcfree' : d0 - dc_part(d0, oc), 'V2 profile' : -grads['prof'],
                   'V3 split' : d0 - dc_part(d0, oc) - grads['prof'], 'V4 mass' : proj(d0),
                   'V5 lowonly' : -grads['lowonly'], 'V6 topfree' : -grads['topfree']}
            dec0 = (grads['cell'] * -d0).sum((1, 2, 3))
            for vn, d in var.items() :
                dc = dc_part(d, oc)
                _, (drho, _, _) = torch.autograd.functional.jvp(lambda q : rho_TS(q, obs), x0h.detach(), dc / abar.sqrt())
                lm = level_mean(drho, obs.tmask).mean(0).cpu().numpy()
                n = oc.sum((-2, -1))
                m = d.sum((-2, -1)) / n
                dec = (grads['cell'] * -d).sum((1, 2, 3))
                row = dict(set=name, t=t, variant=vn, T0=m[:, 0].mean().item(), T0_pos=(m[:, 0] > 0).float().mean().item(),
                           S0=m[:, NT].mean().item(), S0_pos=(m[:, NT] > 0).float().mean().item(),
                           mass=(a * d).sum((1, 2, 3)).mean().item(),
                           cos=(dec / (grads['cell'].pow(2).sum((1, 2, 3)).sqrt() * d.pow(2).sum((1, 2, 3)).sqrt() + 1e-30)).mean().item(),
                           dLcell=(dec / dec0).mean().item(), norm=d.pow(2).sum((1, 2, 3)).sqrt().mean().item(),
                           Lcell=cell(rho).mean().item(), Lprof=prof(rho).mean().item())
                row.update({f'rho{k}' : lm[k] for k in LEVELS})
                rows.append(row)
            print(name, t, flush=True)
    df = pd.DataFrame(rows); df.to_csv(HERE / 'p3_remedies.csv', index=False)
    pd.set_option('display.width', 300); pd.set_option('display.precision', 2); pd.set_option('display.max_rows', 500)
    with open(HERE / 'p3_summary.txt', 'w') as f :
        for (name, t), g in df.groupby(['set', 't']) :
            f.write(f'=== {name} t={t} Lcell={g.Lcell.iloc[0]:.2e} Lprof={g.Lprof.iloc[0]:.2e}\n'
                    + g.drop(columns=['set', 't', 'Lcell', 'Lprof']).to_string(index=False) + '\n')
    print(open(HERE / 'p3_summary.txt').read())
