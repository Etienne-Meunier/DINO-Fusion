"""
Task 2 : DPS guidance step of NegativeGradDensityHuber through the UNet, dx = -grad_{x_t} L(clip(Tweedie(x_t))) (lbd = 1),
at a few t, for data and unconstrained states noised to x_t (fixed noise).

Per strided T / S channel (ocean cells) :
    DC     : ocean mean of dx, divided by |dx| of the state
    DCfrac : mean^2 / mean(dx^2), share of the step energy in the uniform mode of the channel
for the DPS step and for the loss-only step at the same Tweedie estimate (-grad_{x0} L / sqrt(abar), what DPS would be if
d x0_hat / d x_t were 1 / sqrt(abar)).
Retention : project the change of x0_hat after the step x_t + s dx on (a) the DC part, (b) the non-DC part of dx / sqrt(abar).
    1 = the denoiser keeps the push, 0 = the denoiser undoes it.
Induced rho : first-order change of rho(x0_hat) by DC(dx) / sqrt(abar), horizontal ocean mean per file level (kg/m3 per
unit |dx|), the profile a DC-blind denoiser would accumulate.

Output : p2_dps.csv (per set, t, channel), p2_rho.csv (per set, t, level), p2_summary.txt
"""
from common import *
import pandas as pd
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from p1_loss_gradient import rho_TS, level_mean

TS = [50, 200, 500, 800]
N = 4


def chan_stats(dx, oc) :
    """dx (b, 37, i, j) masked on ocean -> per-channel ocean mean / |dx| (b, 37) and DC fraction (b, 37)."""
    n = oc.sum((-2, -1)).clamp(min=1)
    m = dx.sum((-2, -1)) / n
    ms = (dx**2).sum((-2, -1)) / n
    scale = dx.pow(2).sum((1, 2, 3)).sqrt()[:, None]
    return m / scale, m**2 / ms.clamp(min=1e-30)


def dc_part(dx, oc) :
    n = oc.sum((-2, -1)).clamp(min=1)
    return (dx.sum((-2, -1)) / n)[..., None, None] * oc


if __name__ == '__main__' :
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    pipe = DDPMPipeline_Tensor.from_pretrained(MODEL).to(dev)
    unet, sch = pipe.unet, pipe.scheduler
    obs = get_observer(dev)
    loss = NegativeGradDensityHuber(observer=obs)
    oc = ocean_mask_norm(obs).to(dev).float()
    sets = {'DATA' : load_data(N, seed=1).to(dev), 'NONE' : load_none()[:N].to(dev)}
    gen = torch.Generator().manual_seed(0)
    noise = torch.randn(N, *sets['DATA'].shape[1:], generator=gen).to(dev)
    rows, rrows = [], []
    for name, x0 in sets.items() :
        for t in TS :
            abar = sch.alphas_cumprod[t].to(dev)
            xt = (abar.sqrt() * x0 + (1 - abar).sqrt() * noise).requires_grad_(True)
            tt = torch.tensor([t], device=dev)
            def tweedie(z) :
                eps = unet(z, tt).sample
                return ((z - (1 - abar).sqrt() * eps) / abar.sqrt()).clamp(-1, 1)
            x0h = tweedie(xt)
            L = loss(x0h)
            dx = -torch.autograd.grad(L.sum(), xt)[0].detach() * oc
            # loss-only step at the same Tweedie estimate
            z = x0h.detach().requires_grad_(True)
            dl = -torch.autograd.grad(loss(z).sum(), z)[0].detach() * oc / abar.sqrt()
            m_d, f_d = chan_stats(dx, oc)
            m_l, f_l = chan_stats(dl, oc)
            # retention of the DC and non-DC parts
            with torch.no_grad() :
                base = x0h.detach()
                dc = dc_part(dx, oc); ndc = dx - dc
                s = 0.05 / dx.pow(2).sum((1, 2, 3)).sqrt().view(-1, 1, 1, 1)   # |s dx| = 0.05
                ret = {}
                for k, comp in (('DC', dc), ('nonDC', ndc)) :
                    ch = (tweedie(xt.detach() + s * comp) - base) * oc
                    ref = s * comp / abar.sqrt()
                    ret[k] = ((ch * ref).sum((1, 2, 3)) / (ref**2).sum((1, 2, 3))).mean().item()
            # rho induced by the DC part, through the observer
            _, (drho, _, _) = torch.autograd.functional.jvp(lambda q : rho_TS(q, obs), base, dc / abar.sqrt())
            scale = dx.pow(2).sum((1, 2, 3)).sqrt().view(-1, 1)
            ir = (level_mean(drho, obs.tmask) / scale).mean(0).cpu().numpy()
            rrows += [dict(set=name, t=t, k=k, ind_rho_DC=ir[k]) for k in range(36)]
            for c in range(2 * NT) :
                rows.append(dict(set=name, t=t, var='T' if c < NT else 'S', ks=c % NT,
                                 DC_dps=m_d[:, c].mean().item(), DC_dps_sign=(m_d[:, c] > 0).float().mean().item(),
                                 DCfrac_dps=f_d[:, c].mean().item(),
                                 DC_loss=m_l[:, c].mean().item(), DCfrac_loss=f_l[:, c].mean().item(),
                                 L=L.mean().item(), ret_DC=ret['DC'], ret_nonDC=ret['nonDC'],
                                 dx_norm=dx.pow(2).sum((1, 2, 3)).sqrt().mean().item(),
                                 dl_norm=dl.pow(2).sum((1, 2, 3)).sqrt().mean().item()))
            print(name, t, 'L', L.mean().item(), 'ret', ret, flush=True)
    df = pd.DataFrame(rows); df.to_csv(HERE / 'p2_dps.csv', index=False)
    dr = pd.DataFrame(rrows); dr.to_csv(HERE / 'p2_rho.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.precision', 3); pd.set_option('display.max_rows', 500)
    with open(HERE / 'p2_summary.txt', 'w') as f :
        for (name, t), g in df.groupby(['set', 't']) :
            f.write(f'=== {name} t={t} L={g.L.iloc[0]:.3e} |dx|={g.dx_norm.iloc[0]:.3e} |dl|={g.dl_norm.iloc[0]:.3e} '
                    f'retention DC={g.ret_DC.iloc[0]:.3f} nonDC={g.ret_nonDC.iloc[0]:.3f}\n')
            f.write(g[g.ks < 10][['var', 'ks', 'DC_dps', 'DC_dps_sign', 'DCfrac_dps', 'DC_loss', 'DCfrac_loss']].to_string(index=False) + '\n')
        f.write('\n=== induced rho by DC part, level mean per unit |dx| ===\n')
        f.write(dr.pivot_table(index='k', columns=['set', 't'], values='ind_rho_DC').to_string() + '\n')
    print(open(HERE / 'p2_summary.txt').read())
