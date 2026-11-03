"""
V7 : the split remedy written as a pure loss (no change of sampling.py), on the Tweedie estimate x0h :
    L7(x0h) = L_prof(rho(x0h)) + L_cell(rho(x0h - DC(x0h) + DC(x0h).detach()))
DC = ocean mean per channel. The cell hinge then has no gradient on the level means, which only the own-profile hinge moves.
Checks that the DPS step of L7 wrt x_t has no surface DC on data (V0 : T0 > 0 on every state) and how much of the
cell-loss decrease it keeps. Output : p6_loss_level_split.txt
"""
from common import *
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from p1_loss_gradient import rho_TS
from p2_dps_gradient import dc_part, TS, N
from p3_remedies import make_losses

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipe = DDPMPipeline_Tensor.from_pretrained(MODEL).to(dev)
unet, sch = pipe.unet, pipe.scheduler
obs = get_observer(dev)
cell, prof, _ = make_losses(obs)
oc = ocean_mask_norm(obs).to(dev).float()
sets = {'DATA' : load_data(N, seed=1).to(dev), 'NONE' : load_none()[:N].to(dev)}
noise = torch.randn(N, *sets['DATA'].shape[1:], generator=torch.Generator().manual_seed(0)).to(dev)
lines = ['set t | variant | T0 (mean, share>0) | S0 | |DC|/|dx| upper 6 T channels | dLcell / dLcell(V0)']
for name, x0 in sets.items() :
    for t in TS :
        abar = sch.alphas_cumprod[t].to(dev); tt = torch.tensor([t], device=dev)
        xt = (abar.sqrt() * x0 + (1 - abar).sqrt() * noise).requires_grad_(True)
        x0h = ((xt - (1 - abar).sqrt() * unet(xt, tt).sample) / abar.sqrt()).clamp(-1, 1)
        rho = rho_TS(x0h, obs)[0]
        g0 = torch.autograd.grad(cell(rho).sum(), xt, retain_graph=True)[0] * oc
        dcx = dc_part(x0h * oc, oc)
        x7 = x0h - dcx + dcx.detach()
        for vn, L in (('V7 split-loss', prof(rho) + cell(rho_TS(x7, obs)[0])), ('V7 cell part only', cell(rho_TS(x7, obs)[0]))) :
            d = -torch.autograd.grad(L.sum(), xt, retain_graph=True)[0] * oc
            n = oc.sum((-2, -1)); m = d.sum((-2, -1)) / n
            dcn = dc_part(d, oc)[:, :6].pow(2).sum((1, 2, 3)).sqrt() / d[:, :6].pow(2).sum((1, 2, 3)).sqrt()
            eff = ((g0 * -d).sum((1, 2, 3)) / (g0 * g0).sum((1, 2, 3))).mean()
            m0 = (-g0).sum((-2, -1)) / n
            lines.append(f'{name} {t:3d} | {vn:17s} | {m[:, 0].mean():+.2e} ({(m[:, 0] > 0).float().mean():.2f}) [V0 {m0[:, 0].mean():+.2e}] '
                         f'| {m[:, NT].mean():+.2e} | {dcn.mean():.3f} | {eff:.2f}')
        print(lines[-1], flush=True)
open(HERE / 'p6_loss_level_split.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
