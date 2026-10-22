"""Single-step denoiser diagnostics on CPU (no sampling): for x_t = sqrt(abar) x0 + sqrt(1-abar) eps (+ uniform DC offset delta),
measure (1) the DC (per-channel ocean mean) error of the Tweedie x0_hat vs the error of the spatial pattern,
(2) the gain d(x0_hat DC)/d(delta): 0 = the offset is removed as noise (ideal, data DC is pinned), 1/sqrt(abar) = kept and amplified as signal."""
from load import *
from diffusers import UNet2DModel
np.set_printoptions(precision=3, suppress=True, linewidth=220)
torch.manual_seed(0)
unet = UNet2DModel.from_pretrained('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/unet').eval()
ab = torch.tensor(np.load('alphas_cumprod.npy'), dtype=torch.float32)
x0 = torch.from_numpy(load_data(8, seed=3))
oc = torch.from_numpy(ocean_n)
def dc(x): return torch.stack([x[:, c][:, oc[c]].mean(1) for c in range(37)], 1)  # (b,37)
def x0hat(xt, t):
    with torch.no_grad(): e = unet(xt, t).sample
    return (xt - (1 - ab[t]).sqrt() * e) / ab[t].sqrt(), e
eps = torch.randn_like(x0)
print(' t   | DC err of x0hat (rms, normalized) | data DC std | pattern err rms (ocean, DC removed) | land |x0hat| | gain dx0hatDC/ddelta (ocean)  [ideal 0, signal 1/sqrt(abar)=..]  | gain eps_hatDC*sqrt(1-abar) [ideal 1]')
dlm = level_mean_norm(load_data(200)); sdc = dlm.std(0).mean()
rows = []
for t in [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 990]:
    xt = ab[t].sqrt() * x0 + (1 - ab[t]).sqrt() * eps
    xh, e = x0hat(xt, t)
    err_dc = (dc(xh) - dc(x0))
    patt = ((xh - dc(xh)[..., None, None]) - (x0 - dc(x0)[..., None, None]))[:, oc].pow(2).mean().sqrt()
    land = xh[:, ~oc].abs().mean()
    delta = 0.05 * (1 - ab[t]).sqrt()   # offset small relative to the noise level
    xh2, e2 = x0hat(xt + delta, t)
    gain = ((dc(xh2) - dc(xh)) / delta).mean().item()
    gain_eps = ((dc(e2) - dc(e)) * (1 - ab[t]).sqrt() / delta).mean().item()
    rows.append((t, err_dc.pow(2).mean().sqrt().item(), patt.item(), land.item(), gain, 1 / ab[t].sqrt().item(), gain_eps))
    print(f'{t:4d} | {rows[-1][1]:.4f} | {sdc:.4f} | {rows[-1][2]:.4f} | {rows[-1][3]:.4f} | gain {gain:8.3f}  [signal {1/ab[t].sqrt():8.2f}] | {gain_eps:.3f}')
np.save('dc_response.npy', np.array(rows))
