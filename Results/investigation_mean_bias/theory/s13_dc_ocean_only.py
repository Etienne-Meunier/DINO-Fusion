"""Denoiser response to an offset added on ocean cells only (land/pad untouched) vs on the whole grid, and with land/pad
set to 0 in x_t as the border0 projection does. gain_eps = d(eps_hat ocean DC) sqrt(1-abar)/d(delta): 1 = offset removed as noise, 0 = kept as signal."""
from load import *
from diffusers import UNet2DModel
unet = UNet2DModel.from_pretrained('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/unet').eval()
ab = torch.tensor(np.load('alphas_cumprod.npy'), dtype=torch.float32)
torch.manual_seed(0)
x0 = torch.from_numpy(load_data(4, seed=5)); eps = torch.randn_like(x0)
oc = torch.from_numpy(ocean_n)
print('land mask identical across channels?', bool((ocean_n == ocean_n[0]).all()), '; ocean cells per channel', ocean_n[0].sum(), '; tmask ocean cells at surface/deepest strided', tmask_full[0].sum(), tmask_full[34].sum())
def dc(x): return torch.stack([x[:, c][:, oc[c]].mean(1) for c in range(37)], 1)
for t in [100, 300, 500, 700, 900]:
    s = (1 - ab[t]).sqrt(); xt = ab[t].sqrt() * x0 + s * eps; d = 0.05 * s
    with torch.no_grad():
        e0 = unet(xt, t).sample
        e_all = unet(xt + d, t).sample
        e_oc = unet(xt + d * oc.float(), t).sample
        xb = torch.where(oc, xt, torch.zeros_like(xt))
        eb0 = unet(xb, t).sample; eb = unet(xb + d * oc.float(), t).sample
    g = lambda a, b: ((dc(a) - dc(b)) * s / d).mean().item()
    print(f't={t}: gain_eps whole-grid offset {g(e_all, e0):.3f} | ocean-only offset {g(e_oc, e0):.3f} | ocean-only offset with land=0 (border0-like) {g(eb, eb0):.3f}')
