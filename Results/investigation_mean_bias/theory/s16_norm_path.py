"""How much of a uniform input offset survives to the output head: change in per-channel spatial mean of the feature map
before and after the final GroupNorm (conv_norm_out), and in eps_hat, for x_t + delta vs x_t."""
from load import *
from diffusers import UNet2DModel
unet = UNet2DModel.from_pretrained('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/unet').eval()
ab = torch.tensor(np.load('alphas_cumprod.npy'), dtype=torch.float32)
x0 = torch.from_numpy(load_data(4, seed=5)); torch.manual_seed(0); eps = torch.randn_like(x0)
feats = {}
unet.conv_norm_out.register_forward_hook(lambda m, i, o: feats.update(pre=i[0].detach(), post=o.detach()))
unet.conv_in.register_forward_hook(lambda m, i, o: feats.update(cin=o.detach()))
t = 500; s = (1 - ab[t]).sqrt(); xt = ab[t].sqrt() * x0 + s * eps; d = 0.05 * s
out = {}
for k, x in [('0', xt), ('d', xt + d)]:
    with torch.no_grad(): e = unet(x, t).sample
    out[k] = {n: v.clone() for n, v in feats.items()} | {'eps': e}
for n in ['cin', 'pre', 'post', 'eps']:
    a, b = out['0'][n], out['d'][n]
    dmean = (b - a).mean((-2, -1)).abs().mean().item(); dpat = (b - a).std((-2, -1)).mean().item(); scale = a.std().item()
    print(f'{n:5s}: |change of channel spatial means| {dmean:.2e}  spatial std of change {dpat:.2e}  feature scale {scale:.2e}  (input delta {d.item():.2e})')
