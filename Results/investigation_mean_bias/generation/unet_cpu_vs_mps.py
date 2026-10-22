"""Single UNet forward on identical inputs, cpu vs mps, at several timesteps; also times one forward."""
import sys, time, torch, numpy as np
sys.path += ['/Users/emeunier/Desktop/Projets/DINO-Fusion/Diffusion_Model', '/Users/emeunier/Desktop/Projets/DINO-Fusion']
from diffusers import UNet2DModel, DDPMScheduler
M = '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b'
u_cpu = UNet2DModel.from_pretrained(f'{M}/unet').eval()
u_mps = UNet2DModel.from_pretrained(f'{M}/unet').eval().to('mps')
sch = DDPMScheduler.from_pretrained(f'{M}/scheduler')
x0 = torch.from_numpy(np.load(f'{M}/inference/infesteps_1000/constraints_no_constraints/20250203-141645.npy', mmap_mode='r')[:4].copy())
g = torch.Generator().manual_seed(1)
print('params', sum(p.numel() for p in u_cpu.parameters()) / 1e6, 'M')
for t in [999, 900, 500, 100, 10, 0]:
    noise = torch.randn(x0.shape, generator=g)
    xt = sch.add_noise(x0, noise, torch.tensor([t]))
    with torch.no_grad():
        t0 = time.time(); e_c = u_cpu(xt, t).sample; tc = time.time() - t0
        t0 = time.time(); e_m = u_mps(xt.to('mps'), t).sample.cpu(); tm = time.time() - t0
    d = e_m - e_c
    print(f't={t}: rel L2 diff {float(d.norm() / e_c.norm()):.2e}  max|d| {float(d.abs().max()):.2e}  mean(d) {float(d.mean()):.2e} '
          f' per-channel mean d (first 5) {[round(float(v), 5) for v in d.mean((0, 2, 3))[:5]]}  '
          f'eps err vs true noise cpu {float((e_c - noise).pow(2).mean()):.4f} mps {float((e_m - noise).pow(2).mean()):.4f}  time cpu {tc:.2f}s mps {tm:.2f}s')
