"""DDPMScheduler.step on cpu vs mps with identical model_output, sample and variance noise (cpu generator, same seed)."""
import torch
from diffusers import DDPMScheduler
M = '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b'
s = DDPMScheduler.from_pretrained(f'{M}/scheduler'); s.set_timesteps(1000)
g = torch.Generator().manual_seed(3)
x = torch.randn(4, 37, 208, 64, generator=g) * 1.5; e = torch.randn(4, 37, 208, 64, generator=g)
for t in [999, 500, 50, 1, 0]:
    a = s.step(e, t, x, generator=torch.Generator().manual_seed(7)).prev_sample
    b = s.step(e.to('mps'), t, x.to('mps'), generator=torch.Generator().manual_seed(7)).prev_sample.cpu()
    print(t, 'max|diff|', float((a - b).abs().max()), 'mean diff', float((a - b).mean()))
