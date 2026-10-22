"""Check statistics of torch.randn with an mps Generator, as used by sampling.py and DDPMScheduler.step."""
import torch, diffusers
from diffusers.utils.torch_utils import randn_tensor
print(torch.__version__, diffusers.__version__)
shape = (16, 37, 208, 64)
for dev in ['cpu', 'mps']:
    g = torch.Generator(dev).manual_seed(0)
    draws = [randn_tensor(shape, generator=g, device=torch.device(dev)).cpu() for _ in range(6)]
    d0 = draws[0]
    print(dev, 'mean %.4f std %.4f' % (d0.mean(), d0.std()))
    print('  corr successive draws', [round(float(torch.corrcoef(torch.stack([draws[i].flatten(), draws[i+1].flatten()]))[0,1]),4) for i in range(5)])
    print('  corr between batch members of draw0', [round(float(torch.corrcoef(torch.stack([d0[0].flatten(), d0[i].flatten()]))[0,1]),4) for i in range(1,5)])
    print('  per-sample mean of draw0', [round(float(d0[i].mean()),5) for i in range(4)])
    # spatial lag correlation
    print('  lag-1 x corr', float(torch.corrcoef(torch.stack([d0[..., :-1].flatten(), d0[..., 1:].flatten()]))[0,1]))
    print('  identical draws?', bool((draws[0]==draws[1]).all()))
