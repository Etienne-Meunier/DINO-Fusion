"""Unconstrained DDPM sampling, same algorithm as Diffusion_Model/pipelines/sampling.sample with loss_c=None and no projection,
with the compute device and the generator device chosen independently.

python gen_test.py --device mps --gen_device mps --seed 0 --batch 16 --steps 1000 --name X  -> runs/X/generated.npy
"""
import sys, argparse, json, os, time, torch, numpy as np
sys.path += ['/Users/emeunier/Desktop/Projets/DINO-Fusion/Diffusion_Model', '/Users/emeunier/Desktop/Projets/DINO-Fusion']
from diffusers.utils.torch_utils import randn_tensor
from pipelines.pipeline_tensor import DDPMPipeline_Tensor

p = argparse.ArgumentParser()
p.add_argument('--device', default='mps'); p.add_argument('--gen_device', default=None)
p.add_argument('--seed', type=int, default=0); p.add_argument('--batch', type=int, default=16)
p.add_argument('--steps', type=int, default=1000); p.add_argument('--name', required=True)
p.add_argument('--dtype', default='float32')
a = p.parse_args()
a.gen_device = a.gen_device or a.device
out = f'{os.path.dirname(os.path.abspath(__file__))}/runs/{a.name}'
os.makedirs(out, exist_ok=True)
dev = torch.device(a.device)
pipe = DDPMPipeline_Tensor.from_pretrained('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b').to(dev)
pipe.set_progress_bar_config(disable=True)
sch = pipe.scheduler
sch.set_timesteps(a.steps)
g = torch.Generator(a.gen_device).manual_seed(a.seed)
shape = (a.batch, pipe.unet.config.in_channels, *pipe.unet.config.sample_size)
x = randn_tensor(shape, generator=g, device=dev)
t0 = time.time()
with torch.no_grad():
    for i, t in enumerate(sch.timesteps):
        eps = pipe.unet(x, t).sample
        x = sch.step(eps, t, x, generator=g).prev_sample
        if i % 100 == 0: print(i, int(t), f'{time.time() - t0:.0f}s', flush=True)
np.save(f'{out}/generated.npy', x.cpu().numpy())
json.dump(vars(a) | dict(time=time.time() - t0, torch=torch.__version__, timesteps0=sch.timesteps[:3].tolist()), open(f'{out}/args.json', 'w'), indent=2)
print('done', out, time.time() - t0)
