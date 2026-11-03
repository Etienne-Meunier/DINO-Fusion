"""
When does the surface mean of an unconstrained generation leave the data ?

Unconstrained DDPM sampling, batch 8, seed 0, 1000 steps, with or without BorderZero (`--border0`). `--center init`
removes the per-channel spatial mean over the ocean cells of the initial noise, `--center all` also of the noise added
at every step (DDPM step written out, fixed_small variance, x0 clipped as the scheduler). At every step the
area-weighted level-0 T of x0_hat (clamped as in `exp.sample`) is recorded. Saves runs/traj_<tag>.npz with t (steps,)
and T0 (steps, b), and the final T0, and the final batch as runs/traj_<tag>_x.npy.
"""
from exp import *
import argparse
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer
from pipelines.losses import BorderZero

parser = argparse.ArgumentParser()
parser.add_argument('--border0', action='store_true')
parser.add_argument('--border_noise', action='store_true',
                    help='land and padding cells replaced at every step by fresh noise of the next noise level (0 at the end)')
parser.add_argument('--center', choices=['none', 'init', 'all'], default='none')
args = parser.parse_args()
device = Accelerator().device
pipeline = DDPMPipeline_Tensor.from_pretrained(MODEL).to(device)
pipeline.scheduler.set_timesteps(1000)
obs = Observer('data', TrainingConfig(), device=device)
projection = BorderZero(obs) if args.border0 else None
w = torch.as_tensor(np.load(HERE / 'area0.npy'), device=device)


def T0(x) :
    return (torch.nan_to_num(obs.unnormalized(x)['toce'][:, 0]) * w).sum((-2, -1)) / w.sum()


ocean = obs.normalizer.padder((~obs.normalizer.masker.mask).float()).to(device)


def center(z) :
    """z with its per-(member, channel) mean over the ocean cells removed."""
    return z - (z * ocean).sum((-2, -1), keepdim=True) / ocean.sum((-2, -1), keepdim=True) * ocean


def ddpm_step(eps, t, x) :
    """`DDPMScheduler.step` for 1000 inference steps with centred added noise."""
    a_t, a_prev = sch.alphas_cumprod[t].to(device), (sch.alphas_cumprod[t - 1].to(device) if t > 0 else torch.tensor(1., device=device))
    beta_t = 1 - a_t / a_prev
    x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
    mean = a_prev.sqrt() * beta_t / (1 - a_t) * x0 + (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t) * x
    if t == 0 :
        return mean
    std = ((1 - a_prev) / (1 - a_t) * beta_t).clamp(min=1e-20).sqrt()
    return mean + std * center(torch.randn(x.shape, generator=generator, device=device))


sch = pipeline.scheduler
generator = torch.Generator(device).manual_seed(0)
x = torch.randn((8, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size), generator=generator, device=device)
if args.center != 'none' :
    x = center(x)
ts, rec = [], []
with torch.no_grad() :
    for t in pipeline.progress_bar(sch.timesteps) :
        eps = pipeline.unet(x, t).sample
        abar = sch.alphas_cumprod[t].to(device)
        x0h = ((x - (1 - abar).sqrt() * eps) / abar.sqrt()).clamp(-1, 1)
        ts.append(int(t)); rec.append(T0(x0h).cpu().numpy())
        x = ddpm_step(eps, t, x) if args.center == 'all' else sch.step(eps, t, x, generator=generator).prev_sample
        if projection is not None :
            x = projection(x)
        if args.border_noise :
            a_prev = sch.alphas_cumprod[t - 1].to(device) if t > 0 else torch.tensor(1., device=device)
            x = x * ocean + (1 - a_prev).sqrt() * torch.randn(x.shape, generator=generator, device=device) * (1 - ocean)
tag = ('border0' if args.border0 else 'bnoise' if args.border_noise else 'raw') + ('' if args.center == 'none' else f'_c{args.center}')
np.savez(HERE / 'runs' / f'traj_{tag}.npz', t=np.array(ts), T0=np.stack(rec), final=T0(x).cpu().numpy())
np.save(HERE / 'runs' / f'traj_{tag}_x.npy', x.cpu().numpy())
print(tag, 'final T0', T0(x).cpu().numpy().round(2))
