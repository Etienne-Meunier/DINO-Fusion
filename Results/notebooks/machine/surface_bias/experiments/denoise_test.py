"""
Does the denoiser shift the surface mean on its own training data ?

8 training states (normalised, BorderZero layout) are noised to timestep t with a fixed seed ; one call of the unet gives
x0_hat = (x_t - sqrt(1 - abar) eps_hat) / sqrt(abar). Printed per t : mean over the 8 states of the volume-weighted
level-0 T of x0_hat minus the true one, unclamped and clamped to the scheduler clip range (+/-1), and the share of
ocean level-0 values of the data beyond +/-1.
"""
from exp import *
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from DataLoader import DATA_SEED, get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer

device = Accelerator().device
pipeline = DDPMPipeline_Tensor.from_pretrained(MODEL).to(device)
config = TrainingConfig()
obs = Observer('data', config, device=device)
transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device=device)
x0 = next(iter(get_dataloader(config.data_file, transform, batch_size=8, shuffle=True, seed=DATA_SEED))).to(device)
sch = pipeline.scheduler


def T0(x) :
    """Level-0 area-weighted mean T [C] of each member of a normalised batch."""
    t = obs.unnormalized(x)['toce'][:, 0]
    w = torch.as_tensor(np.load(HERE / 'area0.npy'), device=t.device, dtype=t.dtype)
    return (torch.nan_to_num(t) * w).sum((-2, -1)) / w.sum()


lev0 = x0[:, 0]
print(f'data level-0 normalised T : min {lev0.min():.2f} max {lev0.max():.2f}, share |x|>1 {(lev0.abs() > 1).float().mean():.4f}')
ref = T0(x0)
print(f'data T0 {ref.mean():.3f}')
g = torch.Generator(device).manual_seed(0)
for t in [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 990] :
    noise = torch.randn(x0.shape, generator=g, device=device)
    abar = sch.alphas_cumprod[t].to(device)
    xt = abar.sqrt() * x0 + (1 - abar).sqrt() * noise
    with torch.no_grad() :
        eps = pipeline.unet(xt, t).sample
    x0h = (xt - (1 - abar).sqrt() * eps) / abar.sqrt()
    d, dc = T0(x0h) - ref, T0(x0h.clamp(-1, 1)) - ref
    print(f't {t:4d} sqrt(abar) {abar.sqrt():.3f} : dT0 x0_hat {d.mean():+7.3f} +/- {d.std():.3f} | clamped {dc.mean():+7.3f} +/- {dc.std():.3f}')
