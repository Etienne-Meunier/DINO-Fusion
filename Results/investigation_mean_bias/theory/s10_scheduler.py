"""Scheduler numbers: terminal alpha_bar, SNR, how much per-level mean survives at t=T, DC-mode SNR along t."""
import numpy as np, torch
from diffusers import DDPMScheduler
sch = DDPMScheduler.from_pretrained('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/scheduler')
ab = sch.alphas_cumprod.double().numpy(); b = sch.betas.double().numpy()
print('beta max %.4f (clip 0.999), beta[0] %.2e' % (b.max(), b[0]))
for t in [0, 1, 10, 100, 250, 500, 750, 900, 950, 990, 995, 998, 999]:
    print(f't={t:4d} abar={ab[t]:.3e} sqrt(abar)={np.sqrt(ab[t]):.3e} SNR={ab[t]/(1-ab[t]):.3e} x0-amplification sqrt((1-abar)/abar)={np.sqrt((1-ab[t])/ab[t]):.3e}')
sch.set_timesteps(1000); print('inference timesteps first/last', sch.timesteps[:3].tolist(), sch.timesteps[-3:].tolist(), len(sch.timesteps))
N = 13312; Nocean_typ = None
# DC (spatial-mean) mode: data DC std per channel (normalized) ~ s_m, noise DC std = sqrt(1-abar)/sqrt(N)
for s_m in [0.0003, 0.003, 0.009]:
    snr_dc = ab * s_m**2 / ((1 - ab) / N)
    print(f'DC-mode SNR with data DC std {s_m}: t=0 {snr_dc[0]:.2e}, t=500 {snr_dc[500]:.2e}, t=999 {snr_dc[999]:.2e}; t where DC SNR=1: {np.argmin(np.abs(snr_dc-1))}')
# Lin et al leak: fraction of x0 DC in x_T DC
print('signal in x_T: sqrt(abar_T) = %.3e; per-pixel SNR_T = %.2e; DC leak of a 0.3 offset = %.2e vs noise DC std %.2e' % (np.sqrt(ab[999]), ab[999] / (1 - ab[999]), 0.3 * np.sqrt(ab[999]), 1 / np.sqrt(N)))
# zero-SNR rescaled betas for comparison
from diffusers.schedulers.scheduling_ddpm import rescale_zero_terminal_snr
print('rescaled zero-snr abar_T:', rescale_zero_terminal_snr(sch.betas.clone()).pipe(lambda bb: torch.cumprod(1 - bb, 0))[-1].item() if hasattr(torch.Tensor,'pipe') else torch.cumprod(1 - rescale_zero_terminal_snr(sch.betas.clone()), 0)[-1].item())
np.save('alphas_cumprod.npy', ab)
