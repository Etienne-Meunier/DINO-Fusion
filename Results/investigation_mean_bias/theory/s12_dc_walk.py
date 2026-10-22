"""Linear model of the DC (channel spatial mean) along DDPM ancestral sampling, using the measured denoiser DC response.
eps_hat_DC = k_t x_t_DC / sqrt(1-abar_t)  (k measured in s11, ideal Bayes k from a pinned data DC of std s_m)."""
import numpy as np
from load import ocean_n, pad_n
ab = np.load('alphas_cumprod.npy'); T = len(ab)
R = np.load('dc_response.npy')  # t, dcerr, patt, land, gain_x0, 1/sqrt(abar), gain_eps
k_meas = np.interp(np.arange(T), R[:, 0], R[:, 6])
N = 208 * 64
f_ocean = ocean_n.reshape(37, -1).mean(1); print('ocean fraction of the 208x64 grid per channel: min %.2f median %.2f max %.2f' % (f_ocean.min(), np.median(f_ocean), f_ocean.max()))
def bayes_k(s_m):
    # E[eps_DC | x_DC] for x_DC = sqrt(ab) m + sqrt(1-ab) n, m~N(0,s_m^2), n~N(0,1/N)
    vs, vn = ab * s_m**2, (1 - ab) / N
    return vn / (vs + vn)
def walk(k, n=20000, border=None, clip=True, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) / np.sqrt(N)
    for t in range(T - 1, -1, -1):
        abp = ab[t - 1] if t > 0 else 1.0
        beta = 1 - ab[t] / abp
        x0 = x * (1 - k[t]) / np.sqrt(ab[t])
        if clip: x0 = np.clip(x0, -1, 1)
        mu = np.sqrt(abp) * beta / (1 - ab[t]) * x0 + np.sqrt(1 - beta) * (1 - abp) / (1 - ab[t]) * x
        var = (1 - abp) / (1 - ab[t]) * beta if t > 0 else 0.0
        x = mu + np.sqrt(var) * rng.standard_normal(n) / np.sqrt(N)
        if border is not None: x = x * border   # land/pad set to 0 keeps only the ocean fraction of the DC
    return x
for name, k, bd in [('measured k (local unet)', k_meas, None), ('measured k + border0 projection (ocean frac 0.75)', k_meas, 0.75),
                    ('Bayes k, data DC std 0.003', bayes_k(0.003), None), ('k = 0 (DC fully treated as signal)', np.zeros(T), None)]:
    x = walk(k, border=bd)
    print(f'{name:50s} final DC std {x.std():.4f}   frac |DC|>=0.99 {np.mean(np.abs(x) >= 0.99):.3f}')
