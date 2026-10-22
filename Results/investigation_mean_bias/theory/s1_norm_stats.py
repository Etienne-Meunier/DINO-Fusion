"""Normalized-space stats: mean/std infos, data range, clip saturation, land/padding values, level means per run."""
from load import *
np.set_printoptions(precision=4, suppress=True, linewidth=200)
info = transform.concatenated_infos
mu = info['mean'].numpy().ravel(); sd = info['std'].numpy().ravel()
print('channel  depth  mean  std  3std')
for c in [0,2,4,6,8,10,13,17,18,22,26,35,36]:
    print(CH[c], f'{DEPTH_CH[c]:7.1f} {mu[c]:8.3f} {sd[c]:8.3f} {3*sd[c]:8.3f}')
data = load_data(200)
np.save('data200_norm_levelmeans.npy', level_mean_norm(data))
runs = {'data': data} | {k: np.load(p) for k, p in RUNS.items()}
res = {}
for k, x in runs.items():
    oc = x[:, ocean_n]; ld = x[:, ~ocean_n & ~pad_n]; pd = x[:, pad_n]
    lm = level_mean_norm(x)
    res[k] = lm
    print(f'\n== {k} n={len(x)}')
    print(f' ocean: min {oc.min():.3f} max {oc.max():.3f} frac|x|>1 {np.mean(np.abs(oc)>1):.2e} frac|x|>=0.999 {np.mean(np.abs(oc)>=0.999):.2e}')
    print(f' land: mean|x| {np.abs(ld).mean():.4f} max|x| {np.abs(ld).max():.3f};  pad: mean|x| {np.abs(pd).mean():.4f} max|x| {np.abs(pd).max():.3f}')
    per_ch_sat = np.array([np.mean(np.abs(x[:, c][:, ocean_n[c]]) >= 0.999) for c in range(37)])
    print(' frac sat >=0.999 per channel (T0..17 | S0..17 | ssh):', per_ch_sat)
    print(' level-mean (normalized) ensemble mean:', lm.mean(0))
    print(' level-mean (normalized) ensemble std :', lm.std(0))
    # per-state saturation and land
    sat_state = np.array([np.mean(np.abs(s[ocean_n]) >= 0.999) for s in x])
    land_state = np.array([np.abs(s[~ocean_n & ~pad_n]).mean() for s in x])
    print(' per-state sat frac: median %.2e max %.2e ; per-state land mean|x|: median %.4f max %.4f' % (np.median(sat_state), sat_state.max(), np.median(land_state), land_state.max()))
np.savez('norm_levelmeans.npz', **res)
