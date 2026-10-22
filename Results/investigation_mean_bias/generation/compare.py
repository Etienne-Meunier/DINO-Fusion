"""Identity checks between runs and per-state level-mean T, plus bootstrap of 16-state means from the 200-state Feb 2025 run."""
import numpy as np
I = '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference'
L = np.load('run_stats_levels.npz'); d = np.load('data_level_means.npz')['toce']; depth = L['depth']
K = [int(np.abs(depth - z).argmin()) for z in (26, 74, 230, 2892)]
ld = lambda p: np.load(f'{I}/{p}', mmap_mode='r')
a = ld('infesteps_1000/20250129-180927.npy'); b = ld('archived/new_constraints/none/generated.npy')
c = ld('optimal/none/151026-221621/generated.npy'); e = ld('infesteps_1000/constraints_no_constraints/20250203-141645.npy')
f = ld('infesteps_1000/constraints_no_constraints/20250203-171119.npy'); g = ld('test_new_sampler/none/generated.npy')
h = ld('infesteps_1000/constraints_no_constraints/20250203-140245.npy')
print('jan29 vs archived none max|diff|', np.abs(a - b).max())
print('jan29 vs optimal/none[:8] max|diff|', np.abs(a - c[:8]).max())
print('feb3 141645 vs 171119 max|diff|', np.abs(e - f).max())
print('feb3 140245 vs 141645[:8]', np.abs(h - e[:8]).max())
print('test_new_sampler vs jan29', np.abs(g - a).max())
key = lambda k: k.replace('/', '__')
for name in ['optimal/none/151026-221621/generated.npy', 'infesteps_1000/constraints_no_constraints/20250203-141645.npy',
             'optimal/none_border0/151026-150424/generated.npy']:
    t = L[key(name)]
    print('\n', name, 'per-state T at', [f'{depth[k]:.0f}m' for k in K])
    for i in range(min(len(t), 16)): print('  ', i, np.round(t[i, K], 2))
    print('  std over states', np.round(t[:, K].std(0), 2), ' data mean', np.round(d.mean(0)[K], 2), 'data std', np.round(d.std(0)[K], 3))
t200 = L[key('infesteps_1000/constraints_no_constraints/20250203-141645.npy')]
tc = L[key('optimal/none/151026-221621/generated.npy')]
rng = np.random.default_rng(0)
bm = np.array([t200[rng.choice(200, 16, replace=False)][:, K].mean(0) for _ in range(20000)])
obs = tc[:, K].mean(0)
print('\nbootstrap 16-state means from Feb-2025 200 run: mean', np.round(bm.mean(0), 2), 'sd', np.round(bm.std(0), 2))
print('current 16-state run means', np.round(obs, 2), 'z vs bootstrap', np.round((obs - bm.mean(0)) / bm.std(0), 2))
print('two-sided frac as extreme per level', np.round([np.mean(np.abs(bm[:, j] - bm[:, j].mean()) >= abs(obs[j] - bm[:, j].mean())) for j in range(4)], 4))
