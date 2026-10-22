"""Bias indicators of the 8-state samples generated during training (epoch_*.npy) and of the inference runs."""
from load import *
from glob import glob
data = load_data(200); clim = data.mean(0)
dlm = level_mean_norm(data)
def indicators(x):
    a = (x - clim)[:, ocean_n]; a = a - a.mean(1, keepdims=True); a /= np.linalg.norm(a, axis=1, keepdims=True)
    C = a @ a.T; od = C[~np.eye(len(C), dtype=bool)].mean()
    lm = level_mean_norm(x)
    zz = (lm.mean(0) - dlm.mean(0)) / np.sqrt(lm.var(0, ddof=1) / len(lm) + dlm.var(0, ddof=1) / len(dlm))
    return od, np.sqrt(np.mean(zz**2)), np.abs(zz).max(), lm.std(0).mean(), np.mean(np.abs(x[:, ocean_n]) >= 0.999), np.abs(x[:, ~ocean_n & ~pad_n]).mean()
print('name n corr_offdiag rms_z max|z| mean_levelmean_std sat_frac land|x|')
rows = []
def show(name, x):
    r = indicators(x); rows.append((name, len(x)) + r)
    print(f'{name:40s} {len(x):4d} ' + ' '.join(f'{v:.3f}' for v in r))
rng = np.random.default_rng(1)
show('data_8', data[:8]); show('data_8b', data[8:16])
old = np.load(RUNS['old200_a'], mmap_mode='r')
for i in range(4): show(f'old200_subset8_{i}', np.array(old[i*8:(i+1)*8]))
show('old200_all', np.array(old))
show('old8_jan30', np.load(RUNS['old8_jan30'])); show('cur_none', np.load(RUNS['cur_none'])); show('cur_none_first8', np.load(RUNS['cur_none'])[:8]); show('cur_none_last8', np.load(RUNS['cur_none'])[8:])
for p in sorted(glob(f'{INF}/infesteps_1000/*.npy')) + sorted(glob(f'{INF}/infesteps_1000/constraints_no_constraints/*.npy')):
    x = np.load(p, mmap_mode='r'); show(p.split('inference/')[1][-45:], np.array(x[:200]))
for p in sorted(glob(f'{INF}/*/none*/*/generated.npy')) + sorted(glob(f'{INF}/*/none*/generated.npy')):
    show(p.split('inference/')[1][:40], np.load(p))
eps = sorted(glob(f'{os.path.dirname(INF)}/epoch_*.npy'), key=lambda s: int(s.split('_')[-1][:-4]))
for p in eps[::5] + eps[-6:]:
    show(os.path.basename(p), np.load(p))
import pickle; pickle.dump(rows, open('bias_indicators.pkl', 'wb'))
