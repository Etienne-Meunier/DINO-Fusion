"""Inter-state correlation of anomalies (state minus data climatology) in normalized space, ocean cells only."""
from load import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
data = load_data(200)
clim = data.mean(0)
def corrmat(x):
    a = (x - clim)[:, ocean_n]
    a = a - a.mean(1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a @ a.T
def offdiag(C): return C[~np.eye(len(C), dtype=bool)]
old = np.load(RUNS['old200_a'], mmap_mode='r')
cur = np.load(RUNS['cur_none']); b0 = np.load(RUNS['cur_none_border0'])
for name, x in [('data[:16]', data[:16]), ('old200[:16]', np.array(old[:16])), ('old200[100:116]', np.array(old[100:116])), ('cur_none', cur), ('cur_none_border0', b0)]:
    C = corrmat(x); od = offdiag(C)
    print(f'{name:18s} anomaly corr offdiag: mean {od.mean():.3f} median {np.median(od):.3f} max {od.max():.3f}')
    if name.startswith('cur'):
        print(C)
        print('  diag i,i+8:', np.array([C[i, i+8] for i in range(8)]))
