"""(a)(b): per-state spread of T level means and z of 16-member means, old200 random subsets vs current run."""
import numpy as np, xarray as xr
np.set_printoptions(precision=2, suppress=True, linewidth=220)
P = np.load('phys_levelmeans.npz')
depth = xr.open_dataset('/Users/emeunier/Desktop/Projets/DINO-Fusion/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').gdept_1d.values.ravel()
lv = [i for i in range(35)]
D, O, C, B = (P[f'{k}_toce'][:, :35] for k in ['data', 'old200_a', 'cur_none', 'cur_none_border0'])
print('levels (m):', depth[[2,6,11,15,20,25,30,34]].round(0))
sel = [2,6,11,15,20,25,30,34]
for name, X in [('data', D), ('old200', O), ('cur16', C), ('cur_border0', B)]:
    print(f'{name:12s} ens mean', X.mean(0)[sel], '\n             ens std ', X.std(0, ddof=1)[sel])
print('spread ratio old200/data', (O.std(0)/D.std(0))[sel]); print('spread ratio cur/data   ', (C.std(0)/D.std(0))[sel])
def z(X, D):
    return (X.mean(0) - D.mean(0)) / np.sqrt(X.var(0, ddof=1)/len(X) + D.var(0, ddof=1)/len(D))
def rms_off(X, D): return np.sqrt(np.mean((X.mean(0) - D.mean(0))**2))
print('z cur16  :', z(C, D)[sel], ' max|z| %.1f  RMS offset %.2f C' % (np.abs(z(C, D)).max(), rms_off(C, D)))
print('z old200 :', z(O, D)[sel], ' max|z| %.1f  RMS offset %.2f C' % (np.abs(z(O, D)).max(), rms_off(O, D)))
rng = np.random.default_rng(0)
mz, rms, zall = [], [], []
for _ in range(5000):
    s = O[rng.choice(200, 16, replace=False)]
    zz = z(s, D); mz.append(np.abs(zz).max()); rms.append(rms_off(s, D)); zall.append(zz)
mz, rms, zall = np.array(mz), np.array(rms), np.array(zall)
print('old200 16-subsets: max|z| median %.2f p95 %.2f p99 %.2f ; frac >= cur(%.1f): %.4f' % (np.median(mz), *np.percentile(mz, [95, 99]), np.abs(z(C, D)).max(), np.mean(mz >= np.abs(z(C, D)).max())))
print('old200 16-subsets: per-level z std (should be ~1):', zall.std(0)[sel])
print('old200 16-subsets: RMS offset median %.2f p95 %.2f p99 %.2f max %.2f ; frac >= cur(%.2f): %.4f' % (np.median(rms), *np.percentile(rms, [95, 99]), rms.max(), rms_off(C, D), np.mean(rms >= rms_off(C, D))))
# (b) per-state: how many states individually offset in same direction
for k in [2, 6, 15, 30]:
    d = C[:, k] - D[:, k].mean()
    print(f'cur level {depth[k]:.0f} m: per-state offset from data mean (C):', d.round(1), ' n>0: %d/16' % (d > 0).sum())
    do = O[:, k] - D[:, k].mean()
    print(f'   old200 same level: frac>0 %.2f, offset quartiles' % (do > 0).mean(), np.percentile(do, [5, 25, 50, 75, 95]).round(1))
