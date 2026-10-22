"""Clip bias on data, +1/-1 saturation asymmetry, PCA of per-state level-mean offsets, land vs level-mean link."""
from load import *
np.set_printoptions(precision=3, suppress=True, linewidth=220)
info = transform.concatenated_infos; sd3 = 3 * info['std'].numpy().ravel()
data = load_data(200)
lm = level_mean_norm(data); lmc = level_mean_norm(np.clip(data, -1, 1))
cb = (lmc - lm).mean(0)
print('clip bias on data level means (normalized):', cb)
print('clip bias physical (C or g/kg) for T channels:', (cb * sd3)[:18], '\n  S:', (cb * sd3)[18:36])
for k in ['old200_a', 'cur_none', 'cur_none_border0']:
    x = np.load(RUNS[k], mmap_mode='r'); x = np.array(x[:200])
    pos = np.array([np.mean(x[:, c][:, ocean_n[c]] >= 0.999) for c in range(37)]); neg = np.array([np.mean(x[:, c][:, ocean_n[c]] <= -0.999) for c in range(37)])
    print(f'\n{k}: overall frac +1 {pos.mean():.4f}  -1 {neg.mean():.4f}')
    L = level_mean_norm(x) - lm.mean(0)
    # PCA of state level-mean offsets
    Lc = L - L.mean(0); u, s, vt = np.linalg.svd(Lc, full_matrices=False); ev = s**2 / (s**2).sum()
    print(' PCA of per-state level-mean offsets: explained var', ev[:5].round(3))
    print(' PC1 loadings T:', vt[0, :18].round(2), '\n             S:', vt[0, 18:36].round(2), 'ssh', vt[0, 36].round(2))
    # adjacent-level correlation of offsets
    adj = [np.corrcoef(L[:, c], L[:, c + 1])[0, 1] for c in range(17)]
    print(' corr of T offsets between adjacent strided levels:', np.array(adj).round(2))
    print(' corr T offset vs S offset same level:', np.array([np.corrcoef(L[:, c], L[:, c + 18])[0, 1] for c in range(18)]).round(2))
    landabs = np.array([np.abs(s[~ocean_n & ~pad_n]).mean() for s in x])
    landmean = np.array([s[~ocean_n & ~pad_n].mean() for s in x])
    print(' corr(state land mean, state mean level-mean offset over T channels) %.2f ; corr(|land|, RMS level offset) %.2f' % (np.corrcoef(landmean, L[:, :18].mean(1))[0, 1], np.corrcoef(landabs, np.sqrt((L**2).mean(1)))[0, 1]))
    # coastal ring: ocean cells adjacent to land, vs interior: level-mean offset contributions
