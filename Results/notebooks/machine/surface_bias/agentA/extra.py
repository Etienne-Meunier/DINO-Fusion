"""Where the remaining inversions of each run sit (volume share by level band, eval weighting) and b16/b8 overlap check."""
from pathlib import Path
import numpy as np
R = np.load(Path(__file__).resolve().parent / 'measure.npz')
V = R['volume']
print('b16 first 8 members == b8 :', np.allclose(R['none_b16/sig'][:8], R['none_b8/sig'], equal_nan=True),
      np.allclose(R['eta100_b16/sig'][:8], R['eta100_b8/sig'], equal_nan=True))
print('run        | inv vol % (pairs 0-1, 1-9, 10-24, 25-35) | mean -g on inverted pairs 0-1 | loss share of pair 0-1 (%)')
for n in ['data', 'none_b8', 'eta10_b8', 'eta30_b8', 'eta100_b8', 'eta300_b8']:
    s = R[f'{n}/sig'].astype(np.float64)
    g = np.nan_to_num(s[:, 1:] - s[:, :-1])
    w = V[1:] / V[1:].sum()
    hinge = np.maximum(-g, 0) * w
    inv = (g < 0) * w
    parts = [inv[:, a:b].sum((1, 2, 3)).mean() * 100 for a, b in [(0, 1), (1, 9), (9, 24), (24, 35)]]
    print(f'{n:10s} | ' + ' '.join(f'{p:6.2f}' for p in parts) +
          f' | {np.nanmean(np.where(g[:, 0] < 0, -g[:, 0], np.nan)):.4f} | {100 * hinge[:, 0].sum() / hinge.sum():.1f}')
