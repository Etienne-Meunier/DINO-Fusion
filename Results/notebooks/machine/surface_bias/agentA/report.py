"""Tables of the surface density bias (questions 1-5) from `measure.npz`. Prints to stdout; tee into report.txt."""
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
R = np.load(HERE / 'measure.npz')
V, depth, lat = R['volume'], R['depth'], R['lat']
TOP = slice(0, 5)        # levels 0-4, T points 5-49 m, cells down to ~56 m
B8 = sorted([k[:-4] for k in R.files if k.startswith('eta') and k.endswith('_b8/sig')], key=lambda n: float(n[3:-3]))
B16 = sorted([k[:-4] for k in R.files if k.startswith('eta') and k.endswith('_b16/sig')], key=lambda n: float(n[3:-4]))
base = lambda n: 'none_b8' if n.endswith('_b8') else 'none_b16'


def hmean(s, levels=None):
    """Volume-weighted horizontal mean (b, z) of s (b, z, y, x); with `levels` a slice, the volume mean over those levels (b,)."""
    w = V if levels is None else V[levels]
    s = s if levels is None else s[:, levels]
    ax = (-2, -1) if levels is None else (-3, -2, -1)
    return np.nansum(s * w, axis=ax) / w.sum(axis=ax[-2:] if levels is None else ax)


def ms(x):
    return f'{x.mean():+.4f} +/- {x.std(ddof=1) / np.sqrt(len(x)):.4f}'


sig = {k[:-4]: R[k].astype(np.float64) for k in R.files if k.endswith('/sig')}
H = {n: hmean(s) for n, s in sig.items()}
print('levels 0-4 T-point depths', depth[TOP].round(1))

print('\n== Q1: horizontal-mean sigma_0 (kg/m3) per level: mean (member std) ==')
cols = ['data', 'none_b8'] + B8
print('lev depth ' + ' '.join(f'{c:>16s}' for c in cols))
for k in list(range(10)) + list(range(10, 35, 3)):
    print(f'{k:3d} {depth[k]:6.1f} ' + ' '.join(f'{H[c][:, k].mean():8.3f}({H[c][:, k].std(ddof=1):.3f})' for c in cols))
print('\npaired diff eta - none (b8), mean +/- SE over 8 members, per level')
print('lev depth ' + ' '.join(f'{c:>17s}' for c in B8))
for k in list(range(10)) + list(range(10, 35, 3)):
    print(f'{k:3d} {depth[k]:6.1f} ' + ' '.join(f'{ms(H[c][:, k] - H["none_b8"][:, k]):>17s}' for c in B8))

print('\n== Q2: surface bias vs eta ==')
print('run           eta | lev0 eta-none(paired) | lev0 eta-data (SE) | top50 eta-none | top50 eta-data | lbd mean [min,max] | loss/eps mean, max, #sat')
d_data0, d_data50 = H['data'][:, 0], hmean(sig['data'], TOP)
for n in ['none_b8'] + B8 + ['none_b16'] + B16:
    h0, h50 = H[n][:, 0], hmean(sig[n], TOP)
    vd0 = f'{h0.mean() - d_data0.mean():+.4f} ({np.sqrt(h0.var(ddof=1) / len(h0) + d_data0.var(ddof=1) / len(d_data0)):.4f})'
    vd50 = f'{h50.mean() - d_data50.mean():+.4f}'
    if n.startswith('none'):
        pn0 = pn50 = '-'
        lb = ls = '-'
    else:
        pn0, pn50 = ms(h0 - H[base(n)][:, 0]), ms(h50 - hmean(sig[base(n)], TOP))
        l, loss, eps = R[f'{n}/lbd'], R[f'{n}/loss'], float(R[f'{n}/eps'])
        lb = f'{l.mean():8.1f} [{l.min():.1f},{l.max():.1f}]'
        ls = f'{loss.mean() / eps:.2f}, {loss.max() / eps:.2f}, {(loss <= eps).sum()}/{len(loss)}'
    print(f'{n:12s} {float(R[f"{n}/eta"]):5g} | {pn0:>21s} | {vd0:>18s} | {pn50:>17s} | {vd50:>8s} | {lb:>22s} | {ls}')
for n in ['data', 'none_b8', 'none_b16']:
    loss = R[f'{n}/loss']
    print(f'loss/eps {n}: mean {loss.mean() / 8.6e-8:.2f}, max {loss.max() / 8.6e-8:.2f}, median {np.median(loss) / 8.6e-8:.2f}')
eta = np.array([float(R[f'{n}/eta']) for n in B8])
d0 = np.array([(H[n][:, 0] - H['none_b8'][:, 0]).mean() for n in B8])
lb = np.array([R[f'{n}/lbd'].mean() for n in B8])
print('Spearman(eta, d lev0) =', stats.spearmanr(eta, d0)[0].round(3), ' Spearman(mean lbd, d lev0) =', stats.spearmanr(lb, d0)[0].round(3))
print('fit d0 = a + b*log10(eta):', np.polyfit(np.log10(eta), d0, 1)[::-1].round(4))
print('fit d0 = a + b*eta (eta<=100):', np.polyfit(eta[eta <= 100], d0[eta <= 100], 1)[::-1].round(5))

print('\n== Q3: T/S split of the paired surface change (hybrids, sigma(T_a,S_b)) ==')
print('run         | lev0 total | from T  | from S  | resid  | dT lev0 (C) | dS lev0 (psu) || top50 total | from T | from S')
for n in B8 + B16:
    b = base(n)
    tot = sig[n] - sig[b]
    fT = R[f'{n}/sig_Teta_Snone'] - sig[b]
    fS = R[f'{n}/sig_Tnone_Seta'] - sig[b]
    dT = hmean(R[f'{n}/T'] - R[f'{b}/T'])[:, 0]
    dS = hmean(R[f'{n}/S'] - R[f'{b}/S'])[:, 0]
    a = [hmean(x)[:, 0].mean() for x in (tot, fT, fS)]
    t = [hmean(x, TOP).mean() for x in (tot, fT, fS)]
    print(f'{n:11s} | {a[0]:+.4f} | {a[1]:+.4f} | {a[2]:+.4f} | {a[0] - a[1] - a[2]:+.4f} | {dT.mean():+.4f} | {dS.mean():+.5f} || '
          f'{t[0]:+.4f} | {t[1]:+.4f} | {t[2]:+.4f}')

print('\n== Q4: where. Zonal-band mean of the paired change (ensemble mean), lev0 and 0-500 m (levels 0-20) ==')
bands = [(-70, -50), (-50, -30), (-30, -10), (-10, 10), (10, 30), (30, 50), (50, 70)]
for n in ['eta10_b8', 'eta100_b8', 'eta300_b8', 'eta100_b16']:
    d = (sig[n] - sig[base(n)]).mean(0)
    line = []
    for lo, hi in bands:
        m = (lat >= lo) & (lat < hi)
        w0 = V[0][m]
        w5 = V[:21][:, m]
        line.append(f'{np.nansum(d[0][m] * w0) / w0.sum():+.3f}/{np.nansum(d[:21][:, m] * w5) / w5.sum():+.3f}')
    print(f'{n:11s} ' + ' '.join(f'[{lo},{hi}] {s}' for (lo, hi), s in zip(bands, line)))
print('\ncolumn budget: volume-weighted mean paired d sigma by depth range, and its share of the total volume integral')
rng = [(0, 5), (5, 12), (12, 21), (21, 27), (27, 36)]
print('ranges (m):', [f'{depth[a]:.0f}-{depth[b - 1]:.0f}' for a, b in rng])
for n in B8 + B16:
    d = sig[n] - sig[base(n)]
    parts = [np.nansum(d[:, a:b] * V[a:b], axis=(1, 2, 3)) for a, b in rng]
    tot = sum(parts)
    print(f'{n:11s} ' + ' '.join(f'{hmean(d, slice(a, b)).mean():+.4f}' for a, b in rng) +
          f' | whole-ocean mean d sigma {ms(tot / V.sum())} | data-none whole-ocean {np.nansum((sig["data"].mean(0) - sig[base(n)].mean(0)) * V) / V.sum():+.4f}')

print('\n== Q5: surface change vs inversions of the paired none member ==')
for b in ['none_b8', 'none_b16']:
    s = sig[b]
    g = s[:, 1:10] - s[:, :9]
    inv = np.where(np.isnan(g), 0, np.minimum(g, 0))
    invvol = (np.where(inv < 0, 1, 0) * V[1:10]).sum((1, 2, 3)) / V[1:10].sum()
    invmag = (-inv * V[1:10]).sum((1, 2, 3)) / V[1:10].sum()
    colmag = -inv.sum(1)
    print(f'{b}: inversion volume fraction levels 0-9 per member', invvol.round(3), ' mean |g| weighted', invmag.round(4))
    for n in (B8 if b == 'none_b8' else B16):
        d0 = H[n][:, 0] - H[b][:, 0]
        dc = (sig[n] - s)[:, 0]
        ok = ~np.isnan(dc)
        r_col = stats.pearsonr(colmag[ok], dc[ok])[0]
        has = colmag > 0
        print(f'  {n:11s} r(member invvol, d lev0)={stats.pearsonr(invvol, d0)[0]:+.2f} r(member invmag, d lev0)={stats.pearsonr(invmag, d0)[0]:+.2f}'
              f' | per column r={r_col:+.2f}, d lev0 where none col inverted {np.nanmean(dc[has & ok]):+.4f} (n={int((has & ok).sum())}) vs not {np.nanmean(dc[~has & ok]):+.4f}')
gd = sig['data'][:, 1:10] - sig['data'][:, :9]
print('data inversion volume fraction levels 0-9:', ((np.nan_to_num(gd) < 0) * V[1:10]).sum((1, 2, 3)).mean() / V[1:10].sum())
