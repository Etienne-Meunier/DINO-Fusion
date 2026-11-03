"""
Build `profile_board.html` : volume-weighted horizontal-mean profiles of sigma_0, T and S (ensemble mean and member std),
and the zonal sigma_0 sections of `paper/fig1_density_profile.py` (median member by RMSE to the data batch-average
section, batch average, profile over the BAND latitudes as an inset),
of the data batch, the unconstrained run, the C7 eta sweep and the experimental runs of runs/, with the metrics of
`evaluate.py` (bias to the data, level-0 T and S, eval density error E, C7 loss / eps, final lbd). Opens with the local
plotly of Results/notebooks/machine.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
from common import *
from metrics import compute_density_error

INF = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b/inference'
SWEEP = f'{INF}/sweep_density_huber/negative_grad_density_huber_border0'
NONE = f'{INF}/sweep_baselines/none_border0_b8/151026-145216/generated.npy'
OUT = HERE / 'profile_board.html'
OUT_WEB = HERE / 'profile_board_web.html'
PLOTLY_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.0.1/plotly.min.js'

# (run dir name, label, group, shown at load, colour)
RUNS = [('profile_eta30', 'profile hinge, eta 30', 'Profile hinge', False, '#e8a0a8'),
        ('profile_eta100', 'profile hinge, eta 100', 'Profile hinge', True, '#d1495b'),
        ('topfree_eta30', 'C7 surface stop-grad, eta 30', 'C7 surface stop-grad', False, '#c9b3e6'),
        ('topfree_eta100', 'C7 surface stop-grad, eta 100', 'C7 surface stop-grad', False, '#7a5cb8'),
        ('proftop_eta100', 'profile (100) + C7 stop-grad (100)', 'Combinations', True, '#2b8a8f'),
        ('profc7_eta100_30', 'profile (100) + C7 (30)', 'Combinations', False, '#7fc4c7'),
        ('c7surf_eta100_s1e-8', 'C7 100 + surface z, raw 1e-8', 'C7 + surface T/S', False, '#f2c27a'),
        ('c7surf_eta100_sqrt1e-4', 'C7 100 + surface z, sqrt 1e-4', 'C7 + surface T/S', True, '#e07b00'),
        ('c7surf_eta100_sqrt1e-5', 'C7 100 + surface z, sqrt 1e-5', 'C7 + surface T/S', False, '#c49a3a'),
        ('c7zm0_eta100_0.1', 'C7 100 + surface TS, eta 0.1', 'C7 + surface T/S', False, '#9fd39a'),
        ('c7zm0_eta100_0.1_ton500', 'C7 100 + surface TS, eta 0.1, t < 500', 'C7 + surface T/S', False, '#c7e6a8'),
        ('c7zm0_eta100_1', 'C7 100 + surface TS, eta 1', 'C7 + surface T/S', True, '#3c9d4e'),
        ('c7surf_eta100_s1e-7', 'C7 100 + surface z, raw 1e-7 (unstable)', 'C7 + surface T/S', False, '#b0a99f')]
STRENGTH = [10, 30, 50, 60, 70, 80, 90, 100, 200, 300]
RUNS += [(f'surf_c7{e:03d}_e1e-4' if e != 100 else 'c7surf_eta100_sqrt1e-4', f'C7 {e} + surface z, sqrt 1e-4',
          'Surface data: C7 strength', e in (10, 300),
          '#%02x%02x%02x' % tuple(int(255 * v) for v in plt.cm.YlOrBr(0.3 + 0.65 * i / (len(STRENGTH) - 1))[:3]))
         for i, e in enumerate(STRENGTH)]
RUNS += [('surf_c7100_e1e-4_eps1', 'surface z sqrt 1e-4, eps 1', 'Surface data: variants', False, '#f4a259'),
         ('surf_c7100_e1e-4_eps16', 'surface z sqrt 1e-4, eps 16', 'Surface data: variants', False, '#bc6c25'),
         ('surf_c7100_e1e-6', 'surface z sqrt 1e-6', 'Surface data: variants', False, '#dda15e'),
         ('surf_c7100_e3e-4', 'surface z sqrt 3e-4 (unstable)', 'Surface data: variants', False, '#a8a29e'),
         ('surf_c7100_e1e-4_s1', 'surface z sqrt 1e-4, seed 1', 'Surface data: variants', False, '#e76f51'),
         ('surf_c7100_e1e-4_s2', 'surface z sqrt 1e-4, seed 2', 'Surface data: variants', False, '#9c3d1f'),
         ('anchor_c7100_e1e-4', 'C7 + anchor', 'Anchor (no surface data)', False, '#8ecae6'),
         ('anchorrho_c7100_e1e-4_eps1', 'C7 + one-sided density anchor', 'Anchor (no surface data)', False, '#219ebc'),
         ('anchorT_c7100_e1e-4', 'C7 + anchor on T only', 'Anchor (no surface data)', False, '#4d908e'),
         ('anchorprof12_c7100_e1e-4', 'C7 + anchor on top 12 levels', 'Anchor (no surface data)', False, '#577590'),
         ('anchorprof36_c7100_e1e-4', 'C7 + anchor on all levels', 'Anchor (no surface data)', False, '#43aa8b'),
         ('anchorfinal_c7100_e1e-4', 'C7 + anchor to final none', 'Anchor (no surface data)', False, '#90be6d'),
         ('proftopanchor_e1e-4', 'profile + C7 stop-grad + anchor', 'Anchor (no surface data)', False, '#277da1'),
         ('skip_c7100', 'C7 without top pair, alone', 'Anchor (no surface data)', False, '#6d597a'),
         ('skipanchor_c7100_e1e-4', 'C7 without top pair + anchor, eps 1', 'Anchor (no surface data)', True, '#023e8a'),
         ('skipanchor_c7100_e1e-4_eps16', 'C7 without top pair + anchor, eps 16', 'Anchor (no surface data)', False, '#0077b6'),
         ('skipanchor_c7100_e1e-4_eps100', 'C7 without top pair + anchor, eps 100', 'Anchor (no surface data)', False, '#0096c7'),
         ('skipanchor_c7100_e1e-4_eps400', 'C7 without top pair + anchor, eps 400', 'Anchor (no surface data)', False, '#48cae4'),
         ('skipanchor_c7100_e1e-4_s1', 'C7 without top pair + anchor, seed 1', 'Anchor (no surface data)', False, '#3a0ca3'),
         ('skipanchor_c7100_e1e-4_s2', 'C7 without top pair + anchor, seed 2', 'Anchor (no surface data)', False, '#7209b7'),
         ('none_s1', 'unconstrained, seed 1', 'Seeds 1, 2', False, '#9a9a93'),
         ('none_s2', 'unconstrained, seed 2', 'Seeds 1, 2', False, '#6b6b66'),
         ('c7_c7100_s1', 'C7 eta 100, seed 1', 'Seeds 1, 2', False, '#c1121f'),
         ('c7_c7100_s2', 'C7 eta 100, seed 2', 'Seeds 1, 2', False, '#780000'),
         ('c7_c7200', 'C7 eta 200', 'C7 sweep', False, '#5e548e'),
         ('centred_none', 'new sampler (centred noise, noisy land), unconstrained', 'New sampler', True, '#111111'),
         ('centred_none_b32', 'new sampler, unconstrained, b32', 'New sampler', False, '#555555'),
         ('centred_c7010_b32', 'new sampler, C7 10, b32', 'New sampler', False, '#1f77b4'),
         ('centred_zm_e1_b32', 'new sampler, zero_mean 1, b32', 'New sampler', True, '#2ca02c'),
         ('centred_zm_e1_c7010_b32', 'new sampler, zero_mean 1 + C7 10, b32', 'New sampler', True, '#9467bd'),
         ('centred_surf_c7030_b32', 'new sampler, C7 30 + surface T,S mean, b32', 'New sampler', False, '#ff7f0e')]

mask = load_mask()
tmask = mask.tmask.values.astype(bool)
volume = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values
lv = volume.sum((1, 2))
depth = mask.gdepw_1d.values.squeeze() + mask.e3t_1d.values.squeeze() / 2
wet = lv > 0
import cmocean.cm as cm
BAND = (-65, -45)
LEVELS = (22., 28.5, 0.5)
PROFILE_RANGE = (25., 28.5)
DEPTH_POWER, DEPTH_MAX = 0.625, 3500
DEPTH_TICKS = [0, 100, 250, 500, 1000, 1500, 2000, 3000]
latitude = mask.gphit.isel(nav_lon=10).values
sec_depth = mask.gdept_1d.values[:-1]
c7 = get_loss('negative_grad_density_huber', observer)
EPS_C7 = eps_data[False]['negative_grad_density_huber']
LAYERS = {'b lev0' : slice(0, 1), 'b 5-49m' : slice(0, 5), 'b 61-299m' : slice(5, 12)}


def profiles(x) :
    """Volume-weighted horizontal-mean (b, z) profiles of sigma_0, T, S of a normalised batch x, NaN on dry levels."""
    f = observer.unnormalized(torch.as_tensor(x))
    T, S = (f[k].detach().numpy().astype(np.float64) for k in ['toce', 'soce'])
    sig = compute_density({'toce' : T, 'soce' : S}, mask).transpose('batch', 'depth', 'nav_lat', 'nav_lon').values - 1000
    out = {}
    for k, v in [('sig', sig), ('T', T), ('S', S)] :
        p = np.nansum(np.where(tmask[None], v, 0) * volume, axis=(2, 3)) / np.where(wet, lv, np.nan)
        out[k] = p
    out['sig_field'], out['T_field'], out['S_field'] = sig, T, S
    out['E'] = compute_density_error({'toce' : T, 'soce' : S}, mask)
    out['c7/eps'] = c7(torch.as_tensor(x)).numpy() / EPS_C7
    return out


def layer_means(sig) :
    """Volume-weighted mean sigma_0 (b,) of each layer of LAYERS."""
    return {k : np.nansum(np.where(tmask[None, sl], sig[:, sl], 0) * volume[sl], axis=(1, 2, 3)) / volume[sl].sum()
            for k, sl in LAYERS.items()}


def entry(name, label, group, on, x, lbd=None, color=None) :
    p = profiles(x)
    lm = layer_means(p['sig_field'])
    m = {k : float(v.mean() - DATA_LAYERS[k].mean()) for k, v in lm.items()} if DATA_LAYERS else {k : 0. for k in lm}
    m.update({'T0' : float(np.nanmean(p['T'][:, 0])), 'S0' : float(np.nanmean(p['S'][:, 0])), 'E' : float(p['E'].mean()),
              'c7/eps' : float(p['c7/eps'].mean())})
    if lbd is not None :
        m['lbd'] = ' / '.join(f'{v:.3g}' for v in np.atleast_2d(lbd.T).T.mean(0))
    e = dict(name=name, label=label, group=group, on=on, n=len(x), metrics=m, color=color)
    if name == 'data' :
        DATA_LAYERS.update(lm)
    sec = zonal_sections(torch.as_tensor(x), mask)
    mean = np.nanmean(sec, axis=0)
    if name == 'data' :
        DATA_SECTION.append(mean)
    err = rmse(sec, DATA_SECTION[0])
    med = median_member(err)
    e['sec'] = dict(mean=to_list(mean), median=to_list(sec[med]), m=med, rmse=float(err[med]), E=float(p['E'][med]))
    for k, v in ((k, p[k]) for k in ['sig', 'T', 'S']) :
        e[k] = dict(mean=np.round(np.nanmean(v, 0)[wet], 5).tolist(), std=np.round(np.nanstd(v, 0)[wet], 5).tolist(),
                    members=np.round(v[:, wet], 4).tolist())
    return e


def to_list(a) :
    return [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in a]


DATA_LAYERS, DATA_SECTION = {}, []
stretch = lambda d : np.asarray(d) ** DEPTH_POWER
D = dict(depth=np.round(depth[wet], 1).tolist(), runs=[],
         sec=dict(lat=[round(float(v), 3) for v in latitude], y=[round(float(v), 4) for v in stretch(sec_depth)],
                  depth=[round(float(v), 1) for v in sec_depth], band=list(BAND),
                  band_mask=((latitude >= BAND[0]) & (latitude < BAND[1])).tolist(), levels=list(LEVELS),
                  profile_range=list(PROFILE_RANGE), y_max=float(stretch(DEPTH_MAX)), ticks=DEPTH_TICKS,
                  tickvals=stretch(DEPTH_TICKS).round(4).tolist(),
                  colorscale=[[i / 10, '#%02x%02x%02x' % tuple(int(255 * v) for v in cm.deep_r(i / 10)[:3])] for i in range(11)]))
D['runs'].append(entry('data', 'data', 'Reference', True, batch))
D['runs'].append(entry('none', 'unconstrained', 'Reference', True, np.load(NONE)))
for d in sorted(glob(f'{SWEEP}/*/'), key=lambda d : json.load(open(f'{d}/args.json'))['eta']) :
    eta = json.load(open(f'{d}/args.json'))['eta']
    D['runs'].append(entry(f'c7_{eta:g}', f'C7 eta {eta:g}', 'C7 sweep', eta in (30., 100.), np.load(f'{d}/generated.npy'),
                           lbd=np.load(f'{d}/lbd.npy')))
for name, label, group, on, color in RUNS :
    d = HERE / 'runs' / name
    if not (d / 'generated.npy').exists() :
        print(f'skip {name} (not run)')
        continue
    D['runs'].append(entry(name, label, group, on, np.load(d / 'generated.npy'),
                           lbd=np.load(d / 'lbd.npy') if (d / 'lbd.npy').exists() else None, color=color))

html = (HERE / 'board_template.html').read_text().replace('/*DATA*/', 'const D = ' + json.dumps(D, separators=(',', ':')) + ';')
OUT.write_text(html)
print(f'Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')

head, rest = html.split('<body>', 1)
title = head[head.index('<title>'):head.index('</title>') + 8]
style = head[head.index('<style>'):head.index('</style>') + 8]
web = (f'{title}\n<meta name="claude:viewport-fit-sides" content="cover">\n<script src="{PLOTLY_CDN}"></script>\n{style}\n'
       + rest.replace('</body>', '').replace('</html>', ''))
OUT_WEB.write_text(web)
print(f'Saved {OUT_WEB} ({OUT_WEB.stat().st_size / 1e6:.1f} MB), page for the Artifact publish (plotly from cdnjs)')
