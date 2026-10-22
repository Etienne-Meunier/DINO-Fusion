"""
Build `huber_board.html` : Table 1 of the paper (`paper/table1_score_table.py`, box metrics and density error, mean +/- std
over the batch) for the runs of `optimal/` and the eta sweep of the density loss negative_grad_density_huber (C7), and for
every row the zonal sigma_0 section of its median member and of its batch average, with the profile over the BAND
latitudes, as `paper/fig1_density_profile.py`. The median member is computed both by RMSE to the data batch-average
section (as fig1) and by density error, the page switches between the two.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'paper'))
from common import *

from functools import partial
import cmocean.cm as cm
from metrics import compute_density_error, get_density_at_surface, temperature_BWbox_metric, temperature_DWbox_metric

OUT = HERE / 'huber_board.html'
SWEEP = f'{INFERENCE}/sweep_density_huber/negative_grad_density_huber_border0'
SWEEP_B16 = f'{INFERENCE}/sweep_density_huber_b16/negative_grad_density_huber_border0'
NONE_B8 = f'{INFERENCE}/sweep_baselines/none_border0_b8/151026-145216'
BAND = (-65, -45)
LEVELS = (22., 28.5, 0.5)
PROFILE_RANGE = (25., 28.5)
DEPTH_POWER, DEPTH_MAX = 0.625, 3500
DEPTH_TICKS = [0, 100, 250, 500, 1000, 1500, 2000, 3000]
COLUMNS = [('BWbox', 'toce'), ('BWbox', 'soce'), ('DWbox', 'toce'), ('DWbox', 'soce'), ('density_errors', 'density')]

file_mask_LR = load_mask()
box_metrics = {'BWbox' : partial(temperature_BWbox_metric, depth_box=1500),
               'DWbox' : partial(temperature_DWbox_metric, depth_box=1500)}


def compute_density(fields) :
    """Potential density referenced at the surface, (batch, depth, nav_lat, nav_lon), of unnormalized 'toce', 'soce'."""
    bsize = fields['soce'].shape[0]
    data = file_mask_LR.e3t_0.copy().expand_dims({'batch' : bsize})
    soce, toce, tmask = data.copy(), data.copy(), data.copy()
    soce[:] = np.asarray(fields['soce'])
    toce[:] = np.asarray(fields['toce'])
    tmask[:] = np.repeat(file_mask_LR.tmask.values[None], bsize, 0)
    return get_density_at_surface(toce, soce, tmask)


def zonal_sections(fields) :
    """Volume-weighted zonal mean of sigma_0 - 1000, (batch, depth - 1, nav_lat), last (land) level dropped."""
    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0 * file_mask_LR.tmask).squeeze()
    volume = volume.transpose('depth', 'nav_lat', 'nav_lon')
    section = (compute_density(fields) * volume).sum('nav_lon') / volume.sum('nav_lon')
    return (section.transpose('batch', 'depth', 'nav_lat')[:, :-1] - 1000).values


def median_member(errors) :
    """Index of the member at the median of `errors`, the upper median for an even batch."""
    return int(np.argsort(errors)[len(errors) // 2])


def format_mean_std(mean, std, precision=1) :
    """Format `mean ± std`, with the std in scientific notation when below 0.1 (as table1_score_table.py)."""
    std_str = f'{std:.1e}'.replace('e-0', 'e-') if std < 0.1 else f'{std:.{precision}f}'
    return f'{mean:.{precision}f} ± {std_str}'


def to_list(a) :
    return [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in a]


c7 = {}
for d in sorted(glob(f'{SWEEP}/*/')) :
    eta = json.load(open(f'{d}/args.json'))['eta']
    c7[eta] = torch.from_numpy(np.load(f'{d}/generated.npy'))
labels = {'data' : 'Data', 'none' : 'No constraint', 'none_border0' : 'No constraint, border 0 (b16)',
          'mean_density_border0' : 'Density constraint (mean_density)', 'zero_mean_border0' : 'TS constraint (zero_mean)',
          'none_b8' : 'No constraint, border 0 (b8, paired)'}
order = ['data', 'none', 'none_border0', 'mean_density_border0', 'zero_mean_border0', 'negative_grad_density_huber_border0',
         'negative_grad_density_huber_border0_30', 'negative_grad_density_huber_border0_80']
for k in order[5:] :
    if k in gens_args :
        labels[k] = f"Monotonicity C7, eta {gens_args[k]['eta']:g} (b16, optimal)"
runs = {k : (batch if k == 'data' else gens[k]) for k in order if k == 'data' or k in gens}
for d in sorted(glob(f'{SWEEP_B16}/*/')) :
    if os.path.exists(f'{d}/generated.npy') :
        eta = json.load(open(f'{d}/args.json'))['eta']
        runs[f'huber_eta{eta:g}_b16'] = torch.from_numpy(np.load(f'{d}/generated.npy'))
        labels[f'huber_eta{eta:g}_b16'] = f'Monotonicity C7, eta {eta:g} (b16)'
runs['none_b8'] = torch.from_numpy(np.load(f'{NONE_B8}/generated.npy'))
for eta in sorted(c7) :
    runs[f'huber_eta{eta:g}'] = c7[eta]
    labels[f'huber_eta{eta:g}'] = f'Monotonicity C7, eta {eta:g} (b8)'

stretch = lambda d : np.asarray(d) ** DEPTH_POWER
depth = file_mask_LR.gdept_1d.values[:-1]
latitude = file_mask_LR.gphit.isel(nav_lon=10).values
band = (latitude >= BAND[0]) & (latitude < BAND[1])

rows, data_mean = [], None
for name, x in runs.items() :
    fields = {k : v.detach().cpu() for k, v in observer.unnormalized(x).items()}
    stats = {c : [] for c in COLUMNS}
    for field in ['toce', 'soce'] :
        for b in fields[field] :
            sample = file_mask_LR.e3t_0.copy()
            sample[:] = np.asarray(b)
            for key, f in box_metrics.items() :
                stats[(key, field)].append(f(sample, file_mask_LR).item())
    errors = np.asarray([float(v) for v in compute_density_error(fields, file_mask_LR)])
    stats[('density_errors', 'density')] = list(errors)
    sections = zonal_sections({k : v.numpy() for k, v in fields.items()})
    mean = np.nanmean(sections, axis=0)
    if data_mean is None :
        data_mean = mean
    rmse = np.sqrt(np.nanmean((sections - data_mean[None]) ** 2, axis=(1, 2)))
    m_rmse, m_err = median_member(rmse), median_member(errors)
    rows.append({'name' : name, 'label' : labels.get(name, name), 'n' : len(x),
                 'cells' : [format_mean_std(np.mean(stats[c]), np.std(stats[c], ddof=1)) for c in COLUMNS],
                 'mean' : to_list(mean), 'median_rmse' : to_list(sections[m_rmse]), 'median_err' : to_list(sections[m_err]),
                 'm_rmse' : m_rmse, 'm_err' : m_err, 'rmse_m' : float(rmse[m_rmse]), 'err_m' : float(errors[m_err]),
                 'err_at_rmse' : float(errors[m_rmse]), 'rmse_at_err' : float(rmse[m_err])})
    print(name, rows[-1]['cells'], flush=True)

colorscale = [[i / 10, '#%02x%02x%02x' % tuple(int(255 * v) for v in cm.deep_r(i / 10)[:3])] for i in range(11)]
D = {'rows' : rows, 'columns' : [f'{m} {f}' for m, f in COLUMNS], 'lat' : [round(float(v), 3) for v in latitude],
     'y' : [round(float(v), 4) for v in stretch(depth)], 'depth' : [round(float(v), 1) for v in depth],
     'band' : list(BAND), 'band_mask' : band.tolist(), 'levels' : list(LEVELS), 'profile_range' : list(PROFILE_RANGE),
     'y_max' : float(stretch(DEPTH_MAX)), 'ticks' : DEPTH_TICKS, 'tickvals' : stretch(DEPTH_TICKS).round(4).tolist(),
     'colorscale' : colorscale}
html = (HERE / 'huber_board_template.html').read_text().replace('/*DATA*/', 'const D = ' + json.dumps(D, separators=(',', ':')) + ';')
OUT.write_text(html)
print(f'Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')
