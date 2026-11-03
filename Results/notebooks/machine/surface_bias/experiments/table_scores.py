"""
Table 1 of the paper (`paper/table1_score_table.py`) extended with the b16 runs of the surface-bias experiments.

Rows : the data batch and the paper optimal runs (b16, seed 0, {INFERENCE}/optimal), then the runs/b16_* of this folder.
Columns : mean +/- std over the batch of the BW and DW box T and S and of the density error, as table 1, plus the
volume-weighted level-0 T and S (T0, S0) and the level-0 sigma_0 bias to the data batch.
Saves figures/table_scores.txt and figures/table_scores.tex.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
from common import *

from functools import partial
from metrics import compute_density_error, temperature_BWbox_metric, temperature_DWbox_metric

mask = load_mask()
volume0 = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values[0]
ocean0 = mask.tmask.values[0].astype(bool)

box_metrics = {'BWbox' : partial(temperature_BWbox_metric, depth_box=1500),
               'DWbox' : partial(temperature_DWbox_metric, depth_box=1500)}
columns = [('BWbox', 'toce'), ('BWbox', 'soce'), ('DWbox', 'toce'), ('DWbox', 'soce'), ('density_errors', 'density'),
           ('T0', 'toce'), ('S0', 'soce'), ('bias0', 'density')]
LABELS = {('BWbox', 'toce') : r'BW T [$^\circ$C]', ('BWbox', 'soce') : r'BW S [g kg$^{-1}$]',
          ('DWbox', 'toce') : r'DW T [$^\circ$C]', ('DWbox', 'soce') : r'DW S [g kg$^{-1}$]',
          ('density_errors', 'density') : r'Density error [\%]', ('T0', 'toce') : r'Surface T [$^\circ$C]',
          ('S0', 'soce') : r'Surface S [g kg$^{-1}$]', ('bias0', 'density') : r'Surface $\sigma_0$ bias [kg m$^{-3}$]'}
PRECISION = {'density_errors' : 1, 'bias0' : 2}
EXPERIMENTS = {'b16_surf_c7010' : 'Increasing density, eta = 10 + surface T,S mean',
               'b16_surf_c7030' : 'Increasing density, eta = 30 + surface T,S mean',
               'b16_surf_c7100' : 'Increasing density, eta = 100 + surface T,S mean',
               'b16_proftop' : 'Mean-profile hinge + increasing density with surface stop-grad',
               'b16_proftop_p300_t100' : 'Mean-profile hinge (eta 300) + increasing density with surface stop-grad',
               'b16_skipanchor' : 'Increasing density without top pair + anchor to unconstrained',
               'centred_none_b32' : 'Centred sampler, no constraint (b32)',
               'centred_c7010_b32' : 'Centred sampler, increasing density eta = 10 (b32)',
               'centred_zm_e1_b32' : 'Centred sampler, T,S mean eta = 1 (b32)',
               'centred_zm_e1_c7010_b32' : 'Centred sampler, T,S mean eta = 1 + increasing density eta = 10 (b32)',
               'centred_surf_c7030_b32' : 'Centred sampler, increasing density eta = 30 + surface T,S mean (b32)'}
ORDER = ['data', 'none', 'none_border0', 'negative_grad_density_huber_border0', 'negative_grad_density_huber_border0_30',
         'negative_grad_density_huber_border0_50', 'negative_grad_density_huber_border0_80', 'mean_density_border0',
         'zero_mean_border0'] + list(EXPERIMENTS)
LONG_NAMES = NAMES | {'negative_grad_density_huber_border0' : 'Increasing density constraint, eta = 10',
                      'negative_grad_density_huber_border0_50' : 'Increasing density constraint, eta = 50'} | EXPERIMENTS


def level0(fields) :
    """Volume-weighted level-0 mean of each member of the (b, z, y, x) numpy fields T, S and sigma_0."""
    sig = compute_density(fields, mask).transpose('batch', 'depth', 'nav_lat', 'nav_lon').values[:, 0] - 1000
    mean = lambda v : np.nansum(np.where(ocean0, v, 0) * volume0, axis=(1, 2)) / volume0.sum()
    return mean(np.asarray(fields['toce'])[:, 0]), mean(np.asarray(fields['soce'])[:, 0]), mean(sig)


src = dict(sources)
for name in EXPERIMENTS :
    f = HERE / 'runs' / name / 'generated.npy'
    if f.exists() :
        src[name] = torch.from_numpy(np.load(f))
    else :
        print(f'skip {name} (not run)')

stats = []
for source, x in src.items() :
    fields = {k : v.detach().cpu().numpy() for k, v in observer.unnormalized(x).items()}
    for field in ['toce', 'soce'] :
        for bi, b in enumerate(fields[field]) :
            sample = mask.e3t_0.copy()
            sample[:] = np.asarray(b)
            for key, f in box_metrics.items() :
                stats.append({'source' : source, 'index' : bi, 'field' : field, 'metric' : key, 'value' : f(sample, mask).item()})
    for bi, v in enumerate(compute_density_error(fields, mask)) :
        stats.append({'source' : source, 'index' : bi, 'field' : 'density', 'metric' : 'density_errors', 'value' : float(v)})
    for key, field, values in zip(['T0', 'S0', 'bias0'], ['toce', 'soce', 'density'], level0(fields)) :
        stats += [{'source' : source, 'index' : bi, 'field' : field, 'metric' : key, 'value' : float(v)} for bi, v in enumerate(values)]
stats = pd.DataFrame(stats)
data0 = stats.query("source == 'data' and metric == 'bias0'")['value'].mean()
stats.loc[stats.metric == 'bias0', 'value'] -= data0


def format_mean_std(mean, std, precision=2) :
    """`mean ± std` with `precision` decimals, a std that rounds to zero written as `< 10^-precision`."""
    floor = 10 ** -precision
    std_str = f'< {floor:.{precision}f}' if std < floor / 2 else f'{std:.{precision}f}'
    return f'{mean:.{precision}f} ± {std_str}'


grouped = stats.groupby(['source', 'metric', 'field'])['value']
means, stds = grouped.mean(), grouped.std()
table = pd.DataFrame({f'{m} {f}' : [format_mean_std(means[(s, m, f)], stds[(s, m, f)], PRECISION.get(m, 2)) for s in src]
                      for m, f in columns}, index=list(src))
table = table.loc[[s for s in ORDER if s in table.index]]
pd.set_option('display.width', 300)
txt = table.rename(index=LONG_NAMES).to_string()
print(txt)

latex = table.rename(index=LONG_NAMES, columns={f'{m} {f}' : LABELS[(m, f)] for m, f in columns})
latex = latex.map(lambda s : s.replace('±', r'$\pm$')).to_latex(column_format='l' + 'c' * len(columns))
(HERE / 'figures').mkdir(exist_ok=True)
(HERE / 'figures' / 'table_scores.txt').write_text(txt + '\n')
(HERE / 'figures' / 'table_scores.tex').write_text(latex)
