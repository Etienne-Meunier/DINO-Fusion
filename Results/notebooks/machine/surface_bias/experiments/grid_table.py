"""
Grid of sampler x constraint : noise C (raw / centred), land L (model / BorderZero / noisy land), constraint g (none,
zero_mean eta 0.1, C7 eta 10, C7 10 + surface T,S mean). Seed 0 everywhere, batch as available. Every batch has its land
set to 0 before evaluation.

Columns : BW and DW box T and S, density error E, level-0 T, level-0 sigma_0 bias to the data batch (means over the batch).
Missing cells are printed as empty. Prints the table and saves figures/grid_table.txt.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
from common import *

from functools import partial
from metrics import compute_density_error, temperature_BWbox_metric, temperature_DWbox_metric
from pipelines.losses import BorderZero

NOISE, LAND = ['raw', 'centred'], ['model', 'B0', 'NL']
G = ['none', 'zero_mean 0.1', 'C7 10', 'C7 10 + surf TS']
CELLS = {('raw', 'model', 'none') : 'optimal:none',
         ('raw', 'B0', 'none') : 'optimal:none_border0',
         ('raw', 'B0', 'zero_mean 0.1') : 'optimal:zero_mean_border0',
         ('raw', 'B0', 'C7 10') : 'optimal:negative_grad_density_huber_border0',
         ('raw', 'B0', 'C7 10 + surf TS') : 'b16_surf_c7010',
         ('raw', 'NL', 'none') : 'nl_none',
         ('raw', 'NL', 'C7 10') : 'nl_c7010',
         ('centred', 'model', 'none') : 'traj_raw_call_x',
         ('centred', 'B0', 'none') : 'traj_border0_call_x',
         ('centred', 'NL', 'none') : 'centred_none_b32',
         ('centred', 'NL', 'zero_mean 0.1') : 'centred_zm_e0.1',
         ('centred', 'NL', 'C7 10') : 'centred_c7010_b32',
         ('centred', 'NL', 'C7 10 + surf TS') : 'centred_surf_c7010'}
mask = load_mask()
ocean0 = mask.tmask.values[0].astype(bool)
volume0 = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values[0]
project = BorderZero(observer)
boxes = {'BW' : partial(temperature_BWbox_metric, depth_box=1500), 'DW' : partial(temperature_DWbox_metric, depth_box=1500)}


def load(name) :
    """Normalised batch of `data`, `optimal:<constraint>`, runs/<name>.npy (traj outputs) or runs/<name>/generated.npy."""
    if name == 'data' :
        return batch
    if name.startswith('optimal:') :
        return torch.from_numpy(np.load(f'{generation_dirs()[name[8:]]}/generated.npy'))
    f = HERE / 'runs' / f'{name}.npy'
    return torch.from_numpy(np.load(f if f.exists() else HERE / 'runs' / name / 'generated.npy'))


def scores(x) :
    """Batch means of the box T and S, E, level-0 T and sigma_0, and the batch size."""
    f = {k : v.detach().cpu().numpy() for k, v in observer.unnormalized(project(x)).items()}
    sig = compute_density(f, mask).transpose('batch', 'depth', 'nav_lat', 'nav_lon').values[:, 0] - 1000
    mean0 = lambda v : np.nansum(np.where(ocean0, v, 0) * volume0, axis=(1, 2)) / volume0.sum()
    out = {'b' : len(x), 'E' : np.mean(compute_density_error(f, mask)), 'T0' : mean0(f['toce'][:, 0]).mean(),
           'sig0' : mean0(sig).mean()}
    for field in ['toce', 'soce'] :
        for k, m in boxes.items() :
            vals = []
            for b in f[field] :
                sample = mask.e3t_0.copy()
                sample[:] = np.asarray(b)
                vals.append(m(sample, mask).item())
            out[f'{k} {field[0].upper()}'] = np.mean(vals)
    return out


ref = scores(batch)
rows = [('data', '', '', ref | {'source' : 'data batch'})]
for c in NOISE :
    for l in LAND :
        for g in G :
            name = CELLS.get((c, l, g))
            rows.append((c, l, g, (scores(load(name)) | {'source' : name}) if name else {}))
fmt = {'b' : '{:.0f}', 'BW T' : '{:.2f}', 'BW S' : '{:.2f}', 'DW T' : '{:.2f}', 'DW S' : '{:.2f}', 'E' : '{:.1f}', 'T0' : '{:.2f}'}
table = pd.DataFrame([{'C' : c, 'L' : l, 'g' : g} | {k : (v.format(r[k]) if k in r else '') for k, v in fmt.items()}
                      | {'bias0' : f"{r['sig0'] - ref['sig0']:+.2f}" if 'sig0' in r else '', 'source' : r.get('source', '')}
                      for c, l, g, r in rows])
pd.set_option('display.width', 250)
print(table.to_string(index=False))
(HERE / 'figures' / 'grid_table.txt').write_text(table.to_string(index=False) + '\n')
