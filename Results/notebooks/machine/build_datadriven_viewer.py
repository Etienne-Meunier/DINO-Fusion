"""
Build the data of the "A-priori · data-driven" tab of `density_viewer.html` : the generations of the sweeps of SWEEPS
against the whole training set, field by field and level by level, with no physical diagnostic.
    density_viewer_datadriven.js : per-level statistics of every run and of the data (loaded with the tab)
        level_mean : area-weighted basin mean of each level, per member (b, z), data : quantiles over the 1800 states
        bias, spread : per level, rms (over the level, area-weighted) of run mean - data mean, and sqrt of the ratio of
                       the area-weighted batch variance to the data one, raw and with the level mean of each member removed
                       ('dm'), data : mean and std over N_FLOOR disjoint groups of n random states (noise floor), for
                       each batch size n of the runs
        ts_corr : spatial correlation of T and S anomalies (level mean removed) of each level, per member (b, z)
        pattern : spatial correlation of the anomaly of each level with the data climatology anomaly, per member (b, z)
        data quantiles of the per-state values and the 5-95 % range of the median of n random states (floor of the median)
    density_viewer_datadriven_<sweep>_<name>.js : maps of one source (loaded on demand) : batch mean, batch std and the first
        N_MEMBERS members of every field, every level, uint16 gzip base64 ; the data source holds the 1800-state mean and std
        and N_MEMBERS random states.
Fields : toce, soce, sigma (sigma_0 - 1000, from T and S), ssh (one level). The training statistics need two passes over
the training tar and are cached (viewer_cache), as each generation.
"""
import base64
import gzip
import io
import json
import tarfile
from glob import glob
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from build_density_viewer import PRP, INFERENCE, TrainingConfig, Observer, get_density_at_surface
from viewer_cache import cached, code_hash, file_id

HERE = Path(__file__).parent
OUT = HERE / 'density_viewer_datadriven.js'
SWEEPS = ['optimal', 'sweep_density_huber', 'october']  # the first one is shown first
FIELDS = ['toce', 'soce', 'sigma', 'ssh']
N_FLOOR = 16  # disjoint groups of n random training states for the noise floor
N_MEMBERS = 4  # members stored as maps per source
QUANTILES = [5, 25, 50, 75, 95]  # [%] of the per-state data values
SEED = 0
# Maps are stored as uint16 over these ranges (values outside are clipped), NaN as 0
RANGE = {'toce': (-5., 40.), 'soce': (28., 40.), 'sigma': (15., 32.), 'ssh': (-3., 3.)}
STD_RANGE = {'toce': 10., 'soce': 3., 'sigma': 3., 'ssh': 1.}


class Grid:
    """Ocean masks (z, y, x) and cell areas (y, x) of each field, ssh with a single level."""
    def __init__(self):
        m = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').isel(time_counter=0)
        tmask = m.tmask.values.astype(bool)
        self.mask = {f: tmask[:1] if f == 'ssh' else tmask for f in FIELDS}
        self.area = m.e1t.values * m.e2t.values
        self.depth = m.gdept_1d.values
        self.lat = m.gphit.values[:, 0]
        self.lon = m.glamt.values[0] - 60


def add_sigma(fields, tmask):
    """Add sigma_0 - 1000 (b, z, y, x) to unnormalized fields with toce and soce (b, z, y, x), NaN on land."""
    rho = get_density_at_surface(fields['toce'], fields['soce'], np.broadcast_to(tmask, fields['toce'].shape))
    fields['sigma'] = np.where(tmask[None], rho - 1000, np.nan)
    return fields


def level_means(x, mask, area):
    """Area-weighted mean over the ocean of each level of x (b, z, y, x), (b, z), NaN for land-only levels."""
    w = area[None] * mask
    total = w.sum(axis=(1, 2))
    return np.where(total > 0, (np.nan_to_num(x) * w[None]).sum(axis=(2, 3)) / np.maximum(total, 1e-30), np.nan)


def corr_levels(a, b, mask):
    """Pearson correlation over the ocean cells of each level, a (n, z, y, x) with b (n or 1, z, y, x), (n, z)."""
    out = np.full(a.shape[:2], np.nan)
    for k in range(a.shape[1]):
        m = mask[k]
        if m.sum() < 3:
            continue
        x, y = a[:, k][:, m], np.broadcast_to(b[:, k][:, m], (a.shape[0], m.sum()))
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
        out[:, k] = (x * y).sum(axis=1) / np.sqrt((x ** 2).sum(axis=1) * (y ** 2).sum(axis=1))
    return out


def level_scores(mean, var, ref_mean, ref_var, mask, area):
    """Per level : rms of mean - ref_mean and sqrt(var / ref_var), area-weighted over the ocean, (z,) each."""
    w = area[None] * mask
    total = np.maximum(w.sum(axis=(1, 2)), 1e-30)
    bias = np.sqrt((w * np.nan_to_num(mean - ref_mean) ** 2).sum(axis=(1, 2)) / total)
    spread = np.sqrt((w * np.nan_to_num(var)).sum(axis=(1, 2)) / np.maximum((w * np.nan_to_num(ref_var)).sum(axis=(1, 2)), 1e-30))
    return bias, spread


def training_states(path, grid):
    """Yield (archive index, fields) of every training state in archive order, fields {f : (1, z, y, x)} with sigma, NaN on land."""
    with tarfile.open(path) as tar:
        current, fields = None, {}
        for m in tar:
            if not m.isfile() or m.name.startswith('infos/'):
                continue
            key, field, _ = m.name.split('/')[-1].split('.')
            if key != current:
                if fields:
                    yield int(current), add_sigma(fields, grid.mask['toce'])
                current, fields = key, {}
            if field in ('toce', 'soce', 'ssh'):
                x = np.load(io.BytesIO(tar.extractfile(m).read())).astype(np.float64)
                x = x.reshape((1, 1) + x.shape) if field == 'ssh' else x[None]
                fields[field] = np.where(grid.mask[field][None], x, np.nan)
        if fields:
            yield int(current), add_sigma(fields, grid.mask['toce'])


def training_statistics(path, grid, n):
    """
    Statistics of the training states, two passes over the tar :
        mean, var, mean_dm, var_dm : per cell, over all the states, raw and with the level mean of each state removed
        level_mean (N, z), ts_corr (N, z), pattern (N, z) of every state, floor : bias and spread per level of N_FLOOR
        disjoint groups of n random states (raw and dm), members : N_MEMBERS random states (fields)
    """
    rng = np.random.default_rng(SEED)
    keys = []
    acc = {}
    level_mean, ts_corr = {f: [] for f in FIELDS}, []
    for i, fields in training_states(path, grid):
        keys.append(i)
        lm = {f: level_means(fields[f], grid.mask[f], grid.area) for f in FIELDS}
        dm = {f: fields[f] - lm[f][:, :, None, None] for f in FIELDS}
        for f in FIELDS:
            level_mean[f].append(lm[f][0])
            for name, x in [('s1', fields[f][0]), ('s2', fields[f][0] ** 2), ('d1', dm[f][0]), ('d2', dm[f][0] ** 2)]:
                acc[(f, name)] = acc.get((f, name), 0) + np.nan_to_num(x)
        ts_corr.append(corr_levels(dm['toce'], dm['soce'], grid.mask['toce'])[0])
    N = len(keys)
    stats = {}
    for f in FIELDS:
        mean, mean_dm = acc[(f, 's1')] / N, acc[(f, 'd1')] / N
        stats[f] = {'mean': np.where(grid.mask[f], mean, np.nan),
                    'var': np.where(grid.mask[f], (acc[(f, 's2')] - N * mean ** 2) / (N - 1), np.nan),
                    'mean_dm': np.where(grid.mask[f], mean_dm, np.nan),
                    'var_dm': np.where(grid.mask[f], (acc[(f, 'd2')] - N * mean_dm ** 2) / (N - 1), np.nan),
                    'level_mean': np.array(level_mean[f])}
    stats['ts_corr'] = np.array(ts_corr)

    # second pass : pattern correlation with the climatology anomaly, floor groups and stored members
    order = rng.permutation(N)
    group = np.full(N, -1)
    group[order[:n * N_FLOOR]] = np.repeat(np.arange(N_FLOOR), n)
    members = set(order[n * N_FLOOR:n * N_FLOOR + N_MEMBERS].tolist())
    gacc, pattern, stored = {}, {f: [] for f in FIELDS}, []
    for i, fields in training_states(path, grid):
        lm = {f: level_means(fields[f], grid.mask[f], grid.area) for f in FIELDS}
        dm = {f: fields[f] - lm[f][:, :, None, None] for f in FIELDS}
        for f in FIELDS:
            pattern[f].append(corr_levels(dm[f], stats[f]['mean_dm'][None], grid.mask[f])[0])
        if i in members:
            stored.append({f: fields[f][0].astype(np.float32) for f in FIELDS})
        g = group[i]
        if g < 0:
            continue
        for f in FIELDS:
            # shifted by the climatology so that float32 sums of squares keep the small member-to-member variance
            for name, x in [('s1', fields[f][0] - stats[f]['mean']), ('d1', dm[f][0] - stats[f]['mean_dm'])]:
                x = np.nan_to_num(x).astype(np.float32)
                gacc.setdefault((f, name), np.zeros((N_FLOOR,) + x.shape, np.float32))[g] += x
                gacc.setdefault((f, name + '2'), np.zeros((N_FLOOR,) + x.shape, np.float32))[g] += x ** 2
    for f in FIELDS:
        stats[f]['pattern'] = np.array(pattern[f])
        floor = {}
        for kind, name in [('raw', 's1'), ('dm', 'd1')]:
            ref_var = stats[f]['var' if kind == 'raw' else 'var_dm']
            scores = []
            for g in range(N_FLOOR):
                m1, m2 = gacc[(f, name)][g] / n, gacc[(f, name + '2')][g] / n
                scores.append(level_scores(m1, (m2 - m1 ** 2) * n / (n - 1), 0., ref_var, grid.mask[f], grid.area))
            scores = np.array(scores)  # (group, 2, z)
            floor[kind] = {'bias': scores[:, 0], 'spread': scores[:, 1]}
        stats[f]['floor'] = floor
    stats['members'] = stored
    return stats


def run_statistics(fields, grid, data):
    """Per-level statistics of a generation (unnormalized fields with sigma, (b, z, y, x)) against the training `data`."""
    out = {'level_mean': {}, 'bias': {}, 'spread': {}, 'pattern': {}}
    dm = {}
    for f in FIELDS:
        lm = level_means(fields[f], grid.mask[f], grid.area)
        dm[f] = fields[f] - lm[:, :, None, None]
        out['level_mean'][f] = lm
        out['pattern'][f] = corr_levels(dm[f], data[f]['mean_dm'][None], grid.mask[f])
        for kind, x, ref in [('raw', fields[f], ''), ('dm', dm[f], '_dm')]:
            b, s = level_scores(np.nanmean(x, 0), np.nanvar(x, 0, ddof=1), data[f]['mean' + ref], data[f]['var' + ref],
                                grid.mask[f], grid.area)
            out['bias'].setdefault(f, {})[kind] = b
            out['spread'].setdefault(f, {})[kind] = s
    out['ts_corr'] = corr_levels(dm['toce'], dm['soce'], grid.mask['toce'])
    return out


def median_floor(values, n, rng, draws=2000):
    """5 and 95 % quantiles of the median of n states drawn without replacement from values (N, z), (2, z)."""
    medians = np.array([np.nanmedian(values[rng.choice(len(values), n, replace=False)], axis=0) for _ in range(draws)])
    return np.nanpercentile(medians, [5, 95], axis=0)


def gz64(array):
    """gzip then base64 of the bytes of an array, decoded in the page with DecompressionStream."""
    return base64.b64encode(gzip.compress(np.ascontiguousarray(array).tobytes(), compresslevel=6, mtime=0)).decode()


def quantize(x, lo, hi):
    """uint16 of x over [lo, hi] (clipped), NaN as 0, gzip base64."""
    q = np.where(np.isfinite(x), np.clip((x - lo) / (hi - lo), 0, 1) * 65534 + 1, 0).astype('<u2')
    return gz64(q)


def maps(mean, std, members):
    """Quantized maps {f : {'mean', 'std', 'members' : [...]}} of every field."""
    return {f: {'mean': quantize(mean[f], *RANGE[f]), 'std': quantize(std[f], 0., STD_RANGE[f]),
                'members': [quantize(m[f], *RANGE[f]) for m in members]} for f in FIELDS}


def r(a, d=5):
    """Nested lists of a float array rounded to d significant digits, NaN as None."""
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else float(f'{v:.{d}g}') for v in a.ravel()] if a.ndim == 1 else [r(v, d) for v in a]


if __name__ == '__main__':
    grid = Grid()
    config = TrainingConfig()
    observer = Observer('data', config, device='cpu')
    code = code_hash(__file__)
    sweeps = {}
    for sweep in SWEEPS:
        paths = {}
        for path in sorted(glob(f'{INFERENCE}/{sweep}/*/*/generated.npy')):
            paths.setdefault(Path(path).parent.parent.name, []).append(path)  # latest timestamp last
        runs = {}
        for folder, group in paths.items():
            if len(group) == 1:
                runs[folder] = group[0]
                continue
            # several runs in one folder (an eta sweep) : one entry per run, named by eta, in increasing eta
            etas = {path: json.load(open(Path(path).parent / 'args.json'))['eta'] for path in group}
            for path in sorted(group, key=etas.get):
                runs[f'{folder}_{etas[path]:g}'] = path
        if runs:
            sweeps[sweep] = runs
    batch = {(sweep, name): len(np.load(path, mmap_mode='r')) for sweep, runs in sweeps.items() for name, path in runs.items()}
    sizes = sorted(set(batch.values()))
    # batch size of a sweep (its noise floor) : the most common one of its runs
    sweep_n = {sweep: max(sizes, key=[batch[(sweep, r)] for r in runs].count) for sweep, runs in sweeps.items()}

    data_by_n = {n: cached('datadriven', [code, file_id(config.data_file), n, 'training'],
                           lambda n=n: training_statistics(config.data_file, grid, n)) for n in sizes}
    data = data_by_n[sweep_n[next(iter(sweeps))]]
    print(f'training statistics : {len(data["ts_corr"])} states, floors for n = {sizes}')

    def run_entry(path):
        def compute():
            with torch.no_grad():
                fields = {k: v.cpu().numpy().astype(np.float64) for k, v in observer.unnormalized(torch.from_numpy(np.load(path))).items()}
            fields['ssh'] = fields['ssh'][:, None]
            fields = add_sigma(fields, grid.mask['toce'])
            fields = {f: np.where(grid.mask[f][None], fields[f], np.nan) for f in FIELDS}
            stats = run_statistics(fields, grid, data)
            stats['maps'] = maps({f: np.nanmean(fields[f], 0) for f in FIELDS}, {f: np.nanstd(fields[f], 0, ddof=1) for f in FIELDS},
                                 [{f: fields[f][i] for f in FIELDS} for i in range(min(N_MEMBERS, len(fields['toce'])))])
            return stats
        return cached('datadriven', [code, file_id(config.data_file), file_id(path)], compute)

    rng = np.random.default_rng(SEED)
    out = {'n_states': len(data['ts_corr']), 'fields': FIELDS, 'quantiles': QUANTILES,
           'depth': r(grid.depth), 'lat': r(grid.lat), 'lon': r(grid.lon), 'nz': {f: int(grid.mask[f].shape[0]) for f in FIELDS},
           'range': RANGE, 'std_range': STD_RANGE, 'n_members': N_MEMBERS, 'n_floor': N_FLOOR, 'sweeps': {}}
    out['data'] = {
        'level_mean': {f: {'q': r(np.nanpercentile(data[f]['level_mean'], QUANTILES, axis=0)),
                           'mean': r(np.nanmean(data[f]['level_mean'], 0))} for f in FIELDS},
        'pattern': {f: {'q': r(np.nanpercentile(data[f]['pattern'], QUANTILES, axis=0))} for f in FIELDS},
        'ts_corr': {'q': r(np.nanpercentile(data['ts_corr'], QUANTILES, axis=0))},
    }
    # noise floors of n random training states, per batch size n
    out['floors'] = {n: {
        'moments': {f: {k: {s: {'mean': r(v[s].mean(0)), 'std': r(v[s].std(0))} for s in ('bias', 'spread')}
                        for k, v in d[f]['floor'].items()} for f in FIELDS},
        'pattern': {f: r(median_floor(d[f]['pattern'], n, rng)) for f in FIELDS},
        'ts_corr': r(median_floor(d['ts_corr'], n, rng)),
    } for n, d in data_by_n.items()}
    for sweep, runs in sweeps.items():
        entry = out['sweeps'][sweep] = {'n': sweep_n[sweep], 'runs': {}}
        for name, path in runs.items():
            s = run_entry(path)
            with open(Path(path).parent / 'args.json') as f:
                args = json.load(f)
            entry['runs'][name] = {'n': batch[(sweep, name)], 'level_mean': {f: r(v) for f, v in s['level_mean'].items()},
                                   'bias': {f: {k: r(v) for k, v in d.items()} for f, d in s['bias'].items()},
                                   'spread': {f: {k: r(v) for k, v in d.items()} for f, d in s['spread'].items()},
                                   'pattern': {f: r(v) for f, v in s['pattern'].items()}, 'ts_corr': r(s['ts_corr']),
                                   'args': {k: args.get(k) for k in ['constraint', 'eta', 'projection', 'inf_steps', 'seed', 'noise', 'land']}}
            key = f'{sweep}/{name}'
            (HERE / f'density_viewer_datadriven_{sweep}_{name}.js').write_text(
                f'window.DATADRIVEN_MAPS = window.DATADRIVEN_MAPS || {{}}; window.DATADRIVEN_MAPS[{json.dumps(key)}] = {json.dumps(s["maps"])};\n')
            print(f'{key} (b={batch[(sweep, name)]}) : {path}')

    data_maps = maps({f: data[f]['mean'] for f in FIELDS}, {f: np.sqrt(data[f]['var']) for f in FIELDS}, data['members'])
    (HERE / 'density_viewer_datadriven_data.js').write_text(
        f'window.DATADRIVEN_MAPS = window.DATADRIVEN_MAPS || {{}}; window.DATADRIVEN_MAPS["data"] = {json.dumps(data_maps)};\n')
    OUT.write_text(f'window.DATADRIVEN_DATA = {json.dumps(out)};\n')
    print(f'Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')
