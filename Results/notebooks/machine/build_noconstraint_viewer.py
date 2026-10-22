"""
Build `density_viewer_noconstraint.js`, the data of the "No constraint" tab of `density_viewer.html` : how much of the
ocean state the unconstrained diffusion model has learned, data against the unconstrained runs of default_eps (without
and with BorderZero). Figures :
    offsets : basin mean of T and S per member and level, and the share of the member variance it explains
    maps : anomaly maps (field minus its basin mean) at MAP_DEPTHS
    correlation : pattern correlation of each member with the data mean, per level, on the full anomaly and on its
                  zonally asymmetric part (zonal mean removed), with the data members and a latitude-only field as references
    recentred : T-S histograms, k-means cluster maps and cluster profiles, raw and with the basin mean of each level of each
                member moved onto the data one
    mean : ensemble mean zonal sigma_0 section and SST map
    spectra : meridional power spectra of the anomaly at MAP_DEPTHS
    zonal : zonal mean (over the ocean cells) of T and S of the stored map members, every level
    shifts : spatial standard deviation and quantiles of the values of each level of the data, to set the basin mean of each
             member against them
    level_clusters : k-means of the (T, S) of the ocean cells at MAP_DEPTHS, one per level, fit on the data with the basin
                     mean of each member removed, labels of the stored map members, raw (data level mean removed) and
                     anomaly (own level mean removed), gzip compressed, with the agreement with the most frequent data label
"""
import base64
import gzip
import json
import random
from glob import glob
from pathlib import Path

import cmocean.cm as cm
import numpy as np
import torch
import xarray as xr

from sklearn.cluster import KMeans

from build_density_viewer import (PRP, INFERENCE, BASELINES, N_CLUSTERS, TS_T_EDGES, TS_S_EDGES, TrainingConfig, Observer,
                                  TransformationPipeline, get_infos, get_dataloader, DATA_SEED, get_density_at_surface,
                                  ProfileClusters, acc_mask, b64, compute_density, ts_histograms, zonal_sections)

OUT = Path(__file__).parent / 'density_viewer_noconstraint.js'
N_DATA = 32  # data states for the statistics, the first 8 are the states of the other tabs (same seed and shuffle)
N_DATA_MAPS = 8  # data states stored as maps
N_DATA_FIT = 8  # data states the k-means is fit on, as in build_density_viewer.py
MAP_DEPTHS = [5., 136., 514.]  # [m], the nearest levels are used
ANOMALY_RANGE = {'toce': 25., 'soce': 4.}  # anomaly maps are stored as uint16 over [-r, r], NaN as 0
ZONAL_RANGE = {'toce': (-5., 40.), 'soce': (28., 40.)}  # zonal mean sections are stored as uint16 over this range, NaN as 0
FIELDS = ['toce', 'soce']
SHIFT_DEPTHS = [5., 50., 100., 200., 500., 1000., 2000., 3300.]  # [m], levels of the basin mean strip plot, nearest levels
LAYER_QUANTILES = [5, 25, 50, 75, 95]  # [%], spatial distribution of each level of the data
N_LEVEL_CLUSTERS = 5  # k-means of the (T, S) of the ocean cells of each level of MAP_DEPTHS


def gz64(array):
    """gzip then base64 of the bytes of an array, decoded in the page with DecompressionStream."""
    return base64.b64encode(gzip.compress(np.ascontiguousarray(array).tobytes(), compresslevel=9, mtime=0)).decode()


class LevelClusters:
    """
    k-means of the (T, S) of the ocean cells of each level of `levels`, one fit per level, on centred values of a reference
    batch, T and S each scaled by their standard deviation over the reference cells of that level. Clusters are numbered 1..k
    by increasing sigma_0 of their centre plus the data mean of the level (`level_mean`, {f : (z,)}).
        anom : {f : (b, z, y, x)} centred values (NaN on land), tmask (z, y, x) bool
    `labels(values)` assigns the cells of centred values {f : (b, len(levels), y, x)} : uint8 (b, len(levels), y, x), 0 on land.
    """
    def __init__(self, anom, tmask, levels, level_mean, k=N_LEVEL_CLUSTERS):
        self.tmask, self.levels, self.k = tmask[levels], levels, k
        self.km, self.scale, self.rank, self.centres, self.sigma = [], [], [], [], []
        for l, kz in enumerate(levels):
            x = np.stack([anom[f][:, kz][:, tmask[kz]].ravel() for f in FIELDS], axis=1).astype(np.float64)
            scale = x.std(axis=0)
            km = KMeans(k, n_init=10, random_state=0).fit(x / scale)
            centres = km.cluster_centers_ * scale
            sigma = get_density_at_surface(centres[:, 0] + level_mean['toce'][kz], centres[:, 1] + level_mean['soce'][kz], np.ones(k)) - 1000
            order = np.argsort(sigma)
            rank = np.empty(k, int)
            rank[order] = np.arange(k)
            for name, v in [('km', km), ('scale', scale), ('rank', rank), ('centres', centres[order]), ('sigma', sigma[order])]:
                getattr(self, name).append(v)

    def labels(self, values):
        b = values[FIELDS[0]].shape[0]
        out = np.zeros((b, *self.tmask.shape), 'u1')
        for l in range(len(self.levels)):
            m = self.tmask[l]
            x = np.stack([values[f][:, l][:, m].ravel() for f in FIELDS], axis=1).astype(np.float64)
            out[:, l][:, m] = (self.rank[l][self.km[l].predict(x / self.scale[l])] + 1).reshape(b, -1)
        return out

    def counts(self, labels):
        """Number of members in each cluster for every cell, (k, len(levels), y, x)."""
        return np.stack([(labels == c + 1).sum(axis=0) for c in range(self.k)])

    def mode(self, counts):
        """Most frequent cluster of every cell (the lowest on ties), uint8 (len(levels), y, x), 0 on land."""
        return np.where(self.tmask, counts.argmax(axis=0) + 1, 0).astype('u1')

    def agreement(self, labels, ref):
        """Share of the ocean cells of each level of each member with the label of ref, mean over the members : (len(levels),)."""
        return np.stack([(labels[:, l][:, self.tmask[l]] == ref[l][self.tmask[l]]).mean(axis=1) for l in range(len(self.levels))], 1).mean(0)


def level_means(fields, tmask):
    """Mean over the ocean cells of each level, {f : (b, z)}, NaN for the land-only levels."""
    b = fields[FIELDS[0]].shape[0]
    return {f: np.stack([fields[f][:, k][:, tmask[k]].mean(axis=1) if tmask[k].any() else np.full(b, np.nan)
                         for k in range(tmask.shape[0])], axis=1) for f in FIELDS}


def anomalies(fields, tmask, means):
    """Field minus the basin mean of its level, NaN on land, {f : (b, z, y, x)}."""
    return {f: np.where(tmask[None], fields[f] - means[f][:, :, None, None], np.nan) for f in FIELDS}


def zonal_asym(a):
    """Remove the zonal mean (over the ocean, NaN ignored) of each row of a (..., y, x)."""
    return a - np.nanmean(a, axis=-1, keepdims=True)


def corr(a, b):
    """Pearson correlation of a (n, p) with b (p,) over the finite entries of b, per row of a."""
    ok = np.isfinite(b)
    a, b = a[:, ok], b[ok]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean()
    return (a @ b) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum())


def correlations(anom, ref, tmask, nz):
    """Pattern correlation of each member with ref per level, full and zonally asymmetric, {f : {kind : (b, nz)}}."""
    out = {}
    for f in FIELDS:
        full, asym = [], []
        for k in range(nz):
            m = tmask[k]
            full.append(corr(anom[f][:, k][:, m], ref[f][k][m]))
            asym.append(corr(zonal_asym(anom[f][:, k])[:, m], zonal_asym(ref[f][k])[m]))
        out[f] = {'full': np.stack(full, 1), 'asym': np.stack(asym, 1)}
    return out


def spectral_window(tmask_k):
    """Rows (slice) from the first to the last row holding ocean, and the columns (bool) that are ocean on all of them."""
    rows = np.flatnonzero(tmask_k.any(axis=1))
    rows = slice(rows[0], rows[-1] + 1)
    return rows, tmask_k[rows].all(axis=0)


def meridional_spectrum(a, tmask_k):
    """
    Power spectrum along latitude of a (b, y, x) at one level, over the rows that hold ocean, averaged over the members and
    the longitudes whose column is ocean on all these rows, Hann window, mean of each column removed, without the k = 0
    mode : (n // 2,) for n ocean rows.
    """
    rows, cols = spectral_window(tmask_k)
    x = a[:, rows][:, :, cols]
    x = x - x.mean(axis=1, keepdims=True)
    w = np.hanning(x.shape[1])[None, :, None]
    p = np.abs(np.fft.rfft(x * w, axis=1)) ** 2 / (w ** 2).sum()
    return p.mean(axis=(0, 2))[1:x.shape[1] // 2 + 1]


def quantize_range(a, lo, hi):
    """uint16 of a over [lo, hi] (clipped), 0 for NaN."""
    q = np.clip((a - lo) / (hi - lo), 0, 1) * 65534 + 1
    return np.where(np.isfinite(a), q, 0).astype('<u2')


def quantize(a, r):
    return quantize_range(a, -r, r)


def describe(fields, tmask, data_means, data_ref, clusters, file_mask_LR, volume, acc, nz, levels, n_maps, level_clusters, data_mode=None):
    """
    Statistics of one batch of unnormalized fields, see the module docstring. data_mode : most frequent level cluster of the
    data, {kind : (len(levels), y, x)}, None for the data itself (its agreement is set by the caller).
    """
    means = level_means(fields, tmask)
    anom = anomalies(fields, tmask, means)
    var_raw = {f: np.nanmean(np.where(tmask[None], fields[f], np.nan).var(axis=0), axis=(1, 2))[:nz] for f in FIELDS}
    var_anom = {f: np.nanmean(anom[f].var(axis=0), axis=(1, 2))[:nz] for f in FIELDS}
    out = {'offset': {f: means[f][:, :nz].tolist() for f in FIELDS},
           'var_raw': {f: var_raw[f].tolist() for f in FIELDS},
           'var_anom': {f: var_anom[f].tolist() for f in FIELDS},
           'maps': {f: b64(quantize(anom[f][:n_maps, levels], ANOMALY_RANGE[f])) for f in FIELDS},
           'n_maps': min(n_maps, fields['toce'].shape[0])}
    c = correlations(anom, data_ref, tmask, nz)
    out['corr'] = {f: {kind: v.tolist() for kind, v in c[f].items()} for f in FIELDS}
    # recentred : basin mean of every level moved onto the data one
    recentred = {f: fields[f] - np.nan_to_num(means[f] - data_means[f][None])[:, :, None, None] for f in FIELDS}
    for name, fl in [('raw', fields), ('recentred', recentred)]:
        hist = np.stack([ts_histograms(fl, acc, volume), ts_histograms(fl, tmask & ~acc, volume)], axis=1).mean(axis=0)
        labels = clusters.labels(fl)
        profiles, counts = clusters.mean_profiles(fl, labels)
        w = counts[:, :, None, None] * np.isfinite(profiles)
        profile = np.nansum(np.nan_to_num(profiles) * w, axis=0) / np.where(w.sum(axis=0) > 0, w.sum(axis=0), np.nan)
        out[name] = {'ts': b64(hist.astype('<f4')), 'labels': b64(labels), 'profiles': b64(profile.astype('<f4'))}
    out['level_clusters'] = {}
    centred = {'raw': {f: fields[f][:, levels] - data_means[f][levels][None, :, None, None] for f in FIELDS},
               'anom': {f: anom[f][:, levels] for f in FIELDS}}
    for kind, values in centred.items():
        labels = level_clusters.labels(values)
        out['level_clusters'][kind] = {'labels': gz64(labels[:n_maps])}
        if data_mode is not None:
            out['level_clusters'][kind]['agree'] = level_clusters.agreement(labels, data_mode[kind]).tolist()
    out['zonal'] = {f: b64(quantize_range(np.nanmean(np.where(tmask[None, :nz], fields[f][:n_maps, :nz], np.nan), axis=-1), *ZONAL_RANGE[f]))
                    for f in FIELDS}
    sections = zonal_sections(fields, file_mask_LR)
    out['section_mean'] = b64(np.nanmean(sections, axis=0).astype('<f4'))
    out['sst_mean'] = b64(np.where(tmask[0], fields['toce'][:, 0].mean(axis=0), np.nan).astype('<f4'))
    out['spectra'] = {f: [meridional_spectrum(anom[f][:, k], tmask[k]).tolist() for k in levels] for f in FIELDS}
    return out


if __name__ == '__main__':
    file_mask_LR = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').sel(time_counter=0)
    file_mask_LR = file_mask_LR.rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})
    config = TrainingConfig()
    observer = Observer('data', config, device='cpu')
    tmask = file_mask_LR.tmask.transpose('depth', 'nav_lat', 'nav_lon').values.astype(bool)
    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0).transpose('depth', 'nav_lat', 'nav_lon').values
    acc = acc_mask(tmask)
    depth = file_mask_LR.gdept_1d.values
    nz = tmask.shape[0] - 1
    levels = [int(np.abs(depth - d).argmin()) for d in MAP_DEPTHS]

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device='cpu')
    batch = next(iter(get_dataloader(config.data_file, transform, batch_size=N_DATA, shuffle=True, seed=DATA_SEED)))
    with torch.no_grad():
        data = {k: v.cpu().numpy() for k, v in observer.unnormalized(batch).items()}
    clusters = ProfileClusters({f: data[f][:N_DATA_FIT] for f in FIELDS}, tmask)
    data_means = {f: m.mean(axis=0) for f, m in level_means(data, tmask).items()}
    data_anom = anomalies(data, tmask, level_means(data, tmask))
    level_clusters = LevelClusters(data_anom, tmask, levels, data_means)
    data_labels = {'raw': level_clusters.labels({f: data[f][:, levels] - data_means[f][levels][None, :, None, None] for f in FIELDS}),
                   'anom': level_clusters.labels({f: data_anom[f][:, levels] for f in FIELDS})}
    data_counts = {kind: level_clusters.counts(v) for kind, v in data_labels.items()}
    data_mode = {kind: level_clusters.mode(c) for kind, c in data_counts.items()}
    data_ref = {f: a.mean(axis=0) for f, a in anomalies(data, tmask, level_means(data, tmask)).items()}

    out = {'data': describe(data, tmask, data_means, data_ref, clusters, file_mask_LR, volume, acc, nz, levels, N_DATA_MAPS, level_clusters),
           'runs': {}}
    # Data labels against the most frequent label of the other states (leave one out)
    for kind, labels in data_labels.items():
        one_hot = np.stack([labels == c + 1 for c in range(level_clusters.k)], axis=1)
        out['data']['level_clusters'][kind]['agree'] = np.mean(
            [level_clusters.agreement(labels[i:i + 1], level_clusters.mode(data_counts[kind] - one_hot[i])) for i in range(N_DATA)], axis=0).tolist()
    # The data members are compared with the mean of the others (leave one out), not with a mean that holds them
    anom = anomalies(data, tmask, level_means(data, tmask))
    loo = {f: [] for f in FIELDS}
    for i in range(N_DATA):
        ref = {f: (anom[f].sum(axis=0) - anom[f][i]) / (N_DATA - 1) for f in FIELDS}
        c = correlations({f: anom[f][i:i + 1] for f in FIELDS}, ref, tmask, nz)
        for f in FIELDS:
            loo[f].append({kind: v[0] for kind, v in c[f].items()})
    out['data']['corr'] = {f: {kind: np.stack([l[kind] for l in loo[f]]).tolist() for kind in ['full', 'asym']} for f in FIELDS}
    # Latitude-only reference : the zonal mean of the data mean anomaly, broadcast over longitude
    lat_only = {f: np.broadcast_to(np.nanmean(data_ref[f], axis=-1, keepdims=True), data_ref[f].shape) for f in FIELDS}
    out['lat_only'] = {f: [float(corr(lat_only[f][k][tmask[k]][None], data_ref[f][k][tmask[k]])[0]) for k in range(nz)]
                       for f in FIELDS}

    for name in BASELINES:
        run_dir = sorted(glob(f'{INFERENCE}/default_eps/{name}/*/'))[-1]
        with open(f'{run_dir}/args.json') as f:
            projection = json.load(f).get('projection', 'none')
        x = torch.from_numpy(np.load(f'{run_dir}/generated.npy'))
        with torch.no_grad():
            fields = {k: v.cpu().numpy() for k, v in observer.unnormalized(x).items()}
        out['runs'][name] = describe(fields, tmask, data_means, data_ref, clusters, file_mask_LR, volume, acc, nz, levels, 10 ** 6,
                                     level_clusters, data_mode)
        out['runs'][name]['projection'] = projection
        out['runs'][name]['dir'] = run_dir.replace(INFERENCE, '')
        print(f'{name} : {run_dir}')

    out['depth'] = depth[:nz].tolist()
    out['spatial_std'] = {f: np.stack([data[f][:, k][:, tmask[k]].std(axis=1) for k in range(nz)], axis=1).mean(axis=0).tolist()
                          for f in FIELDS}
    # Spatial distribution of each level, ocean cells of all data states pooled, plain cell count : (len(LAYER_QUANTILES), nz)
    out['layer_quantiles'] = {'q': LAYER_QUANTILES, **{f: np.stack([np.percentile(data[f][:, k][:, tmask[k]], LAYER_QUANTILES)
                                                                   for k in range(nz)], axis=1).tolist() for f in FIELDS}}
    out['shift_levels'] = sorted({int(np.abs(depth[:nz] - d).argmin()) for d in SHIFT_DEPTHS})
    out['level_clusters'] = {'n': level_clusters.k, 'T': [c[:, 0].tolist() for c in level_clusters.centres],
                             'S': [c[:, 1].tolist() for c in level_clusters.centres], 'sigma': [s.tolist() for s in level_clusters.sigma]}
    out['map_levels'] = levels
    out['map_depths'] = depth[levels].tolist()
    out['anomaly_range'] = ANOMALY_RANGE
    out['zonal_range'] = ZONAL_RANGE
    out['colorscales'] = {f: [[float(v), f'rgb{tuple(int(255 * c) for c in cmap(v)[:3])}'] for v in np.linspace(0, 1, 11)]
                          for f, cmap in [('toce', cm.thermal), ('soce', cm.haline)]}
    out['lon'] = (file_mask_LR.glamt.values[0] - 60).tolist()
    out['lat'] = file_mask_LR.gphit.values[:, 0].tolist()
    out['n_clusters'] = N_CLUSTERS
    out['ts_t'] = ((TS_T_EDGES[1:] + TS_T_EDGES[:-1]) / 2).tolist()
    out['ts_s'] = ((TS_S_EDGES[1:] + TS_S_EDGES[:-1]) / 2).tolist()
    t_grid, s_grid = np.meshgrid(out['ts_t'], out['ts_s'], indexing='ij')
    out['ts_sigma'] = (get_density_at_surface(t_grid, s_grid, np.ones_like(t_grid)) - 1000).tolist()
    out['cluster_centres'] = {'T': clusters.centres[:, :nz].tolist(), 'S': clusters.centres[:, clusters.nz:clusters.nz + nz].tolist(),
                              'surface_sigma': clusters.surface_sigma.tolist()}
    windows = [spectral_window(tmask[k]) for k in levels]
    out['spectrum_columns'] = [int(cols.sum()) for _, cols in windows]
    n_rows = [rows.stop - rows.start for rows, _ in windows]
    out['spectrum_wavelength'] = [(n / np.arange(1, n // 2 + 1)).tolist() for n in n_rows]  # [grid cells along latitude]
    OUT.write_text(f'window.NOCONSTRAINT_DATA = {json.dumps(out)};\n')
    print(f'Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')
