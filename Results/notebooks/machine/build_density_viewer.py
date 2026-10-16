"""
Build `density_viewer_data.js` and `density_viewer_apriori.js`, the data of `density_viewer.html` : zonal sections of sigma_0 for 8 random data states and
every run of the strength sweep (TS : zero_mean, C1 : mean_density) and the unconstrained runs of default_eps (with and
without BorderZero), with the constraint value C(x) and the density error of each member.
The a-priori file holds, for the same states, the SST map, the mixed layer depth map, the volume-weighted T-S histogram
and the batch variance of SST and SSS (as `graphs/graph4_snapshots.py`).
"""
from pathlib import Path
import sys
PRP = f'{Path(__file__).resolve().parents[3]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/analysis_scripts/')
sys.path.append(PRP)

import base64
import random
import json
import os
from glob import glob

import cmocean.cm as cm
import numpy as np
import torch
import xarray as xr

from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.losses import get_loss
from metrics import compute_density_error, get_density_at_surface

INFERENCE = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b/inference'
SWEEP = f'{INFERENCE}/sweep_strength'
BASELINES = ['none', 'none_border0']  # latest run of each in default_eps
OUT = Path(__file__).parent / 'density_viewer_data.js'
OUT_APRIORI = Path(__file__).parent / 'density_viewer_apriori.js'
SST_RANGE = (-2., 32.)  # SST maps are stored as uint8 over this range, NaN as 0
MLD_DSIGMA, MLD_ZREF = 0.03, 10.  # mixed layer depth : sigma_0 exceeds its value at MLD_ZREF [m] by MLD_DSIGMA [kg m-3]
TS_T_EDGES = np.arange(-3., 32.01, 0.5)  # T-S histogram bins, values outside fall in the edge bins
TS_S_EDGES = np.arange(32., 38.001, 0.05)
TS_LOG_RANGE = (-10., 0.)  # T-S histograms are stored as uint16 of log10(volume fraction) over this range, 0 for empty bins
N_DATA = 8
SIGMA_RANGE = (18., 30.)  # sections are stored as uint16 over this range, NaN as 0


def compute_density(fields, file_mask_LR):
    """Potential density referenced at the surface, (batch, depth, nav_lat, nav_lon), of unnormalized 'toce', 'soce'."""
    bsize = fields['soce'].shape[0]
    data = file_mask_LR.e3t_0.copy().expand_dims({'batch': bsize})
    soce, toce, tmask = data.copy(), data.copy(), data.copy()
    soce[:] = np.asarray(fields['soce'])
    toce[:] = np.asarray(fields['toce'])
    tmask[:] = np.repeat(file_mask_LR.tmask.values[None], bsize, 0)
    return get_density_at_surface(toce, soce, tmask)


def zonal_sections(fields, file_mask_LR):
    """Volume-weighted zonal mean of sigma_0, (batch, depth - 1, nav_lat), last (land) level dropped."""
    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0 * file_mask_LR.tmask).squeeze()
    volume = volume.transpose('depth', 'nav_lat', 'nav_lon')
    density = compute_density(fields, file_mask_LR)
    section = (density * volume).sum('nav_lon') / volume.sum('nav_lon')
    return (section.transpose('batch', 'depth', 'nav_lat')[:, :-1] - 1000).values


def encode(sections):
    """Quantize float sections to uint16 over SIGMA_RANGE (0 = NaN) and return them base64 encoded."""
    lo, hi = SIGMA_RANGE
    q = np.clip((sections - lo) / (hi - lo), 0, 1) * 65534 + 1
    q = np.where(np.isfinite(sections), q, 0).astype('<u2')
    return base64.b64encode(q.tobytes()).decode()


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def mixed_layer_depth(sigma, tmask, depth):
    """
    Depth [m] where sigma_0 first exceeds its value at MLD_ZREF by MLD_DSIGMA, linearly interpolated between levels,
    the deepest ocean level if never reached, NaN on land.
        sigma (b, z, y, x), tmask (z, y, x) bool, depth (z,)
    """
    k1 = int(np.searchsorted(depth, MLD_ZREF))
    w = (MLD_ZREF - depth[k1 - 1]) / (depth[k1] - depth[k1 - 1])
    target = (1 - w) * sigma[:, k1 - 1] + w * sigma[:, k1] + MLD_DSIGMA
    above = (sigma > target[:, None]) & tmask[None]
    above[:, :k1] = False
    reached = above.any(axis=1)
    k = above.argmax(axis=1).clip(min=1)
    take = lambda a, kk: np.take_along_axis(a, kk[:, None], axis=1)[:, 0]
    s0, s1 = take(sigma, k - 1), take(sigma, k)
    frac = np.clip((target - s0) / np.where(s1 != s0, s1 - s0, 1), 0, 1)
    mld = depth[k - 1] + frac * (depth[k] - depth[k - 1])
    bottom = depth[(tmask.sum(axis=0) - 1).clip(min=0)]
    return np.where(tmask[0], np.where(reached, mld, bottom), np.nan)


def acc_mask(tmask):
    """ACC region (z, y, x) : the latitude rows of the reentrant channel, where the surface ocean spans every longitude."""
    channel = tmask[0].all(axis=1)
    return tmask & channel[None, :, None]


def ts_histograms(fields, region, volume):
    """Fraction of the volume of `region` (z, y, x) in each (T, S) bin, (b, len(TS_T_EDGES) - 1, len(TS_S_EDGES) - 1)."""
    t = np.clip(fields['toce'][:, region], TS_T_EDGES[0], TS_T_EDGES[-1] - 1e-6)
    s = np.clip(fields['soce'][:, region], TS_S_EDGES[0], TS_S_EDGES[-1] - 1e-6)
    w = volume[region] / volume[region].sum()
    return np.stack([np.histogram2d(tb, sb, bins=[TS_T_EDGES, TS_S_EDGES], weights=w)[0] for tb, sb in zip(t, s)])


def describe(x, observer, losses, file_mask_LR):
    """
    Of a normalized batch x (b, c, i, j), return :
        entry : zonal sections, C(x) per loss and density error of each member
        apriori : SST maps (uint8 (b, y, x)), mixed layer depth maps (uint16 [m] (b, y, x), 0 on land), T-S histograms
                  of the ACC and of the rest of the basin, each normalized by its volume (uint16 of log10 volume
                  fraction (b, 2, nT, nS)), batch variance of SST and SSS (float32 (2, y, x)) and its mean over the ocean
    """
    with torch.no_grad():
        fields = {k: v.detach().cpu().numpy() for k, v in observer.unnormalized(x).items()}
        c = {name: f(x).cpu().numpy().tolist() for name, f in losses.items()}
    tmask = file_mask_LR.tmask.transpose('depth', 'nav_lat', 'nav_lon').values.astype(bool)
    variance = np.stack([np.where(tmask[0], fields[f][:, 0].var(axis=0), np.nan) for f in ['toce', 'soce']])
    entry = {'sections': encode(zonal_sections(fields, file_mask_LR)),
             'C': c,
             'density_error': compute_density_error(fields, file_mask_LR).tolist()}
    sigma = np.where(tmask[None], compute_density(fields, file_mask_LR).transpose('batch', 'depth', 'nav_lat', 'nav_lon').values - 1000, np.nan)
    depth = file_mask_LR.gdept_1d.values
    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0).transpose('depth', 'nav_lat', 'nav_lon').values
    lo, hi = SST_RANGE
    sst = np.where(tmask[0], fields['toce'][:, 0], np.nan)
    sst = np.where(np.isfinite(sst), np.clip((sst - lo) / (hi - lo), 0, 1) * 254 + 1, 0).astype('u1')
    mld = np.nan_to_num(np.clip(mixed_layer_depth(sigma, tmask, depth), 1, 65535), nan=0).astype('<u2')
    acc = acc_mask(tmask)
    hist = np.stack([ts_histograms(fields, acc, volume), ts_histograms(fields, tmask & ~acc, volume)], axis=1)
    lo_ts, hi_ts = TS_LOG_RANGE
    log_hist = np.log10(np.where(hist > 0, hist, 1))
    ts = np.where(hist > 0, np.clip((log_hist - lo_ts) / (hi_ts - lo_ts), 0, 1) * 65534 + 1, 0).astype('<u2')
    apriori = {'sst': b64(sst), 'mld': b64(mld), 'ts': b64(ts),
               'var': b64(variance.astype('<f4')),
               'var_mean': np.nanmean(variance, axis=(1, 2)).tolist()}
    return entry, apriori


if __name__ == '__main__':
    file_mask_LR = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').sel(time_counter=0)
    file_mask_LR = file_mask_LR.rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})
    config = TrainingConfig()
    observer = Observer('data', config, device='cpu')
    losses = {name: get_loss(name, observer) for name in ['zero_mean', 'mean_density']}

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device='cpu')
    batch = next(iter(get_dataloader(config.data_file, transform, batch_size=N_DATA, shuffle=True)))
    out = {'runs': [], 'baselines': {}}
    apriori = {'runs': [], 'baselines': {}}
    out['data'], apriori['data'] = describe(batch, observer, losses, file_mask_LR)

    for name in BASELINES:
        run_dir = sorted(glob(f'{INFERENCE}/default_eps/{name}/*/'))[-1]
        with open(f'{run_dir}/args.json') as f:
            projection = json.load(f).get('projection', 'none')
        out['baselines'][name], apriori['baselines'][name] = describe(torch.from_numpy(np.load(f'{run_dir}/generated.npy')),
                                                                      observer, losses, file_mask_LR)
        out['baselines'][name]['projection'] = projection
        print(f'{name} : {run_dir}')

    for args_path in sorted(glob(f'{SWEEP}/*/*/args.json')):
        with open(args_path) as f:
            args = json.load(f)
        x = torch.from_numpy(np.load(f'{os.path.dirname(args_path)}/generated.npy'))
        run, run_apriori = describe(x, observer, {args['loss']: losses[args['loss']]}, file_mask_LR)
        apriori['runs'].append(run_apriori)
        out['runs'].append(run | {k: args[k] for k in ['loss', 'eps', 'eta', 'lbd0', 'projection']})
        print(f"{args['loss']} eps={args['eps']} eta={args['eta']} lbd0={args['lbd0']}")

    depth = file_mask_LR.gdept_1d.values[:-1]
    out['depth'] = depth.tolist()
    out['latitude'] = file_mask_LR.gphit.isel(nav_lon=10).values.tolist()
    out['sigma_range'] = SIGMA_RANGE
    out['colorscale'] = [[float(v), f'rgb{tuple(int(255 * c) for c in cm.deep_r(v)[:3])}'] for v in np.linspace(0, 1, 11)]
    OUT.write_text(f'window.DENSITY_DATA = {json.dumps(out)};\n')
    print(f'Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')

    apriori['lon'] = (file_mask_LR.glamt.values[0] - 60).tolist()
    apriori['lat'] = file_mask_LR.gphit.values[:, 0].tolist()
    apriori['sst_range'] = SST_RANGE
    apriori['ts_t'] = ((TS_T_EDGES[1:] + TS_T_EDGES[:-1]) / 2).tolist()
    apriori['ts_s'] = ((TS_S_EDGES[1:] + TS_S_EDGES[:-1]) / 2).tolist()
    apriori['ts_log_range'] = TS_LOG_RANGE
    channel_lat = file_mask_LR.gphit.values[:, 0][acc_mask(file_mask_LR.tmask.transpose('depth', 'nav_lat', 'nav_lon').values.astype(bool))[0].any(axis=1)]
    apriori['acc_lat'] = [float(channel_lat.min()), float(channel_lat.max())]
    t_grid, s_grid = np.meshgrid(apriori['ts_t'], apriori['ts_s'], indexing='ij')
    apriori['ts_sigma'] = (get_density_at_surface(t_grid, s_grid, np.ones_like(t_grid)) - 1000).tolist()
    apriori['mld_def'] = f'sigma_0 > sigma_0({MLD_ZREF:.0f} m) + {MLD_DSIGMA}'
    apriori['colorscales'] = {f: [[float(v), f'rgb{tuple(int(255 * c) for c in cmap(v)[:3])}'] for v in np.linspace(0, 1, 11)]
                              for f, cmap in [('toce', cm.thermal), ('soce', cm.haline), ('mld', cm.deep), ('ts', cm.amp), ('ts_acc', cm.ice_r)]}
    OUT_APRIORI.write_text(f'window.APRIORI_DATA = {json.dumps(apriori)};\n')
    print(f'Saved {OUT_APRIORI} ({OUT_APRIORI.stat().st_size / 1e6:.1f} MB)')
