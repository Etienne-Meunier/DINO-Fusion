"""
Steps 0 and 1 of PLAN.md : reproduce the eval density error E (`compute_density_error`) and split it by the magnitude of
the inversion |g|, g = rho[k+1] - rho[k] < 0, with the density computed in float32 (as the torch observer) and in float64
(as the eval), on the training data and on saved generations.

Output : `step1_bins.csv`, `step1_depth.csv`, `step1_bins.png` in this folder.
"""
from pathlib import Path
import sys
PRP = f'{Path(__file__).resolve().parents[4]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/analysis_scripts/')
sys.path.append(PRP)

import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from metrics import get_density_at_surface_tensor, compute_density_error

HERE = Path(__file__).parent
R = '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference/test_new_sampler_clip'
RUNS = {'NONE' : 'none', 'NGD' : 'negative_grad_density_sqrt_eta0.3_eps0',
        'X64' : 'combo_iso5e-6_zm0.1_eps1.66e-1', 'X64J' : 'combo_iso5e-6_zm0.1_eps1.66e-1_joint0.01'}
MESH = f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc'
BINS = [0.0, 1.2e-4, 1e-3, 1e-2, np.inf]
BIN_NAMES = ['<1.2e-4', '1.2e-4-1e-3', '1e-3-1e-2', '>=1e-2']
BANDS = {'0-9' : slice(0, 10), '10-24' : slice(10, 25), '25-35' : slice(25, 36)}


def physical(x, observer) :
    """Temperature and salinity (b, 36, i, j) float32 on the file levels of normalised states x (b, c, i, j)."""
    with torch.no_grad() :
        u = observer.unnormalized(x)
    return u['toce'], u['soce']


def split_E(T, S, volume, dtype) :
    """
        Density error split by inversion magnitude, density in `dtype`.
    Returns :
        E (b,) total %, E_bins (b, 4) %, E_depth (b, 35) % (contribution of the inversions below each level k),
        ties (b,) % volume with g == 0 exactly, g (b, 35, i, j)
    """
    rho = get_density_at_surface_tensor(T.to(dtype), S.to(dtype))
    g = rho[:, 1:] - rho[:, :-1]
    v = volume[1:].to(dtype)                      # xarray diff keeps the lower level, weighted by its volume
    total = volume.sum().to(dtype)
    inv = (g < 0) & torch.isfinite(g)
    mag = -g
    E_bins = torch.stack([((inv & (mag >= lo) & (mag < hi)) * v).nansum(dim=(1, 2, 3)) for lo, hi in zip(BINS[:-1], BINS[1:])], 1)
    E_depth = (inv * v).nansum(dim=(2, 3))
    ties = ((g == 0) * v).nansum(dim=(1, 2, 3))
    return 100 * E_bins.sum(1) / total, 100 * E_bins / total, 100 * E_depth / total, 100 * ties / total, g


def eval_E(T, S, mesh) :
    """E of the eval function itself, on the same T, S."""
    batch = {'toce' : np.nan_to_num(T.numpy()), 'soce' : np.nan_to_num(S.numpy())}
    return compute_density_error(batch, mesh)


if __name__ == '__main__' :
    config = TrainingConfig()
    observer = Observer('data', config)
    volume = observer.volume
    mesh = xr.open_dataset(MESH).isel(time_counter=0).rename({'nav_lev' : 'depth', 'y' : 'nav_lat', 'x' : 'nav_lon'})

    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)])
    rng = np.random.default_rng(0)
    sets = {'DATA' : states[rng.choice(len(states), 64, replace=False)]}
    del states
    sets.update({k : torch.tensor(np.load(f'{R}/{v}/generated.npy')) for k, v in RUNS.items()})

    rows, depth_rows, f32_ties = [], [], {}
    for name, x in sets.items() :
        T, S = physical(x, observer)
        E_ref = eval_E(T, S, mesh)
        for dt, dtype in [('f32', torch.float32), ('f64', torch.float64)] :
            E, E_bins, E_depth, ties, g = split_E(T, S, volume, dtype)
            if dt == 'f32' :
                f32_ties[name] = (g == 0)
            else :
                on_tie = (((g < 0) & f32_ties[name]) * volume[1:]).nansum(dim=(1, 2, 3)) / (((g < 0) & torch.isfinite(g)) * volume[1:]).nansum(dim=(1, 2, 3)).clamp(min=1e-30)
            for b in range(len(x)) :
                row = {'set' : name, 'dtype' : dt, 'sample' : b, 'E' : E[b].item(), 'E_eval' : float(E_ref[b]), 'ties' : ties[b].item(),
                       **{f'E_{n}' : E_bins[b, i].item() for i, n in enumerate(BIN_NAMES)}}
                for band, s in BANDS.items() :
                    row[f'E_lev{band}'] = E_depth[b, s].sum().item()
                if dt == 'f64' :
                    row['f64_inv_on_f32_tie'] = on_tie[b].item()
                rows.append(row)
            depth_rows += [{'set' : name, 'dtype' : dt, 'level' : k + 1, 'E' : E_depth[:, k].mean().item()} for k in range(E_depth.shape[1])]
        print(name, 'done')

    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'step1_bins.csv', index=False)
    pd.DataFrame(depth_rows).to_csv(HERE / 'step1_depth.csv', index=False)
    cols = ['E_eval', 'E', 'ties'] + [f'E_{n}' for n in BIN_NAMES] + [f'E_lev{b}' for b in BANDS] + ['f64_inv_on_f32_tie']
    summary = df.groupby(['set', 'dtype'], sort=False)[cols].mean()
    pd.set_option('display.width', 250)
    print(summary.to_string(float_format=lambda v : f'{v:.4g}'))
    summary.to_csv(HERE / 'step1_summary.csv')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dt in zip(axes, ['f32', 'f64']) :
        s = summary.xs(dt, level='dtype')[[f'E_{n}' for n in BIN_NAMES]]
        s.plot.bar(stacked=True, ax=ax, rot=0)
        ax.set_title(f'density in {dt}')
        ax.set_ylabel('E (% ocean volume)')
    fig.tight_layout()
    fig.savefig(HERE / 'step1_bins.png', dpi=120)
