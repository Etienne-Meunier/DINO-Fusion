"""
Database of the constrained generations of `sweep_run.py` and `campaign_run.py` : one row per run directory of `sweep_path` with
its arguments, the trajectory of its dual variables and its a priori metrics, plus reference rows (data, paper generations).

    enforcement : density errors (% of volume with a density inversion, Table 1),
                  RMSE of the mean density profile and of its vertical gradient against the training mean
    variance    : mean point by point variance of T and S over the batch (Table 2)

The metrics of a run are cached in `<run>/a_priori_metrics.json`. Writes `<sweep_path>/database.csv` and a copy in
`figures/sweep/database.csv`.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from load_data import load_data_batch, load_mask, paper_path, LABELS
from metrics import get_density_at_surface
from sweep_run import sweep_path
from __init__ import PRP

figures_path = f'{PRP}/Results/a_priori/figures/sweep/'
mask = load_mask()
tmask = mask.tmask.transpose('depth', 'nav_lat', 'nav_lon').values
volume = (mask.e1t * mask.e2t * mask.e3t_0).transpose('depth', 'nav_lat', 'nav_lon').values * tmask
mean_density = np.loadtxt(f'{PRP}/Results/analysis_scripts/mean_density_train.txt')[:-1]


def compute_metrics(toce, soce) :
    """
        toce, soce (b, 36, 199, 62) : un-normalised states
    Returns :
        dict of enforcement and variance metrics, averaged over the batch ({'diverged' : True} if a state is not finite)
    """
    ocean = np.broadcast_to(tmask[None] == 1, toce.shape)
    if not (np.isfinite(toce[ocean]).all() and np.isfinite(soce[ocean]).all()) :
        return {'diverged' : True}
    density = get_density_at_surface(toce, soce, tmask[None])
    inversions = (density[:, 1:] - density[:, :-1]) < 0.0
    density_errors = 100 * (inversions * volume[None, 1:]).sum(axis=(1, 2, 3)) / volume.sum()
    profile = np.nanmean(np.where(tmask[None] == 1, density, np.nan), axis=(-2, -1))[:, :-1]
    gradient = profile[:, 1:] - profile[:, :-1]
    return {'diverged' : False,
            'density_errors' : density_errors.mean(),
            'density_errors_std' : density_errors.std(),
            'profile_rmse' : np.sqrt(((profile - mean_density[None])**2).mean(axis=1)).mean(),
            'gradient_rmse' : np.sqrt(((gradient - np.diff(mean_density)[None])**2).mean(axis=1)).mean(),
            'var_T' : np.nanmean(np.var(toce, axis=0)),
            'var_S' : np.nanmean(np.var(soce, axis=0))}


def describe(name) :
    """
        Arguments of a run (args.json) and summary of its dual variables along sampling (metrics.csv).
    Returns :
        dict with family (method, loss or projection, x_t or Tweedie), beta_type, beta, eta, eps, eta_rho, tau and
        final / mean beta, final rho and norm of lambda, number of sampling steps done
    """
    args = json.load(open(f'{sweep_path}/{name}/args.json'))
    tweedie = bool(args.get('tweedie', False))
    if args['method'] == 'gradient' :
        family = f'gradient{"_tweedie" if tweedie else ""}_{args["loss"]}'
        beta_type = args['beta_type']
    else :
        family = {'none' : 'none', 'projection' : f'projection_{args.get("projection")}'}[args['method']]
        beta_type = None
    d = {'family' : family, 'method' : args['method'], 'loss' : args.get('loss') if args['method'] == 'gradient' else None,
         'projection' : args.get('projection') if args['method'] == 'projection' else None, 'tweedie' : tweedie,
         'beta_type' : beta_type, 'beta' : args['beta'], 'eta' : args['eta'], 'eps' : args.get('eps', 0.0),
         'eta_rho' : args['eta_rho'], 'tau' : args['tau']}
    if name == 'border_zero_only' :
        d |= {'family' : 'border_zero_only', 'loss' : None}
    if Path(f'{sweep_path}/{name}/metrics.csv').exists() :
        m = pd.read_csv(f'{sweep_path}/{name}/metrics.csv')
        d |= {'n_steps' : len(m),
              'final_beta' : m['mean(beta)'].iloc[-1] if 'mean(beta)' in m else np.nan,
              'mean_beta' : m['mean(beta)'].mean() if 'mean(beta)' in m else np.nan,
              'final_rho' : m['rho'].iloc[-1] if 'rho' in m else np.nan,
              'final_norm_lambda' : m['norm(lbd)'].iloc[-1] if 'norm(lbd)' in m else np.nan}
    return d


def run_metrics(name) :
    """
        A priori metrics of a run, cached in `<run>/a_priori_metrics.json`.
    """
    cache = Path(f'{sweep_path}/{name}/a_priori_metrics.json')
    if cache.exists() :
        return json.load(open(cache))
    clean = f'{sweep_path}/{name}/clean'
    m = {k : (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in compute_metrics(np.load(f'{clean}/toce.npy'), np.load(f'{clean}/soce.npy')).items()}
    json.dump(m, open(cache, 'w'), indent=2)
    return m


def references() :
    """
        Metrics of 64 training states and of the paper generations.
    """
    batch = load_data_batch(batch_size=64)
    refs = [{'run' : 'data', 'family' : 'reference'} | compute_metrics(batch['toce'].numpy(), batch['soce'].numpy())]
    for key, p in paper_path.items() :
        refs.append({'run' : f'paper {LABELS[key]}', 'family' : 'reference'} | compute_metrics(np.load(f'{p}/toce.npy'), np.load(f'{p}/soce.npy')))
    return refs


def build() :
    """
        Returns the database (pd.DataFrame) of all finished runs and the references, and writes it.
    """
    rows = []
    for run in sorted(Path(sweep_path).iterdir()) :
        if (run / 'clean' / 'toce.npy').exists() and (run / 'args.json').exists() :
            rows.append({'run' : run.name} | describe(run.name) | run_metrics(run.name))
    df = pd.DataFrame(references() + rows)
    df.to_csv(f'{sweep_path}/database.csv', index=False)
    Path(figures_path).mkdir(exist_ok=True, parents=True)
    df.to_csv(f'{figures_path}/database.csv', index=False)
    return df


if __name__ == '__main__' :
    df = build()
    print(f'{len(df)} rows, {df["diverged"].fillna(False).astype(bool).sum()} diverged runs')
