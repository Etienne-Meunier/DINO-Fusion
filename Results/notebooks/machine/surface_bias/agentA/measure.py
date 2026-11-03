"""
Surface density bias of the C7 constraint (negative_grad_density_huber): per-run sigma_0 fields, horizontal-mean
profiles, T/S hybrids, paired differences against the unconstrained run of the same seed. Saves `measure.npz`;
`report.py` prints the tables.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
from common import *

OUT = HERE / 'measure.npz'
SWEEP = f'{INFERENCE}/sweep_density_huber/negative_grad_density_huber_border0'
B16 = f'{INFERENCE}/sweep_density_huber_b16/negative_grad_density_huber_border0/181026-165339'
OPT = f'{INFERENCE}/optimal'

mask = load_mask()
tmask = mask.tmask.values.astype(bool)
volume = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values
loss_fn = get_loss('negative_grad_density_huber', observer)


def fields_of(x):
    """Unnormalized (T, S) numpy arrays (b, 36, 208, 64) of a normalized batch."""
    f = observer.unnormalized(x)
    return f['toce'].detach().cpu().numpy().astype(np.float64), f['soce'].detach().cpu().numpy().astype(np.float64)


def sigma(T, S):
    """sigma_0 = rho - 1000 (b, z, y, x), NaN on land."""
    rho = compute_density({'toce': T, 'soce': S}, mask).transpose('batch', 'depth', 'nav_lat', 'nav_lon').values - 1000
    return np.where(tmask[None], rho, np.nan)


def run_dirs():
    """(name, eta, batch, dir) of every run used, the b8 paired set then the b16 paired set."""
    out = [('none_b8', 0.0, 8, f'{INFERENCE}/sweep_baselines/none_border0_b8/151026-145216')]
    for d in sorted(glob(f'{SWEEP}/*/')):
        eta = json.load(open(f'{d}/args.json'))['eta']
        out.append((f'eta{eta:g}_b8', eta, 8, d))
    out.append(('none_b16', 0.0, 16, generation_dirs()['none_border0']))
    for k in ['negative_grad_density_huber_border0', 'negative_grad_density_huber_border0_30',
              'negative_grad_density_huber_border0_80']:
        d = generation_dirs()[k]
        out.append((f"eta{json.load(open(f'{d}/args.json'))['eta']:g}_b16", None, 16, d))
    out.append(('eta100_b16', 100.0, 16, B16))
    return out


res = {}
T, S = fields_of(batch)
res['data/T'], res['data/S'], res['data/sig'] = T, S, sigma(T, S)
res['data/loss'] = loss_fn(batch).numpy()
for name, eta, b, d in run_dirs():
    x = torch.from_numpy(np.load(f'{d}/generated.npy'))
    a = json.load(open(f'{d}/args.json'))
    T, S = fields_of(x)
    res[f'{name}/T'], res[f'{name}/S'], res[f'{name}/sig'] = T, S, sigma(T, S)
    res[f'{name}/loss'] = loss_fn(x).numpy()
    res[f'{name}/eta'] = np.float64(a['eta'])
    res[f'{name}/eps'] = np.float64(a['eps'])
    res[f'{name}/lbd'] = np.load(f'{d}/lbd.npy') if os.path.exists(f'{d}/lbd.npy') else np.zeros(b)
    print(name, a['eta'], a['batch'], a['seed'], flush=True)
for name, eta, b, d in run_dirs():
    if name.startswith('none'):
        continue
    base = 'none_b8' if b == 8 else 'none_b16'
    res[f'{name}/sig_Teta_Snone'] = sigma(res[f'{name}/T'], res[f'{base}/S'])
    res[f'{name}/sig_Tnone_Seta'] = sigma(res[f'{base}/T'], res[f'{name}/S'])
np.savez(OUT, volume=volume, depth=mask.gdept_1d.values.squeeze(), lat=mask.gphit.isel(nav_lon=10).values,
         **{k: np.asarray(v, dtype=np.float32) for k, v in res.items()})
print('saved', OUT)
