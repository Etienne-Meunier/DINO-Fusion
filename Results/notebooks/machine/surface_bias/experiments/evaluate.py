"""
Table of the experimental runs (runs/*) against the data batch of `paper/common.py`, the unconstrained run none_border0_b8
and the C7 eta sweep, paired member by member with the unconstrained run of the same seed (ref.npy of an anchor run, runs/none_s<seed>, or
none_border0_b8 for seed 0) :
    sigma_0 (volume-weighted horizontal mean) of level 0, 5-49 m, 61-299 m, whole ocean, as a paired difference to none
    and as a bias to the data ; T and S of level 0 ; eval density error E (% volume) ; C7 loss / eps ; RMS distance to
    none (normalised) ; final lbd.
    python evaluate.py [run names or glob patterns ...]     default : every run of runs/
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
EPS_C7 = eps_data[False]['negative_grad_density_huber']

mask = load_mask()
tmask = mask.tmask.values.astype(bool)
volume = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values
depth = mask.gdept_1d.values.squeeze() if 'gdept_1d' in mask else None
c7 = get_loss('negative_grad_density_huber', observer)
LAYERS = {'lev0' : slice(0, 1), '5-49m' : slice(0, 5), '61-299m' : slice(5, 12), 'all' : slice(0, 36)}


def summarise(x) :
    """Per-member diagnostics (dict of (b,) arrays) of a normalised batch x."""
    f = observer.unnormalized(x)
    T, S = (f[k].detach().numpy().astype(np.float64) for k in ['toce', 'soce'])
    sig = np.where(tmask[None], compute_density({'toce' : T, 'soce' : S}, mask)
                   .transpose('batch', 'depth', 'nav_lat', 'nav_lon').values - 1000, np.nan)
    out = {}
    for k, sl in LAYERS.items() :
        v = volume[sl]
        out[k] = np.nansum(sig[:, sl] * v, axis=(1, 2, 3)) / v.sum()
    v0 = volume[0] / volume[0].sum()
    out['T0'], out['S0'] = np.nansum(T[:, 0] * v0, axis=(1, 2)), np.nansum(S[:, 0] * v0, axis=(1, 2))
    out['E'] = compute_density_error({'toce' : T, 'soce' : S}, mask)
    out['c7/eps'] = c7(x).numpy() / EPS_C7
    return out


def paired_none(d) :
    """Unconstrained batch of the seed of run directory d, runs/centred_none_b<batch> for a batch other than 8."""
    args = json.load(open(d / 'args.json'))
    seed = args['seed']
    if args['batch'] != 8 :
        return np.load(HERE / 'runs' / f'centred_none_b{args["batch"]}' / 'generated.npy')
    if (d / 'ref.npy').exists() :
        return np.load(d / 'ref.npy')
    return np.load(NONE if seed == 0 else HERE / 'runs' / f'none_s{seed}' / 'generated.npy')


def runs(names) :
    """(name, normalised batch, lbd or None, paired none batch) of the reference C7 runs and of the experimental runs."""
    out = []
    for d in sorted(glob(f'{SWEEP}/*/')) :
        eta = json.load(open(f'{d}/args.json'))['eta']
        if eta in (30., 100.) :
            out.append((f'C7 eta{eta:g} (repo)', np.load(f'{d}/generated.npy'), np.load(f'{d}/lbd.npy'), XN))
    for n in names :
        d = HERE / 'runs' / n
        lbd = np.load(d / 'lbd.npy') if (d / 'lbd.npy').exists() else None
        out.append((n, np.load(d / 'generated.npy'), lbd, paired_none(d)))
    return out


patterns = sys.argv[1:] or ['*']
names = sorted({p.name for q in patterns for p in (HERE / 'runs').glob(q) if p.is_dir() and not p.name.startswith('check')})
XN = np.load(NONE)
ref = {'data' : summarise(batch), 'none' : summarise(torch.from_numpy(XN))}
cache = {}
rows = []
for name, x, lbd, xn in [('data', None, None, None), ('none', None, None, None)] + runs(names) :
    s = ref[name] if x is None else summarise(torch.from_numpy(x))
    if xn is not None and id(xn) not in cache :
        cache[id(xn)] = summarise(torch.from_numpy(xn))
    sn = ref['none'] if xn is None else cache[id(xn)]
    r = {'run' : name}
    for k in LAYERS :
        r[f'd {k}'] = np.mean(s[k] - sn[k]) if name != 'data' else np.nan
        r[f'b {k}'] = np.mean(s[k]) - np.mean(ref['data'][k])
    for k in ['T0', 'S0', 'E', 'c7/eps'] :
        r[k] = np.mean(s[k])
    r['d T0'] = np.mean(s['T0'] - sn['T0']) if name != 'data' else np.nan
    r['se d lev0'] = np.std(s['lev0'] - sn['lev0'], ddof=1) / np.sqrt(len(s['lev0'])) if x is not None else np.nan
    r['dist none'] = np.sqrt(np.mean((x - xn)**2)) if x is not None else np.nan
    lbd = None if lbd is None else np.atleast_2d(lbd.T).T
    r['lbd'] = '' if lbd is None else ' / '.join(f'{v:.3g}' for v in lbd.mean(0))
    rows.append(r)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
df = pd.DataFrame(rows).set_index('run')
cols = ['d lev0', 'se d lev0', 'b lev0', 'b 5-49m', 'b 61-299m', 'd all', 'T0', 'd T0', 'S0', 'E', 'c7/eps', 'dist none', 'lbd']
txt = df[cols].to_string(float_format=lambda v : f'{v:.3g}')
print(txt)
(HERE / 'evaluate.txt').write_text(__doc__ + '\nd = paired difference to none, b = bias to the data batch (kg/m3, C, psu)\n\n' + txt + '\n')
df.to_csv(HERE / 'evaluate.csv')
