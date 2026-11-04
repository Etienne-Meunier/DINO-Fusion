"""Table 2: first two moments of each generation against the whole training set, per field (T, S, SSH).

For a source of n states, at every ocean cell, the batch mean mu and the batch variance var (ddof = 1) are compared to
those of all the training states (mu_D, var_D):

    bias   = sqrt(sum w (mu - mu_D)^2 / sum w)          [field units]
    spread = sqrt(sum w var / sum w var_D)              [1 = same variance, < 1 under-dispersive]

with w the ocean cell volume e1t e2t e3t_0 of GRID (area e1t e2t of the surface ocean cells for SSH). The noise floor is
the same scores for N_FLOOR disjoint random groups of n training states (mean ± std over the groups), the data row the
first n states of `batch`, the data states of the a posteriori data run, drawn from the start of the time-ordered
training set."""
from common import *

import io
import tarfile

N_FLOOR = 32
FIELDS = ['toce', 'soce', 'ssh']
LABELS = {'toce': r'T [$^\circ$C]', 'soce': r'S [g kg$^{-1}$]', 'ssh': 'SSH [m]'}
PRECISION = {'toce': 3, 'soce': 4, 'ssh': 3}


def training_moments(n, n_floor, seed=DATA_ARGS['data_seed']):
    """Mean and variance (ddof = 1) of every field over all the training states, and over `n_floor` disjoint random
    groups of `n` states, read once from the training tar and cached next to it as moments_<n>x<n_floor>.npz, land as 0.

    Returns {field: (mean, var)} and {field: (group means, group vars)} with a leading group axis.
    """
    cache = f'{os.path.dirname(config.data_file)}/moments_{n}x{n_floor}.npz'
    if not os.path.exists(cache):
        with tarfile.open(config.data_file) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and not m.name.startswith('infos/')]
            keys = sorted({m.name.split('/')[-1].split('.')[0] for m in members})
            group = np.full(len(keys), -1)
            group[np.random.default_rng(seed).permutation(len(keys))[:n * n_floor]] = np.repeat(np.arange(n_floor), n)
            group = dict(zip(keys, group))
            s1, s2, g1, g2 = {}, {}, {}, {}
            for m in members:
                key, field, _ = m.name.split('/')[-1].split('.')
                if field not in FIELDS:
                    continue
                x = np.nan_to_num(np.load(io.BytesIO(tar.extractfile(m).read())).astype(np.float64))
                s1[field] = s1.get(field, 0) + x
                s2[field] = s2.get(field, 0) + x ** 2
                if group[key] >= 0:
                    g1.setdefault(field, np.zeros((n_floor,) + x.shape))[group[key]] += x
                    g2.setdefault(field, np.zeros((n_floor,) + x.shape))[group[key]] += x ** 2
        N = len(keys)
        out = {}
        for f in FIELDS:
            out[f'{f}_mean'] = s1[f] / N
            out[f'{f}_var'] = (s2[f] - s1[f] ** 2 / N) / (N - 1)
            out[f'{f}_gmean'] = g1[f] / n
            out[f'{f}_gvar'] = (g2[f] - g1[f] ** 2 / n) / (n - 1)
        np.savez(cache, **{k: v.astype(np.float32) for k, v in out.items()})
    c = np.load(cache)
    return ({f: (c[f'{f}_mean'], c[f'{f}_var']) for f in FIELDS},
            {f: (c[f'{f}_gmean'], c[f'{f}_gvar']) for f in FIELDS})


def scores(mean, var, reference, w):
    """(bias, spread) of a (mean, var) pair against the reference (mean, var), weighted by w."""
    bias = np.sqrt((w * (mean - reference[0]) ** 2).sum() / w.sum())
    spread = np.sqrt((w * var).sum() / (w * reference[1]).sum())
    return np.array([bias, spread])


def batch_moments(x):
    """Mean and variance (ddof = 1) over the batch of every field of a normalized batch x, land as 0."""
    ds = to_dataset(x).fillna(0.).astype(np.float64)
    return {f: (ds[f].mean('member').values, ds[f].var('member', ddof=1).values) for f in FIELDS}


def compute():
    """(bias, spread) of the noise floor (mean, std over the groups), of the first n states of `batch` and of
    every generation, per field: {'n', 'rows': {row: {field: (scores (2,), std (2,) or None)}}}."""
    n = next(iter(gens.values())).shape[0]
    volume = cell_volume(GRID).values
    w = {'toce': volume, 'soce': volume, 'ssh': (GRID.mask.e1t * GRID.mask.e2t * GRID.mask.tmask.isel(z_c=0)).values}
    full, floor = training_moments(n, N_FLOOR)

    rows = {'floor': {}, 'data_seed': {}}
    for f in FIELDS:
        group_scores = np.array([scores(floor[f][0][g], floor[f][1][g], full[f], w[f]) for g in range(N_FLOOR)])
        rows['floor'][f] = (group_scores.mean(0), group_scores.std(0))
    for name, x in [('data_seed', batch[:n])] + list(gens.items()):
        m = batch_moments(x)
        rows[name] = {f: (scores(*m[f], full[f], w[f]), None) for f in FIELDS}
    return {'n': n, 'rows': rows}


def table(computed):
    """Print the table of `computed` (`compute`) and its LaTeX version, return the LaTeX."""
    n, rows = computed['n'], computed['rows']

    cell = lambda value, std, k, p: f'{value[k]:.{p}f}' + ('' if std is None else f' ± {std[k]:.{p}f}')
    columns = {(f, k): [cell(*rows[s][f], k, PRECISION[f] if k == 0 else 2) for s in rows] for f in FIELDS for k in (0, 1)}
    kind = lambda k: 'bias' if k == 0 else 'spread'
    print(pd.DataFrame({f'{f} {kind(k)}': v for (f, k), v in columns.items()}, index=list(rows)).to_string())

    row_names = NAMES | {'floor': f'Data, {n} random states ({N_FLOOR} groups)',
                         'data_seed': f'Data, {n} states of the data run'}
    order = ['floor', 'data_seed'] + [s for s in CONFIGS['table'] if s in gens]
    latex = pd.DataFrame({f'{LABELS[f]} {kind(k)}': v for (f, k), v in columns.items()}, index=list(rows)).loc[order]
    latex.index = [row_names[s] for s in order]
    latex = latex.map(lambda s: s.replace('±', r'$\pm$')).to_latex(column_format='l' + 'c' * len(columns))
    print(latex)
    return latex


def main():
    """Compute the table and save its LaTeX version as table2_moments.tex in FIGURES."""
    save_tex(table(compute()), f'{FIGURES}/table2_moments.tex')


if __name__ == '__main__':
    main()
