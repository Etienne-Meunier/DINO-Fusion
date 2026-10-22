"""Physical basin level means of T,S per state for data and runs; z of the ensemble mean, random 16-subsets of old200."""
from load import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
data = load_data(200)
out = {'data': phys_level_means(data)}
for k in ['old200_a', 'cur_none', 'cur_none_border0']:
    out[k] = phys_level_means(np.load(RUNS[k]))
np.savez('phys_levelmeans.npz', **{f'{k}_{f}': v[f] for k, v in out.items() for f in v})
