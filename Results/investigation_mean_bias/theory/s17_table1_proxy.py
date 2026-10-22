"""Table-1 metrics (BW/DW box T,S with depth_box=1500, density error) for old200 (final-weights proxy) next to the current runs,
plus the RMS of the ensemble-mean T level offset and the median per-state T level-mean spread."""
from load import *
from glob import glob
from functools import partial
import pandas as pd
sys.path.append(f'{PRP}/Results/analysis_scripts/')
from metrics import compute_density_error, temperature_BWbox_metric, temperature_DWbox_metric
from pipelines.State import Observer
observer = Observer('data', config)
mask = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').sel(time_counter=0).rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})
box = {'BW': partial(temperature_BWbox_metric, depth_box=1500), 'DW': partial(temperature_DWbox_metric, depth_box=1500)}
srcs = {'data': load_data(200), 'old200 (final-weights proxy)': np.array(np.load(RUNS['old200_a'], mmap_mode='r'))}
for n in ['none', 'none_border0', 'zero_mean_border0', 'negative_grad_density_huber_border0', 'mean_density_border0']:
    srcs[n] = np.load(sorted(glob(f'{INF}/optimal/{n}/*/generated.npy'))[-1])
P = np.load('phys_levelmeans.npz'); DT = P['data_toce'][:, :35].mean(0)
rows = {}
for s, x in srcs.items():
    f = {k: v.detach() for k, v in observer.unnormalized(torch.from_numpy(x)).items()}
    r = {}
    for fld in ['toce', 'soce']:
        for kb, fn in box.items():
            vals = []
            for b in f[fld]:
                smp = mask.e3t_0.copy(); smp[:] = np.asarray(b); vals.append(fn(smp, mask).item())
            r[f'{kb} {fld[0].upper()}'] = f'{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}'
    de = np.asarray(compute_density_error(f, mask), dtype=float); r['dens err %'] = f'{de.mean():.1f} ± {de.std(ddof=1):.1f}'
    L = phys_level_means(x)['toce'][:, :35]
    r['RMS ens-mean T offset (C)'] = f'{np.sqrt(((L.mean(0) - DT)**2).mean()):.2f}'
    r['median per-state T level-mean std (C)'] = f'{np.median(L.std(0, ddof=1)):.2f}'
    r['upper-500m per-state T std (C)'] = f'{L.std(0, ddof=1)[:20].mean():.2f}'
    rows[s] = r; print(s, r, flush=True)
print(pd.DataFrame(rows).T.to_string())
pd.DataFrame(rows).T.to_csv('table1_proxy.csv')
