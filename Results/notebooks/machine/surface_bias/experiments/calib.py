"""
Statistics of the training states for the experimental constraints : mean / std of the volume-weighted level-0 T and S
(target of SurfaceMean), and the distribution of the C7, profile, topfree and surface losses on the data (choice of eps).
Saves `calib.npz` and prints quantiles.
"""
from exp import *
sys.path.append(f'{PRP}/Results/notebooks/machine/surface_bias/agentB/')
from common import load_data, get_observer

obs = get_observer()
states = load_data(n=1800)
c7, prof, top = NegativeGradDensityHuber(observer=obs), ProfileHinge(observer=obs), HuberTopFree(observer=obs)
surf = SurfaceMean(observer=obs, stats=dict(T0_mean=0, T0_std=1, S0_mean=0, S0_std=1))

out = {k : [] for k in ['c7', 'profile', 'topfree', 'T0', 'S0']}
with torch.no_grad() :
    for b in states.split(64) :
        out['c7'].append(c7(b)); out['profile'].append(prof(b)); out['topfree'].append(top(b))
        T0, S0 = surf.means(b)
        out['T0'].append(T0); out['S0'].append(S0)
out = {k : torch.cat(v).double().numpy() for k, v in out.items()}
stats = dict(T0_mean=out['T0'].mean(), T0_std=out['T0'].std(), S0_mean=out['S0'].mean(), S0_std=out['S0'].std())
z2 = ((out['T0'] - stats['T0_mean']) / stats['T0_std'])**2 + ((out['S0'] - stats['S0_mean']) / stats['S0_std'])**2
np.savez(HERE / 'calib.npz', **out, surface=z2, **stats)

print(f'n = {len(z2)} ; level 0 : T {stats["T0_mean"]:.3f} +/- {stats["T0_std"]:.3f} C, S {stats["S0_mean"]:.4f} +/- {stats["S0_std"]:.4f}')
for k, v in [('c7', out['c7']), ('topfree', out['topfree']), ('profile', out['profile']), ('surface', z2)] :
    q = np.quantile(v, [0, 0.5, 0.9, 0.99, 1])
    print(f'{k:8s} mean {v.mean():.3e} | min/med/q90/q99/max ' + ' '.join(f'{x:.3e}' for x in q) + f' | frac 0 {np.mean(v == 0):.2f}')

zm0 = SurfaceZeroMean(observer=obs)
with torch.no_grad() :
    v = torch.cat([zm0(b) for b in states.split(64)]).double().numpy()
q = np.quantile(v, [0, 0.5, 0.9, 0.99, 1])
print(f'zm0      mean {v.mean():.3e} | min/med/q90/q99/max ' + ' '.join(f'{x:.3e}' for x in q))
