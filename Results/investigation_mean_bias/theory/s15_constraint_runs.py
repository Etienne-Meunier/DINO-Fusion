"""Level-mean T and S offsets of the constrained runs; density-compensation check for the mean-density run."""
from load import *
from glob import glob
from metrics import get_density_at_surface_tensor as eos
rho = lambda S, T: np.array([eos(torch.full((1,1,1,1), float(t), dtype=torch.float64), torch.full((1,1,1,1), float(s_), dtype=torch.float64)).item() for s_, t in zip(S, T)])
np.set_printoptions(precision=2, suppress=True, linewidth=220)
P = np.load('phys_levelmeans.npz'); DT, DS = P['data_toce'].mean(0), P['data_soce'].mean(0)
sel = [2, 6, 11, 15, 20, 25, 30]
print('depths', depth_full[sel].round(0))
for name in ['mean_density_border0', 'zero_mean_border0', 'none_border0', 'none']:
    p = sorted(glob(f'{INF}/optimal/{name}/*/generated.npy'))[-1]
    L = phys_level_means(np.load(p)); dT = L['toce'].mean(0) - DT; dS = L['soce'].mean(0) - DS
    r0 = rho(DS, DT); rT = rho(DS, DT + dT) - r0; rS = rho(DS + dS, DT) - r0; drho = rho(DS + dS, DT + dT) - r0
    print(f'{name:22s} dT {dT[sel]}  dS {dS[sel]}')
    print(f'{"":22s} drho_T {rT[sel]} drho_S {rS[sel]} -> drho {drho[sel]}  ; per-state T level-mean std {L["toce"].std(0)[sel]}')
