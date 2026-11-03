"""
Endpoint bias of a monotonicity-only correction ("spiking" of isotonic regression at the boundary) on unconstrained
states : shift of the level-0 (and level k) density when
    (a) each column is projected on the isotonic cone (perfect cell-wise constraint, IsotonicDensity, unweighted)
    (b) only the sample's own horizontal-mean profile (volume-weighted level means) is projected (own-profile constraint)
Reference : the paired shift of the eta runs (agent A), lev0 -0.45 (eta 10) to -1.75 (eta 100) kg/m3.
Output : p5_isotonic_bound.txt
"""
from common import *
from pipelines.losses import IsotonicDensity
from p1_loss_gradient import rho_TS

obs = get_observer()
iso = IsotonicDensity()
x = torch.tensor(np.load(f'{MODEL}/inference/optimal/none_border0/151026-150424/generated.npy')).float()
vol, tm = obs.volume, obs.tmask
lv = vol.sum((-2, -1))
out = []
with torch.no_grad() :
    for i0 in range(0, len(x), 4) :
        rho = rho_TS(x[i0:i0 + 4], obs)[0].double()
        r = torch.nan_to_num(rho)
        # (a) column isotonic : columns with land below keep NaN -> fill land with +inf-like by repeating last ocean value
        filled = rho.clone()
        for k in range(1, 36) :
            filled[:, k] = torch.where(torch.isnan(filled[:, k]), filled[:, k - 1], filled[:, k])
        ic = iso(filled)
        da = ((torch.nan_to_num(ic - filled) * vol).sum((-2, -1)) / lv)
        # (b) own profile
        prof = ((r * vol).sum((-2, -1)) / lv.clamp(min=1))[:, :35]
        db = torch.nn.functional.pad(iso(prof[..., None])[..., 0] - prof, (0, 1))
        out.append((da, db, prof))
da = torch.cat([o[0] for o in out]); db = torch.cat([o[1] for o in out]); prof = torch.cat([o[2] for o in out])
lines = [f'{len(x)} unconstrained border0 states (optimal/none_border0), shift of the level-mean density (kg/m3), mean (std) :',
         'lev | column isotonic | own-profile isotonic']
for k in [0, 1, 2, 3, 4, 5, 8, 13, 16, 22, 34] :
    lines.append(f'{k:3d} | {da[:, k].mean():+.3f} ({da[:, k].std():.3f}) | {db[:, k].mean():+.3f} ({db[:, k].std():.3f})')
lines.append(f'share of states with own-profile surface pair inverted : {(prof[:, 1] < prof[:, 0]).float().mean():.2f}')
open(HERE / 'p5_isotonic_bound.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
