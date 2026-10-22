"""Is the per-state level-mean error a channel-wide DC offset (ocean and land/padding shifted together)?"""
from load import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
data = load_data(200); dlm = level_mean_norm(data).mean(0)
nonocean = ~ocean_n  # land + padding (data value exactly 0)
for k in ['old200_a', 'cur_none']:
    x = np.array(np.load(RUNS[k], mmap_mode='r')[:200])
    oc = level_mean_norm(x) - dlm
    ld = np.stack([x[:, c][:, nonocean[c]].mean(1) for c in range(37)], 1)
    ldm = np.stack([np.median(x[:, c][:, nonocean[c]], 1) for c in range(37)], 1)
    a, b = oc.ravel(), ld.ravel()
    slope = (a * b).sum() / (b * b).sum()
    print(f'{k}: over (state,channel) pairs corr(ocean level-mean offset, land/pad mean) = {np.corrcoef(a, b)[0,1]:.3f}, slope = {slope:.2f}; land/pad mean std {b.std():.3f}, ocean offset std {a.std():.3f}; corr with land median {np.corrcoef(a, ldm.ravel())[0,1]:.3f}')
    r2 = 1 - ((a - slope * b)**2).sum() / (a**2).sum()
    print(f'   fraction of ocean level-mean offset variance explained by the land/pad DC value: {r2:.3f}')
    print('   per-channel corr:', np.array([np.corrcoef(oc[:, c], ld[:, c])[0, 1] for c in range(37)]))
