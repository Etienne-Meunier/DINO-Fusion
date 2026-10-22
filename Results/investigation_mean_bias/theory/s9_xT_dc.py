"""Does the final per-(state,channel) DC offset follow the DC of the initial noise x_T?
x_T reconstructed as torch.randn((16,37,208,64), generator=Generator('mps').manual_seed(0), device='mps'), as in sampling.py."""
from load import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
g = torch.Generator('mps').manual_seed(0)
xT = torch.randn((16, 37, 208, 64), generator=g, device='mps').cpu().numpy()
data = load_data(200); dlm = level_mean_norm(data).mean(0)
for k in ['cur_none', 'cur_none_border0']:
    x = np.load(RUNS[k])
    dc_final_all = x.mean((-2, -1))                     # whole-channel mean
    dc_final_ocean = level_mean_norm(x) - dlm
    for nm, dcT in [('xT whole-grid mean', xT.mean((-2, -1))), ('xT ocean mean', level_mean_norm(xT)), ('xT land+pad mean', np.stack([xT[:, c][:, ~ocean_n[c]].mean(1) for c in range(37)], 1))]:
        a, b = dcT.ravel(), dc_final_ocean.ravel()
        print(f'{k:17s} corr({nm}, final ocean level-mean offset) = {np.corrcoef(a, b)[0,1]:.3f}  ; with whole-channel final mean {np.corrcoef(a, dc_final_all.ravel())[0,1]:.3f};  regression gain {np.polyfit(a, b, 1)[0]:.1f}')
    print('   std of xT whole-grid mean %.4f (1/sqrt(N)=%.4f)' % (xT.mean((-2, -1)).std(), 1 / np.sqrt(208 * 64)))
# sanity: the reconstructed noise should be the true x_T: check that the final states correlate pixelwise with x_T more than with another seed
g2 = torch.Generator('mps').manual_seed(1); xT2 = torch.randn((16, 37, 208, 64), generator=g2, device='mps').cpu().numpy()
x = np.load(RUNS['cur_none'])
cc = lambda u, v: np.corrcoef(u.ravel(), v.ravel())[0, 1]
print('pixelwise corr final vs xT(seed0) %.3f, vs xT(seed1) %.3f' % (cc(x - data.mean(0), xT), cc(x - data.mean(0), xT2)))
def lowpass(a, k=16):
    b, c, h, w = a.shape; return a.reshape(b, c, h // k, k, w // k, k).mean((3, 5))
print('low-pass (16x16 blocks) corr final anomaly vs xT(seed0) %.3f, vs seed1 %.3f' % (cc(lowpass(x - data.mean(0)), lowpass(xT)), cc(lowpass(x - data.mean(0)), lowpass(xT2))))
