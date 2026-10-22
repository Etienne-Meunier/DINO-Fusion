"""Systematic DC bias b_t of the eps prediction on in-distribution x_t (mean over data states and noise draws), per channel,
then propagated with the linear DC walk (s12) to predict the ensemble-mean level offset of an unconstrained run.
Compared to the observed ensemble-mean offset of cur_none (local unet) channel by channel."""
from load import *
from diffusers import UNet2DModel
unet = UNet2DModel.from_pretrained('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/unet').eval()
ab = np.load('alphas_cumprod.npy'); abt = torch.tensor(ab, dtype=torch.float32)
oc = torch.from_numpy(ocean_n)
def dc(x): return torch.stack([x[:, c][:, oc[c]].mean(1) for c in range(37)], 1)
data = torch.from_numpy(load_data(32, seed=7))
ts = [5, 50, 150, 300, 450, 600, 750, 850, 925, 975, 999]
B = []
torch.manual_seed(0)
for t in ts:
    errs = []
    for rep in range(2):
        eps = torch.randn_like(data); xt = abt[t].sqrt() * data + (1 - abt[t]).sqrt() * eps
        with torch.no_grad(): e = unet(xt, t).sample
        errs.append(dc(e) - dc(eps))
    errs = torch.cat(errs)
    B.append(errs.mean(0).numpy()); print(f't={t}: eps DC bias rms over channels {np.sqrt((B[-1]**2).mean()):.4f}, SE {errs.std(0).mean().item()/np.sqrt(len(errs)):.4f}', flush=True)
B = np.array(B)  # (len(ts), 37)
np.save('eps_dc_bias.npy', np.concatenate([np.array(ts)[:, None], B], 1))
# propagate: eps_hat_DC = k x/sqrt(1-ab) + b_t, deterministic mean (noise averages out in the linear part)
R = np.load('dc_response.npy'); T = len(ab); k = np.interp(np.arange(T), R[:, 0], R[:, 6])
bt = np.stack([np.interp(np.arange(T), ts, B[:, c]) for c in range(37)], 1)
x = np.zeros(37)
for t in range(T - 1, -1, -1):
    abp = ab[t - 1] if t > 0 else 1.0; beta = 1 - ab[t] / abp
    epsh = k[t] * x / np.sqrt(1 - ab[t]) + bt[t]
    x0 = np.clip((x - np.sqrt(1 - ab[t]) * epsh) / np.sqrt(ab[t]), -1, 1)
    x = np.sqrt(abp) * beta / (1 - ab[t]) * x0 + np.sqrt(1 - beta) * (1 - abp) / (1 - ab[t]) * x
L = np.load('norm_levelmeans.npz'); obs = L['cur_none'].mean(0) - L['data'].mean(0); obs_old = L['old200_a'].mean(0) - L['data'].mean(0)
np.set_printoptions(precision=2, suppress=True, linewidth=220)
print('predicted common offset:', x); print('observed cur_none     :', obs)
print('corr(pred, obs cur) %.2f  sign agreement %.2f ; corr(pred, old200) %.2f ; rms pred %.3f obs %.3f' % (np.corrcoef(x, obs)[0, 1], np.mean(np.sign(x) == np.sign(obs)), np.corrcoef(x, obs_old)[0, 1], np.sqrt((x**2).mean()), np.sqrt((obs**2).mean())))
