"""Bias indicator = mean off-diagonal correlation across states of (state - data climatology), normalised space, all channels/cells.
Also max|diff| of the training-eval reproduction vs epoch_3650 / epoch_4950."""
PRP = '/Users/emeunier/Desktop/Projets/DINO-Fusion/'
import sys, os, glob
sys.path += [f'{PRP}/Diffusion_Model/', PRP]
import numpy as np, torch
OUT = os.path.dirname(os.path.abspath(__file__))
M = '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b'; I = f'{M}/inference'
cache = f'{OUT}/data_clim_normalised.npy'
if not os.path.exists(cache):
    from configs.base_config import TrainingConfig
    from DataLoader import get_dataloader, get_infos
    from TransformFields import TransformationPipeline
    c = TrainingConfig(); torch.manual_seed(0)
    tr = TransformationPipeline(get_infos(c.data_file), c.fields, c.normalisation, device='cpu')
    np.save(cache, next(iter(get_dataloader(c.data_file, tr, batch_size=200, shuffle=True))).mean(0).numpy())
clim = np.load(cache)
def ind(x):
    a = (np.asarray(x[:64], dtype=np.float64) - clim).reshape(min(len(x), 64), -1)
    a = np.nan_to_num(a); C = np.corrcoef(a); n = len(C)
    return (C.sum() - n) / (n * (n - 1))
files = {'epoch_3650': f'{M}/epoch_3650.npy', 'epoch_4950': f'{M}/epoch_4950.npy',
         'old200 (feb2025)': f'{I}/infesteps_1000/constraints_no_constraints/20250203-141645.npy',
         'feb2025 140245 b8': f'{I}/infesteps_1000/constraints_no_constraints/20250203-140245.npy',
         'optimal/none': f'{I}/optimal/none/151026-221621/generated.npy',
         'optimal/none_border0': f'{I}/optimal/none_border0/151026-150424/generated.npy',
         'jan29 180359': f'{I}/infesteps_1000/20250129-180359.npy', 'jan29 181348': f'{I}/infesteps_1000/20250129-181348.npy',
         'prelim_sweep none s0 100st': f'{I}/prelim_sweep/none_s0_n100_b8/generated.npy'}
files |= {'mine ' + p.split('/')[-2]: p for p in sorted(glob.glob(f'{OUT}/runs/*/generated.npy'))}
for k, p in files.items():
    print(f'{k:40s} n={len(np.load(p, mmap_mode="r"))} indicator={ind(np.load(p, mmap_mode="r")):.3f}')
r = np.load(f'{OUT}/runs/mps_cpugen_s0_b8_trainingeval/generated.npy')
for e in (3600, 3650, 3700, 4950):
    b = np.load(f'{M}/epoch_{e}.npy')
    print(f'repro vs epoch_{e}: max|diff| {np.abs(r - b).max():.3f}  rms diff {np.sqrt(((r - b) ** 2).mean()):.4f}  corr {np.corrcoef(r.ravel(), b.ravel())[0, 1]:.4f}')
b1, b2 = np.load(f'{M}/epoch_3600.npy'), np.load(f'{M}/epoch_3650.npy')
print(f'ref: epoch_3600 vs epoch_3650 rms diff {np.sqrt(((b1 - b2) ** 2).mean()):.4f} corr {np.corrcoef(b1.ravel(), b2.ravel())[0, 1]:.4f}')
