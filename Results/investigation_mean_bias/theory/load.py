"""Shared loading for the theory investigation: data batch, normalized runs, masks, level means."""
import sys, os, json
PRP = '/Users/emeunier/Desktop/Projets/DINO-Fusion/'
sys.path.append(f'{PRP}/Diffusion_Model/'); sys.path.append(f'{PRP}/Results/analysis_scripts/'); sys.path.append(PRP)
import numpy as np, torch, xarray as xr
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline

INF = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b/inference'
RUNS = {
    'old200_a': f'{INF}/infesteps_1000/constraints_no_constraints/20250203-141645.npy',
    'old200_b': f'{INF}/infesteps_1000/constraints_no_constraints/20250203-171119.npy',
    'old8_jan30': f'{INF}/infesteps_1000/constraints_no_constraints/20250130-134954.npy',
    'cur_none': f'{INF}/optimal/none/151026-221621/generated.npy',
    'cur_none_border0': f'{INF}/optimal/none_border0/151026-150424/generated.npy',
}
config = TrainingConfig()
transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
mesh = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').isel(time_counter=0)
tmask_full = mesh.tmask.values.astype(bool)  # (36,199,62)
depth_full = mesh.gdept_1d.values
# normalized-space ocean mask (37,208,64): True = ocean (not land, not padding)
land = transform.concatenated_infos['mask'].numpy()  # True = land, (37,199,62)
ocean_n = np.zeros((37, 208, 64), bool); ocean_n[:, 5:-4, 1:-1] = ~land
pad_n = np.ones((37, 208, 64), bool); pad_n[:, 5:-4, 1:-1] = False
CH = [f'T{k}' for k in range(18)] + [f'S{k}' for k in range(18)] + ['ssh']
lev_idx = np.arange(36)[0:-1:2]  # strided levels
DEPTH_CH = np.concatenate([depth_full[lev_idx], depth_full[lev_idx], [0]])

def load_data(n=200, seed=0):
    torch.manual_seed(seed)
    dl = get_dataloader(config.data_file, transform, batch_size=n, shuffle=True)
    return next(iter(dl)).numpy()

def level_mean_norm(x, mask=ocean_n):
    """(b,37) mean over ocean cells of each normalized channel."""
    return np.stack([x[:, c][:, mask[c]].mean(1) for c in range(37)], 1)

def phys_level_means(x):
    """Physical basin level means of T,S per full-res level (b,36), via the training un-normalization (NaN on land)."""
    out = {'toce': [], 'soce': []}
    for g in torch.from_numpy(x):
        f = transform.__uncall__(g)
        for k in out:
            v = f[k].numpy()
            out[k].append(np.array([v[z][tmask_full[z]].mean() if tmask_full[z].any() else np.nan for z in range(36)]))
    return {k: np.array(v) for k, v in out.items()}
