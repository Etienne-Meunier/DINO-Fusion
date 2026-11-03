"""
Shared helpers of the agent B surface-bias probes : paths, data / generated states, observer, per-level diagnostics.
"""
from pathlib import Path
import os, sys
PRP = f'{Path(__file__).resolve().parents[5]}/'
os.environ.setdefault('OCEANDATA', '/Volumes/LoCe/oceandata/')
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/analysis_scripts/')
sys.path.append(PRP)

import numpy as np
import torch
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.losses import NegativeGradDensityHuber

HERE = Path(__file__).parent
MODEL = f"{os.environ['OCEANDATA']}/models/dino-fusion/tav0h83b"
NONE_RUN = f'{MODEL}/inference/sweep_baselines/none_border0_b8/151026-145216/generated.npy'
NT = 18  # strided T levels = strided S levels


def load_data(n=16, seed=0) :
    """n random normalised training states (n, 37, 208, 64)."""
    config = TrainingConfig()
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=64, shuffle=False)])
    rng = np.random.default_rng(seed)
    return states[rng.choice(len(states), n, replace=False)].float()


def load_none() :
    """Unconstrained BorderZero generations (8, 37, 208, 64)."""
    return torch.tensor(np.load(NONE_RUN)).float()


def get_observer(device='cpu') :
    return Observer('data', TrainingConfig(), device=device)


def ocean_mask_norm(observer) :
    """(37, 208, 64) bool, True on ocean cells of the normalised (padded) state."""
    m = observer.normalizer.masker.mask
    ocean = ~m
    return observer.normalizer.padder(ocean.float()) > 0.5
