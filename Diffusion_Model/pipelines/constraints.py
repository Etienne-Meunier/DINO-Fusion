import torch
from ipdb import set_trace
import numpy as np
from configs.base_config import TrainingConfig
from utils import get_dataloader
import xarray as xr
import sys
import torch.nn.functional as F

# adding Folder_2 to the system path
sys.path.insert(0, 'Results/analysis_scripts/')

from data_analytics import get_transformed_data
from metrics import compute_density_tensor

class DiffusionConstraint:
    def __init__(self):
        pass

    def apply(self, x):
        raise NotImplementedError

class ZeroMeanConstraint(DiffusionConstraint):
    def __init__(self, dims=[-1, -2]):
        super().__init__()
        self.dims = dims

    def apply(self, x, t=None):
        mean = x.mean(dim=self.dims, keepdim=True)
        #print(f'apply constraint : {mean} {mean.shape}')
        return x - mean

class BorderZeroConstraint(DiffusionConstraint):
    def __init__(self, mask='Diffusion_Model/pipelines/border_mask.npy'):
        self.mask = torch.tensor(np.load(mask))

    def apply(self, x, t=None):
        x[:, self.mask] = 0.0
        return x


class ConditionalGeneration_SSHLow(DiffusionConstraint):
    def __init__(self, field='pipelines/ssh_low.npy'):
        self.field = torch.tensor(np.load(field))

    def apply(self, x, t=None):
        x[:, -1] = self.field
        return x

class ConditionalGeneration_SSHHigh(DiffusionConstraint):
    def __init__(self, field='pipelines/ssh_high.npy'):
        self.field = torch.tensor(np.load(field))

    def apply(self, x, t=None):
        x[:, -1] = self.field
        return x


class ConditionalGeneration_STempLow(DiffusionConstraint):
    def __init__(self, field='pipelines/surface_temp_low.npy'):
        self.field = torch.tensor(np.load(field))

    def apply(self, x, t=None):
        x[:, 0] = self.field
        return x

class ConditionalGeneration_STempHigh(DiffusionConstraint):
    def __init__(self, field='pipelines/surface_temp_high.npy'):
        self.field = torch.tensor(np.load(field))

    def apply(self, x, t=None):
        x[:, 0] = self.field
        return x


class Beta:
    def __init__(self, beta, beta_type='constant'):
        self.beta_type = beta_type
        self.beta = beta

    def get_beta(self, t, k=20, l=40):
        if self.beta_type == 'constant':
            return self.beta
        elif self.beta_type == 'decreasing':
            # Example: stronger constraint at the end of sampling
            return self.beta + l*self.beta * torch.exp((1-t/1000) * k - k)
        else:
            raise ValueError(f"Unknown beta type: {self.beta_type}. Available beta type: constant, decreasing ")


class GradientZeroMeanConstraint(DiffusionConstraint):
    def __init__(self, beta, beta_type='constant', dims=[-1, -2]):
        super().__init__()
        self.beta = Beta(beta=beta, beta_type=beta_type)
        self.dims = dims

    def apply(self, x, t=None, interior=None):#(slice(5, -5), slice(5, -5))
        # Handle interior region indexing for older Python versions
        if interior:
            region = x[..., interior[0], interior[1]]
            grad = region.mean(dim=self.dims, keepdim=True)
        else:
            grad = x.mean(dim=self.dims, keepdim=True)

        beta = self.beta.get_beta(t) if t is not None else self.beta
        print(f'apply constraint {beta}: {grad[0, :, 0,0]}')

        if interior:
            x[..., interior[0], interior[1]] -= beta * grad
        else:
            x -= beta * grad
        return x

class GradientZeroMeanDensityConstraint(DiffusionConstraint):
    def __init__(self, beta, beta_type='constant', dims=[-1, -2], batch=8):
        super().__init__()
        self.beta = Beta(beta=beta, beta_type=beta_type)
        self.dims = dims
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch = batch
        
        # get the extractor function to unnormalise the data
        config = TrainingConfig()
        config.normalisation = '3-std'
        config.data_file = '../DATA_DINOFusion/dino_1_4_degree_coarse_240125.tar'
        train_dataloader = get_dataloader(config.data_file, batch_size=batch,
                                                fields=config.fields, normalisation='3-std', transform=True, shuffle=True, device=self.device)
        self.extractor = train_dataloader.get_transform().uncall
        self.mask_infos = train_dataloader.get_transform().infos['mask']['toce'] #verifier que les mask toce et soce sont identiques
        self.fields = train_dataloader.get_transform().fields

        #mask for density metric
        file_mask_LR = xr.open_dataset("Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
        self.mask = file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})


    def format_density(self, density):
        #replace nan values 
        density[self.mask_infos.repeat(self.batch,1,1,1)] = 0
        #add correct padding
        padding = (1, 1, 5, 4)
        density_padded = F.pad(density, padding, mode='constant', value=0)
        #reshape levels and concatenate (zero for ssh layer)
        density_sliced = density_padded[:, 0:-1:2]
        return density_sliced


    def apply(self, x, t=None): 
        #un-normalize
        x_norm = get_transformed_data(x, function=self.extractor)

        #compute density
        density = compute_density_tensor(x_norm, self.mask)
        density_padded = self.format_density(density)
        grad = density_padded.mean(dim=self.dims, keepdim=True)

        beta = self.beta.get_beta(t) if t is not None else self.beta
        print(f'apply constraint {beta}: {grad[0, :, 0,0]}')
        
        ssh_layer = torch.zeros([self.batch, 1, 1, 1], device=self.device)
        grad_all = torch.cat([grad, grad, ssh_layer], dim=1)
        
        x -= beta * grad_all
        return x
    
class GradientDensityConstraint(DiffusionConstraint):
    def __init__(self, beta, beta_type='constant', dims=[-1, -2], batch=8):
        super().__init__()
        self.beta = Beta(beta=beta, beta_type=beta_type)
        self.dims = dims
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch = batch
        
        # get the extractor function to unnormalise the data
        config = TrainingConfig()
        config.normalisation = '3-std'
        config.data_file = '../DATA_DINOFusion/dino_1_4_degree_coarse_240125.tar'
        train_dataloader = get_dataloader(config.data_file, batch_size=batch,
                                                fields=config.fields, normalisation='3-std', transform=True, shuffle=True, device=self.device)
        self.extractor = train_dataloader.get_transform().uncall
        self.mask_infos = train_dataloader.get_transform().infos['mask']['toce'] #verifier que les mask toce et soce sont identiques
        self.fields = train_dataloader.get_transform().fields

        #mask for density metric
        file_mask_LR = xr.open_dataset("Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
        self.mask = file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})


    def format_density(self, density):
        #replace nan values 
        density[self.mask_infos.repeat(self.batch,1,1,1)] = 0
        #add correct padding
        padding = (1, 1, 5, 4)
        density_padded = F.pad(density, padding, mode='constant', value=0)
        #reshape levels and concatenate (zero for ssh layer)
        density_sliced = density_padded[:, 0:-1:2]
        return density_sliced


    def apply(self, x, t=None): 
        #un-normalize
        h= x.shape[2]
        w = x.shape[3]
        x_norm = get_transformed_data(x, function=self.extractor)

        #compute density
        density = compute_density_tensor(x_norm, self.mask)
        density_padded = self.format_density(density)
        grad = density_padded - torch.roll(density_padded, -1, dims=1)
        grad[:,-1,:,:] = torch.zeros([self.batch,h,w], device=self.device)

        beta = self.beta.get_beta(t) if t is not None else self.beta
        print(f'apply constraint {beta}: {grad[0, :, 120,30]}')
        #add ssh layer
        ssh_layer = torch.zeros([self.batch,1,h,w], device=self.device)
        grad_all = torch.cat([grad, grad, ssh_layer], dim=1)
        
        x -= beta * grad_all
        return x