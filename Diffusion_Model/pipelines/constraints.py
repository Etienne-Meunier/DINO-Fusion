import torch
import numpy as np
from configs.base_config import TrainingConfig
from utils import get_dataloader
import xarray as xr

#adding folder to system path
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Results/analysis_scripts/')
sys.path.append(f'{PRP}/Diffusion_Model/')

from data_analytics import get_transformed_data
from metrics import get_density_at_surface_tensor

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
        return x - mean

class BorderZeroConstraint(DiffusionConstraint):
    def __init__(self, mask=f'{PRP}/Diffusion_Model/pipelines/border_mask.npy'):
        self.mask = torch.tensor(np.load(mask))

    def apply(self, x, t=None):
        x[:, self.mask] = 0.0
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

    def update(self, esp, eta=0.001, type='eq'):
        if type=='eq':
            self.beta = self.beta + eta * esp
        elif type=='ineq':
            self.beta = torch.max(torch.tensor(0), self.beta + eta * esp)
        else:
            raise ValueError(f"Unknown type. Available types: 'ineq' for inequality constraints and 'eq' for equality constraints")



class GradientZeroMeanConstraint(DiffusionConstraint):
    """
        Constraint TS : zero spatial mean of the normalised temperature and salinity at each level.
    """
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

            self.beta.update(esp=torch.mean(torch.nansum(grad**2, dim=-1)), eta=0.001)
        return x

class GradientZeroMeanDensityConstraint(DiffusionConstraint):
    """
        Constraint C1 : mean density per level follows the mean density of the training set.
    """
    def __init__(self, beta, beta_type='constant', dims=[-1, -2], batch=8):
        super().__init__()
        self.beta = Beta(beta=beta, beta_type=beta_type)
        self.dims = dims
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch = batch

        # get the extractor function to unnormalise the data
        config = TrainingConfig()
        config.normalisation = '3-std'
        train_dataloader = get_dataloader(config.data_file, batch_size=batch,
                                                fields=config.fields, normalisation='3-std', transform=True, shuffle=True, device=self.device)
        self.extractor = train_dataloader.get_transform().uncall
        #mask for density metric
        file_mask_LR = xr.open_dataset(f"{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
        self.mask = file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})

        #mean density of the training set
        self.mean_density = torch.tensor(np.loadtxt(f'{PRP}/Results/analysis_scripts/mean_density_train.txt'), device=self.device, dtype=torch.float32)


    def apply(self, x, t=None, n_iter=1):

        for i in range(n_iter):
            #detach to remove gradients
            x = x.detach()

            with torch.enable_grad():
                if x.grad is not None:
                    x.grad.zero_()
                x.requires_grad =True
                samples = get_transformed_data(x, function=self.extractor)

                #compute density
                tmask = torch.tensor(self.mask.tmask.values, device=self.device, dtype=torch.float32).repeat(self.batch, 1, 1, 1)
                density = get_density_at_surface_tensor(samples['toce.npy'], samples['soce.npy'], tmask)

                mean_density = self.mean_density[None, :].expand(self.batch, -1)

                diff = (mean_density - density.nanmean(dim=self.dims))**2
                loss = torch.nansum(diff)

                loss.backward()

                grad = x.grad

            beta = self.beta.get_beta(t) if t is not None else self.beta
            print(f'apply constraint {beta}: {grad[0, :, 10,10]}')

            x -= beta * grad

        self.beta.update(esp=torch.mean(torch.nansum(diff, dim=-1)), eta=0.001)

        return x.detach()


class MeanGradientDensityConstraint(DiffusionConstraint):
    """
        Constraint C2 : mean vertical density gradient follows the one of the training set.
    """
    def __init__(self, beta, beta_type='constant', dims=[-1, -2], batch=8):
        super().__init__()
        self.beta = Beta(beta=beta, beta_type=beta_type)
        self.dims = dims
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch = batch

        # get the extractor function to unnormalise the data
        config = TrainingConfig()
        config.normalisation = '3-std'
        train_dataloader = get_dataloader(config.data_file, batch_size=batch,
                                                fields=config.fields, normalisation='3-std', transform=True, shuffle=True, device=self.device)
        self.extractor = train_dataloader.get_transform().uncall
        #mask for density metric
        file_mask_LR = xr.open_dataset(f"{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
        self.mask = file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})

        #mean density of the training set
        self.mean_density = torch.tensor(np.loadtxt(f'{PRP}/Results/analysis_scripts/mean_density_train.txt'), device=self.device, dtype=torch.float32)


    def apply(self, x, t=None, n_iter=1):

        for i in range(n_iter):
            #detach to remove gradients
            x = x.detach()

            with torch.enable_grad():
                if x.grad is not None:
                    x.grad.zero_()
                x.requires_grad =True
                samples = get_transformed_data(x, function=self.extractor)
                #compute density
                tmask = torch.tensor(self.mask.tmask.values, device=self.device, dtype=torch.float32).repeat(self.batch, 1, 1, 1)
                density = get_density_at_surface_tensor(samples['toce.npy'], samples['soce.npy'], tmask)

                mean_density = self.mean_density[None, :].expand(self.batch, -1)
                grad_mean_density = mean_density[:, :-1] - mean_density[:, 1:]
                grad_mean_density[:, -1] = torch.zeros_like(grad_mean_density[:, -1])

                #compute density vertical gradient
                grad_density = (density[:,:-1,:,:] - density[:,1:,:,:])
                grad_density[:, -1,:,:]= torch.zeros_like(grad_density[:, -1,:,:])

                diff = (grad_mean_density - grad_density.nanmean(dim=self.dims))**2
                loss = torch.nansum(diff)

                loss.backward()

                grad = x.grad

            beta = self.beta.get_beta(t) if t is not None else self.beta
            print(f'apply constraint {beta}: {grad[0, :, 10,10]}')

            x -= beta * grad

        self.beta.update(esp=torch.mean(torch.nansum(diff, dim=-1)), eta=0.0001)
        return x.detach()
