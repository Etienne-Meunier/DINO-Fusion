import torch
from ipdb import set_trace
import numpy as np
from configs.base_config import TrainingConfig
from utils import get_dataloader
import xarray as xr
import sys
import torch.nn.functional as F
from einops import rearrange

# adding Folder_2 to the system path
sys.path.insert(0, 'Results/analysis_scripts/')

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
        self.unormalize = train_dataloader.get_transform().unstandardize_4D
        self.mask_infos = train_dataloader.get_transform().infos['mask']['toce'] #verifier que les mask toce et soce sont identiques
        self.fields = train_dataloader.get_transform().fields
        self.infos_shape = train_dataloader.get_transform().infos['shape']

        #mask for density metric
        file_mask_LR = xr.open_dataset("Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
        self.mask = file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})

        #mean density of the training set
        self.mean_density = torch.tensor(np.loadtxt('././Results/analysis_scripts/mean_density_train.txt'), device='mps', dtype=torch.float32)

    def format_density(self, density):
        #replace nan values 
        density[self.mask_infos.repeat(self.batch,1,1,1)] = 0
        #add correct padding
        padding = (1, 1, 5, 4)
        density_padded = F.pad(density, padding, mode='constant', value=0)
        #reshape levels and concatenate (zero for ssh layer)
        density_sliced = density_padded[:, 0:-1:2]
        return density_sliced
    
    def un_stride_concat(self, data, interpolation=True) :
            idx = 0
            sample = {}
            for key in self.fields.keys() :
                oz = self.infos_shape[key][0] # original z
                levels = len(torch.arange(oz)[self.fields[key]]) #torch version
                field =  data[:, idx:levels+idx]
                idx += levels
                if interpolation :
                    interp_field = F.interpolate(rearrange(field, 'b z x y -> b (x y) z'), size=(oz), mode='linear')
                    sample[key] = rearrange(interp_field, 'b (x y) z -> b z x y', x=field.shape[2], y=field.shape[3])
                else :
                    sample[key] = field
            return sample


    def apply(self, x, t=None): 
        #unconcat x
        x = x.detach()
        sample = self.un_stride_concat(x, interpolation=True)
        #unpad 
        sal_unpad = sample['soce'][:, :,5:-4, 1:-1]
        temp_unpad = sample['toce'][:, :,5:-4, 1:-1]

        with torch.enable_grad():
            sal_unpad.requires_grad = True
            temp_unpad.requires_grad = True

            sal = self.unormalize(sal_unpad, 'soce')
            temp = self.unormalize(temp_unpad, 'toce')

            #compute density
            tmask = torch.tensor(self.mask.tmask.values, device=self.device, dtype=torch.float32).repeat(self.batch, 1, 1, 1)
            density = get_density_at_surface_tensor(temp, sal, tmask)

            mean_density = self.mean_density[None, :].expand(self.batch, -1)

            loss = torch.sum((mean_density - density.mean(dim=self.dims))**2, dim=1)
            
            grads_sal = []
            grads_temp = []
            for i in range(self.batch):
                if sal_unpad.grad is not None:
                    sal_unpad.grad.zero_()
                if temp_unpad.grad is not None:
                    temp_unpad.grad.zero_()

                loss[i].backward(retain_graph=True)
                grads_sal.append(sal_unpad.grad[i].clone())
                grads_temp.append(temp_unpad.grad[i].clone())

        #set_trace()
        grad_sal_padded = self.format_density(torch.stack(grads_sal))
        grad_temp_padded = self.format_density(torch.stack(grads_temp))

        beta = self.beta.get_beta(t) if t is not None else self.beta
        print(f'apply constraint {beta}: {grad_sal_padded[0, :, 10,10]}')
        
        ssh_layer = torch.zeros([self.batch, 1, grad_temp_padded.shape[2], grad_temp_padded.shape[3]], device=self.device)
        grad_all = torch.cat([grad_temp_padded, grad_sal_padded, ssh_layer], dim=1)
        
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


    def format_tensor(self, tensor):
        #add padding
        padding = (1, 1, 5, 4)
        tensor_padded = F.pad(tensor, padding, mode='constant', value=0)
        #reshape levels and concatenate (zero for ssh layer)
        tensor_sliced = tensor_padded[:, 0:-1:2]
        return tensor_sliced


    def apply(self, x, t=None): 
        #un-normalize
        h= x.shape[2]
        w = x.shape[3]
        x_norm = get_transformed_data(x, function=self.extractor)

        #compute density
        density = compute_density_tensor(x_norm, self.mask)
        #remove nan 
        density[self.mask_infos.repeat(self.batch,1,1,1)] = 0
        #compute density vertical gradient
        dz = torch.tensor(self.mask.e3t_0.values, device=self.device, dtype=torch.float32)
        grad = (density - torch.roll(density, -1, dims=1)) /dz
        grad[:,-1,:,:] = torch.zeros([self.batch,grad.shape[2],grad.shape[3]], device=self.device)

        grad_padded = self.format_tensor(grad)
    
        beta = self.beta.get_beta(t) if t is not None else self.beta
        print(f'apply constraint {beta}: {grad_padded[0, :, 120,30]}')
        #add ssh layer
        ssh_layer = torch.zeros([self.batch,1,h,w], device=self.device)
        grad_all = torch.cat([grad_padded, grad_padded, ssh_layer], dim=1)
        
        x -= beta * torch.max(grad_all, torch.tensor(0.0))
        return x