
import torch
import xarray as xr
from __init__ import PRP; import sys; sys.path.append(PRP)

from Results.analysis_scripts.metrics import get_density_at_surface_tensor
from DataLoader import get_infos
from functools import partial
from TransformFields import TransformationPipeline

class Observer:
    """
        Observation operators on normalised states (b, c, i, j) : un-normalised fields, density and its profiles.
        All operators are differentiable with respect to the normalised state.
        Grid metrics of the full resolution levels (file grid, (36, 199, 62)), zero on land :
            area (i, j)          : horizontal area of the T cells (m^2)
            tmask (k, i, j)      : 1 on ocean T cells
            volume (k, i, j)     : volume of the T cells (m^3)
            volume_w (k-1, i, j) : volume between the T points k and k+1, ocean if both are (m^3), weight of density_gradient
    """
    def __init__(self, name, config=None, device='cpu',
                 mesh_file=f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc'):
        self.name = name
        if config is not None : 
            self.normalizer, self.unnormalizer = self.build_extractor(config, device)
        self.area, self.tmask, self.volume, self.volume_w = self.build_grid(mesh_file, device)

    @staticmethod
    def build_grid(mesh_file, device):
        """Grid metrics (area, tmask, volume, volume_w) of the NEMO mesh mask `mesh_file`, see the class docstring."""
        mesh = xr.open_dataset(mesh_file).isel(time_counter=0)
        get = lambda v : torch.tensor(mesh[v].values, dtype=torch.float32, device=device)
        area, tmask = get('e1t') * get('e2t'), get('tmask')
        volume = area * get('e3t_0') * tmask
        # e3w_0[k] is the distance between the T points k-1 and k
        volume_w = area * get('e3w_0')[1:] * tmask[:-1] * tmask[1:]
        return area, tmask, volume, volume_w

    def cell_volume(self, result=None):
        """Ocean volume of the cells (k, i, j) (m^3) of the file levels for result None / 'file', of the strided levels for 'strided'."""
        if result != 'strided' :
            return self.volume
        return self.volume[self.normalizer.striders['toce'].stride_pattern]

    def level_volume(self, result=None):
        """Ocean volume of each level (k,) (m^3), see `cell_volume`."""
        return self.cell_volume(result).sum(dim=(-2, -1))

    def build_extractor(self, config, device):
        transform = TransformationPipeline(get_infos(config.data_file),
                                           config.fields, config.normalisation, device=device)
        return transform, transform.__uncall__

    def unnormalized(self, normalized, result=None):
        """Unnormalized state getter."""
        return self.get_transformed_data(normalized, partial(self.unnormalizer, result=result))
    
    @staticmethod
    def get_transformed_data(batch: torch.Tensor, function):
        """Extract and stack transformed data from a batch."""
        samples = [function(g) for g in batch]
        return torch.stack(samples, dim=0)
        
    def density(self, normalized, result=None) : 
        unnormalized = self.unnormalized(normalized, result)
        density = get_density_at_surface_tensor(unnormalized['toce'], unnormalized['soce'],
                                                 tmask=None)
        return density

    def density_profile(self, normalized, result=None) : 
        density = self.density(normalized, result)
        return density.nanmean(axis=(-2,-1)) # Spatial mean density -> (N, Z)

    
    def density_gradient(self, normalized, result=None) : 
        density = self.density(normalized, result)
        gradient_density = density[:, :-1] -  density[:, 1:]
        return gradient_density
    
    def density_gradient_profile(self, normalized, result=None) : 
        density_gradient = self.density_gradient(normalized, result)
        return density_gradient.nanmean(axis=(-2,-1)) # Spatial mean density -> (N, Z)
    
