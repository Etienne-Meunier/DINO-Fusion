
import torch
from __init__ import PRP; import sys; sys.path.append(PRP)

from Results.analysis_scripts.metrics import get_density_at_surface_tensor
from DataLoader import get_dataloader, get_infos
from functools import partial
from TransformFields import TransformationPipeline

class Observer:
    def __init__(self, name, config=None):
        self.name = name
        if config is not None : 
            self.normalizer, self.unnormalizer = self.build_extractor(config)

    def build_extractor(self, config):
        tr = TransformationPipeline(get_infos(config.data_file),
                                     config.fields, config.normalisation, device='mps')
        train_dataloader = get_dataloader(config.data_file, tr, batch_size=1)
        transform = train_dataloader.get_transform()
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
    
