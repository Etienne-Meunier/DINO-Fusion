
import torch
from __init__ import PRP; import sys; sys.path.append(PRP)

from Results.analysis_scripts.metrics import get_density_at_surface_tensor
from utils import get_dataloader

class State:
    def __init__(self, name, config):
        self.name = name
        self.normalizer, self.unnormalizer = self.build_extractor(config)
        self._normalized = None
        self._unnormalized = None

    def build_extractor(self, config):
        train_dataloader = get_dataloader(
            config.data_file,
            batch_size=1,
            fields=config.fields,
            normalisation=config.normalisation,
            shuffle=False,
        )
        transform = train_dataloader.get_transform()
        return transform, transform.uncall

    @property
    def normalized(self):
        """Normalized state getter."""
        if (self._normalized is None) and (self._unnormalized is not None) :
            self._normalized = self.get_transformed_data(self._unnormalized, self.normalizer)
        return self._normalized

    @normalized.setter
    def normalized(self, normalized):
        self._normalized = normalized
        self._unnormalized =  None

    @property
    def unnormalized(self):
        """Unnormalized state getter."""
        if (self._unnormalized is None) and (self._normalized is not None) :
            self._unnormalized = self.get_transformed_data(self._normalized, self.unnormalizer)
        return self._unnormalized

    @unnormalized.setter
    def unnormalized(self, unnormalized):
        self._unnormalized = unnormalized
        self._normalized = None

    @staticmethod
    def get_transformed_data(batch: torch.Tensor, function):
        """Extract and stack transformed data from a batch."""
        samples = [function(b) for b in batch]  # list[dict[str, Tensor]]
        return {k: torch.stack([s[k] for s in samples], 0) for k in samples[0]}
        
    @property
    def density(self) : 
        density = get_density_at_surface_tensor(self.unnormalized['toce.npy'], self.unnormalized['soce.npy'], tmask=None)
        return density

    @property
    def density_gradient(self) : 
        density = get_density_at_surface_tensor(self.unnormalized['toce.npy'], self.unnormalized['soce.npy'], tmask=None)
        gradient_density = density[:, :-1] -  density[:, 1:]
        return gradient_density