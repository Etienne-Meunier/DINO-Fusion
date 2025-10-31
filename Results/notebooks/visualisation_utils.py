from __init__ import PRP; import sys
sys.path.append(PRP)
sys.path.append(PRP + 'Diffusion_model')
sys.path.append(PRP + 'Results/analysis_scripts/')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import warnings
import torch
from utils import get_dataloader
from metrics import get_density_at_surface_tensor


# Suppress deprecation/future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Try new API, fallback to old one
try:
    import matplotlib.colormaps as mcm
    get_cmap = lambda name, n=None: mcm.get_cmap(name, n)
except ImportError:
    from matplotlib.cm import get_cmap  # for older versions

from matplotlib.colors import ListedColormap, to_rgba

def histogram(ar, ax, bins=50, cmap_name='viridis', label=None):
    """Plot a histogram of array values with discrete colors."""
    flat = ar.flatten()
    flat = flat[flat != 0.0]

    counts, edges, patches = ax.hist(flat, bins=bins, alpha=0.7,
                                     label=label, edgecolor='black')

    # Detect if cmap_name is a single color (no variation)
    if cmap_name in plt.colormaps():  # standard colormap name
        cmap = get_cmap(cmap_name, len(edges) - 1)
    else:  # interpret as a plain color
        color = to_rgba(cmap_name)
        cmap = ListedColormap([color] * (len(edges) - 1))

    norm = BoundaryNorm(edges, cmap.N)

    # Get discrete colors robustly
    try:
        colors = cmap.colors
    except AttributeError:
        colors = [cmap(i / (len(edges) - 2)) for i in range(len(edges) - 1)]

    # Color bars
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    if label is not None:
        ax.legend()

    return edges, cmap, norm


def show_image_with_hist(img, bins=50, cmap_name='viridis', title=None):
    """Show image and corresponding histogram with shared discrete color mapping."""
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 4))

    edges, cmap, norm = histogram(img, ax=ax_hist, bins=bins, cmap_name=cmap_name)

    im = ax_img.imshow(img, cmap=cmap, norm=norm)
    ax_img.set_title(title if title else "Image")
    fig.colorbar(im, ax=ax_hist, label="Value bins")

    plt.tight_layout()
    plt.show()



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
        self._normalized = normalized.detach()
        self._unnormalized =  None

    @property
    def unnormalized(self):
        """Unnormalized state getter."""
        if (self._unnormalized is None) and (self._normalized is not None) :
            self._unnormalized = self.get_transformed_data(self._normalized, self.unnormalizer)
        return self._unnormalized

    @unnormalized.setter
    def unnormalized(self, unnormalized):
        self._unnormalized = unnormalized.detach()
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