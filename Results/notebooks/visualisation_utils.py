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