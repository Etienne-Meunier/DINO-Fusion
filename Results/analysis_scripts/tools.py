import torch
import sys

sys.path.append('../../Diffusion_Model/')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import numpy as np
from ShapeChecker import ShapeCheck
from sklearn.cluster import KMeans
mpl.rcParams['image.origin'] = 'lower'



def operate_on_dataset(dataloader, operations, fields, data_extractor):
        """
        Calculate averages for multiple variables across defined areas.

        Args:
            dataloader: DataLoader containing the data
            operations: A pair {key:function} that will be applied on each batch
            areas: Dictionary of areas with their slices
            data_extractor: Function to extract data from batch (with all arguments bound)

        Returns:
            Dictionary of structure {key: {area: tensor}} containing averages
        """
        results = {field: {area: [] for area in operations.keys()} for field in fields}

        for batch in tqdm(dataloader):
            batch = data_extractor(batch)
            for field in fields :
                data = batch[field]
                for key, function in operations.items():
                    results[field][key].append(function(data))


        return {field : {key : torch.cat(results[field][key]) for key in results[field].keys()} for field in results.keys()}

def calculate_area_averages(dataloader, keys, areas, data_extractor):
    """
    Calculate averages for multiple variables across defined areas.

    Args:
        dataloader: DataLoader containing the data
        keys: List of keys to process (e.g., ['soce.npy', 'toce.npy', 'ssh.npy'])
        areas: Dictionary of areas with their slices
        data_extractor: Function to extract data from batch (with all arguments bound)

    Returns:
        Dictionary of structure {key: {area: tensor}} containing averages
    """
    results = {key: {area: [] for area in areas.keys()} for key in keys}

    for batch in tqdm(dataloader):
        batch = data_extractor(batch)
        for key in keys:
            data = batch[key]
            for area_name, area_slice in areas.items():
                area_mean = data[..., area_slice, :].nanmean(dim=(-2, -1))
                results[key][area_name].append(area_mean)

    return {key: {area: torch.concat(values) for area, values in areas_dict.items()} for key, areas_dict in results.items()}


def concat_dict(ds) :
    catds = {k : [] for k in ds[0].keys()}
    for d in ds :
        for k in d.keys() :
            catds[k].append(d[k])
    return {k : torch.stack(catds[k], 0) for k in catds}


def plot_area_distributions(data_dict, z_index=0, figsize=(15, 5), title=''):
    """
    Plot distributions of different fields for different areas.

    Args:
        data_dict: Dictionary from calculate_area_averages
        z_index: Index to use for fields with z dimension (default=0)
        figsize: Figure size (width, height)
    """
    n_fields = len(data_dict)
    fig, axes = plt.subplots(1, n_fields, figsize=figsize)
    fig.suptitle(title)
    for ax, (field, areas_data) in zip(axes, data_dict.items()):
        # Prepare data for this field
        plot_data = {}
        for area, values in areas_data.items():
            # If data has z dimension, select specific z
            if len(values.shape) > 1:
                plot_data[f"{area}"] = values[:, z_index]
            else:
                plot_data[f"{area}"] = values

        # Create boxplot
        sns.boxplot(data=plot_data, ax=ax)

        # Clean up field name for title
        field_name = field.replace('.npy', '')
        ax.set_title(field_name)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_cluster_data(data, n_cluster=6, name='', fill_na=1020, show_layer=0) : 
    """
    Cluster and plot spatial data 
    
    :param data: (k, i, j) -> spatial array to cluster
    :param n_cluster: int, number of cluster
    """

    # From here on, *only* use centers_sorted and clusters_sorted
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    fig.suptitle(f'{name}')
    axs[0].set_title(f'data layer={show_layer}')
    im = axs[0].imshow(
        data[show_layer],
        vmin=np.nanmin(data[show_layer].numpy()),
        vmax=np.nanmax(data[show_layer].numpy())
    )
    plt.colorbar(im, ax=axs[0])


    data = data.nan_to_num(fill_na)
    sc = ShapeCheck(data.shape, 'k i j')
    data_flat = sc.rearrange(data, 'k i j -> (i j) k')
    km = KMeans(n_cluster)
    #valid_mask = data_flat.isnan().sum(axis=-1) == 0 # (i, j)
    #clusters = np.full((data_flat.shape[0],), fill_value=np.nan)
    #clusters[valid_mask] = km.fit_predict(data_flat[valid_mask])
    clusters = km.fit_predict(data_flat)

    # --- Sort clusters by surface density and relabel ---
    surface_data = km.cluster_centers_[:, 0]          # "surface" value for each cluster
    order = np.argsort(surface_data)                  # indices of clusters sorted by surface density

    # Build mapping old_label -> new_label such that new_label encodes sorted order
    old_to_new = np.empty_like(order)
    old_to_new[order] = np.arange(len(order))

    # Apply mapping to labels and centers
    clusters_sorted = old_to_new[clusters]               # relabeled cluster IDs
    centers_sorted = km.cluster_centers_[order]          # centers in increasing surface density order


    n_clusters = km.n_clusters  # or 8
    cmap = plt.cm.get_cmap('Set1', n_clusters)

    # --- Line plot with colors following sorted order ---
    axs[1].set_title('Clusters')
    axs[1].set_prop_cycle(color=[cmap(i) for i in range(n_clusters)])
    axs[1].plot(centers_sorted.T)   # centers are already ordered by surface density

    # --- Cluster map with same cmap/index convention ---
    axs[2].set_title('Cluster Map')
    cluster_map = sc.rearrange(clusters_sorted, '(i j) -> i j')
    axs[2].imshow(
        cluster_map,
        cmap=cmap,
        vmin=0,
        vmax=n_clusters - 1,
    )
    return fig, axs