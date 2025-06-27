#%%
# Libraries
import torch
import numpy as np
import sys
sys.path.append('../../Diffusion_Model/')
from configs.base_config import TrainingConfig
from utils import get_dataloader, TransformFields
from data_analytics import get_transformed_data
import matplotlib.pyplot as plt
import einops
from torch.nn.functional import interpolate
from glob import glob
import xarray as xr
import numpy as np
from ipdb import set_trace

# Metrics implementated by G. Gachon (IPSL)

def temperature_surface_metric(temperature,   file_mask):
    """
        Metric Extraction Function :
        Unit : °C


        Input :
           -  thetao    : xarray.DataArray
           -  file_mask : xarray.Dataset
        Output :
           - np.float32 or np.float64 depending on recording precision of simulation files

    """

    # Taking Temperature At 500m depth and between 30N and 30S.

    tsurf=temperature.sel(depth=0,method='nearest')

    # Computing Area Weights from Mask over 30N-30S latitude zone and @500m depth
    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    e3t=file_mask.e3t_0.squeeze()
    tmask=file_mask.tmask.squeeze()
    area_surf=e1t*e2t*e3t*tmask.sel(depth=0,method='nearest')

    #Returning Average Temperature at 500m depth as a numpy scalar
    return ((tsurf*area_surf).sum(dim=["nav_lat","nav_lon"])/area_surf.sum(dim=["nav_lat","nav_lon"]))

def temperature_mean_metric(temperature,   file_mask):
    """
        Metric Extraction Function :
        Unit : °C


        Input :
           -  thetao    : xarray.DataArray
           -  file_mask : xarray.Dataset
        Output :
           - np.float32 or np.float64 depending on recording precision of simulation files

    """


    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    e3t=file_mask.e3t_0.squeeze()
    tmask=file_mask.tmask.squeeze()
    area_surf=e1t*e2t*e3t*tmask

    #Returning Average Temperature at 500m depth as a numpy scalar
    return ((temperature*area_surf).sum(dim=["nav_lat","nav_lon"])/area_surf.sum(dim=["nav_lat","nav_lon"]))





def temperature_500m_30NS_metric(temperature,   file_mask):
    """
        Metric Extraction Function :
        Average Temperature at 500m depth between 30N and 30S.
        Unit : °C


        Input :
           -  thetao    : xarray.DataArray
           -  file_mask : xarray.Dataset
        Output :
           - np.float32 or np.float64 depending on recording precision of simulation files

    """

    # Taking Temperature At 500m depth and between 30N and 30S.

    t500_30NS=temperature.sel(depth=500,method='nearest').where(abs(temperature.nav_lat)<30,drop=False)

    # Computing Area Weights from Mask over 30N-30S latitude zone and @500m depth
    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    e3t=file_mask.e3t_0.squeeze()
    tmask=file_mask.tmask.squeeze()
    area_500m_30NS=e1t*e2t*e3t*tmask.sel(depth=500,method='nearest').where(abs(temperature.nav_lat)<30,drop=False)

    #Returning Average Temperature at 500m depth as a numpy scalar
    return ((t500_30NS*area_500m_30NS).sum(dim=["nav_lat","nav_lon"])/area_500m_30NS.sum(dim=["nav_lat","nav_lon"]))



def temperature_BWbox_metric(thetao,   file_mask, depth_box=3000):
    """
        Metric Extraction Function :
        Average Temperature in a U-shaped "Bottom Water" box corresponding to waters below 3000m or beyond 30 degrees of latitude North and South.

        ________________________________________________ _Surface
        | . . . . |__________________________| . . . . |_500m
        | . . . . |                          | . . . . |
        | . . . . |        Deep Water        | . . . . |
        | . . . . |__________________________| . . . . |_3000m
        | . . . . . . . . Bottom Water . . . . . . . . |
        |______________________________________________|_Bottom
        S        30S           Eq.          30N        N

        Figure : Schematic Representation of the Bottom Water box used in this metric.

        Unit : °C

        Input :
           -  thetao    : xarray.DataArray
           -  file_mask : xarray.Dataset
        Output :
           - np.float32 or np.float64 depending on recording precision of simulation files

    """

    t_BW=thetao.where(1-(thetao.depth<depth_box)*(abs(thetao.nav_lat)<30))

    # Computing Area Weights from Mask over Box
    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    e3t=file_mask.e3t_0.squeeze()
    tmask=file_mask.tmask.squeeze()
    area_BW=e1t*e2t*e3t*tmask.where(1-(thetao.depth<depth_box)*(abs(thetao.nav_lat)<30))

    #Returning Average Temperature on Box
    return ((t_BW*area_BW).sum(dim=["nav_lat","nav_lon","depth"])/area_BW.sum(dim=["nav_lat","nav_lon","depth"]))



def temperature_DWbox_metric(thetao,   file_mask, depth_box=3000):
    """
        Metric Extraction Function :
        Average Temperature in a "Deep Water" box corresponding to waters between 500m and 3000m depth and 30°N and 30°S.

        ________________________________________________ _Surface
        |         |__________________________|         |_500m
        |         | . . . . . . . . . . . . .|         |
        |         | . . . .Deep Water . . . .|         |
        |         |__________________________|         |_3000m
        |                 Bottom Water                 |
        |______________________________________________|_Bottom
        S        30S           Eq.          30N        N

        Figure : Schematic Representation of the Deep Water box used in this metric.

        Unit : °C

        Input :
           -  thetao    : xarray.DataArray
           -  file_mask : xarray.Dataset
        Output :
           - np.float32 or np.float64 depending on recording precision of simulation files

    """
    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    e3t=file_mask.e3t_0.squeeze()
    tmask=file_mask.tmask.squeeze()
    condition = (thetao.depth<depth_box)*(thetao.depth>500)*(abs(thetao.nav_lat)<30)
    t_DW=thetao.where(condition)

    # Computing Area Weights from Mask over Box
    area_DW=e1t*e2t*e3t*tmask.where(condition)

    #Returning Average Temperature on Box
    return ((t_DW*area_DW).sum(dim=["nav_lat","nav_lon","depth"])/area_DW.sum(dim=["nav_lat","nav_lon","depth"]))


def volume_avg(data, file_mask, condition=1) :
    """
        Compute the average over the data ponderated by volume
    """
    volume = (file_mask.e3t_0 * file_mask.e1t * file_mask.e2t) # Compute the volume for each cll
    tmask = file_mask.tmask # Extract water areas
    mask = tmask * condition
    return (data * volume * mask).sum() / (volume * mask).sum()


def get_density_at_surface(thetao, so, tmask):
    """
    Compute potential density referenced at the surface.

    Parameters:
        thetao (numpy.array) : Temperature array - (t,z,y,x).
        so (numpy.array)     : Salinity array    - (t,z,y,x).
        tmask (numpy.array)  : Mask array        - (t,z,y,x).

    Returns:
        tuple: A tuple containing:
            array: Potential density referenced at the surface.
    """
    rdeltaS = 32.0
    r1_S0 = 0.875 / 35.16504
    r1_T0 = 1.0 / 40.0
    r1_Z0 = 1.0e-4

    EOS000 = 8.0189615746e02
    EOS100 = 8.6672408165e02
    EOS200 = -1.7864682637e03
    EOS300 = 2.0375295546e03
    EOS400 = -1.2849161071e03
    EOS500 = 4.3227585684e02
    EOS600 = -6.0579916612e01
    EOS010 = 2.6010145068e01
    EOS110 = -6.5281885265e01
    EOS210 = 8.1770425108e01
    EOS310 = -5.6888046321e01
    EOS410 = 1.7681814114e01
    EOS510 = -1.9193502195
    EOS020 = -3.7074170417e01
    EOS120 = 6.1548258127e01
    EOS220 = -6.0362551501e01
    EOS320 = 2.9130021253e01
    EOS420 = -5.4723692739
    EOS030 = 2.1661789529e01
    EOS130 = -3.3449108469e01
    EOS230 = 1.9717078466e01
    EOS330 = -3.1742946532
    EOS040 = -8.3627885467
    EOS140 = 1.1311538584e01
    EOS240 = -5.3563304045
    EOS050 = 5.4048723791e-01
    EOS150 = 4.8169980163e-01
    EOS060 = -1.9083568888e-01
    EOS001 = 1.9681925209e01
    EOS101 = -4.2549998214e01
    EOS201 = 5.0774768218e01
    EOS301 = -3.0938076334e01
    EOS401 = 6.6051753097
    EOS011 = -1.3336301113e01
    EOS111 = -4.4870114575
    EOS211 = 5.0042598061
    EOS311 = -6.5399043664e-01
    EOS021 = 6.7080479603
    EOS121 = 3.5063081279
    EOS221 = -1.8795372996
    EOS031 = -2.4649669534
    EOS131 = -5.5077101279e-01
    EOS041 = 5.5927935970e-01
    EOS002 = 2.0660924175
    EOS102 = -4.9527603989
    EOS202 = 2.5019633244
    EOS012 = 2.0564311499
    EOS112 = -2.1311365518e-01
    EOS022 = -1.2419983026
    EOS003 = -2.3342758797e-02
    EOS103 = -1.8507636718e-02
    EOS013 = 3.7969820455e-01

    zt = thetao * r1_T0  # temperature
    zs = np.sqrt(np.abs(so + rdeltaS) * r1_S0)  # square root salinity
    ztm = tmask.squeeze()

    zn3 = EOS013 * zt + EOS103 * zs + EOS003
    zn2 = (
        (EOS022 * zt + EOS112 * zs + EOS012) * zt + (EOS202 * zs + EOS102) * zs + EOS002
    )
    zn1 = (
        (
            (
                (EOS041 * zt + EOS131 * zs + EOS031) * zt
                + (EOS221 * zs + EOS121) * zs
                + EOS021
            )
            * zt
            + ((EOS311 * zs + EOS211) * zs + EOS111) * zs
            + EOS011
        )
        * zt
        + (((EOS401 * zs + EOS301) * zs + EOS201) * zs + EOS101) * zs
        + EOS001
    )
    zn0 = (
        (
            (
                (
                    (
                        (EOS060 * zt + EOS150 * zs + EOS050) * zt
                        + (EOS240 * zs + EOS140) * zs
                        + EOS040
                    )
                    * zt
                    + ((EOS330 * zs + EOS230) * zs + EOS130) * zs
                    + EOS030
                )
                * zt
                + (((EOS420 * zs + EOS320) * zs + EOS220) * zs + EOS120) * zs
                + EOS020
            )
            * zt
            + ((((EOS510 * zs + EOS410) * zs + EOS310) * zs + EOS210) * zs + EOS110)
            * zs
            + EOS010
        )
        * zt
        + (
            ((((EOS600 * zs + EOS500) * zs + EOS400) * zs + EOS300) * zs + EOS200) * zs
            + EOS100
        )
        * zs
        + EOS000
    )
    rhop = zn0 * ztm  # potential density referenced at the surface
    return rhop

def compute_density(batch, file_mask_LR):
    """
        Take a batch with toce (b, z, h, w) and soce along with the maks
        and return the corresponding density 
    """
    bsize = batch['soce.npy'].shape[0]
    device = batch["soce.npy"].device
    dtype = batch["soce.npy"].dtype

    data = file_mask_LR.e3t_0.copy().expand_dims({'batch': bsize})
    soce = data.copy().astype(np.float32)
    soce[:] = batch['soce.npy'].cpu().numpy()
    toce = data.copy().astype(np.float32)
    toce[:] = batch['toce.npy'].cpu().numpy()

    tmask = data.copy().astype(np.float32)
    tmask[:] = np.repeat(file_mask_LR.tmask.values[None], bsize, 0)

    density= get_density_at_surface(toce, soce, tmask)

    density = torch.tensor(density.values, device=device, dtype=dtype)

    return density

import torch

def get_density_at_surface_tensor(thetao, so, tmask):
    """
    Compute potential density referenced at the surface using PyTorch tensors.

    Parameters:
        thetao (torch.Tensor): Temperature tensor - (t, z, y, x).
        so (torch.Tensor): Salinity tensor - (t, z, y, x).
        tmask (torch.Tensor): Mask tensor - (t, z, y, x).

    Returns:
        torch.Tensor: Potential density referenced at the surface.
    """
    # Constants
    rdeltaS = 32.0
    r1_S0 = 0.875 / 35.16504
    r1_T0 = 1.0 / 40.0

    # EOS coefficients
    EOS000 = 8.0189615746e02
    EOS100 = 8.6672408165e02
    EOS200 = -1.7864682637e03
    EOS300 = 2.0375295546e03
    EOS400 = -1.2849161071e03
    EOS500 = 4.3227585684e02
    EOS600 = -6.0579916612e01
    EOS010 = 2.6010145068e01
    EOS110 = -6.5281885265e01
    EOS210 = 8.1770425108e01
    EOS310 = -5.6888046321e01
    EOS410 = 1.7681814114e01
    EOS510 = -1.9193502195
    EOS020 = -3.7074170417e01
    EOS120 = 6.1548258127e01
    EOS220 = -6.0362551501e01
    EOS320 = 2.9130021253e01
    EOS420 = -5.4723692739
    EOS030 = 2.1661789529e01
    EOS130 = -3.3449108469e01
    EOS230 = 1.9717078466e01
    EOS330 = -3.1742946532
    EOS040 = -8.3627885467
    EOS140 = 1.1311538584e01
    EOS240 = -5.3563304045
    EOS050 = 5.4048723791e-01
    EOS150 = 4.8169980163e-01
    EOS060 = -1.9083568888e-01
    EOS001 = 1.9681925209e01
    EOS101 = -4.2549998214e01
    EOS201 = 5.0774768218e01
    EOS301 = -3.0938076334e01
    EOS401 = 6.6051753097
    EOS011 = -1.3336301113e01
    EOS111 = -4.4870114575
    EOS211 = 5.0042598061
    EOS311 = -6.5399043664e-01
    EOS021 = 6.7080479603
    EOS121 = 3.5063081279
    EOS221 = -1.8795372996
    EOS031 = -2.4649669534
    EOS131 = -5.5077101279e-01
    EOS041 = 5.5927935970e-01
    EOS002 = 2.0660924175
    EOS102 = -4.9527603989
    EOS202 = 2.5019633244
    EOS012 = 2.0564311499
    EOS112 = -2.1311365518e-01
    EOS022 = -1.2419983026
    EOS003 = -2.3342758797e-02
    EOS103 = -1.8507636718e-02
    EOS013 = 3.7969820455e-01

    # Tensor computations
    zt = thetao * r1_T0  # temperature
    zs = torch.sqrt(torch.abs(so + rdeltaS) * r1_S0)  # square root salinity
    ztm = tmask.squeeze()

    

    zn3 = EOS013 * zt + EOS103 * zs + EOS003
    zn2 = ((EOS022 * zt + EOS112 * zs + EOS012) * zt + (EOS202 * zs + EOS102) * zs + EOS002)
    zn1 = (((((EOS041 * zt + EOS131 * zs + EOS031) * zt + (EOS221 * zs + EOS121) * zs + EOS021) * zt
            + ((EOS311 * zs + EOS211) * zs + EOS111) * zs + EOS011) * zt
            + (((EOS401 * zs + EOS301) * zs + EOS201) * zs + EOS101) * zs + EOS001))
    zn0 = (
        (
            (
                (
                    (
                        (EOS060 * zt + EOS150 * zs + EOS050) * zt
                        + (EOS240 * zs + EOS140) * zs
                        + EOS040
                    )
                    * zt
                    + ((EOS330 * zs + EOS230) * zs + EOS130) * zs
                    + EOS030
                )
                * zt
                + (((EOS420 * zs + EOS320) * zs + EOS220) * zs + EOS120) * zs
                + EOS020
            )
            * zt
            + ((((EOS510 * zs + EOS410) * zs + EOS310) * zs + EOS210) * zs + EOS110)
            * zs
            + EOS010
        )
        * zt
        + (
            ((((EOS600 * zs + EOS500) * zs + EOS400) * zs + EOS300) * zs + EOS200) * zs
            + EOS100
        )
        * zs
        + EOS000
    )
    rhop = zn0 * ztm  # potential density referenced at the surface
    return rhop

def compute_density_tensor(batch, file_mask_LR):
    """
    Compute density from temperature, salinity, and mask tensors.

    Parameters:
        batch (dict): Contains tensors for "soce" (salinity) and "toce" (temperature).
        file_mask_LR: Contains mask tensors.

    Returns:
        torch.Tensor: Computed density.
    """
    bsize = batch["soce.npy"].shape[0]
    device = batch["soce.npy"].device
    dtype = batch["soce.npy"].dtype

    data = file_mask_LR.e3t_0.copy().expand_dims({'batch': bsize})  

    #pass to tensor
    #data = torch.tensor(data.values, device=device, dtype=dtype)
    #soce = data.clone()
    soce = batch["soce.npy"]
    #toce = data.clone()
    toce = batch["toce.npy"]

    #tmask = data.clone()
    tmask = torch.tensor(np.repeat(file_mask_LR.tmask.values[None], bsize, 0), device=device, dtype=dtype)

    # Compute density
    density= get_density_at_surface_tensor(toce, soce, tmask)
    return density

#%%
if __name__ =='__main__' :

    config = TrainingConfig()
    config.normalisation = '3-std'
    config.data_file = '../../../DATA_DINOFusion/dino_1_4_degree_coarse_240125.tar'
    train_dataloader = get_dataloader(config.data_file, batch_size=50,
                                                fields=config.fields, normalisation='3-std', transform=True, shuffle=False, device='mps')
    extractor = train_dataloader.get_transform().uncall

    idt = iter(train_dataloader)
    batch = next(idt)
    #print(batch.shape)

    file_mask_LR = xr.open_dataset("data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
    mask = file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})

    batch_norm = get_transformed_data(batch, function=extractor)

#%%
    density, zn0 = compute_density_tensor(batch_norm, mask)
    density_np, zn0_np= compute_density(batch_norm, mask)
    #grad = density.mean(dim=[-1, -2], keepdim=True)
    #density[density != density] = 0

    #vertical_grad = density - torch.roll(density, -1, dims=1)
    #dz = torch.tensor(mask.e3t_0.values, device='mps', dtype=torch.float32)
    #vertical_grad = vertical_grad/dz
    #vertical_grad[:,-1,:,:] = torch.zeros([8,vertical_grad.shape[2],vertical_grad.shape[3]], device='mps')



#%%
    #compute mean density per z level for the training dataset
    density_training = torch.zeros([1,36], device='mps')
    nan_count = torch.zeros([1,36], device='mps')
    BATCH_SIZE = 50
    num_elt = 0
    
    for step, batch in enumerate(train_dataloader):
        batch_norm = get_transformed_data(batch, function=extractor)
        density = compute_density_tensor(batch_norm, mask)
        print(density.nanmean(dim=[0, -1,-2])[0])
        density_training = density_training + torch.nansum(density, dim=[0,-1, -2])
        num_elt += BATCH_SIZE
        if step ==0:
            nan_count = torch.isnan(density[0,:,:,:]).sum(dim=(1,2))

    N =  num_elt * 199 * 62 * torch.ones([1,36], device='mps') - nan_count * num_elt

    mean_density_train = density_training / N
    mean_density_train[0, -1]= 0.0 #remove nan

    mean_dens = mean_density_train.cpu().numpy()

    plt.figure(figsize=(6, 8))
    scatter = plt.scatter(np.zeros_like(mask.e3t_0.depth[:-1]), mask.e3t_0.depth[:-1], c=mean_dens[0, :-1], cmap="viridis", label="Mean Density")
    #plt.scatter(mean_dens[0, :-1], , label="Mean Density")
    plt.gca().invert_yaxis()  
    plt.xlabel("Mean Density")
    #plt.ylabel("Depth")
    plt.title("Mean Density vs Depth")
    plt.colorbar(scatter, label="Mean Density")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()


#%%
# Save to text file
    np.savetxt('mean_density_train.txt', mean_dens, fmt='%.6f')


# %%
