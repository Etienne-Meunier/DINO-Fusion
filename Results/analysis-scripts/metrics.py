# Libraries
import torch
import numpy as np
import sys
sys.path.append('../../Diffusion_Model/')
from configs.base_config import TrainingConfig
from utils import get_dataloader, TransformFields
import matplotlib.pyplot as plt
import einops
from torch.nn.functional import interpolate
from glob import glob
import xarray as xr
import numpy as np


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
