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
    tmask=file_mask.tmask.squeeze()
    area_500m_30NS=e1t*e2t*tmask.sel(depth=500,method='nearest').where(abs(temperature.nav_lat)<30,drop=False)

    #Returning Average Temperature at 500m depth as a numpy scalar
    return ((t500_30NS*area_500m_30NS).sum(dim=["nav_lat","nav_lon"])/area_500m_30NS.sum(dim=["nav_lat","nav_lon"]))



def temperature_BWbox_metric(thetao,   file_mask):
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

    t_BW=thetao.where(1-(thetao.depth<3000)*(abs(thetao.nav_lat)<30))

    # Computing Area Weights from Mask over Box
    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    tmask=file_mask.tmask.squeeze()
    area_BW=e1t*e2t*tmask.where(1-(thetao.depth<3000)*(abs(thetao.nav_lat)<30))

    #Returning Average Temperature on Box
    return ((t_BW*area_BW).sum(dim=["nav_lat","nav_lon","depth"])/area_BW.sum(dim=["nav_lat","nav_lon","depth"]))



def temperature_DWbox_metric(thetao,   file_mask):
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
    tmask=file_mask.tmask.squeeze()
    t_DW=thetao.where(abs((thetao.depth-1750)<1250)*(abs(thetao.nav_lat)<30))

    # Computing Area Weights from Mask over Box
    area_DW=e1t*e2t*tmask.where(abs((thetao.depth-1750)<1250)*(abs(thetao.nav_lat)<30))


    #Returning Average Temperature on Box
    return ((t_DW*area_DW).sum(dim=["nav_lat","nav_lon","depth"])/area_DW.sum(dim=["nav_lat","nav_lon","depth"]))
