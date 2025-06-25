from typing import Union
from datetime import date

import numpy as np

import rasters as rt
from rasters import Raster, RasterGeometry

from .VIIRS.VIIRSDownloader import VIIRSDownloaderAlbedo

def generate_albedo_coarse_image(
    date_UTC: Union[date, str], VIIRS_connection: VIIRSDownloaderAlbedo, geometry: RasterGeometry = None
) -> Raster:
    """
    Generates a coarse-resolution albedo image from VIIRS data.

    Args:
        date_UTC (Union[date, str]): The UTC date for which to retrieve albedo data.
        VIIRS_connection (VIIRSDownloaderAlbedo): An initialized VIIRS albedo downloader object.
        geometry (RasterGeometry, optional): The target geometry for the VIIRS image.
                                            If None, the native VIIRS geometry is used.

    Returns:
        Raster: A Raster object representing the coarse-resolution albedo image.
                Zero values are converted to NaN.
    """
    coarse_image = VIIRS_connection.albedo(date_UTC=date_UTC, geometry=geometry)
    # Convert zero values to NaN for consistency
    coarse_image = rt.where(coarse_image == 0, np.nan, coarse_image)
    return coarse_image
