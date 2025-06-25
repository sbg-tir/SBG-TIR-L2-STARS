from typing import Union
from datetime import date
import numpy as np
import rasters as rt
from rasters import Raster

from harmonized_landsat_sentinel import HLS

def generate_NDVI_fine_image(
        date_UTC: Union[date, str], 
        tile: str, 
        HLS_connection: HLS) -> Raster:
    """
    Generates a fine-resolution NDVI image from HLS data.

    Args:
        date_UTC (Union[date, str]): The UTC date for which to retrieve NDVI data.
        tile (str): The HLS tile ID.
        HLS_connection (HLS): An initialized HLS data connection object.

    Returns:
        Raster: A Raster object representing the fine-resolution NDVI image.
                Zero values are converted to NaN.
    """
    fine_image = HLS_connection.NDVI(tile=tile, date_UTC=date_UTC)
    # Convert zero values to NaN for consistency
    fine_image = rt.where(fine_image == 0, np.nan, fine_image)
    return fine_image
