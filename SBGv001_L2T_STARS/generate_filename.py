from typing import Union
from os import makedirs
from os.path import join, dirname, abspath
from datetime import date

from dateutil import parser

def generate_filename(
    directory: str, variable: str, date_UTC: Union[date, str], tile: str, cell_size: int
) -> str:
    """
    Generates a standardized filename for a raster product and ensures its directory exists.

    The filename format is STARS_{variable}_{YYYY-MM-DD}_{tile}_{cell_size}m.tif.

    Args:
        directory (str): The base directory where the file will be saved.
        variable (str): The name of the variable (e.g., "NDVI", "albedo").
        date_UTC (Union[date, str]): The UTC date of the data. Can be a date object or a string.
        tile (str): The HLS tile ID (e.g., 'H09V05').
        cell_size (int): The spatial resolution in meters (e.g., 70, 490, 980).

    Returns:
        str: The full, standardized path to the generated filename.
    """
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    variable = str(variable)
    timestamp = date_UTC.strftime("%Y-%m-%d")
    tile = str(tile)
    cell_size = int(cell_size)
    filename = join(directory, f"STARS_{variable}_{timestamp}_{tile}_{cell_size}m.tif")
    # Ensure the directory structure for the file exists
    makedirs(dirname(filename), exist_ok=True)

    return filename
