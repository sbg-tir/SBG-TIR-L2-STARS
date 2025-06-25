from typing import Union
from datetime import date
from os import makedirs
from os.path import join

from dateutil import parser

def generate_model_state_tile_date_directory(
    model_directory: str, tile: str, date_UTC: Union[date, str]
) -> str:
    """
    Generates a directory for storing model state files (e.g., priors, posteriors)
    organized by tile and date, and ensures it exists.

    Args:
        model_directory (str): The base directory for model state files.
        tile (str): The HLS tile ID.
        date_UTC (Union[date, str]): The UTC date for the model state.

    Returns:
        str: The full path to the created or existing model state directory.
    """
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    directory = join(model_directory, tile, f"{date_UTC:%Y-%m-%d}")
    makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    return directory
