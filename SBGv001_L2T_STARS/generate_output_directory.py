from typing import Union
from datetime import date
from os import makedirs
from os.path import join

from dateutil import parser

def generate_output_directory(
        working_directory: str, 
        date_UTC: Union[date, str], 
        tile: str) -> str:
    """
    Generates a dated output directory for Julia model results and ensures it exists.

    Args:
        working_directory (str): The main working directory.
        date_UTC (Union[date, str]): The UTC date for the output.
        tile (str): The HLS tile ID.

    Returns:
        str: The full path to the created or existing output directory.
    """
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    directory = join(working_directory, f"julia_output_{date_UTC:%y.%m.%d}_{tile}")
    makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    return directory
