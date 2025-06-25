from os import makedirs
from os.path import join

def generate_input_staging_directory(
        input_staging_directory: str, 
        tile: str, 
        prefix: str) -> str:
    """
    Generates a path for an input staging directory and ensures it exists.

    This is used to organize temporary input files for the Julia processing.

    Args:
        input_staging_directory (str): The base input staging directory.
        tile (str): The HLS tile ID.
        prefix (str): A prefix for the sub-directory name (e.g., "NDVI_coarse").

    Returns:
        str: The full path to the created or existing staging directory.
    """
    directory = join(input_staging_directory, f"{prefix}_{tile}")
    makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    return directory
