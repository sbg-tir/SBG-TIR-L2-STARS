
from .generate_input_staging_directory import generate_input_staging_directory

def generate_albedo_fine_directory(input_staging_directory: str, tile: str) -> str:
    """
    Generates the specific staging directory for fine albedo images.

    Args:
        input_staging_directory (str): The base input staging directory.
        tile (str): The HLS tile ID.

    Returns:
        str: The full path to the fine albedo staging directory.
    """
    return generate_input_staging_directory(
        input_staging_directory, tile, "albedo_fine"
    )
