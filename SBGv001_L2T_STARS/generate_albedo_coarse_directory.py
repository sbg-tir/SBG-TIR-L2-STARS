
from .generate_input_staging_directory import generate_input_staging_directory


def generate_albedo_coarse_directory(input_staging_directory: str, tile: str) -> str:
    """
    Generates the specific staging directory for coarse albedo images.

    Args:
        input_staging_directory (str): The base input staging directory.
        tile (str): The HLS tile ID.

    Returns:
        str: The full path to the coarse albedo staging directory.
    """
    return generate_input_staging_directory(
        input_staging_directory, tile, "albedo_coarse"
    )
