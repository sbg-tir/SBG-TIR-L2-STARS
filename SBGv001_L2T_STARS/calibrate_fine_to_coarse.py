import logging

import numpy as np
from scipy import stats

from rasters import Raster

logger = logging.getLogger(__name__)

def calibrate_fine_to_coarse(fine_image: Raster, coarse_image: Raster) -> Raster:
    """
    Calibrates a fine-resolution raster image to a coarse-resolution raster image
    using linear regression.

    This function aggregates the fine image to the geometry of the coarse image,
    then performs a linear regression between the aggregated fine image and the
    original coarse image. The derived slope and intercept are then applied to
    the original fine image for calibration.

    Args:
        fine_image (Raster): The higher-resolution raster image to be calibrated.
        coarse_image (Raster): The lower-resolution raster image used as the reference
                                for calibration.

    Returns:
        Raster: The calibrated fine-resolution raster image. If too few valid
                data points are available for regression (less than 30), the
                original fine_image is returned.
    """
    # Aggregate the fine image to the coarse image's geometry for comparison
    aggregated_image = fine_image.to_geometry(coarse_image.geometry, resampling="average")
    x = np.array(aggregated_image).flatten()  # Independent variable (aggregated fine)
    y = np.array(coarse_image).flatten()  # Dependent variable (coarse)

    # Create a mask to remove NaN values from both arrays, ensuring valid data points for regression
    mask = ~np.isnan(x) & ~np.isnan(y)

    # Check if there are enough valid data points for a meaningful linear regression
    if np.count_nonzero(mask) < 30:
        logger.warning(
            f"Insufficient valid data points ({np.count_nonzero(mask)}) for calibration. "
            "Returning original fine image."
        )
        return fine_image

    # Apply the mask to get only valid data points
    x = x[mask]
    y = y[mask]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    logger.info(
        f"Linear regression for calibration: slope={slope:.4f}, intercept={intercept:.4f}, "
        f"R-squared={r_value**2:.4f}"
    )

    # Apply the derived calibration to the original fine image
    calibrated_image = fine_image * slope + intercept

    return calibrated_image
