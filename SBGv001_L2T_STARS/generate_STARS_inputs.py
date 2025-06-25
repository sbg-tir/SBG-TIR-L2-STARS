from typing import Union
from datetime import date, datetime
from dateutil.rrule import rrule, DAILY
import logging

import colored_logging as cl
from rasters import Raster, RasterGeometry
from harmonized_landsat_sentinel import HLS2CMR

from SBGv001_exit_codes import AuxiliaryLatency

from .constants import VIIRS_GIVEUP_DAYS
from .generate_filename import generate_filename
from .daterange import get_date
from .generate_NDVI_coarse_image import generate_NDVI_coarse_image
from .generate_NDVI_fine_image import generate_NDVI_fine_image
from .generate_albedo_coarse_image import generate_albedo_coarse_image
from .generate_albedo_fine_image import generate_albedo_fine_image
from .calibrate_fine_to_coarse import calibrate_fine_to_coarse
from .VIIRS.VIIRSDownloader import VIIRSDownloaderAlbedo, VIIRSDownloaderNDVI

logger = logging.getLogger(__name__)

def generate_STARS_inputs(
    tile: str,
    date_UTC: date,
    HLS_start_date: date,
    HLS_end_date: date,
    VIIRS_start_date: date,
    VIIRS_end_date: date,
    NDVI_resolution: int,
    albedo_resolution: int,
    target_resolution: int,
    NDVI_coarse_geometry: RasterGeometry,
    albedo_coarse_geometry: RasterGeometry,
    working_directory: str,
    NDVI_coarse_directory: str,
    NDVI_fine_directory: str,
    albedo_coarse_directory: str,
    albedo_fine_directory: str,
    HLS_connection: HLS2CMR,
    NDVI_VIIRS_connection: VIIRSDownloaderNDVI,
    albedo_VIIRS_connection: VIIRSDownloaderAlbedo,
    calibrate_fine: bool = False,
):
    """
    Generates and stages the necessary coarse and fine resolution input images
    for the STARS data fusion process.

    This function iterates through the VIIRS date range, retrieving and saving
    coarse NDVI and albedo images. For dates within the HLS range, it also
    retrieves and saves fine NDVI and albedo images. It can optionally
    calibrate the fine images to the coarse images.

    Args:
        tile (str): The HLS tile ID.
        date_UTC (date): The target UTC date for the L2T_STARS product.
        HLS_start_date (date): The start date for HLS data retrieval for the fusion period.
        HLS_end_date (date): The end date for HLS data retrieval for the fusion period.
        VIIRS_start_date (date): The start date for VIIRS data retrieval for the fusion period.
        VIIRS_end_date (date): The end date for VIIRS data retrieval for the fusion period.
        NDVI_resolution (int): The resolution of the coarse NDVI data.
        albedo_resolution (int): The resolution of the coarse albedo data.
        target_resolution (int): The desired output resolution of the fused product.
        NDVI_coarse_geometry (RasterGeometry): The target geometry for coarse NDVI images.
        albedo_coarse_geometry (RasterGeometry): The target geometry for coarse albedo images.
        working_directory (str): The main working directory.
        NDVI_coarse_directory (str): Directory for staging coarse NDVI images.
        NDVI_fine_directory (str): Directory for staging fine NDVI images.
        albedo_coarse_directory (str): Directory for staging coarse albedo images.
        albedo_fine_directory (str): Directory for staging fine albedo images.
        HLS_connection (HLS2CMR): An initialized HLS data connection object.
        NDVI_VIIRS_connection (VIIRSDownloaderNDVI): An initialized VIIRS NDVI downloader.
        albedo_VIIRS_connection (VIIRSDownloaderAlbedo): An initialized VIIRS albedo downloader.
        calibrate_fine (bool, optional): If True, calibrate fine images to coarse images.
                                         Defaults to False.

    Raises:
        AuxiliaryLatency: If coarse VIIRS data is missing within the VIIRS_GIVEUP_DAYS window.
    """
    missing_coarse_dates = set()  # Track dates where coarse data could not be generated

    # Process each day within the VIIRS data fusion window
    for processing_date in [
        get_date(dt) for dt in rrule(DAILY, dtstart=VIIRS_start_date, until=VIIRS_end_date)
    ]:
        logger.info(
            f"Preparing coarse image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}"
        )

        try:
            # Generate coarse NDVI image
            NDVI_coarse_image = generate_NDVI_coarse_image(
                date_UTC=processing_date,
                VIIRS_connection=NDVI_VIIRS_connection,
                geometry=NDVI_coarse_geometry,
            )

            # Define filename for coarse NDVI and save
            NDVI_coarse_filename = generate_filename(
                directory=NDVI_coarse_directory,
                variable="NDVI",
                date_UTC=processing_date,
                tile=tile,
                cell_size=NDVI_resolution,
            )
            logger.info(
                f"Saving coarse image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}: {NDVI_coarse_filename}"
            )
            NDVI_coarse_image.to_geotiff(NDVI_coarse_filename)

            # If the processing date is within the HLS range, generate fine NDVI
            if processing_date >= HLS_start_date:
                logger.info(
                    f"Preparing fine image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}"
                )
                try:
                    NDVI_fine_image = generate_NDVI_fine_image(
                        date_UTC=processing_date,
                        tile=tile,
                        HLS_connection=HLS_connection,
                    )

                    # Optionally calibrate the fine NDVI image to the coarse NDVI image
                    if calibrate_fine:
                        logger.info(
                            f"Calibrating fine image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}"
                        )
                        NDVI_fine_image = calibrate_fine_to_coarse(
                            NDVI_fine_image, NDVI_coarse_image
                        )

                    # Define filename for fine NDVI and save
                    NDVI_fine_filename = generate_filename(
                        directory=NDVI_fine_directory,
                        variable="NDVI",
                        date_UTC=processing_date,
                        tile=tile,
                        cell_size=target_resolution,
                    )
                    logger.info(
                        f"Saving fine image for STARS NDVI at {cl.place(tile)} on {cl.time(processing_date)}: {NDVI_fine_filename}"
                    )
                    NDVI_fine_image.to_geotiff(NDVI_fine_filename)
                except Exception:  # Catch any exception during HLS fine image generation
                    logger.info(f"HLS NDVI is not available on {processing_date}")
        except Exception as e:
            logger.exception(e)
            logger.warning(
                f"Unable to produce coarse NDVI for date {processing_date}"
            )
            missing_coarse_dates.add(processing_date)  # Add date to missing set

        logger.info(
            f"Preparing coarse image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}"
        )
        try:
            # Generate coarse albedo image
            albedo_coarse_image = generate_albedo_coarse_image(
                date_UTC=processing_date,
                VIIRS_connection=albedo_VIIRS_connection,
                geometry=albedo_coarse_geometry,
            )

            # Define filename for coarse albedo and save
            albedo_coarse_filename = generate_filename(
                directory=albedo_coarse_directory,
                variable="albedo",
                date_UTC=processing_date,
                tile=tile,
                cell_size=albedo_resolution,
            )
            logger.info(
                f"Saving coarse image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}: {albedo_coarse_filename}"
            )
            albedo_coarse_image.to_geotiff(albedo_coarse_filename)

            # If the processing date is within the HLS range, generate fine albedo
            if processing_date >= HLS_start_date:
                logger.info(
                    f"Preparing fine image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}"
                )
                try:
                    albedo_fine_image = generate_albedo_fine_image(
                        date_UTC=processing_date,
                        tile=tile,
                        HLS_connection=HLS_connection,
                    )

                    # Optionally calibrate the fine albedo image to the coarse albedo image
                    if calibrate_fine:
                        logger.info(
                            f"Calibrating fine image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}"
                        )
                        albedo_fine_image = calibrate_fine_to_coarse(
                            albedo_fine_image, albedo_coarse_image
                        )

                    # Define filename for fine albedo and save
                    albedo_fine_filename = generate_filename(
                        directory=albedo_fine_directory,
                        variable="albedo",
                        date_UTC=processing_date,
                        tile=tile,
                        cell_size=target_resolution,
                    )
                    logger.info(
                        f"Saving fine image for STARS albedo at {cl.place(tile)} on {cl.time(processing_date)}: {albedo_fine_filename}"
                    )
                    albedo_fine_image.to_geotiff(albedo_fine_filename)
                except Exception:  # Catch any exception during HLS fine image generation
                    logger.info(f"HLS albedo is not available on {processing_date}")
        except Exception as e:
            logger.exception(e)
            logger.warning(
                f"Unable to produce coarse albedo for date {processing_date}"
            )
            missing_coarse_dates.add(processing_date)  # Add date to missing set

    # Check for missing coarse dates within the give-up window
    coarse_latency_dates = [
        d
        for d in missing_coarse_dates
        if (datetime.utcnow().date() - d).days <= VIIRS_GIVEUP_DAYS
    ]

    if len(coarse_latency_dates) > 0:
        raise AuxiliaryLatency(
            f"Missing coarse dates within {VIIRS_GIVEUP_DAYS}-day window: "
            f"{', '.join([str(d) for d in sorted(list(coarse_latency_dates))])}"
        )
