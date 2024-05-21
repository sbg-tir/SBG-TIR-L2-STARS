# SBG-TIR OTTER STARS Data Fusion System

Gregory H. Halverson (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

Margaret C. Johnson (she/her)<br>
[maggie.johnson@jpl.nasa.gov](mailto:maggie.johnson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 398L

Kerry Cawse-Nicholson (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

This is the main repository for the Suface Biology and Geology Thermal Infrared (SBG-TIR) STARS NDVI and albedo data product. This product will utilize the [Spatial Timeseries for Automated high-Resolution multi-Sensor (STARS)](https://github.com/STARS-Data-Fusion) data fusion system to produce normalized difference vegetation index (NDVI) and albedo estimates corresponding to SBG-TIR OTTER surface temperature measurements, to support the [evapotranspiration product](https://github.com/sbg-tir/SBG-TIR-L3-ET).

NDVI and albedo are estimated at 60 m SBG standard resolution for each daytime SBG overpass by fusing temporally sparse but fine spatial resolution images from the Harmonized Landsat Sentinel (HLS) 2.0 product with daily, moderate spatial resolution images from the Suomi NPP Visible Infrared Imaging Radiometer Suite (VIIRS) VNP09GA product. The data fusion is performed using a variant of the Spatial Timeseries for Automated high-Resolution multi-Sensor data fusion (STARS) algorithm developed by Dr. Margaret Johnson and Gregory Halverson at the Jet Propulsion Laboratory. STARS is a Bayesian timeseries methodology that provides streaming data fusion and uncertainty quantification through efficient Kalman filtering.

Operationally, each L2T STARS tile run loads the means and covariances of the STARS model saved from the most recent tile run, then iteratively advances the means and covariances forward each day updating with fine imagery from HLS and/or moderate resolution imagery from VIIRS up to the day of the target SBG overpass. A pixelwise, lagged 16-day implementation of the VNP43 algorithm (Schaaf, 2017) is used for a near-real-time BRDF correction on the VNP09GA products to produce VIIRS NDVI and albedo.

The layers of the L2T STARS product are listed in Table 4. All layers of this product are represented by 32-bit floating point arrays. The NDVI estimates and 1$$\sigma$$ uncertainties (-UQ) are unitless from -1 to 1. The albedo estimates and 1$$\sigma$$ uncertainties (-UQ) are proportions from 0 to 1.

L2T STARS Data Layers
- Normalized Difference Vegetation Index [-1, 1] (NDVI)
- NDVI Uncertainty [-1, 1] (NDVI-UQ)
- albedo [0, 1] (albedo)
- albedo uncertainty [0, 1] (albedo-UQ)
