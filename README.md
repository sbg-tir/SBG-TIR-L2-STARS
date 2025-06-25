# SBG-TIR OTTER LST STARS NDVI & Albedo Product

This is the main repository for the Suface Biology and Geology Thermal Infrared (SBG-TIR) STARS NDVI and albedo data product. This product will utilize the [Spatial Timeseries for Automated high-Resolution multi-Sensor (STARS)](https://github.com/STARS-Data-Fusion) data fusion system to produce normalized difference vegetation index (NDVI) and albedo estimates corresponding to SBG-TIR OTTER surface temperature measurements, to support the [evapotranspiration product](https://github.com/sbg-tir/SBG-TIR-L3-ET).

The SBG collection 1 level 2 vegetation index and albedo data product is being developed based on the [ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (SBG) collection 3 level 2 vegetation index and albedo data product](https://github.com/SBG-Collection-3/SBGv001-L2-STARS).

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

[Margaret C. Johnson](https://github.com/majohnso) (she/her)<br>
[maggie.johnson@jpl.nasa.gov](mailto:maggie.johnson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 398L

[Kerry Cawse-Nicholson](https://github.com/kcawse) (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

[Claire Villanueva-Weeks](https://github.com/clairesvw) (she/her)<br>
[claire.s.villanueva-weeks@jpl.nasa.gov](mailto:claire.s.villanueva-weeks@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

The code for the SBG level 2 STARS PGE will be developed using open-science practices based on the [SBG collection 2 gridded and tiled product generation software](https://github.com/SBG-Collection-2/SBG-Collection-2).

This software will produce estimates of:
- Normalized Difference Vegetation Index (NDVI)
- albedo

NDVI and albedo are estimated at 60 m SBG standard resolution for each daytime SBG overpass by fusing temporally sparse but fine spatial resolution images from the Harmonized Landsat Sentinel (HLS) 2.0 product with daily, moderate spatial resolution images from the Suomi NPP Visible Infrared Imaging Radiometer Suite (VIIRS) VNP09GA product. The data fusion is performed using a variant of the Spatial Timeseries for Automated high-Resolution multi-Sensor data fusion (STARS) algorithm developed by Dr. Margaret Johnson and Gregory Halverson at the Jet Propulsion Laboratory. STARS is a Bayesian timeseries methodology that provides streaming data fusion and uncertainty quantification through efficient Kalman filtering.

Operationally, each L2T STARS tile run loads the means and covariances of the STARS model saved from the most recent tile run, then iteratively advances the means and covariances forward each day updating with fine imagery from HLS and/or moderate resolution imagery from VIIRS up to the day of the target SBG overpass. A pixelwise, lagged 16-day implementation of the VNP43 algorithm (Schaaf, 2017) is used for a near-real-time BRDF correction on the VNP09GA products to produce VIIRS NDVI and albedo.

## 1. Introduction to Data Products

The data format for the SBG products is described in the [SBG-TIR OTTER landing page](https://github.com/sbg-tir).

## 2. L2T STARS NDVI and Albedo Product

```mermaid
flowchart TB
    subgraph VIREO[SBG-TIR VIREO]
        VIREO_NDVI(SBG-TIR<br>VIREO<br>30m<br>NDVI)
        VIREO_upsampled[Upsampled<br>VIREO<br>60m<br>NDVI]
    end

    subgraph VNP43NRT[VNP43NRT.jl]
        VNP09GA_I[VNP09GA<br>I-Band<br>500m<br>Surface<br>Reflectance]
        VNP09GA_M[VNP09GA<br>M-Band<br>1000m<br>Surface<br>Reflectance]
        VIIRS_downscaling[VIIRS<br>Downscaling]
        VNP09GA_downscaled[Downscaled<br>500m<br>VIIRS<br>Surface<br>Reflectance]
        VNP43_BRDF[VNP43NRT.jl<br>BRDF<br>Correction]
        VIIRS_corrected[VIIRS<br>BRDF-Corrected<br>500m<br>Surface<br>Reflectance]
        VIIRS_NDVI[VIIRS<br>500m<br>NDVI]
        VIIRS_albedo[VIIRS<br>500m<br>Albedo]
    end

    subgraph HLS_aquisition[HLS.jl]
        direction TB
        Landsat_reflectance[HLS<br>Landsat<br>30m<br>Surface<br>Reflectance]
        Landsat_upsampled[Upsampled<br>Landsat<br>60m<br>Surface<br>Reflectance]
        Landsat_NDVI[Landsat<br>60m<br>NDVI]
        Sentinel_reflectance[HLS<br>Sentinel<br>30m<br>Surface<br>Reflectance]
        Sentinel_upsampled[Upsampled<br>Sentinel<br>60m<br>Surface<br>Reflectance]
        Sentinel_NDVI[Sentinel<br>60m<br>NDVI]
        Landsat_albedo[Landsat<br>60m<br>Albedo]
        Sentinel_albedo[Sentinel<br>60m<br>Albedo]
    end

    subgraph bayesian_state[Bayesian State]
        NDVI_covariance_prior[NDVI<br>Fine-Coarse<br>Covariance<br>Prior<br>from<br>Previous<br>Overpass]
        NDVI_covariance_posterior[NDVI<br>Fine-Coarse<br>Covariance<br>Posterior<br>for<br>Next<br>Overpass]
        albedo_covariance_prior[Albedo<br>Fine-Coarse<br>Covariance<br>Prior<br>from<br>Previous<br>Overpass]
        albedo_covariance_posterior[Albedo<br>Fine-Coarse<br>Covariance<br>Posterior<br>for<br>Next<br>Overpass]
    end

    fine_NDVI_input[NDVI<br>60m<br>Composite]
    NDVI_data_fusion[STARS.jl<br>NDVI<br>Data<br>Fusion]
    fine_NDVI_output[Fused<br>30m<br>NDVI]
    fine_NDVI_uncertainty[NDVI<br>Uncertainty]

    fine_albedo_input[Albedo<br>60m<br>Composite]
    albedo_data_fusion[STARS.jl<br>Albedo<br>Data<br>Fusion]
    fine_albedo_output[Fused<br>30m<br>Albedo]
    fine_albedo_uncertainty[Albedo<br>Uncertainty]

    SBG_L2T_STARS(SBG-TIR<br>OTTER<br>L2T<br>STARS<br>NDVI<br>&<br>Albedo<br>Product)

    VNP09GA_I --> VIIRS_downscaling
    VNP09GA_M --> VIIRS_downscaling
    VIIRS_downscaling --> VNP09GA_downscaled
    VNP09GA_downscaled --> VNP43_BRDF
    VNP43_BRDF --> VIIRS_corrected
    VIIRS_corrected --> VIIRS_NDVI
    VIIRS_corrected --> VIIRS_albedo

    VIREO_NDVI --> VIREO_upsampled
    Landsat_reflectance --> Landsat_upsampled
    Sentinel_reflectance --> Sentinel_upsampled

    Landsat_upsampled --> Landsat_NDVI
    Sentinel_upsampled --> Sentinel_NDVI

    Landsat_upsampled --> Landsat_albedo
    Sentinel_upsampled --> Sentinel_albedo

    VIREO_upsampled --> fine_NDVI_input
    Landsat_NDVI --> fine_NDVI_input
    Sentinel_NDVI --> fine_NDVI_input
    fine_NDVI_input --> NDVI_data_fusion
    VIIRS_NDVI --> NDVI_data_fusion
    NDVI_covariance_prior --> NDVI_data_fusion
    NDVI_data_fusion --> fine_NDVI_output
    NDVI_data_fusion --> fine_NDVI_uncertainty
    NDVI_data_fusion --> NDVI_covariance_posterior

    Landsat_albedo --> fine_albedo_input
    Sentinel_albedo --> fine_albedo_input
    fine_albedo_input --> albedo_data_fusion
    VIIRS_albedo --> albedo_data_fusion
    albedo_covariance_prior --> albedo_data_fusion
    albedo_data_fusion --> fine_albedo_output
    albedo_data_fusion --> fine_albedo_uncertainty
    albedo_data_fusion --> albedo_covariance_posterior

    fine_NDVI_output --> SBG_L2T_STARS
    fine_NDVI_uncertainty --> SBG_L2T_STARS
    fine_albedo_output --> SBG_L2T_STARS
    fine_albedo_uncertainty --> SBG_L2T_STARS

    click VNP43_BRDF "https://github.com/STARS-Data-Fusion/VNP43NRT.jl"
    click NDVI_data_fusion "https://github.com/STARS-Data-Fusion/STARS.jl"
    click albedo_data_fusion "https://github.com/STARS-Data-Fusion/STARS.jl"
```

*Figure 1. Flowchart of the SBG-TIR L2T STARS processing workflow.*

NDVI and albedo are estimated at 60 m SBG standard resolution with uncertainty for each UTC day in which there is an SBG overpass by fusing temporally sparse but fine spatial resolution images from the Harmonized Landsat Sentinel (HLS) 2.0 product with daily, moderate spatial resolution images from the Suomi NPP Visible Infrared Imaging Radiometer Suite (VIIRS) VNP09GA product.

Landsat and Sentinel surface reflectances are collected using the [HLS.jl](https://github.com/STARS-Data-Fusion/HLS.jl) package.

VIIRS surface reflectance is downscaled and BRDF corrected using the [VNP43NRT.jl](https://github.com/STARS-Data-Fusion/VNP43NRT.jl) package. A pixelwise, lagged 16-day implementation of the VNP43 algorithm (Schaaf, 2017) is used for a near-real-time BRDF correction on the VNP09GA products to produce VIIRS NDVI and albedo.

The data fusion is performed with a variant of the Spatial Timeseries for Automated high-Resolution multi-Sensor data fusion (STARS) algorithm developed by Dr. Margaret Johnson and Gregory H. Halverson at the Jet Propulsion Laboratory using the [STARS.jl](https://github.com/STARS-Data-Fusion/STARS.jl) package. STARS is a Bayesian timeseries methodology that provides streaming data fusion and uncertainty quantification through efficient Kalman filtering. Operationally, each L2T STARS tile run loads the means and covariances of the STARS model saved from the most recent tile run, then iteratively advances the means and covariances forward each day updating with fine imagery from HLS and/or moderate resolution imagery from VIIRS up to the day of the target SBG overpass. 

The layers of the L2T STARS product are listed in Table 2. All layers of this product are represented by 32-bit floating point arrays. The NDVI estimates and 1σ uncertainties (-UQ) are unitless from -1 to 1. The albedo estimates and 1σ uncertainties (-UQ) are proportions from 0 to 1. 

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** |**Scale Factor** | **Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| NDVI | Normalized Difference Vegetation Index | float32 | Index | NaN | N/A | -1 | 1 | N/A | 13.4 mb |
| NDVI-UQ | Normalized Difference Vegetation Index Uncertainty | float32 | Index | NaN | N/A | -1 | 1 | N/A | 13.4 mb |
| albedo | Albedo | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 13.4 mb |
| albedo-UQ | Albedo Uncertainty | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 13.4 mb |

*Table 2. Listing of L2T STARS data layers.*

## References

Schaaf, C. B. et al. (2017). *Algorithm Theoretical Basis Document for MODIS Bidirectional Reflectance Distribution Function and Albedo (MOD43) Products*. NASA. [Link to source](https://lpdaac.usgs.gov/documents/110/MOD43_ATBD.pdf)
