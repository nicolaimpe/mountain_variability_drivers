import logging
import os
from typing import Tuple

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio.enums import Resampling

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


def slope_map_gdal(input_file: str, output_file: str) -> xr.DataArray:
    """Wrap-up GDAL command to generate a slope map. See scripts/process_mnt.sh for the GDAL routine."""
    os.system(f"gdaldem slope -alg ZevenbergenThorne {input_file} {output_file}")


def aspect_map_gdal(input_file: str, output_file: str) -> xr.DataArray:
    """Wrap-up GDAL command to generate an aspect map. See scripts/process_mnt.sh for the GDAL routine."""
    os.system(f"gdaldem aspect -alg ZevenbergenThorne {input_file} {output_file}")


def preprocess_topography(input_dem_filepath: str, distributed_data_filepath: str, output_folder: str) -> Tuple[str, str, str]:
    """Create auxiliary data for topography based data binning of a spatial dataset using GDAL.

    1. Regrid DEM on the spatial data grid
    2. Generate slope and aspect map from the regridded DEM

    Args:
        input_dem_filepath (str): Path to a Digital Elevation Model in any projection covering your area of interest.
        distributed_data_filepath (str): Path to some geospatial dataset.
        output_folder (str): path to a folder that stores the regridded DEM, the slope and aspect map.

    Returns:
        Tuple[str, str, str]: the filepaths to regridded DEM, slope and aspect maps
    """
    logger.info("Opening distributed dataset in target geometry")
    distributed_data = xr.open_dataset(distributed_data_filepath, engine="rasterio")
    logger.info("Opening DEM data")
    input_dem = xr.open_dataset(input_dem_filepath, engine="rasterio")
    logger.info("Resampling DEM to output grid")
    output_dem_filepath = f"{output_folder}/dem.tif"
    resampled_dem = input_dem.rio.reproject_match(distributed_data, resampling=Resampling.lanczos)

    # Need to save this using rasterio to export a GeoTiff as GDAL would do in order to be consistent with slope and aspect maps
    with rasterio.open(
        output_dem_filepath,
        "w",
        width=resampled_dem.rio.width,
        height=resampled_dem.rio.height,
        count=1,
        dtype=np.float32,
        nodata=-9999,
        transform=resampled_dem.rio.transform(),
        crs=resampled_dem.rio.crs,
    ) as dst:
        dst.write(resampled_dem.data_vars["band_data"].values)

    logger.info(f"Exporting to {output_dem_filepath}")
    logger.info("Generating slope map")
    output_slope_filepath = f"{output_folder}/slope.tif"
    slope_map_gdal(input_file=output_dem_filepath, output_file=output_slope_filepath)
    logger.info(f"Exported to {output_slope_filepath}")
    output_aspect_filepath = f"{output_folder}/aspect.tif"
    aspect_map_gdal(input_file=output_dem_filepath, output_file=output_aspect_filepath)
    logger.info(f"Exported to {output_aspect_filepath}")
    distributed_data.close()
    return output_dem_filepath, output_slope_filepath, output_aspect_filepath


def preprocess(
    input_dem_filepath: str, forest_mask_filepath: str, distributed_data_filepath: str, output_folder: str
) -> Tuple[str, str, str, str]:
    """Create auxiliary data for topography and landcover data binning of a spatial dataset using GDAL.

    1. Use preprocess_topography to generate a regridded DEM on target grid, a slope and an aspect map
    2. Regrid a forest mask (or any landcover map) to target grid

    The target grid and projections correspond to the geospatial (distributed) dataset used.

    Args:
        input_dem_filepath (str): path to to a Digital Elevation Model in any projection covering your area of interest
        forest_mask_filepath (str): path to a forest mask (or any landcover dataset) in any projection covering your area of interest
        distributed_data_filepath (str): path to some geospatial dataset.
        output_folder (str): path to a folder that stores the regridded DEM, the slope and aspect map.

    Returns:
        Tuple[str, str, str, str]: the filepaths to regridded DEM, slope map, aspect map and regridded forest mask (or landcover map)
    """
    output_dem_filepath, output_slope_filepath, output_aspect_filepath = preprocess_topography(
        input_dem_filepath=input_dem_filepath, distributed_data_filepath=distributed_data_filepath, output_folder=output_folder
    )
    logger.info("Opening distributed dataset un target geometry")
    distributed_data = xr.open_dataset(distributed_data_filepath)
    logger.info("Opening forest mask data")
    input_forest_mask_data = xr.open_dataset(forest_mask_filepath)
    resampled_forest_mask = input_forest_mask_data.rio.reproject_match(distributed_data, resampling=Resampling.lanczos)
    output_forest_mask_filepath = f"{output_folder}/forest_mask.tif"
    # Need to save this using rasterio to export a GeoTiff as GDAL would do in order to be consistent with slope and aspect maps
    with rasterio.open(
        output_forest_mask_filepath,
        "w",
        width=resampled_forest_mask.rio.width,
        height=resampled_forest_mask.rio.height,
        count=1,
        dtype=np.float32,
        nodata=-9999,
        transform=resampled_forest_mask.rio.transform(),
        crs=resampled_forest_mask.rio.crs,
    ) as dst:
        dst.write(resampled_forest_mask.data_vars["band_data"].values)
    return output_dem_filepath, output_slope_filepath, output_aspect_filepath, output_forest_mask_filepath
