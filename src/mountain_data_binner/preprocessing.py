import logging
import os

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio.enums import Resampling

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


def slope_map_gdal(input_file: str, output_file: str) -> xr.DataArray:
    os.system(f"gdaldem slope -alg ZevenbergenThorne {input_file} {output_file}")


def aspect_map_gdal(input_file: str, output_file: str) -> xr.DataArray:
    os.system(f"gdaldem aspect -alg ZevenbergenThorne {input_file} {output_file}")


def preprocess_topography(input_dem_filepath: str, distributed_data_filepath: str, output_folder: str):
    logger.info("Opening distributed dataset in target geometry")
    distributed_data = xr.open_dataset(distributed_data_filepath, engine="rasterio")
    logger.info("Opening DEM data")
    input_dem = xr.open_dataset(input_dem_filepath, engine="rasterio")
    logger.info("Resampling DEM to output grid")
    output_dem_filepath = f"{output_folder}/dem.tif"
    resampled_dem = input_dem.rio.reproject_match(distributed_data)

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


# def preprocess(input_dem_filepath: str, forest_mask_filepath: str, distributed_data_filepath: str, output_folder: str):
#     logger.info("Opening distributed dataset un target geometry")
#     distributed = xr.open_dataset(distributed_data_filepath)
#     logger.info("Opening DEM data")
#     input_dem_data = xr.open_dataset(input_dem_dilepath)
#     preprocess_topography(input_dem=input_dem_data)
#     forest_mask =


# output_grid = GSGrid(x=, y0=,resolution=)
# output_grid = ""
# if __name__ == "__main__":
#     input_dem_dilepath = ""
#     forest_mask__filepath = ""

#     logger.info("Opening forest mask")
#     forest_mask = xr.open_dataset(forest_mask__filepath)
#     logger.info("Reprojecting forest mask to the output grid")
#     forest_mask_resampled = reproject_using_grid(data=forest_mask, output_grid=output_grid)
#     forest_mask_resampled = reproject_using_grid(data=forest_mask, output_grid=output_grid)
