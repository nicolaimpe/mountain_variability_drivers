import logging

import rioxarray
import xarray as xr
import xdem
from geospatial_grid.gsgrid import GSGrid
from geospatial_grid.reprojections import reproject_using_grid
from rasterio.enums import Resampling

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


def preprocess_topography(input_dem_filepath: str, distributed_data_filepath: str, output_folder: str):
    logger.info("Opening distributed dataset un target geometry")
    distributed_data = xr.open_dataset(distributed_data_filepath)
    logger.info("Opening DEM data")
    input_dem = xr.open_dataset(input_dem_filepath)
    logger.info("Resampling DEM to output grid")
    resampled_dem = input_dem.rio.reproject_match(distributed_data)
    resampled_dem.to_netcdf(f"{output_folder}/dem.nc")
    dem = xdem.DEM(filename_or_dataset=resampled_dem)
    logger.info(f"Exporting to {output_folder}/dem.nc")
    logger.info("Generating slope map")
    slope_map = xdem.terrain.slope(dem=dem, method="ZevenbergThorne", degrees=True)
    slope_map.to_netcdf(f"{output_folder}/slope_map.nc")
    logger.info("Generating aspect map")
    logger.info(f"Exporting to {output_folder}/slope.nc")
    aspect_map = xdem.terrain.aspect(dem=dem, method="ZevenbergThorne", degrees=True)
    aspect_map.to_netcdf(f"{output_folder}/aspect_map.nc")
    logger.info(f"Exporting to {output_folder}/aspect.nc")
    distributed_data.close()


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
