from dataclasses import dataclass
from typing import Dict

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from mountain_variability_drivers.preprocessing import preprocess_topography


class SemidistributedError(Exception):
    pass


@dataclass
class SemidistributedConfig:
    slope_map_path: str | None = None
    aspect_map_path: str | None = None
    dem_path: str | None = None


class Semidistributed:
    def __init__(self, config: SemidistributedConfig):
        self.config = config

    @staticmethod
    def user_bins(user_class_edges: np.array) -> BinGrouper:
        return BinGrouper(user_class_edges, user_class_edges[:-1])

    def aspect_map_transform(aspect_map: xr.DataArray) -> xr.DataArray:
        """
        Aspect map in degrees azimuth

        Transform the aspect map so that its values are monotonically incresing from N to NW,
        without dividing the North in two bins (NNW [337.5-360] and NNE [0-315])
        This is convenient for BinGrouper object

        """
        # Transform the aspect map so that its values are monotonically incresing from N to NW,
        # without dividing the North in two bins (NNW [337.5-360] and NNE [0-315])
        # This is convenient for BinGrouper object

        aspect_map = aspect_map.where(aspect_map < 360 - 22.5, aspect_map - 360)
        return aspect_map

    @staticmethod
    def regular_slope_bins(slope_sampling_step: int = 20) -> BinGrouper:
        bin_edges = np.arange(0, 90, slope_sampling_step)
        return BinGrouper(bin_edges, labels=bin_edges[:-1], include_lowest=True)

    @staticmethod
    def regular_elevation_bins(elevation_step: int = 600) -> BinGrouper:
        bin_edges = np.arange(0, 90, elevation_step)
        return BinGrouper(bin_edges, labels=bin_edges[:-1], include_lowest=True)

    @staticmethod
    def regular_aspect_bins() -> BinGrouper:
        return BinGrouper(
            np.arange(-45 / 2, 360, 45),
            labels=np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        )

    @staticmethod
    def create_default_bin_dict(slope_step: int = 20, elevation_step: int = 300):
        return dict(
            slope=Semidistributed.regular_slope_bins(slope_sampling_step=slope_step),
            aspect=Semidistributed.regular_aspect_bins(),
            elevation=Semidistributed.regular_elevation_bins(elevation_step=elevation_step),
        )

    @staticmethod
    def create_user_bin_dict(slope_edges: np.ndarray, elevation_edges: np.ndarray):
        return dict(
            slope=Semidistributed.user_bins(user_class_edges=slope_edges),
            aspect=Semidistributed.regular_aspect_bins(),
            elevation=Semidistributed.user_bins(user_class_edges=elevation_edges),
        )

    @classmethod
    def from_dem_filepath(cls, dem_filepath: str, distributed_data_filepath: str, output_folder: str):
        preprocess_topography(
            input_dem_filepath=dem_filepath, distributed_data_filepath=distributed_data_filepath, output_folder=output_folder
        )
        return cls(
            SemidistributedConfig(
                slope_map_path=f"{output_folder}/slope.nc",
                aspect_map_path=f"{output_folder}/aspect.nc",
                dem_path=f"{output_folder}/dem.nc",
            )
        )

    def stack_driver_data(self, distributed_data: xr.DataArray | xr.Dataset):
        if self.config.slope_map_path is not None:
            slope_map = xr.open_dataarray(self.config.slope_map_path)
            dataset = distributed_data.assign(slope=slope_map)

        if self.config.aspect_map_path is not None:
            aspect_map = xr.open_dataarray(self.config.aspect_map_path)
            aspect_map = self.aspect_map_transform(aspect_map)
            dataset = dataset.assign(aspect=aspect_map)

        if self.config.dem_path is not None:
            dem_map = xr.open_dataarray(self.config.dem_path)
            dataset = dataset.assign(altitude=dem_map)

        if "band" in dataset.dims:
            dataset = dataset.sel(band=1).drop_vars("band")

        return dataset

    def prepare(self, distributed_data: xr.DataArray | xr.Dataset, analysis_bin_dict: Dict[str, BinGrouper]) -> xr.Dataset:
        variable_and_auxiliary = self.stack_driver_data(distributed_data=distributed_data)
        return variable_and_auxiliary.groupby_bins(analysis_bin_dict)


# Semidistributed(config=config).prepare(data, analysis_bin_dict).map(my_fun)


# if config.forest_mask_path is not None:
#     forest_mask = xr.open_dataarray(config.forest_mask_path)
#     dataset = dataset.assign(forest_mask=forest_mask.sel(band=1).drop_vars("band"))
#     analysis_bin_dict.update(forest_mask=self.forest_bins())

#     @staticmethod
#     def forest_bins() -> BinGrouper:
#         return BinGrouper(np.array([-1, 0, 1]), labels=["no_forest", "forest"], right=True)

#     @staticmethod
#     def sub_roi_bins() -> BinGrouper:
#         return BinGrouper(
#             np.array([0, 1, 2, 3, 4, 5, 6]), labels=["Alps", "Pyrenees", "Corse", "Massif Central", "Jura", "Vosges"]
#         )
