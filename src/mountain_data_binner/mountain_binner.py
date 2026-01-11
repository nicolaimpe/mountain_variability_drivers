from dataclasses import dataclass
from typing import Callable, Dict, List, Self

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from mountain_data_binner.preprocessing import preprocess


class MountainBinnerError(Exception):
    pass


@dataclass
class MountainBinnerConfig:
    """Regroup user inputs. All datasets should be in the same geospatial grid and projection.

    Args:
        slope_map_path (str): path to the slope map. Optional.
        aspect_map_path (str): path to the aspect map. Optional.
        dem_path (str): path to the Digital Elevation Model. Optional.
        forest_mask_path (str): path to a forest mask (or a landcover map). Optional
        regular_8_aspects_bins: whether the aspect binning should be done in a 8-points compass rose (N, NE, E, SE, S, SW, W, NW).
                                Defaults to True.

    """

    slope_map_path: str | None = None
    aspect_map_path: str | None = None
    dem_path: str | None = None
    forest_mask_path: str | None = None
    regular_8_aspects: bool = True


class MountainBinner:
    """Use binning approach to characterize the variability of a geophysical variable in mountains.

    Recall: a bin is define by a sequence of edges.

            --------------- --------------- ------
        |               |               |
            bin_label_1     bin_label_2
    bin_edge_1       bin_edge_2

    """

    def __init__(self, config: MountainBinnerConfig):
        self.config = config

    @staticmethod
    def aspect_map_transform(aspect_map: xr.DataArray) -> xr.DataArray:
        """
        Transform the aspect map so that its values are monotonically incresing from N to NW.

        This allows to prevent dividing the North in two bins (NNW [337.5-360] and NNE [0-315]) and is convenient for BinGrouper object

        Args:
            aspect_map (xr.DataArray): The aspect map, i.e. slope azimuth in degrees

        """

        aspect_map = aspect_map.where(aspect_map < 360 - 22.5, aspect_map - 360)
        return aspect_map

    @staticmethod
    def user_bins(bin_edges: np.ndarray) -> BinGrouper:
        """Helper to create custom bins. Left bins by default: the left edge is included in the bin, the right excluded.

        Args:
            bin_edges (np.ndarray): Sequence of bin edges. It needs to be sorted from lower to higher but not necessarily
                                    uniformly spaced

        Returns:
            BinGrouper: a bin grouper object ot use coupled with xarray groupby function.
        """
        return BinGrouper(
            bins=bin_edges,
            labels=MountainBinner.create_labels_from_bin_edges(bin_edges),
            include_lowest=True,
            right=False,
        )

    @staticmethod
    def create_labels_from_bin_edges(bin_edges: np.ndarray) -> List[str]:
        """Define bin labels that explicit both bin edges '<bin_edge_left> - <bin_edge_right>'."""
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            bin_labels.append(f"{bin_edges[i]} - {bin_edges[i + 1]}")
        return bin_labels

    @staticmethod
    def default_slope_bands() -> BinGrouper:
        """
        Create the bins [0-10)°, [10-30)°, [30-50)° matching the S2M snowpack modelling system [1].

        [1] M. Vernay et al., « The S2M meteorological and snow cover reanalysis over the French mountainous areas:
        description and evaluation (1958–2021) », Earth System Science Data, vol. 14, nᵒ 4, p. 1707‑1733, avr. 2022,
        doi: 10.5194/essd-14-1707-2022.
        """
        bin_edges = np.array([0, 10, 30, 50])
        return BinGrouper(
            bin_edges, labels=MountainBinner.create_labels_from_bin_edges(bin_edges), include_lowest=True, right=False
        )

    @staticmethod
    def forest_mask_bins() -> BinGrouper:
        """Create the bins open, forest, assuming a forest mask where canopy is marked with 1."""
        return BinGrouper([0, 1, 2], labels=["open", "forest"], include_lowest=True, right=False)

    @staticmethod
    def altitude_bands(altitude_step: int = 300, altitude_min: int = 0, altitude_max: int = 4800) -> BinGrouper:
        """
        Create bins for altitude consisting in band of 300 m elevation difference matching the S2M snowpack modelling system [1].
        """
        bin_edges = np.arange(altitude_min, altitude_max, altitude_step)

        return BinGrouper(
            bin_edges, labels=MountainBinner.create_labels_from_bin_edges(bin_edges), include_lowest=True, right=False
        )

    def bins_max(self, MountainBinner_data: xr.DataArray, coordinate_name: str) -> np.ndarray:
        """Right edge of a bin."""
        return np.array(
            [float(bin_max_str.split("-")[1]) for bin_max_str in MountainBinner_data.coords[coordinate_name].values]
        )

    def bins_min(self, MountainBinner_data: xr.DataArray, coordinate_name: str) -> np.ndarray:
        """Left edge of a bin."""
        return np.array(
            [float(bin_min_str.split("-")[0]) for bin_min_str in MountainBinner_data.coords[coordinate_name].values]
        )

    @staticmethod
    def regular_8_aspect_bins() -> BinGrouper:
        """Define bins consisting in a compass rose with the cardinal and intercardinal points.

        It assumes that MountainBinner.aspect_map_transform was applied to an aspect map."""
        return BinGrouper(
            np.arange(-45 / 2, 360, 45),
            labels=np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        )

    @staticmethod
    def create_default_bin_dict(altitude_step: int = 300, altitude_max: int = 4801) -> Dict[str, BinGrouper]:
        """Create bins for topographic chracterization by slope, aspect and elevation (semidistributed geometry) and forest.

        Match S2M [1]."""
        return dict(
            slope=MountainBinner.default_slope_bands(),
            aspect=MountainBinner.regular_8_aspect_bins(),
            altitude=MountainBinner.altitude_bands(altitude_step=altitude_step, altitude_max=altitude_max),
            forest_mask=MountainBinner.forest_mask_bins(),
        )

    def create_user_bin_dict(
        self,
        slope_edges: np.ndarray | None = None,
        aspect_edges: np.ndarray | None = None,
        altitude_edges: np.ndarray | None = None,
        landcover_classes: np.ndarray | None = None,
    ) -> Dict[str, BinGrouper]:
        """Simplify user input to create binning.

        The user defines the edge sequence for each paraleter and this function does the rest

        Args:
            slope_edges (np.ndarray): sequence of slope bin edges. Ascending order. Optional.
            aspect_edges (np.ndarray): sequence of aspect bin edges. Ascending order. Optional.
            altitude_edges (np.ndarray): sequence of altitude bin edges. Ascending order. Optional.
            landcover_classes (np.ndarray): landcover classes. Optional.

        Raises:
            MountainBinnerError: negative altitude or slopes are not accepted.

        Returns:
            Dict[str, BinGrouper]: the dictionary of bin to be used in xarray groupby
        """
        if np.any(altitude_edges < 0) or np.any(slope_edges < 0):
            raise MountainBinnerError(
                f"Negative altitudes and slopes not supported. Your altitude {altitude_edges}. Your slopes {slope_edges}"
            )
        output_bin_dict = {}
        if slope_edges is not None:
            output_bin_dict.update(slope=MountainBinner.user_bins(bin_edges=slope_edges))
        if aspect_edges is not None:
            output_bin_dict.update(aspect=MountainBinner.user_bins(bin_edges=aspect_edges))
        if altitude_edges is not None:
            output_bin_dict.update(altitude=MountainBinner.user_bins(bin_edges=altitude_edges))
        if landcover_classes is not None:
            landcover_classes_sorted = np.sort(landcover_classes)
            # landcover classes are defined on a bin whose lowest values correspond to the discrete value of the class
            # we need to define an extra bin edge for the last class in order to keep the bin definition
            # this way is should be transparent to users
            output_bin_dict.update(
                forest_mask=BinGrouper(
                    bins=np.array([*landcover_classes_sorted, landcover_classes_sorted[-1] + 1]),
                    labels=landcover_classes_sorted,
                    include_lowest=True,
                    right=False,
                )
            )
        return output_bin_dict

    @classmethod
    def from_dem_and_forest_mask_filepath(
        cls, dem_filepath: str, forest_mask_filepath: str, distributed_data_filepath: str, output_folder: str
    ) -> Self:
        """Default initialization of Mountain binner object. It takes care of preprocessing data.

        Args:
            dem_filepath (str): path to a Digital Elevation Model in any projection covering your area of interest.
            forest_mask_filepath (str): path to a forest mask (or any landcover dataset) in any projection covering your area of interest
            distributed_data_filepath (str): path to some geospatial dataset.
            output_folder (str): path to a folder that stores the regridded DEM, the slope and aspect map.

        Returns:
            Self: a MountainBinner object
        """
        output_dem_filepath, output_slope_filepath, output_aspect_filepath, output_forest_mask_filepath = preprocess(
            input_dem_filepath=dem_filepath,
            forest_mask_filepath=forest_mask_filepath,
            distributed_data_filepath=distributed_data_filepath,
            output_folder=output_folder,
        )
        return cls(
            MountainBinnerConfig(
                slope_map_path=output_slope_filepath,
                aspect_map_path=output_aspect_filepath,
                dem_path=output_dem_filepath,
                regular_8_aspects=True,
                forest_mask_path=output_forest_mask_filepath,
            )
        )

    def stack_auxiliary_data(self, distributed_data: xr.DataArray | xr.Dataset) -> xr.Dataset:
        """Create a Dataset whose DataArrays are slope, aspect, elevation, landcover maps and a geospatial (distributed) dataset.

        This is essential to apply binning in an Xarray flavour.

        Args:
            distributed_data (xr.DataArray | xr.Dataset): your geospatial dataset.

        Returns:
            xr.Dataset: output dataset with geospatial dataset and auxiliary data.
        """
        if self.config.slope_map_path is not None:
            slope_map = xr.open_dataarray(self.config.slope_map_path)
            dataset = distributed_data.assign(slope=slope_map)

        if self.config.aspect_map_path is not None:
            aspect_map = xr.open_dataarray(self.config.aspect_map_path)
            if self.config.regular_8_aspects:
                aspect_map = MountainBinner.aspect_map_transform(aspect_map)
            dataset = dataset.assign(aspect=aspect_map)

        if self.config.dem_path is not None:
            dem_map = xr.open_dataarray(self.config.dem_path)
            dataset = dataset.assign(altitude=dem_map)

        if self.config.forest_mask_path is not None:
            forest_mask = xr.open_dataarray(self.config.forest_mask_path)
            dataset = dataset.assign(forest_mask=forest_mask)

        # Drop band dimension if rioxarray was used as engine
        if "band" in dataset.dims:
            dataset = dataset.sel(band=1).drop_vars("band")

        return dataset

    def prepare(self, distributed_data: xr.DataArray | xr.Dataset, bin_dict: Dict[str, BinGrouper]):
        """Use groupby to prepare binning.

        The resulting dataset is ready to apply a custom reduction using the map function. Example:

        mountain_binner.prepare(distributed_data, bin_dict).map(<your function>)

        Args:
            distributed_data (xr.DataArray | xr.Dataset): your geospatial dataset.
            bin_dict (Dict[str, BinGrouper]): the dictionary of bin to be used in xarray groupby

        """
        variable_and_auxiliary = self.stack_auxiliary_data(distributed_data=distributed_data)
        return variable_and_auxiliary.groupby(bin_dict)

    def rename_coords(self, binned_data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        """Clean output of groupby for easier user handling.

        Args:
            binned_data (xr.DataArray | xr.Dataset): Result of a groupby().map()

        Returns:
            xr.DataArray | xr.Dataset: the same as binned_data but with coordinates renamed.
        """
        sd = binned_data
        if self.config.dem_path:
            sd = sd.assign_coords(altitude_min=("altitude_bins", self.bins_min(sd, "altitude_bins")))
            sd = sd.assign_coords(altitude_max=("altitude_bins", self.bins_max(sd, "altitude_bins")))
            sd = sd.set_xindex("altitude_min")
            sd = sd.set_xindex("altitude_max")
        if self.config.slope_map_path:
            sd = sd.assign_coords(slope_min=("slope_bins", self.bins_min(sd, "slope_bins")))
            sd = sd.assign_coords(slope_max=("slope_bins", self.bins_max(sd, "slope_bins")))
            sd = sd.set_xindex("slope_min")
            sd = sd.set_xindex("slope_max")
        if self.config.aspect_map_path:
            if not self.config.regular_8_aspects:
                sd = sd.assign_coords(aspect_min=("aspect_bins", self.bins_min(sd, "aspect_bins")))
                sd = sd.assign_coords(aspect_max=("aspect_bins", self.bins_max(sd, "aspect_bins")))
                sd = sd.set_xindex("aspect_min")
                sd = sd.set_xindex("aspect_max")
            else:
                sd = sd.rename({"aspect_bins": "aspect"})
        if self.config.forest_mask_path:
            sd = sd.rename({"forest_mask_bins": "landcover"})
        return sd

    def transform(
        self,
        distributed_data: xr.DataArray | xr.Dataset,
        bin_dict: Dict[str, BinGrouper],
        function: Callable,
        *args,
    ) -> xr.DataArray | xr.Dataset:
        """Wrap-up data binning in one function.

        Args:
            distributed_data (xr.DataArray | xr.Dataset): your geospatial dataset.
            bin_dict (Dict[str, BinGrouper]): the dictionary of bin to be used in xarray groupby
            function (Callable): the reduction function you want to apply

        Returns:
            xr.DataArray | xr.Dataset: the geospatial data transformed in the output binned space
        """
        transformed = self.prepare(
            distributed_data=distributed_data,
            bin_dict=bin_dict,
        ).map(func=function, args=args)
        postprocess = self.rename_coords(transformed)

        return postprocess
