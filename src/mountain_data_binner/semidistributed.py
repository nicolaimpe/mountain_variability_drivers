from dataclasses import dataclass
from typing import Dict, Self

import numpy as np
from xarray.groupers import BinGrouper

from mountain_data_binner.mountain_binner import MountainBinner, MountainBinnerConfig
from mountain_data_binner.preprocessing import preprocess_topography


class SemidistributedError(Exception):
    pass


@dataclass
class SemidistributedConfig:
    """Regroup user inputs. All datasets should be in the same geospatial grid and projection.

    Args:
        slope_map_path (str): path to the slope map. Optional.
        aspect_map_path (str): path to the aspect map. Optional.
        dem_path (str): path to the Digital Elevation Model. Optional.
        regular_8_aspects_bins: whether the aspect binning should be done in a 8-points compass rose (N, NE, E, SE, S, SW, W, NW).
                                Defaults to True.

    """

    slope_map_path: str | None = None
    aspect_map_path: str | None = None
    dem_path: str | None = None
    regular_8_aspects: bool = True


class Semidistributed(MountainBinner):
    """Same as MountainBinner but we only consider topography.

    Implement semidistributed geometry as described in [1]

    [1] M. Vernay et al., « The S2M meteorological and snow cover reanalysis over the French mountainous areas:
    description and evaluation (1958–2021) », Earth System Science Data, vol. 14, nᵒ 4, p. 1707‑1733, avr. 2022,
    doi: 10.5194/essd-14-1707-2022.
    """

    def __init__(self, config: SemidistributedConfig):
        print(config)
        super().__init__(
            MountainBinnerConfig(
                slope_map_path=config.slope_map_path,
                aspect_map_path=config.aspect_map_path,
                dem_path=config.dem_path,
                regular_8_aspects=config.regular_8_aspects,
                forest_mask_path=None,
            )
        )

    @classmethod
    def from_dem_filepath(cls, dem_filepath: str, distributed_data_filepath: str, output_folder: str) -> Self:
        """Default initialization of Semidistributed object. It takes care of preprocessing data.

        Args:
            dem_filepath (str): path to a Digital Elevation Model in any projection covering your area of interest.
            distributed_data_filepath (str): path to some geospatial dataset.
            output_folder (str): path to a folder that stores the regridded DEM, the slope and aspect map.

        Returns:
            Self: a Semidistributed object
        """
        output_dem_filepath, output_slope_filepath, output_aspect_filepath = preprocess_topography(
            input_dem_filepath=dem_filepath, distributed_data_filepath=distributed_data_filepath, output_folder=output_folder
        )
        return cls(
            SemidistributedConfig(
                slope_map_path=output_slope_filepath,
                aspect_map_path=output_aspect_filepath,
                dem_path=output_dem_filepath,
                regular_8_aspects=True,
            )
        )

    @staticmethod
    def create_default_bin_dict(altitude_step: int = 300, altitude_max: int = 4801) -> Dict[str, BinGrouper]:
        """Create bins for topographic chracterization by slope, aspect and elevation (semidistributed geometry).

        Match S2M [1]."""
        return dict(
            slope=MountainBinner.default_slope_bands(),
            aspect=MountainBinner.regular_8_aspect_bins(),
            altitude=MountainBinner.altitude_bands(altitude_step=altitude_step, altitude_max=altitude_max),
        )

    @staticmethod
    def create_user_bin_dict(
        slope_edges: np.ndarray | None = None, aspect_edges: np.ndarray | None = None, altitude_edges: np.ndarray | None = None
    ) -> Dict[str, BinGrouper]:
        """Simplify user input to create binning.

        The user defines the edge sequence for each paraleter and this function does the rest

        Args:
            slope_edges (np.ndarray): sequence of slope bin edges. Ascending order. Optional.
            aspect_edges (np.ndarray): sequence of aspect bin edges. Ascending order. Optional.
            altitude_edges (np.ndarray): sequence of altitude bin edges. Ascending order. Optional.

        Raises:
            SemidistributedError: negative altitude or slopes are not accepted.

        Returns:
            Dict[str, BinGrouper]: the dictionary of bin to be used in xarray groupby
        """
        if np.any(altitude_edges < 0) or np.any(slope_edges < 0):
            raise SemidistributedError(
                f"Negative altitudes and slopes not supported. Your altitude {altitude_edges}. Your slopes {slope_edges}"
            )
        output_bin_dict = {}
        if slope_edges is not None:
            output_bin_dict.update(slope=MountainBinner.user_bins(bin_edges=slope_edges))
        if aspect_edges is not None:
            output_bin_dict.update(aspect=MountainBinner.user_bins(bin_edges=aspect_edges))
        if altitude_edges is not None:
            output_bin_dict.update(altitude=MountainBinner.user_bins(bin_edges=altitude_edges))

        return output_bin_dict
