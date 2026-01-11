import numpy as np
import pytest
import rasterio
import rioxarray
import xarray as xr
from affine import Affine

from mountain_data_binner.semidistributed import Semidistributed, SemidistributedConfig, SemidistributedError

"""Minimal representative example documented in test_preprocessing.py"""


@pytest.fixture(scope="session")
def test_dem_file(tmp_path_factory):
    dem_data = np.pad(np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]), pad_width=2)
    file_name = tmp_path_factory.mktemp("data") / "dem.tif"
    transform = Affine(1, 0, 0, 0, -1, 7)
    with rasterio.open(
        file_name, "w", width=7, height=7, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(dem_data, 1)
    return file_name


@pytest.fixture(scope="session")
def test_dem_file_regrid_true(tmp_path_factory):
    dem_regrid_data = np.pad(np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]), pad_width=1)
    file_name = tmp_path_factory.mktemp("data") / "dem_regrid_true.tif"
    transform = Affine(1, 0, 1, 0, -1, 6)
    with rasterio.open(
        file_name, "w", width=5, height=5, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(dem_regrid_data, 1)
    return file_name


@pytest.fixture(scope="session")
def test_slope_file_true(tmp_path_factory):
    slope_data = np.pad(
        np.array([[35.26439, 45, 35.26439], [45, 0, 45], [35.26439, 45, 35.26439]]), pad_width=1, constant_values=-9999
    )

    file_name = tmp_path_factory.mktemp("data") / "slope_true.tif"
    transform = Affine(1, 0, 1, 0, -1, 6)
    with rasterio.open(
        file_name, "w", width=5, height=5, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(slope_data, 1)
    return file_name


@pytest.fixture(scope="session")
def test_aspect_file_true(tmp_path_factory):
    aspect_data = np.pad(np.array([[315, 0, 45], [270, -9999, 90], [225, 180, 135]]), pad_width=1, constant_values=-9999)

    file_name = tmp_path_factory.mktemp("data") / "aspect_true.tif"
    transform = Affine(1, 0, 1, 0, -1, 6)
    with rasterio.open(
        file_name, "w", width=5, height=5, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(aspect_data, 1)
    return file_name


@pytest.fixture(scope="session")
def test_distributed_data_file(tmp_path_factory):
    distributed_data = np.ones(shape=(5, 5))
    file_name = tmp_path_factory.mktemp("data") / "distributed.tif"
    transform = Affine(1, 0, 1, 0, -1, 6)
    with rasterio.open(
        file_name, "w", width=5, height=5, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(distributed_data, 1)
    return file_name


# contents of test_image.py
def test_semidistributed_init_methods(
    test_dem_file,
    test_dem_file_regrid_true,
    test_slope_file_true,
    test_aspect_file_true,
    test_distributed_data_file,
    tmp_path_factory,
):
    semidistributed_from_config = Semidistributed(
        SemidistributedConfig(
            slope_map_path=test_slope_file_true, dem_path=test_dem_file_regrid_true, aspect_map_path=test_aspect_file_true
        )
    )
    semidistributed_from_dem = Semidistributed.from_dem_filepath(
        dem_filepath=test_dem_file,
        distributed_data_filepath=test_distributed_data_file,
        output_folder=tmp_path_factory.mktemp("data"),
    )

    assert xr.open_dataarray(semidistributed_from_config.config.dem_path, engine="rasterio").equals(
        xr.open_dataarray(semidistributed_from_dem.config.dem_path, engine="rasterio")
    )
    assert xr.open_dataarray(semidistributed_from_config.config.slope_map_path, engine="rasterio").equals(
        xr.open_dataarray(semidistributed_from_dem.config.slope_map_path, engine="rasterio")
    )
    assert xr.open_dataarray(semidistributed_from_config.config.aspect_map_path, engine="rasterio").equals(
        xr.open_dataarray(semidistributed_from_dem.config.aspect_map_path, engine="rasterio")
    )


def sum_data_array(data: xr.DataArray) -> xr.DataArray:
    return data["test_data"].sum()


def test_semidistributed_defaults_bins(test_dem_file, test_distributed_data_file, tmp_path_factory):
    semidistributed = Semidistributed.from_dem_filepath(
        dem_filepath=test_dem_file,
        distributed_data_filepath=test_distributed_data_file,
        output_folder=tmp_path_factory.mktemp("data"),
    )
    distributed_data = xr.Dataset({"test_data": xr.open_dataarray(test_distributed_data_file)})
    default_created_bins = Semidistributed.create_default_bin_dict(altitude_step=1, altitude_max=5)

    result = semidistributed.transform(
        distributed_data=distributed_data, bin_dict=default_created_bins, function=sum_data_array
    )
    # Summit point slope=0, altitude=2 doesn't have a defined aspect (very special case)
    # So we cannot test it here
    assert result.sel(slope_min=30).sum() == 8
    assert result.sel(slope_max=50).sum() == 8
    assert result.sel(slope_bins="30 - 50").sum() == 8
    assert result.sel(altitude_min=0).sum() == 4
    assert result.sel(altitude_max=1).sum() == 4
    assert result.sel(altitude_bins="0 - 1").sum() == 4
    assert result.sel(altitude_min=1).sum() == 4
    for aspect in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        assert result.sel(aspect=aspect).sum() == 1


def test_semidistributed_user_bins(
    test_dem_file_regrid_true, test_distributed_data_file, test_slope_file_true, test_aspect_file_true
):
    semidistributed = Semidistributed(
        SemidistributedConfig(
            slope_map_path=test_slope_file_true,
            dem_path=test_dem_file_regrid_true,
            aspect_map_path=test_aspect_file_true,
            regular_8_aspects=False,
        )
    )
    distributed_data = xr.Dataset({"test_data": xr.open_dataarray(test_distributed_data_file)})
    with pytest.raises(SemidistributedError):
        user_created_bins = semidistributed.create_user_bin_dict(
            slope_edges=np.arange(0, 60, 10),
            aspect_edges=np.arange(0, 361, 90),
            altitude_edges=np.arange(-2, 5, 2),  # Altitude of -2
        )
    user_created_bins = semidistributed.create_user_bin_dict(
        slope_edges=np.arange(0, 60, 10),
        aspect_edges=np.arange(0, 361, 90),
        altitude_edges=np.arange(0, 5, 2),
    )

    result = semidistributed.transform(
        distributed_data=distributed_data,
        bin_dict=user_created_bins,
        function=sum_data_array,
    )
    # Summit point slope=0, altitude=2 doesn't have a defined aspect (very special case)
    # So we cannot test it here
    assert result.sel(slope_min=0).sum() == 0
    assert result.sel(slope_max=30).sum() == 0
    assert result.sel(slope_bins="30 - 40").sum() == 4
    assert result.sel(slope_max=50).sum() == 4
    assert result.sel(altitude_bins="0 - 2").sum() == 8
    assert result.sel(altitude_max=4).sum() == 0
    assert result.sel(aspect_bins="0 - 90").sum() == 2
    assert result.sel(aspect_min=90).sum() == 2
    assert result.sel(aspect_max=270).sum() == 2
    assert result.sel(aspect_min=slice(180, None)).sum() == 4
    assert result.sel(aspect_max=slice(None, 180)).sum() == 4
