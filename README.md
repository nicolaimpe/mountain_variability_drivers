# mountain_variability_drivers
Efficient xarray-based workflow to perform data binning of a geospatial quantity in a mountain landscape.

## Description
Map distributed (gridded) information into a multi-dimensional space whose coordinates are the main drivers of
spatial  variability in a mountain landscape: topography (elevation, aspect, slope), landcover (presence of canopy).

```bash
$ tree
├── data
├── notebooks
│   └── example_usage.ipynb         : Use cases
├── scripts
│   └── process_mnt.sh              : GDAL based script to regrid and prepare data (same of preprocessing.py)
├── src
│   └── mountain_data_binner
│       ├── __init__.py
│       ├── mountain_binner.py      : class to project a geospatial dataset into the binned space
│       ├── preprocessing.py        : regrid and prepare data for efficient binning
│       └── semidistributed.py      : particular case of mountain_binner where we are interested in a topographic 
|                                     representation (semidistributed geometry)
└──  tests
```
## Installation

```bash
git clone git@github.com:nicolaimpe/mountain_data_binner.git
cd mountain_data_binner
pip install .
```

## Usage
```python
# 

```
```

```

```python
# 
```
See `notebooks/example_usage.ipynb` for use cases.

## Contributing

Contributions are welcome.

PDM is recommended for environment management.

```bash
pip install pdm
pdm install
```

To add a package to the project

```bash
pip add <your_package>
```