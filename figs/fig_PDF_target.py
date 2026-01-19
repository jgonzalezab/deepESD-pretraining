import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/')
import src.utils as utils

# Add deep4downscaling to path
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/deep4downscaling/')
from deep4downscaling.deep.xai import get_closest_gridpoints_to_stations
import deep4downscaling.trans as trans

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_path = paths['data']
data_stations_eca = paths['data_stations_eca']
figs_path = paths['figs']

##### Configuration #####
var_target = 'pr'
#########################

# Period to plot
period = ('1980', '2010') # Training set

# Load grid data (target used for training/prediction)
grid_filename = f'{data_path}/{var_target}_AEMET.nc'
grid_data = xr.open_dataset(grid_filename).load()
grid_data = grid_data.sel(time=slice(*period))

# Load station data (actual observations)
var_eca_map = {'tasmin': 'tn', 'tasmax': 'tx', 'pr': 'rr'}
var_eca = var_eca_map[var_target]
station_filename = f'{data_stations_eca}/ECA_blend_{var_eca}.nc'
station_data = xr.open_dataset(station_filename).load()
station_data = station_data.sel(time=slice(*period))

# Remove stations with no values in the training period
station_data, _ = trans.remove_stations_with_nans(station_data, station_data)

# Align both datasets in time
grid_data, station_data = trans.align_datasets(grid_data, station_data, 'time')

# Create mask for grid data to use with get_closest_gridpoints_to_stations
# We use the grid target data to define the mask of available grid points
grid_mask = (grid_data[[var_target]].isel(time=0, drop=True).notnull() * 1) if 'time' in grid_data.dims else (grid_data[[var_target]].notnull() * 1)

# Use get_closest_gridpoints_to_stations to obtain the unique closest gridpoints
# This function finds which grid points are closest to the stations
closest_indices_unique = get_closest_gridpoints_to_stations(grid_mask, station_data)

# Stack grid data to 1D for indexing
grid_stack = grid_data[var_target].stack(gridpoint=('lat', 'lon'))
grid_mask_stack = grid_mask[var_target].stack(gridpoint=('lat', 'lon'))
grid_mask_filt = grid_mask_stack.where(grid_mask_stack == 1, drop=True)
grid_stack_filt = grid_stack.sel(gridpoint=grid_mask_filt['gridpoint'].values)

# Dataset with the unique closest gridpoints
grid_unique = grid_stack_filt.isel(gridpoint=closest_indices_unique)

# Build per-station mapping to the closest of these unique gridpoints
unique_lats = grid_unique['lat'].values
unique_lons = grid_unique['lon'].values
station_lats = station_data['lat'].values
station_lons = station_data['lon'].values

station_to_grid_idx = []
for s in range(len(station_lats)):
    dists = np.sqrt((unique_lats - station_lats[s])**2 +
                    (unique_lons - station_lons[s])**2)
    station_to_grid_idx.append(int(np.argmin(dists)))

# Extract grid values at the closest gridpoint for each station
# grid_unique has dims (time, gridpoint) or (gridpoint, time)
# We use xarray's isel to safely select the grid points for each station
grid_at_stations = grid_unique.isel(gridpoint=station_to_grid_idx)

# Flatten data for PDF computation
# station_data[var_eca] has dims (time, station)
# grid_at_stations has dims (time, gridpoint)
station_vals_flat = station_data[var_eca].values.flatten()
grid_vals_flat = grid_at_stations.values.flatten()

# Ensure consistent mask for both (only use points where both have valid data)
valid_mask = (~np.isnan(station_vals_flat)) & (~np.isnan(grid_vals_flat))

# For precipitation, also exclude negative values if any
if var_target == 'pr':
    valid_mask = valid_mask & (station_vals_flat >= 0) & (grid_vals_flat >= 0)

station_vals_flat = station_vals_flat[valid_mask]
grid_vals_flat = grid_vals_flat[valid_mask]

# Plotting
plt.figure(figsize=(9, 7))

# Histogram parameters
if var_target == 'pr':
    bins = np.linspace(0, 200, 150)
    plt.yscale('log')
    plt.xlabel('Precipitation amount (mm/day)', fontsize=14, fontweight='bold')
elif var_target == 'tasmin':
    bins = np.linspace(-10, 25, 25)
    plt.xlabel('Minimum temperature (°C)', fontsize=14, fontweight='bold')
elif var_target == 'tasmax':
    bins = np.linspace(0, 45, 30)
    plt.xlabel('Maximum temperature (°C)', fontsize=14, fontweight='bold')

# The image shows something similar to a step histogram
plt.hist(station_vals_flat, bins=bins, histtype='step', color='black', label='Stations (Observations)', linewidth=2.5, density=False)
plt.hist(grid_vals_flat, bins=bins, histtype='step', color='red', label='Grid (Closest Points)', linewidth=1.5, density=False, alpha=0.8)

plt.ylabel('Spain Count', fontsize=14, fontweight='bold')

plt.grid(True, which='both', linestyle='-', alpha=0.3)
plt.legend(fontsize=12, loc='upper right')

# Match the ticks style from the image
plt.tick_params(axis='both', which='major', labelsize=12)

# Save the figure
output_file = f'{figs_path}/pdf_targets_{var_target}.pdf'
plt.savefig(output_file, bbox_inches='tight')
plt.close()
