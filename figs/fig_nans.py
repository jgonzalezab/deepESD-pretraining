import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/')
import src.utils as utils

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling')
import deep4downscaling.metrics as metrics
import deep4downscaling.metrics_ccs as metrics_ccs

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_path = paths['data']
figs_path = paths['figs']
data_stations_eca = paths['data_stations_eca']

##### Configuration #####
var_target = 'tasmax'
var_target_eca = 'tn' if var_target == 'tasmin' else 'tx' if var_target == 'tasmax' else 'rr'
#########################

# Periods to plot
periods = {
    'TRAIN': ('1980', '2010'),
    'TEST': ('2011', '2020'),
}

# Load stations dataset (full period)
stations_filename = f'{data_stations_eca}/ECA_blend_{var_target_eca}.nc'
stations = xr.open_dataset(stations_filename)
stations = stations.drop_vars(('elevation', 'country'))  # Remove projection and altitude variables
stations = stations.rename({var_target_eca: var_target})  # Rename variable to match target
stations = stations.load()

# Load gridded dataset (full period)
gridded_filename = f'{data_path}/{var_target}_AEMET.nc'
gridded = xr.open_dataset(gridded_filename)
gridded = gridded.load()

# Subset gridded dataset (spatial only)
gridded = gridded.sel(lat=slice(34.5, 44.2)).sel(lon=slice(-10.5, 6.2))

# Function to compute percentage of NaNs
def compute_nan_percentage(dataset):
    """
    Compute percentage of NaN values along time dimension.
    Returns a dataset with percentage of NaNs for each spatial point.
    """
    nan_count = dataset[var_target].isnull().sum(dim='time')
    total_count = dataset[var_target].sizes['time']
    nan_percentage = (nan_count / total_count) * 100
    
    # Create a new dataset with the same structure
    result = dataset.isel(time=0).drop_vars('time')
    result[var_target] = nan_percentage
    
    return result

############################################################
# Summary tables and figures per period
############################################################

# Set colorbar parameters for percentage data
vmin, vmax = 0, 100
num_levels = 21
colormap = 'Reds'
continuous_cmap = plt.get_cmap(colormap)
discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))

for period_name, period in periods.items():
    # Subset stations dataset for current period
    stations_period = stations.sel(time=slice(*period))

    # Datasets to plot
    datasets_to_plot = {'STATIONS-CAT': stations_period}

    # Compute NaN percentages
    nan_percentages = {}
    for dataset_name, dataset in datasets_to_plot.items():
        nan_percentages[dataset_name] = compute_nan_percentage(dataset)

    # Print NaN percentage for each station
    print(f"\n{'='*60}")
    print(f"NaN Percentage for each station ({var_target}) - {period_name}")
    print(f"Period: {period[0]} - {period[1]}")
    print(f"{'='*60}\n")
    print(f"{'Station':<10} {'Lon':<10} {'Lat':<10} {'% NaN':<10}")
    print(f"{'-'*60}")

    station_data = nan_percentages['STATIONS-CAT']
    for i in range(len(station_data['station'])):
        station_id = station_data['station'].values[i]
        lon_val = station_data['lon'].values[i]
        lat_val = station_data['lat'].values[i]
        nan_pct = station_data[var_target].values[i]
        print(f"{station_id:<10} {lon_val:<10.3f} {lat_val:<10.3f} {nan_pct:<10.2f}")

    print(f"\n{'='*60}")
    print(f"Summary Statistics ({period_name}):")
    print(f"  Mean NaN %: {float(station_data[var_target].mean()):.2f}")
    print(f"  Min NaN %:  {float(station_data[var_target].min()):.2f}")
    print(f"  Max NaN %:  {float(station_data[var_target].max()):.2f}")
    print(f"  Stations with >50% NaN: {int((station_data[var_target] > 50).sum())}")
    print(f"  Total stations: {len(station_data['station'])}")
    print(f"{'='*60}\n")

    # Plot figure for current period
    fig = plt.figure(figsize=(8, 6))

    dataset_name = 'STATIONS-CAT'
    dataset_nan = nan_percentages[dataset_name]

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')

    ax.set_title(
        f'% NaN Values - {dataset_name} ({period_name}: {period[0]}-{period[1]})',
        size=16,
    )

    point_size = 20

    im = plt.scatter(
        dataset_nan['lon'],
        dataset_nan['lat'],
        c=dataset_nan[var_target],
        s=point_size,
        edgecolor='k',
        linewidth=0.5,
        zorder=2,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=discrete_cmap,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046)
    cbar.set_label('% NaN Values', size=12)

    fig_name = f'nans_{var_target}_{period_name.lower()}.pdf'
    plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight')
    plt.close(fig)