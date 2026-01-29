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

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/deep4downscaling')
import deep4downscaling.trans as trans
import deep4downscaling.metrics as metrics
import deep4downscaling.metrics_ccs as metrics_ccs

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_path = paths['data']
figs_path = paths['figs']
data_stations_eca = paths['data_stations_eca']

##### Configuration #####
var_target = 'pr'
var_target_eca = 'tn' if var_target == 'tasmin' else 'tx' if var_target == 'tasmax' else 'rr'
#########################

# Periods to plot
periods = {'TRAIN': ('2009', '2018'),
           'TEST': ('2020', '2022')}

# Load stations dataset (full period)
stations_filename = f'{data_stations_eca}/ECA_blend_{var_target_eca}.nc'
stations = xr.open_dataset(stations_filename)
stations = stations.drop_vars(('elevation', 'country'))  # Remove projection and altitude variables
stations = stations.rename({var_target_eca: var_target})  # Rename variable to match target
stations = stations.load()

# Remove stations with no values in the training period
_, stations = trans.remove_stations_with_nans(stations.sel(time=slice(*periods['TRAIN'])), 
                                              stations)

# Function to compute data
def compute_data_availability(dataset):

    data_availability = dataset[var_target].notnull().sum(dim='time') / 365

    # Create a new dataset with the same structure
    result = dataset.isel(time=0).drop_vars('time')
    result[var_target] = data_availability
    
    return result

############################################################
# Summary tables and figures per period
############################################################

for period_name, period in periods.items():

    if period_name == 'TRAIN':
        vmin, vmax = 0, 10
    elif period_name == 'TEST':
        vmin, vmax = 0, 3

    # Set colorbar parameters for percentage data
    num_levels = 21
    colormap = 'turbo_r'
    continuous_cmap = plt.get_cmap(colormap)
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))
    
    # Subset stations dataset for current period
    stations_period = stations.sel(time=slice(*period))

    # Datasets to plot
    datasets_to_plot = {'STATIONS-CAT': stations_period}

    # Compute data availabilities
    data_availabilities = {}
    for dataset_name, dataset in datasets_to_plot.items():
        data_availabilities[dataset_name] = compute_data_availability(dataset)

    station_data = data_availabilities['STATIONS-CAT']
    for i in range(len(station_data['station'])):
        station_id = station_data['station'].values[i]
        lon_val = station_data['lon'].values[i]
        lat_val = station_data['lat'].values[i]
        nan_pct = station_data[var_target].values[i]

    # Plot figure for current period
    fig = plt.figure(figsize=(8, 6))

    dataset_name = 'STATIONS-CAT'
    dataset_data_availability = data_availabilities[dataset_name]

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')

    ax.set_title(
        f'Data Availability - {dataset_name} ({period_name}: {period[0]}-{period[1]})',
        size=16,
    )

    point_size = 40

    im = plt.scatter(
        dataset_data_availability['lon'],
        dataset_data_availability['lat'],
        c=dataset_data_availability[var_target],
        s=point_size,
        edgecolor='k',
        linewidth=0.5,
        zorder=2,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=discrete_cmap)

    # Add text with statistics summary in bottom-right corner
    mean_data_availability = float(station_data[var_target].mean())
    ax.text(0.97, 0.03, f'Average Data Availability = {mean_data_availability:.2f}',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046)
    cbar.set_label('Data Availability', size=12)

    fig_name = f'data_availability_{var_target}_{period_name.lower()}.pdf'
    plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight')
    plt.close(fig)