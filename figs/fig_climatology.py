import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import matplotlib.patches as patches

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
var_target = 'tasmin'
var_target_eca = 'tn' if var_target == 'tasmin' else 'tx' if var_target == 'tasmax' else 'rr'
#########################

# Period to plot
period = ('2009', '2018') # Training set

# Load stations dataset
stations_filename = f'{data_stations_eca}/ECA_blend_{var_target_eca}.nc'
stations = xr.open_dataset(stations_filename)
stations = stations.drop_vars(('elevation', 'country')) # Remove projection and altitude variables
stations = stations.rename({var_target_eca: var_target})  # Rename variable to match target
stations = stations.sel(time=slice(*period))
stations = stations.load()

# Remove stations with no values in the training period
stations, _ = trans.remove_stations_with_nans(stations, stations)

# Load gridded dataset
gridded_filename = f'{data_path}/{var_target}_AEMET.nc'
gridded = xr.open_dataset(gridded_filename)
gridded = gridded.sel(time=slice(*period))
gridded = gridded.load()

# Subset gridded dataset
gridded = gridded.sel(lat=slice(34.5, 44.2)).sel(lon=slice(-10.5, 6.2))

# Set climatology functions
if var_target in ('tasmin'):
    metrics_to_plot = {'Mean': metrics_ccs.mean,
                       'TNn': metrics_ccs.TNn}
elif var_target in ('tasmax'):
    metrics_to_plot = {'Mean': metrics_ccs.mean,
                       'TXx': metrics_ccs.TXx}
elif var_target in ('pr'):
    metrics_to_plot = {'Mean': metrics_ccs.mean,
                       'RX1day': metrics_ccs.RX1day}

# Set colorbars
if var_target == 'tasmin':
    vmin, vmax, num_levels = -10, 15, 25
    colormap = 'plasma_r'
elif var_target == 'tasmax':
    vmin, vmax, num_levels = 15, 45, 30 
    colormap = 'inferno_r'
elif var_target == 'pr':
    limits_metric = {'Mean': (0, 6, 12),
                     'RX1day': (30, 80, 25)}
    colormap_metric = {'Mean': 'viridis',
                      'RX1day': 'plasma_r'}

# Datasets to plot
datasets_to_plot = {'ROCIO-IBEB': gridded,
                    'STATIONS-CAT': stations}

# Compute plot
n_rows, n_cols = len(datasets_to_plot), len(metrics_to_plot)
fig = plt.figure(figsize=(8, 5))
figure_idx = 1

for dataset in datasets_to_plot.keys():
    for metric in metrics_to_plot.keys():

        dataset_clim = metrics_to_plot[metric](datasets_to_plot[dataset])

        ax = fig.add_subplot(n_rows, n_cols, figure_idx,
                             projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')

        if var_target in ('tasmin', 'tasmax'):
            vmin_plot, vmax_plot = vmin, vmax 
            continuous_cmap = plt.get_cmap(colormap)
            discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))
        elif var_target in ('pr'):
            vmin_plot, vmax_plot = limits_metric[metric][0], limits_metric[metric][1] 
            continuous_cmap = plt.get_cmap(colormap_metric[metric])
            discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, limits_metric[metric][2])))            

        if figure_idx in (1, 2):
            ax.set_title(metric, size=16)

        if figure_idx in (1, 3):
            ax.set_yticks([])
            ax.set_ylabel(dataset, size=16)

        if var_target in ('tasmin', 'tasmax'):
            vmin, vmax = vmin, vmax
            discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))

        if dataset == 'ROCIO-IBEB':

            im = plt.pcolormesh(dataset_clim.coords['lon'].values,
                                dataset_clim.coords['lat'].values,
                                dataset_clim[var_target],
                                transform=ccrs.PlateCarree(),
                                vmin=vmin_plot, vmax=vmax_plot, cmap=discrete_cmap)
            
            # Add black square to indicate station coverage area
            station_lats = stations['lat'].values
            station_lons = stations['lon'].values
            lat_min, lat_max = station_lats.min(), station_lats.max()
            lon_min, lon_max = station_lons.min(), station_lons.max()
            
            rect = patches.Rectangle((lon_min, lat_min), 
                                   lon_max - lon_min, 
                                   lat_max - lat_min,
                                   linewidth=2, edgecolor='black', 
                                   facecolor='none', transform=ccrs.PlateCarree())
            ax.add_patch(rect)

        elif dataset == 'STATIONS-CAT':

            point_size = 20

            im = plt.scatter(dataset_clim['lon'],
                             dataset_clim['lat'], c=dataset_clim[var_target],
                             s=point_size, edgecolor='k', linewidth=0.5, zorder=2,
                             transform=ccrs.PlateCarree(),
                             vmin=vmin_plot, vmax=vmax_plot, cmap=discrete_cmap)

        figure_idx = figure_idx + 1

        if (var_target == 'pr') and (figure_idx in (3, 4)):
            if metric == 'Mean':
                colorbar_pos = [0.15, 0.06, 0.3, 0.02]
            elif metric == 'RX1day':
                colorbar_pos = [0.58, 0.06, 0.3, 0.02]

            cbar_ax = fig.add_axes(colorbar_pos)
            fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')

    if var_target in ('tasmin', 'tasmax'):
        cbar_ax = fig.add_axes([0.16, 0.06, 0.7, 0.02])
        fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')

fig_name = f'climatology_{var_target}.pdf'
plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight')
plt.close()