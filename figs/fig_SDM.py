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
xai_path = paths['xai']
figs_path = paths['figs']

##### Configuration #####
var_target = 'tasmin'
var_target_eca = 'tn' if var_target == 'tasmin' else 'tx' if var_target == 'tasmax' else 'rr'
num_ensemble = 1
#########################

# SDM to load (test set)
sdm_to_load = {'Original trained model': f'{xai_path}/SDM_deepESD_{var_target}_ens{num_ensemble}_test_period.nc',
               'No pre-training': f'{xai_path}/SDM_deepESD_stations_eca_original_{var_target}_ens{num_ensemble}_test_period.nc',
               'Pre-trained': f'{xai_path}/SDM_deepESD_stations_eca_pretrained_{var_target}_ens{num_ensemble}_test_period.nc',
               'Pre-trained w/ fine-tuning': f'{xai_path}/SDM_deepESD_stations_eca_pretrained_finetuning_{var_target}_ens{num_ensemble}_test_period.nc'}
sdm_to_load = {x: xr.open_dataset(sdm_to_load[x]).load() for x in sdm_to_load}

# Set colorbar
colormap = 'magma_r'
vmin_plot, vmax_plot, num_levels = 0, 2000, 40 
continuous_cmap = plt.get_cmap(colormap)
discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))

# Create the plot
n_rows, n_cols = 1, len(sdm_to_load)
fig = plt.figure(figsize=(15, 10))
figure_idx = 1

# Plot the SDMs
for key in sdm_to_load.keys():

    ax = fig.add_subplot(n_rows, n_cols, figure_idx,
                         projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.set_title(key)

    data_to_plot = sdm_to_load[key]

    if key == 'Original trained model':
        data_to_plot = data_to_plot.sel(lat=slice(34.5, 44.2)).sel(lon=slice(-10.5, 6.2))
        im = plt.pcolormesh(data_to_plot.coords['lon'].values,
                            data_to_plot.coords['lat'].values,
                            data_to_plot[var_target],
                            transform=ccrs.PlateCarree(),
                            vmin=vmin_plot, vmax=vmax_plot, cmap=discrete_cmap)
    else:
        im = plt.scatter(data_to_plot['lon'],
                         data_to_plot['lat'], c=data_to_plot[var_target],
                         s=10, edgecolor='k', linewidth=0, zorder=2,
                         transform=ccrs.PlateCarree(),
                         vmin=vmin_plot, vmax=vmax_plot, cmap=discrete_cmap)       

    figure_idx = figure_idx + 1

cbar_ax = fig.add_axes([0.16, 0.38, 0.7, 0.02])
fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')

fig_name = f'SDM_test_{var_target}.pdf'
plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight')
plt.close()