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
ccs_path = paths['ccs']
figs_path = paths['figs']

##### Configuration #####
var_target = 'pr'
num_total_ensemble = 10
training_routine_list = ['original', 'pretrained', 'pretrained_finetuning']

gcm_name = 'EC-Earth3-Veg_r1i1p1f1'
scenario = 'ssp370'
#########################

# Split the GCM name
gcm_model_name, gcm_run = gcm_name.split('_')

# Compute plot
periods_fut = {'fut3': ('01-01-2071', '31-12-2100')}

if var_target in ('tasmin'):
    metric = 'TNn'
elif var_target in ('tasmax'):
    metric = 'TXx'
elif var_target in ('pr'):
    metric = 'RX1day'

if var_target == 'tasmin':
    vmin, vmax, num_levels = 0, 0.25, 15
    continuous_cmap = plt.get_cmap('Reds')
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))
elif var_target == 'tasmax':
    vmin, vmax, num_levels = 0, 0.5, 15
    continuous_cmap = plt.get_cmap('Reds')
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))
elif var_target in ('pr'):
    vmin, vmax, num_levels = 0, 5, 15
    continuous_cmap = plt.get_cmap('Reds')
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))

# Dict mapping training routine names
maps_routine_name = {'original': 'No pre-training',
                     'pretrained': 'Pre-trained',
                     'pretrained_finetuning': 'Pre-trained w/ fine-tuning'}

n_rows, n_cols = len(periods_fut), len(training_routine_list)
fig = plt.figure(figsize=(20, 10))
figure_idx = 1

for period in periods_fut.keys():
    for routine in training_routine_list:

        # Iterate over members of the ensemble
        ccs_ensemble = []
        for num_ensemble in range(1, num_total_ensemble+1):

            # Set model name
            model_name = f'deepESD_stations_eca_{routine}_{var_target}_ens{num_ensemble}'

            # Load data
            file_name = f'{ccs_path}/{gcm_model_name}_{scenario}_{gcm_run}_{period}_{model_name}_{metric}.nc'
            data = xr.open_dataset(file_name)

            # Append to list
            ccs_ensemble.append(data)

        # Compute std over the ensemble
        ccs_ensemble = xr.concat(ccs_ensemble, dim='ensemble_member')
        ccs_ensemble_std = ccs_ensemble.std(dim='ensemble_member')

        # Add subplot
        ax = fig.add_subplot(n_rows, n_cols, figure_idx,
                            projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.set_title(maps_routine_name[routine], size=14)

        # Plot the corresponding projection
        im = ax.scatter(ccs_ensemble_std['lon'], ccs_ensemble_std['lat'],
                        c=ccs_ensemble_std[var_target],
                        s=20, edgecolor='k', linewidth=0,
                        transform=ccrs.PlateCarree(), zorder=2,
                        vmin=vmin, vmax=vmax,
                        cmap=discrete_cmap)

        figure_idx = figure_idx + 1

cbar_ax = fig.add_axes([0.16, 0.3, 0.7, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
cbar.ax.tick_params(labelsize=10)

plt.savefig(f'{figs_path}/CCS_std_{var_target}_{metric}.pdf', bbox_inches='tight')
plt.close()