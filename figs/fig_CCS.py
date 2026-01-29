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
num_ensemble = 1
training_routine_list = ['raw', '', 'original', 'pretrained', 'pretrained_finetuning']

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

if var_target in ('tasmin', 'tasmax'):
    vmin, vmax, num_levels = 0, 8, 16
    continuous_cmap = plt.get_cmap('hot_r')
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))
elif var_target in ('pr'):
    vmin, vmax, num_levels = -40, 40, 20
    continuous_cmap = plt.get_cmap('BrBG')
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))

# Dict mapping training routine names
maps_routine_name = {'raw': 'Global Climate Model',
                     '': 'Pre-training',
                     'original': 'Full-training',
                     'pretrained': 'Partial fine-tuning',
                     'pretrained_finetuning': 'Full fine-tuning'}

# Compute figure
n_rows, n_cols = len(periods_fut), len(training_routine_list)
fig = plt.figure(figsize=(25, 10))
figure_idx = 1

for period in periods_fut.keys():
    for routine in training_routine_list:

        # Set model name
        if routine == '':
            model_name = f'deepESD_{var_target}_ens{num_ensemble}'
        else:
            model_name = f'deepESD_stations_eca_{routine}_{var_target}_ens{num_ensemble}'

        # Add subplot
        ax = fig.add_subplot(n_rows, n_cols, figure_idx,
                            projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.set_title(maps_routine_name[routine], size=16)

        # Load data
        if routine == 'raw':
            file_name = f'{ccs_path}/{gcm_model_name}_{scenario}_{gcm_run}_{period}_raw_{var_target}_{metric}.nc'
            data = xr.open_dataset(file_name)
            mask = utils.compute_gcm_mask_PNACC(data)
            data = data * mask
        else:
            file_name = f'{ccs_path}/{gcm_model_name}_{scenario}_{gcm_run}_{period}_{model_name}_{metric}.nc'
            data = xr.open_dataset(file_name)

        # Spatially subset
        if routine in (''):
            data = data.sel(lat=slice(40.5, 42.9)).sel(lon=slice(0.18, 3.45))

        # Plot the corresponding projection
        if routine in ('original', 'pretrained', 'pretrained_finetuning'):
            im = ax.scatter(data['lon'], data['lat'], c=data[var_target],
                            s=40, edgecolor='k', linewidth=0.5,
                            transform=ccrs.PlateCarree(), zorder=2,
                            vmin=vmin, vmax=vmax,
                            cmap=discrete_cmap)
        else:
            im = plt.pcolormesh(data.coords['lon'].values, data.coords['lat'].values,
                                data[var_target],
                                transform=ccrs.PlateCarree(),
                                vmin=vmin, vmax=vmax, cmap=discrete_cmap)

        figure_idx = figure_idx + 1

cbar_ax = fig.add_axes([0.92, 0.38, 0.02, 0.23])
cbar = fig.colorbar(im, cax=cbar_ax, orientation = 'vertical')
cbar.ax.tick_params(labelsize=10)

plt.savefig(f'{figs_path}/CCS_{var_target}_{metric}.pdf', bbox_inches='tight')
plt.close()