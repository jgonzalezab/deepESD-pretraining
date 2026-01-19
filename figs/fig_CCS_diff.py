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

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/deep4downscaling/')
from deep4downscaling.deep.xai import get_closest_gridpoints_to_stations
import deep4downscaling.trans as trans

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
ccs_path = paths['ccs']
data_stations_eca = paths['data_stations_eca']
figs_path = paths['figs']

##### Configuration #####
var_target = 'pr'
num_ensemble = 1
training_routine_comparison = ['original', 'pretrained', 'pretrained_finetuning']

gcm_name = 'EC-Earth3-Veg_r1i1p1f1'
scenario = 'ssp370'
period = 'fut3'
#########################

# Split the GCM name
gcm_model_name, gcm_run = gcm_name.split('_')

# Get metric
if var_target in ('tasmin'):
    metric = 'TNn'
elif var_target in ('tasmax'):
    metric = 'TXx'
elif var_target in ('pr'):
    metric = 'RX1day'

# Set colormap for differences
if var_target in ('tasmin', 'tasmax'):
    vmin, vmax, num_levels = -1, 1, 16
    continuous_cmap = plt.get_cmap('RdBu_r')
elif var_target in ('pr'):
    vmin, vmax, num_levels = -20, 20, 20
    continuous_cmap = plt.get_cmap('RdBu_r')

discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))

# Dict mapping training routine names
maps_routine_name = {'original': 'No pre-training',
                     'pretrained': 'Pre-trained',
                     'pretrained_finetuning': 'Pre-trained w/ fine-tuning'}

# Load grid CCS data (reference)
grid_model_name = f'deepESD_{var_target}_ens1'
grid_file = f'{ccs_path}/{gcm_model_name}_{scenario}_{gcm_run}_{period}_{grid_model_name}_{metric}.nc'
grid_ccs = xr.open_dataset(grid_file).load()

# Load station coordinates to find closest grid points
var_eca_map = {'tasmin': 'tn', 'tasmax': 'tx', 'pr': 'rr'}
var_eca = var_eca_map[var_target]
eca_data = xr.open_dataset(f'{data_stations_eca}/ECA_blend_{var_eca}.nc').load()
eca_data = eca_data.drop_vars(('elevation', 'country'), errors='ignore')

# Remove stations with no values in the training period
train_period = ('1980', '2010')
eca_data = eca_data.sel(time=slice(*train_period))
eca_data, _ = trans.remove_stations_with_nans(eca_data, eca_data)

# Create mask for grid data to use with get_closest_gridpoints_to_stations
grid_mask = (grid_ccs[[var_target]].isel(time=0, drop=True).notnull() * 1) if 'time' in grid_ccs.dims else (grid_ccs[[var_target]].notnull() * 1)

# Use get_closest_gridpoints_to_stations to obtain the unique closest gridpoints
closest_indices_unique = get_closest_gridpoints_to_stations(grid_mask, eca_data)

# Stack grid data to 1D for indexing
grid_stack = grid_ccs.stack(gridpoint=('lat', 'lon'))
grid_mask_stack = grid_mask.stack(gridpoint=('lat', 'lon'))
grid_mask_filt = grid_mask_stack.where(grid_mask_stack[var_target] == 1, drop=True)
grid_stack_filt = grid_stack.sel(gridpoint=grid_mask_filt['gridpoint'].values)

# Dataset with the unique closest gridpoints
grid_unique = grid_stack_filt.isel(gridpoint=closest_indices_unique)

# Build per-station mapping to the closest of these unique gridpoints
unique_lats = grid_unique['lat'].values
unique_lons = grid_unique['lon'].values
station_lats = eca_data['lat'].values
station_lons = eca_data['lon'].values

station_to_grid_idx = []
for s in range(len(station_lats)):
    dists = np.sqrt((unique_lats - station_lats[s])**2 +
                    (unique_lons - station_lons[s])**2)
    station_to_grid_idx.append(int(np.argmin(dists)))


# Create figure with 3 subplots (one for each comparison)
fig = plt.figure(figsize=(18, 6))
figure_idx = 1

for routine in training_routine_comparison:
    # Load station CCS data
    station_model_name = f'deepESD_stations_eca_{routine}_{var_target}_ens{num_ensemble}'
    station_file = f'{ccs_path}/{gcm_model_name}_{scenario}_{gcm_run}_{period}_{station_model_name}_{metric}.nc'
    station_ccs = xr.open_dataset(station_file).load()

    # Get station dimension name and coordinates from the CCS file
    station_da = station_ccs[var_target]
    station_dim = station_da.dims[0]

    # Extract grid CCS values at the closest gridpoint for each station
    grid_unique_vals = grid_unique[var_target].values
    grid_vals_per_station = grid_unique_vals[station_to_grid_idx]

    grid_at_stations_da = xr.DataArray(
        grid_vals_per_station,
        dims=[station_dim],
        coords={station_dim: station_da[station_dim],
                'lat': station_ccs['lat'],
                'lon': station_ccs['lon']},
        name=var_target,
    )

    # Compute difference (station - grid at closest point)
    diff = station_da - grid_at_stations_da
    
    # Add subplot
    ax = fig.add_subplot(1, 3, figure_idx, projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.set_title(maps_routine_name[routine], size=16)
    
    # Plot as scatter (one value per station)
    im = ax.scatter(station_ccs['lon'], station_ccs['lat'], c=diff,
                    s=50, edgecolor='k', linewidth=0.5,
                    transform=ccrs.PlateCarree(), zorder=2,
                    vmin=vmin, vmax=vmax,
                    cmap=discrete_cmap)
    
    # Plot mean difference in the bottom-right corner of the subplot
    mean_diff = float(abs(diff).mean().values)
    ax.text(0.97, 0.03, f'Abs. diff. mean = {mean_diff:.2f}',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))
    
    figure_idx = figure_idx + 1

# Add colorbar
cbar_ax = fig.add_axes([0.16, 0.15, 0.7, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar.set_label(f'CCS Difference ({metric}): Station - Grid', fontsize=12)

plt.savefig(f'{figs_path}/CCS_diff_{var_target}_{metric}.pdf', bbox_inches='tight')
plt.close()
