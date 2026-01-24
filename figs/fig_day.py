import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/')
import src.utils as utils

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/deep4downscaling')
import deep4downscaling.trans as trans

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_path = paths['data']
figs_path = paths['figs']
data_stations_eca = paths['data_stations_eca']

##### Configuration #####
var_target = 'tasmin'  # Can be 'tasmin', 'tasmax', or 'pr'
var_target_eca = 'tn' if var_target == 'tasmin' else 'tx' if var_target == 'tasmax' else 'rr'
#########################

# Select a specific day to plot (format: YYYY-MM-DD)
specific_day = '2018-06-15'

# Load stations dataset
stations_filename = f'{data_stations_eca}/ECA_blend_{var_target_eca}.nc'
stations = xr.open_dataset(stations_filename)
stations = stations.drop_vars(('elevation', 'country')) # Remove projection and altitude variables
stations = stations.rename({var_target_eca: var_target})  # Rename variable to match target

# Remove stations with no values in the training period
training_period = ('1980', '2010')
test_period = ('2011', '2020')
stations_training = stations.sel(time=slice(*training_period))
stations_test = stations.sel(time=slice(*test_period))

# Remove stations with no values in the training and test periods
stations_training, stations_test = trans.remove_stations_with_nans(stations_training, stations_test)
print(stations_training)
print(stations_test)

# Select the specific day
stations_day = stations_test.sel(time=specific_day)

# Set up the plot - single panel for stations only
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')

# Set colorbar limits based on variable (for consistent colors)
if var_target == 'tasmin':
    vmin, vmax = -10, 25
    cmap = 'RdBu_r'
elif var_target == 'tasmax':
    vmin, vmax = 5, 45
    cmap = 'RdBu_r'
elif var_target == 'pr':
    vmin, vmax = 0, 50
    cmap = 'Blues'

point_size = 100

im = plt.scatter(stations_day['lon'],
                 stations_day['lat'], c=stations_day[var_target],
                 s=point_size, edgecolor='k', linewidth=0.5, zorder=2,
                 transform=ccrs.PlateCarree(),
                 vmin=vmin, vmax=vmax, cmap=cmap)

# Remove titles and labels
ax.set_title('') 
ax.set_xlabel('')
ax.set_ylabel('')

# Remove ticks and tick labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# Save figure
fig_name = f'day_stations_eca_{var_target}_{specific_day}.pdf'
plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight', dpi=300)
plt.close()