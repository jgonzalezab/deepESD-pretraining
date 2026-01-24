import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs

# Add project root to sys.path
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/')
import src.utils as utils

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
xai_path = paths['xai']
figs_path = paths['figs']

##### Configuration #####
var_target = 'pr'
var_target_eca = 'tn' if var_target == 'tasmin' else 'tx' if var_target == 'tasmax' else 'rr'
num_ensemble = 1
coord_ism = (41.5, 2.1)
day_to_plot = None  # Set to specific date (e.g., '2016-03-04') or None/'all' to plot mean across all days
variables_to_plot = ['t850', 'q700', 'v850', 'msl']  # Set to list of specific variables (e.g., ['pr', 'tasmax']) or None to plot all available variables

# Plotting config
cmap = 'CMRmap_r'
vmin, vmax = 0, 0.002 # Adjust these as needed for sensitivity values
n_levels = 30  # Number of discrete levels
figsize_per_subplot = (4, 3)

# Generate output filename based on configuration
if day_to_plot is None or (isinstance(day_to_plot, str) and day_to_plot.lower() == 'all'):
    output_filename = f'ISM_{var_target}_comparison_mean_all_days.pdf'
else:
    output_filename = f'ISM_{var_target}_comparison_{day_to_plot}.pdf'
#########################

# ISM files to load
lat_st, lon_st = coord_ism
ism_to_load = {
    'Pre-training': f'{xai_path}/ISM_deepESD_{var_target}_ens{num_ensemble}_only_eca_stations_lat{lat_st}_lon{lon_st}_test_period.nc',
    'Full-training': f'{xai_path}/ISM_deepESD_stations_eca_original_{var_target}_ens{num_ensemble}_lat{lat_st}_lon{lon_st}_test_period.nc',
    'Partial fine-tuning': f'{xai_path}/ISM_deepESD_stations_eca_pretrained_{var_target}_ens{num_ensemble}_lat{lat_st}_lon{lon_st}_test_period.nc',
    'Full fine-tuning': f'{xai_path}/ISM_deepESD_stations_eca_pretrained_finetuning_{var_target}_ens{num_ensemble}_lat{lat_st}_lon{lon_st}_test_period.nc'
}

# Load and process data (select specific day or compute mean across all days)
ism_data = {}
for name, path in ism_to_load.items():
    print(f"Loading {name} from {path}...")
    ds = xr.open_dataset(path)
    if variables_to_plot is not None:
        ds = ds[variables_to_plot]
    # Select specific day or compute mean across all days
    if day_to_plot is None or (isinstance(day_to_plot, str) and day_to_plot.lower() == 'all'):
        ds_day = ds.mean(dim='time')
        print(f"  Computing mean across {len(ds.time)} days...")
    else:
        ds_day = ds.sel(time=day_to_plot)
        print(f"  Selected day: {day_to_plot}")
    ism_data[name] = ds_day
    ds.close()

# Identify variables to plot (all data variables in the dataset)
variables = list(ism_data[list(ism_data.keys())[0]].data_vars)
models = list(ism_data.keys())

n_rows = len(variables)
n_cols = len(models)

fig, axes = plt.subplots(n_rows, n_cols, 
                         figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
                         subplot_kw={'projection': ccrs.PlateCarree()},
                         constrained_layout=True)

# If only one variable or one model, axes might not be a 2D array
if n_rows == 1:
    axes = axes[np.newaxis, :]
if n_cols == 1:
    axes = axes[:, np.newaxis]

for r, var in enumerate(variables):
    for c, model_name in enumerate(models):
        ax = axes[r, c]
        data = ism_data[model_name][var]
        
        # Plot the data
        im = data.plot(ax=ax, 
                       transform=ccrs.PlateCarree(),
                       cmap=cmap,
                       vmin=vmin, 
                       vmax=vmax,
                       levels=n_levels,
                      add_colorbar=False)
        
        # Add geographical features
        ax.coastlines()
        ax.add_feature(ccrs.cartopy.feature.BORDERS, linestyle=':')
        
        # Add station location
        ax.plot(lon_st, lat_st, 'ro', markersize=5, transform=ccrs.PlateCarree(), label='Station')
        
        # Titles and labels
        if r == 0:
            ax.set_title(model_name, fontsize=12)
        else:
            ax.set_title("")
            
        if c == 0:
            ax.text(-0.07, 0.5, var.upper(), transform=ax.transAxes, 
                    va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)

# Add a single colorbar for the whole figure
cbar = fig.colorbar(im, 
                    ax=axes, 
                    orientation='vertical', 
                    shrink=0.5, 
                    aspect=30, 
                    pad=0.02,
                    label='Input Sensitivity')

# Save the plot
save_path = f"{figs_path}/{output_filename}"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
