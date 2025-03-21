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
var_target = 'pr'
num_ensemble = 1
#########################

# ASM to load (test set)
asm_to_load = {'Original trained model': f'{xai_path}/ASM_deepESD_{var_target}_ens{num_ensemble}_test_period.nc',
               'No pre-training': f'{xai_path}/ASM_deepESD_stations_original_{var_target}_ens{num_ensemble}_test_period.nc',
               'Pre-trained': f'{xai_path}/ASM_deepESD_stations_pretrained_{var_target}_ens{num_ensemble}_test_period.nc',
               'Pre-trained w/ fine-tuning': f'{xai_path}/ASM_deepESD_stations_pretrained_finetuning_{var_target}_ens{num_ensemble}_test_period.nc'}
asm_to_load = {x: xr.open_dataset(asm_to_load[x]).load() for x in asm_to_load}

# Get keys (assuming all ASMs have the same keys)
keys = list(asm_to_load['Original trained model'].keys())

# Get the values of each ASM
values = []
for key in asm_to_load.keys():
    asm_sum = {x: asm_to_load[key][x].sum().values.item() * 100 for x in list(asm_to_load[key].data_vars)}
    values.append(list(asm_sum.values()))

# Number of groups (categories) and bars per group
num_keys = len(keys)
num_bars = len(asm_to_load)

# Colors
bar_colors = {'Original trained model': 'silver',
              'No pre-training': 'lightcoral',
              'Pre-trained': 'lightgreen',
              'Pre-trained w/ fine-tuning': 'lightskyblue'}

# Bar width
bar_width = 0.25
group_space = 0.3

# X positions for each group
x_pos = np.arange(num_keys) * (num_bars * bar_width + group_space)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot bars for each dictionary
for i, val in enumerate(values):
    label = list(asm_to_load.keys())[i]
    ax.bar(x_pos + i * bar_width, val, width=bar_width,
           label=label,
           color=bar_colors[label],
           edgecolor='black',
           zorder=3)

# Set x-axis labels and ticks
ax.set_xticks(x_pos + (num_bars - 1) * bar_width / 2)  # Center labels
ax.set_xticklabels(keys)

# Add labels, title, and legend
ax.set_xlabel('Predictor variable', fontsize=14)
ax.set_ylabel('ASM (%)', fontsize=14)
ax.legend(fontsize=14)

# Add grid
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

fig_name = f'ASM_test_{var_target}.pdf'
plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight')
plt.close()