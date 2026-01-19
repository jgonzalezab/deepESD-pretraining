import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
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
loss_path = paths['loss']
figs_path = paths['figs']

##### Configuration #####
var_target = 'pr'
num_total_ensemble = 10
mode = 'train'
#########################
# Set losses to load
if mode == 'train':
    path_appendix = 'train'
elif mode == 'val':
    path_appendix = 'val'

# Iterate over ensembles
for num_ensemble in range(1, num_total_ensemble+1):

    loss_values = {'No pre-training': np.load(f'{loss_path}/{path_appendix}_deepESD_stations_eca_original_{var_target}_ens{num_ensemble}.npy'),
                  'Pre-trained': np.load(f'{loss_path}/{path_appendix}_deepESD_stations_eca_pretrained_{var_target}_ens{num_ensemble}.npy'),
                  'Pre-trained w/ fine-tuning': np.load(f'{loss_path}/{path_appendix}_deepESD_stations_eca_pretrained_finetuning_{var_target}_ens{num_ensemble}.npy')}

    colors = {'No pre-training': 'orange',
              'Pre-trained': 'blue',
              'Pre-trained w/ fine-tuning': 'green'}

    var_name = {'tasmin': 'Minimum Temperature',
                'tasmax': 'Maximum Temperature',
                'pr': 'Precipitation'}

    epoch_init = 10

    for model in loss_values.keys():

        plt.plot(list(range(epoch_init+1, len(loss_values[model])+1)),
                loss_values[model][epoch_init:],
                color=colors[model],
                linestyle='solid', linewidth=0.5,
                alpha=0.5)

if var_target == 'pr':
    y_label = 'Loss (ASYM)'
else:
    y_label = 'Loss (MSE)'

# Set legend
legend_handles = []
for model, _ in colors.items():
    handle = mlines.Line2D([], [], color=colors[model], linestyle='solid', linewidth=2, label=model)
    legend_handles.append(handle)
plt.legend(handles=legend_handles)

plt.title(f'{mode.capitalize()} - {var_name[var_target]}')
plt.xlabel('Epoch')
plt.ylabel(y_label)
plt.grid(True)

plt.savefig(f'{figs_path}/loss_{var_target}_{mode}.pdf',
            bbox_inches='tight')
plt.close()