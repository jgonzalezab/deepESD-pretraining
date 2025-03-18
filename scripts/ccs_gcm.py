# Import libraries
import sys
import json
import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining')
import src.utils as utils

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/deep4downscaling')
import deep4downscaling.trans as trans
import deep4downscaling.deep.utils as deep_utils
import deep4downscaling.deep.models as deep_models
import deep4downscaling.deep.loss as deep_loss
import deep4downscaling.deep.train as deep_train
import deep4downscaling.deep.pred as deep_pred
import deep4downscaling.metrics_ccs as metrics_ccs

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_path = paths['data']
gcm_raw_path = paths['gcm_raw']
preds_path = paths['data_preds']
model_path = paths['models']
mask_path = paths['mask']
asym_path = paths['asym']
loss_path = paths['loss']
gcm_proj_path = paths['data_proj']
ccs_path = paths['ccs']

##### Configuration #####
var_target = sys.argv[1] # tasmin, tasmax, pr

gcm_name = 'EC-Earth3-Veg_r1i1p1f1' # GCM to downscale
scenario = 'ssp370' # Scenario to downscale
years_hist_ccs = ('01-01-1980', '31-12-2014') # Interval for the historic period when computing the Cs
#########################

#### Compute the CCs ####

# Get GCM and run separately
gcm_model_name, gcm_run = gcm_name.split('_')

# Load GCM data
pred_hist = utils.load_surface_gcm(gcm=gcm_name, var=var_target, scenario='historical',
                                   gcm_path=gcm_raw_path)

pred_fut = utils.load_surface_gcm(gcm=gcm_name, var=var_target, scenario=scenario,
                                  gcm_path=gcm_raw_path)

# Convert units
if var_target in ('pr'):
    pred_hist = pred_hist * 86400
    pred_fut = pred_fut * 86400
elif var_target in ('tasmin', 'tasmax'):
    pred_hist = pred_hist - 273.15
    pred_fut = pred_fut - 273.15     

# Subset the historical data
pred_hist = pred_hist.sel(time=slice(*years_hist_ccs))

# Periods to downscale
periods_fut = {'fut1': ('01-01-2015', '31-12-2040'),
               'fut2': ('01-01-2041', '31-12-2070'),
               'fut3': ('01-01-2071', '31-12-2100')}

# Metrics per variable
if var_target in ('tasmin'):
    metrics = {'mean': metrics_ccs.mean,
               'TNn': metrics_ccs.TNn}
    relative=False
elif var_target in ('tasmax'):
    metrics = {'mean': metrics_ccs.mean,
               'TXx': metrics_ccs.TXx}
    relative=False
elif var_target in ('pr'):
    metrics = {'mean': metrics_ccs.mean,
               'RX1day': metrics_ccs.RX1day}
    relative=True

# Iterate over periods to downscale
for period in periods_fut:
    pred_fut_period = pred_fut.sel(time=slice(*periods_fut[period]))

    for metric in metrics:
        ccs_values = metrics_ccs.compute_ccs(hist_data=pred_hist, fut_data=pred_fut_period,
                                             reduction_function=metrics[metric],
                                             relative=relative)
        ccs_values.to_netcdf(f'{ccs_path}/{gcm_model_name}_{scenario}_{gcm_run}_{period}_raw_{metric}.nc')
####