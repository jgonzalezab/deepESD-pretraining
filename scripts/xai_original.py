# Import libraries
import sys
import json
import xarray as xr
import numpy as np
import torch
import captum

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining')
import src.utils as utils

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/deep4downscaling')
import deep4downscaling.trans as trans
import deep4downscaling.deep.utils as deep_utils
import deep4downscaling.deep.models as deep_models
import deep4downscaling.deep.xai as deep_xai

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
xai_path = paths['xai']

##### Configuration #####
var_target = sys.argv[1] # tasmin, tasmax, pr
num_ensemble = 1 # Member of the ensemble to use to compute the XAI techniques
years_train = ('1980', '2010'); years_test = ('2011', '2020') # Train and test sets
#########################

#### Preprocess the data as done for test prediction ####

# Get the avaialble device for training the model
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Load predictor
predictor_filename = f'{data_path}/ERA5_NorthAtlanticRegion_1-5dg_full.nc'
predictor = xr.open_dataset(predictor_filename).load()

# Load predictand
predictand_filename = f'{data_path}/{var_target}_AEMET.nc'
predictand = xr.open_dataset(predictand_filename).load()

# Remove days with nans in the predictor
predictor = trans.remove_days_with_nans(predictor)

# Align both datasets in time
predictor, predictand = trans.align_datasets(predictor, predictand, 'time')

# Subset the training and test set
x_train = predictor.sel(time=slice(*years_train))
y_train = predictand.sel(time=slice(*years_train))

x_test = predictor.sel(time=slice(*years_test))
y_test = predictand.sel(time=slice(*years_test))

# Standardize the predictors
x_train_stand = trans.standardize(data_ref=x_train, data=x_train)
x_test_stand = trans.standardize(data_ref=x_train, data=x_test)

# Mask the predictors
# This is done to handle the NaNs of some GCMs from CMIP6
mask_predictors = xr.open_dataset(f'{mask_path}/CMIP6-1.5_level850-NAs_mask.nc').load()
mask_predictors = mask_predictors.mean('time')

x_train_stand = x_train_stand * mask_predictors['clt']
x_test_stand = x_test_stand * mask_predictors['clt']

# Compute mask of non-nans values of the predictand
# This is required to reshape the prediction of the DL model to a valid format
y_mask = trans.compute_valid_mask(y_train) 

# Stack in one dimension (gridpoint) and remove nans following y_mask
y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))
y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))

y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True) # Select the non-nan gridpoints
y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                         drop=True) # Filter y_train w.r.t. y_mask

# Convert data from xarray to numpy (test set)
x_test_stand_arr = trans.xarray_to_numpy(x_test_stand)
y_train_arr = trans.xarray_to_numpy(y_train_stack_filt)

# Get the name of the model
model_name = f'deepESD_{var_target}_ens{num_ensemble}'

# Select the proper DeepESD model and initialize it
if var_target == 'pr':
    dl_model = deep_models.DeepESDpr
    model = dl_model(x_shape=x_test_stand_arr.shape,
                     y_shape=y_train_arr.shape,
                     filters_last_conv=1,
                     stochastic=False)
elif var_target in ('tasmin', 'tasmax'):
    dl_model = deep_models.DeepESDtas
    model = dl_model(x_shape=x_test_stand_arr.shape,
                        y_shape=y_train_arr.shape,
                        filters_last_conv=1,
                        stochastic=False)

# Load the weights
model.load_state_dict(torch.load(f'{model_path}/{model_name}.pt'))
####

#### Compute XAI metrics for the test set ####

# ASM
asm = deep_xai.compute_asm(data=x_test_stand,
                           mask=y_mask.copy(deep=True),
                           model=model, device=device,
                           xai_method=captum.attr.Saliency(model),
                           batch_size=1024,
                           postprocess=True)
asm.to_netcdf(f'{xai_path}/ASM_{model_name}_test_period.nc')

# SDM
sdm = deep_xai.compute_sdm(data=x_test_stand,
                           mask=y_mask.copy(deep=True),
                           var_target=var_target,
                           model=model, device=device,
                           xai_method=captum.attr.Saliency(model),
                           batch_size=1024,
                           postprocess=True)
sdm.to_netcdf(f'{xai_path}/SDM_{model_name}_test_period.nc')
####