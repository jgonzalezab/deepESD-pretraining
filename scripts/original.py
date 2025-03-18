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
num_ensemble = 1 # Member of the ensemble of deep learning models to run
years_train = ('1980', '2010'); years_test = ('2011', '2020') # Train and test sets

gcm_name = 'EC-Earth3-Veg_r1i1p1f1' # GCM to downscale
scenario = 'ssp370' # Scenario to downscale
years_hist_ccs = ('01-01-1980', '31-12-2014') # Interval for the historic period when computing the Cs
#########################

#### Train the model ####

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

# Subset the training set
x_train = predictor.sel(time=slice(*years_train))
y_train = predictand.sel(time=slice(*years_train))

# Standardize the predictors
x_train_stand = trans.standardize(data_ref=x_train, data=x_train)

# Mask the predictors
# This is done to handle the NaNs of some GCMs from CMIP6
mask_predictors = xr.open_dataset(f'{mask_path}/CMIP6-1.5_level850-NAs_mask.nc').load()
mask_predictors = mask_predictors.mean('time')
x_train_stand = x_train_stand * mask_predictors['clt']

# Compute mask of non-nans values of the predictand
# This is required to reshape the prediction of the DL model to a valid format
y_mask = trans.compute_valid_mask(y_train) 

# Stack in one dimension (gridpoint) and remove nans following y_mask
y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))
y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))

y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True) # Select the non-nan gridpoints
y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                         drop=True) # Filter y_train w.r.t. y_mask

# If downscaling precipitation, initialize the Asym loss function
if var_target == 'pr':
    loss_function = deep_loss.Asym(ignore_nans=False,
                                   asym_path=asym_path)

    if loss_function.parameters_exist():
        loss_function.load_parameters()
    else:
        loss_function.compute_parameters(data=y_train_stack_filt,
                                        var_target=var_target)

# Convert data from xarray to numpy
x_train_stand_arr = trans.xarray_to_numpy(x_train_stand)
y_train_arr = trans.xarray_to_numpy(y_train_stack_filt)

# Create Dataset
train_dataset = deep_utils.StandardDataset(x=x_train_stand_arr,
                                           y=y_train_arr)

# Split into training and validation sets
train_dataset, valid_dataset = random_split(train_dataset,
                                            [0.9, 0.1])

# Create DataLoaders
batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True)

# Name the model
model_name = f'deepESD_{var_target}_ens{num_ensemble}'

# Select the proper DeepESD model and initialize it
if var_target == 'pr':
    dl_model = deep_models.DeepESDpr
    model = dl_model(x_shape=x_train_stand_arr.shape,
                     y_shape=y_train_arr.shape,
                     filters_last_conv=1,
                     stochastic=False)
elif var_target in ('tasmin', 'tasmax'):
    dl_model = deep_models.DeepESDtas
    model = dl_model(x_shape=x_train_stand_arr.shape,
                     y_shape=y_train_arr.shape,
                     filters_last_conv=1,
                     stochastic=False)

# Configure the training of the DL model
num_epochs = 10000
patience_early_stopping = 60

learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate)

# If downscaling temperature, select the proper loss function
if var_target in ('tasmin', 'tasmax'):
    loss_function = deep_loss.MseLoss(ignore_nans=False)

# If downscaling precipitation, prepare the ASYM loss function
if var_target == 'pr':
    loss_function.prepare_parameters(device=device)

train_loss, val_loss = deep_train.standard_training_loop(model=model, model_name=model_name, model_path=model_path,
                                                         device=device, num_epochs=num_epochs,
                                                         loss_function=loss_function, optimizer=optimizer,
                                                         train_data=train_dataloader, valid_data=valid_dataloader,
                                                         patience_early_stopping=patience_early_stopping)

# Save losses
train_loss, val_loss = np.array(train_loss), np.array(val_loss)
np.save(f'{loss_path}/train_{model_name}.npy', arr=train_loss)
np.save(f'{loss_path}/val_{model_name}.npy', arr=val_loss)
####

#### Compute predictions on test set ####

# Load model (just in case)
model.load_state_dict(torch.load(f'{model_path}/{model_name}.pt'))

# Subset the test set
x_test = predictor.sel(time=slice(*years_test))

# Standardize the predictors
x_test_stand = trans.standardize(data_ref=x_train, data=x_test)

# Compute and save the predictions on the test set
pred_test = deep_pred.compute_preds_standard(x_data=x_test_stand, model=model,
                                             device=device, var_target=var_target,
                                             mask=y_mask)

file_name = f'{model_name}_preds_test'
pred_test.to_netcdf(f'{preds_path}/{file_name}.nc')
####

#### Downscale the GCM ####

# Load the historical GCM
gcm_hist = utils.load_gcm(gcm=gcm_name,
                          scenario='historical',
                          gcm_path=gcm_raw_path)

# Load the future GCM
gcm_fut = utils.load_gcm(gcm=gcm_name, scenario=scenario,
                         gcm_path=gcm_raw_path)

# Bias-correct the historical GCM
gcm_hist_corrected = trans.scaling_delta_correction(data=gcm_hist,
                                                    gcm_hist=gcm_hist, obs_hist=x_train)

# Bias-correct the future GCM independently for each future period
future_periods = {'fut_1': ('01-01-2015', '31-12-2040'),
                  'fut_2': ('01-01-2041', '31-12-2070'),
                  'fut_3': ('01-01-2071', '31-12-2100')}

gcm_fut_list = []
for period in future_periods:
    gcm_fut_period = gcm_fut.sel(time=slice(*future_periods[period]))
    gcm_fut_period_corrected = trans.scaling_delta_correction(data=gcm_fut_period,
                                                              gcm_hist=gcm_hist, obs_hist=x_train)
    gcm_fut_list.append(gcm_fut_period_corrected)

gcm_fut_corrected = xr.merge(gcm_fut_list)

# Standardize the GCM predictors
gcm_hist_corrected_stand = trans.standardize(data_ref=x_train, data=gcm_hist_corrected)
gcm_fut_corrected_stand = trans.standardize(data_ref=x_train, data=gcm_fut_corrected)

# Mask the predictors
gcm_hist_corrected_stand = gcm_hist_corrected_stand.fillna(value=0) # Convert nans to 0
gcm_hist_corrected_stand = gcm_hist_corrected_stand * mask_predictors['clt']

gcm_fut_corrected_stand = gcm_fut_corrected_stand.fillna(value=0) # Convert nans to 0
gcm_fut_corrected_stand = gcm_fut_corrected_stand * mask_predictors['clt']

# Get GCM name
gcm_model_name, gcm_run = gcm_name.split('_')

# Compute and save the predictions for the historical GCM
pred_hist = deep_pred.compute_preds_standard(x_data=gcm_hist_corrected_stand, model=model,
                                             device=device, var_target=var_target,
                                             mask=y_mask)
file_name = f'{gcm_model_name}_historical_{gcm_run}_{model_name}.nc'
pred_hist.to_netcdf(f'{gcm_proj_path}/{file_name}')

# Compute and save the predictions for the future GCM
pred_fut = deep_pred.compute_preds_standard(x_data=gcm_fut_corrected_stand, model=model,
                                            device=device, var_target=var_target,
                                            mask=y_mask)
file_name = f'{gcm_model_name}_{scenario}_{gcm_run}_{model_name}.nc'
pred_fut.to_netcdf(f'{gcm_proj_path}/{file_name}')
####

#### Compute the CCs ####
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
        ccs_values.to_netcdf(f'{ccs_path}/{gcm_model_name}_{scenario}_{gcm_run}_{period}_{model_name}_{metric}.nc')
####