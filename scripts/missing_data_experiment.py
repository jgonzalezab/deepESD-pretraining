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

# Training routine of the deep learning model
training_routine = sys.argv[2] # original, pretrained, pretrained_finetuning 
# original: The model is trained with no finetuning
# pretrained: Convolutional layers are initialized (and frozen) from the original model trained on grid data (original.py)
# pretrained_finetuning: Convolutional layers are initialized (and finetuned) from the original model trained on grid data (original.py)

nan_perc = float(sys.argv[3]) # Percentage of points of the predictand to mask with nans
#########################

#### Train the model ####

# Get the avaialble device for training the model
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Load predictor
predictor_filename = f'{data_path}/ERA5_NorthAtlanticRegion_1-5dg_full.nc'
predictor = xr.open_dataset(predictor_filename).load()

# Load predictand
predictand_filename = f'{data_path}/PENINSULAYBALEARES_{var_target}_19750101-20201231.nc'
predictand = xr.open_dataset(predictand_filename).load()
predictand = predictand.drop_vars(('projection', 'alt')) # Remove projection and altitude variables

# Remove days with nans in the predictor
predictor = trans.remove_days_with_nans(predictor)

# Align both datasets in time
predictor, predictand = trans.align_datasets(predictor, predictand, 'time')

# Subset the training set
x_train = predictor.sel(time=slice(*years_train))
y_train = predictand.sel(time=slice(*years_train))

# Randomly mask (with nans) the predictand (train set)
y_train = utils.nan_masking_xarray(data=y_train, var_target=var_target,
                                   nan_perc=nan_perc)

# Standardize the predictors
x_train_stand = trans.standardize(data_ref=x_train, data=x_train)

# Mask the predictors
# This is done to handle the NaNs of some GCMs from CMIP6
mask_predictors = xr.open_dataset(f'{mask_path}/CMIP6-1.5_level850-NAs_mask.nc').load()
mask_predictors = mask_predictors.mean('time')
x_train_stand = x_train_stand * mask_predictors['clt']

# If downscaling precipitation, initialize the Asym loss function
if var_target == 'pr':
    loss_function = deep_loss.Asym(ignore_nans=True, # We need to ignore the value of masked points
                                   asym_path=asym_path,
                                   appendix='stations')

    if loss_function.parameters_exist():
        loss_function.load_parameters()
    else:
        loss_function.compute_parameters(data=y_train,
                                        var_target=var_target)

# Convert data from xarray to numpy
x_train_stand_arr = trans.xarray_to_numpy(x_train_stand)
y_train_arr = trans.xarray_to_numpy(y_train,
                                    ignore_vars=['station_id'])

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
model_name = f'deepESD_stations_{training_routine}_{var_target}_ens{num_ensemble}_mask{nan_perc}'

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

# Set training routine
# By default, we load the weights for the num_ensemble = 1
if training_routine == 'original':
    pass

elif training_routine == 'pretrained':
    # Load pretrained weights
    weights_original = torch.load(f'{model_path}/deepESD_{var_target}_ens1.pt')

    # Assign to convolutional layers of the new model
    with torch.no_grad():
        for (name_original, param_original), (name_model, param_model) in \
            zip(weights_original.items(), model.named_parameters()):
            if (name_original == name_model) and ('conv' in name_original):
                param_model.data.copy_(param_original)

    # Freeze the convolutional layers
    for name, param in model.named_parameters():
        if 'conv' in name:
            param.requires_grad = False

elif training_routine == 'pretrained_finetuning':
    # Load pretrained weights
    weights_original = torch.load(f'{model_path}/deepESD_{var_target}_ens1.pt')

    # Assign to convolutional layers of the new model
    with torch.no_grad():
        for (name_original, param_original), (name_model, param_model) in \
            zip(weights_original.items(), model.named_parameters()):
            if (name_original == name_model) and ('conv' in name_original):
                param_model.data.copy_(param_original)

# Configure the training of the DL model
num_epochs = 10000
patience_early_stopping = 60

learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate)

# If downscaling temperature, select the proper loss function
if var_target in ('tasmin', 'tasmax'):
    loss_function = deep_loss.MseLoss(ignore_nans=True) # We need to ignore the value of masked points

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

# Mask the predictors
# This is done to handle the NaNs of some GCMs from CMIP6
x_test_stand = x_test_stand * mask_predictors['clt']

# Create output template
y_train = predictand.sel(time=slice(*years_train))
template = y_train.mean('time')

# Compute and save the predictions on the test set
pred_test = deep_pred.compute_preds_standard(x_data=x_test_stand, model=model,
                                             device=device, var_target=var_target,
                                             template=template)

file_name = f'{model_name}_preds_test'
pred_test.to_netcdf(f'{preds_path}/{file_name}.nc')
####
