import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling')
import deep4downscaling.metrics as metrics

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_path = paths['data']
preds_path = paths['data_preds']
figs_path = paths['figs']

##### Configuration #####
var_target = 'pr' # tasmin, tasmax, pr
num_ensemble = 1 
years_train = ('1980', '2010'); years_test = ('2011', '2020') # Train and test sets

# Training routine of the deep learning model
training_routine_list = ['original', 'pretrained', 'pretrained_finetuning']

nan_perc_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
#########################

# Load predictand
predictand_filename = f'{data_path}/PENINSULAYBALEARES_{var_target}_19750101-20201231.nc'
predictand = xr.open_dataset(predictand_filename).load()
predictand = predictand.drop_vars(('projection', 'alt')) # Remove projection and altitude variables

# Get test data
y_test = predictand.sel(time=slice(*years_test))

# Metric to plot
def metric_func(target, pred, var_target=var_target):
    return metrics.rmse(target=target, pred=pred, var_target=var_target)

def aggregation_func(data): return data.mean().values.item()

# Set names
names_dict = {'original': 'No pre-training',
              'pretrained': 'Pre-trained',
              'pretrained_finetuning': 'Pre-trained w/ fine-tuning'}

# Compute metric
routine_time_evolution = {}
for training_routine in training_routine_list:
    print(f'Computing metric for {training_routine} model...')

    routine_time_evolution[training_routine] = []
    for nan_perc in nan_perc_list:

        # Load prediction
        file_name = f'deepESD_stations_{training_routine}_{var_target}_ens{num_ensemble}_mask{nan_perc}_preds_test.nc'
        data_pred = xr.open_dataset(f'{preds_path}/{file_name}').load()

        # Compute metric
        metric_value = metric_func(target=y_test[[var_target]],
                                   pred=data_pred[[var_target]])
        metric_value = aggregation_func(metric_value[var_target])

        # Append to dict 
        routine_time_evolution[training_routine].append(metric_value)

# Compute plot
colors_routine = {'original': 'blue',
                  'pretrained': 'green',
                  'pretrained_finetuning': 'purple'}

for training_routine in training_routine_list:

    plt.plot(range(len(nan_perc_list)),
             routine_time_evolution[training_routine],
             color=colors_routine[training_routine],
             linestyle='solid', marker='.',
             label=names_dict[training_routine])

plt.xticks(ticks=range(len(nan_perc_list)), labels=nan_perc_list)

plt.legend()
plt.xlabel('Missing data (%)')
plt.ylabel(var_target)
plt.grid(True)

plt.savefig(f'{figs_path}/missing_data_experiment_{var_target}.pdf',
            bbox_inches='tight')
plt.close()