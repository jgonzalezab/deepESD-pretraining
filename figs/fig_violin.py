import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/deep4downscaling/')
import deep4downscaling.metrics as metrics
import deep4downscaling.trans as trans

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_path = paths['data']
preds_path = paths['data_preds']
figs_path = paths['figs']
data_stations_eca = paths['data_stations_eca']

##### Configuration #####
var_target = 'pr'
var_target_eca = 'tn' if var_target == 'tasmin' else 'tx' if var_target == 'tasmax' else 'rr'
num_total_ensemble = 10
training_routine_list = ['original', 'pretrained', 'pretrained_finetuning']
years_train = ('2009', '2018'); years_test = ('2019', '2021') # Train and test sets
#########################

# Load predictand for original model
predictand_original_filename = f'{data_path}/{var_target}_AEMET.nc'
predictand_original = xr.open_dataset(predictand_original_filename).load()
y_test_original = predictand_original.sel(time=slice(*years_test))

# Load predictand for comparison models
predictand_comparison_filename = f'{data_stations_eca}/ECA_blend_{var_target_eca}.nc'  
predictand_comparison = xr.open_dataset(predictand_comparison_filename).load()
predictand_comparison = predictand_comparison.drop_vars(('elevation', 'country'))
predictand_comparison = predictand_comparison.rename({var_target_eca: var_target})

y_train_comparison = predictand_comparison.sel(time=slice(*years_train))
y_test_comparison = predictand_comparison.sel(time=slice(*years_test))

# Remove stations with no values in the training period
y_train_comparison, y_test_comparison = trans.remove_stations_with_nans(y_train_comparison, y_test_comparison)

# Define the functions
def bias_tnn(target, pred):
    return metrics.bias_tnn(target=target, pred=pred, var_target=var_target)

def bias_txx(target, pred):
    return metrics.bias_txx(target=target, pred=pred, var_target=var_target)

def bias_mean(target, pred):
    return metrics.bias_mean(target=target, pred=pred, var_target=var_target)

def rmse(target, pred):
    return metrics.rmse(target=target, pred=pred, var_target=var_target)

def rmse_wet(target, pred):
    return metrics.rmse_wet(target=target, pred=pred, var_target=var_target)

def bias_dry_days(target, pred):
    return metrics.bias_dry_days(target=target, pred=pred, var_target=var_target)

def bias_rx1day(target, pred):
    return metrics.bias_rx1day(target=target, pred=pred, var_target=var_target)

def bias_SDII(target, pred):
    return metrics.bias_SDII(target=target, pred=pred, var_target=var_target)

# Create the dictionary
if var_target == 'tasmin':
    metrics_func = {
        'RMSE': rmse,
        'Bias Mean': bias_mean,
        'Bias TNn': bias_tnn,
    }

elif var_target == 'tasmax':
    metrics_func = {
        'RMSE': rmse,
        'Bias Mean': bias_mean,
        'Bias TXx': bias_txx,
    }

elif var_target == 'pr':
    metrics_func = {
        'RMSE (wet days)': rmse_wet,
        'Bias Freq. Dry Days': bias_dry_days,
        'Bias SDII': bias_SDII,
        'Bias Rx1day': bias_rx1day,
    }

# Define units for each metric
metrics_units = {
    'Bias Mean': '°C',
    'Bias TNn': '°C',
    'Bias TXx': '°C',
    'RMSE': '°C',
    'RMSE (wet days)': 'mm/day',
    'Bias Freq. Dry Days': '%',
    'Bias SDII': 'mm/day',
    'Bias Rx1day': 'mm/day',
}

# Define violin limits
metrics_limits = {
    'Bias Mean': (-4, 4),
    'Bias TNn': (-4, 4),
    'Bias TXx': (-4, 4),
    'RMSE': (0, 4),
    'RMSE (wet days)': (0, 30),
    'Bias Freq. Dry Days': (-20, 20),
    'Bias SDII': (-10, 10),
    'Bias Rx1day': (-90, 90),
}

# Routine names
routines_names = {
    'original': 'Full-training',
    'pretrained': 'Partial fine-tuning',
    'pretrained_finetuning': 'Full fine-tuning'
}

# Compute plot
subplot_width = 4
subplot_height = 4

if var_target in ('tasmin', 'tasmax'):
    n_rows, n_cols = 1, 3
elif var_target == 'pr':
    n_rows, n_cols = 1, 4

fig = plt.figure(figsize=(n_cols * subplot_width, n_rows * subplot_height))

subplot_idx = 1

for metric in metrics_func.keys():

    ax = fig.add_subplot(n_rows, n_cols, subplot_idx)
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.7)
    ax.set_ylim(metrics_limits[metric][0],
                metrics_limits[metric][1]) 
    ax.set_title(metric, fontsize=16)

    # Set y-axis label with units
    unit = metrics_units.get(metric, '')
    ax.set_ylabel(unit, fontsize=14)

    pos_idx = 1

    # Plot comparison violins
    for routine_idx, routine_value in enumerate(training_routine_list):

        metric_value_members = []
        for num_ensemble in range(1, num_total_ensemble+1):

            file_name = f'deepESD_stations_eca_{routine_value}_{var_target}_ens{num_ensemble}_preds_test.nc'
            data_pred = xr.open_dataset(f'{preds_path}/{file_name}')

            if var_target == 'pr':
                data_pred = data_pred.where(data_pred[var_target] > 0, other=0)

            metric_value = metrics_func[metric](target=y_test_comparison[[var_target]], pred=data_pred[[var_target]])
            metric_value = metric_value[var_target].values
            metric_value = metric_value.flatten()
            metric_value = metric_value[~np.isnan(metric_value)]
            metric_value_members.append(metric_value)

        # Min and max of the ensemble metrics
        ensemble_members_median = [np.median(x) for x in metric_value_members]
        ensemble_members_median_min = np.min(ensemble_members_median)
        ensemble_members_median_max = np.max(ensemble_members_median)

        # Plot violin
        member_to_plot = metric_value_members[0]
        bp = ax.boxplot(member_to_plot,
                        positions=[pos_idx],
                        patch_artist=True,
                        widths=0.6)

        for box in bp['boxes']:
            box.set_facecolor('lightblue')
            box.set_edgecolor('lightblue')

        for median in bp['medians']:
            median.set_color('gray')

        # Plot medians
        ax.hlines(y=ensemble_members_median_max,
                  xmin=pos_idx-0.2,
                  xmax=pos_idx+0.2,
                  color='blue',
                  linestyle='dashed',
                  zorder=3)

        ax.hlines(y=ensemble_members_median_min,
                  xmin=pos_idx-0.2,
                  xmax=pos_idx+0.2,
                  color='blue',
                  linestyle='dashed',
                  zorder=3)

        pos_idx = pos_idx + 1

    xticks = [pos for pos in range(1, len(training_routine_list)+1)]
    ax.set_xticks(xticks)
    plot_labels = [routines_names[name] for name in training_routine_list]
    ax.set_xticklabels(plot_labels, rotation=45, ha='right')

    subplot_idx = subplot_idx + 1

plt.tight_layout()
fig_name = f'violin_merged_comparison_{var_target}.pdf'
plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight')
plt.close()
