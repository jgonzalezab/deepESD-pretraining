import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/deep4downscaling/')
import deep4downscaling.metrics as metrics

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_path = paths['data_asturias']
preds_path = paths['data_preds']
figs_path = paths['figs_asturias']

##### Configuration #####
var_target = 'pr'
num_total_ensemble = 1
training_routine_list = ['original', 'pretrained', 'pretrained_finetuning']
years_train = ('1980', '2010'); years_test = ('2011', '2020') # Train and test sets
#########################

# Load gridded dataset
gridded_filename = f'{data_path}/{var_target}/day/*.nc'
gridded = xr.open_mfdataset(gridded_filename).load()
gridded = gridded.assign_coords({'time': gridded.indexes['time'].normalize()})
gridded_test = gridded.sel(time=slice(*years_test))

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
    'Bias Mean': (-3, 3),
    'Bias TNn': (-3, 3),
    'Bias TXx': (-3, 3),
    'RMSE': (0, 4),
    'RMSE (wet days)': (0, 20),
    'Bias Freq. Dry Days': (-20, 20),
    'Bias SDII': (-5, 5),
    'Bias Rx1day': (-40, 40),
}

# Routine names
routines_names = {
    'original': 'No pre-training',
    'pretrained': 'Pre-training',
    'pretrained_finetuning': 'Fine-tuning'
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

    # Plot violins
    for routine_idx, routine_value in enumerate(training_routine_list):

        metric_value_members = []
        for num_ensemble in range(1, num_total_ensemble+1):

            file_name = f'deepESD_grid_asturias_{routine_value}_{var_target}_ens{num_ensemble}_preds_test.nc'
            data_pred = xr.open_dataset(f'{preds_path}/{file_name}')

            metric_value = metrics_func[metric](target=gridded_test[[var_target]], pred=data_pred[[var_target]])
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
        vc = plt.violinplot(member_to_plot,
                            positions=[pos_idx],
                            showextrema=False, showmedians=True,
                            quantiles=None, points=1000)

        for pc in vc['bodies']:
            pc.set_facecolor('blue')
            pc.set_edgecolor('blue')

        for partAxis in ['cmedians']:
            vc[partAxis].set_colors('blue')

        # Plot medians
        ax.hlines(y=ensemble_members_median_max,
                  xmin=pos_idx-0.2,
                  xmax=pos_idx+0.2,
                  color='black',
                  linestyle='dashed')

        ax.hlines(y=ensemble_members_median_min,
                  xmin=pos_idx-0.2,
                  xmax=pos_idx+0.2,
                  color='black',
                  linestyle='dashed')

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
