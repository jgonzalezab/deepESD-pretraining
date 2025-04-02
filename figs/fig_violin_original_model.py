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
var_target = 'tasmin'
num_ensemble = 1
years_train = ('1980', '2010'); years_test = ('2011', '2020') # Train and test sets
#########################

# Load predictand
predictand_filename = f'{data_path}/{var_target}_AEMET.nc'
predictand = xr.open_dataset(predictand_filename).load()

# Subset into training and test sets
y_test = predictand.sel(time=slice(*years_test))

# Define the functions
def bias_tnn(target, pred):
    return metrics.bias_tnn(target=target, pred=pred, var_target=var_target)

def bias_txx(target, pred):
    return metrics.bias_txx(target=target, pred=pred, var_target=var_target)

def bias_P02(target, pred):
    return metrics.bias_quantile(target=target, pred=pred, quantile=0.02, var_target=var_target)

def bias_mean(target, pred):
    return metrics.bias_mean(target=target, pred=pred, var_target=var_target)

def bias_P98(target, pred):
    return metrics.bias_quantile(target=target, pred=pred, quantile=0.98, var_target=var_target)

def ratio_sd(target, pred):
    return metrics.ratio_std(target=target, pred=pred, var_target=var_target)

def bias_rel_mean(target, pred):
    return metrics.bias_rel_mean(target=target, pred=pred, var_target=var_target)

def bias_rel_P99(target, pred):
    return metrics.bias_rel_quantile(target=target, pred=pred, quantile=0.99, var_target=var_target)

def bias_rel_rx1day(target, pred):
    return metrics.bias_rel_rx1day(target=target, pred=pred, var_target=var_target)

def bias_rel_SDII(target, pred):
    return metrics.bias_rel_SDII(target=target, pred=pred, var_target=var_target)

def ratio_interannual_var(target, pred):
    return metrics.ratio_interannual_var(target=target, pred=pred, var_target=var_target)

def rmse(target, pred):
    return metrics.rmse(target=target, pred=pred, var_target=var_target)

def rmse_wet(target, pred):
    return metrics.rmse_wet(target=target, pred=pred, var_target=var_target)

# Create the dictionary
if var_target == 'tasmin':
    metrics_func = {
        'Bias P02': bias_P02,
        'Bias Mean': bias_mean,
        'Bias P98': bias_P98,
        'Bias TNn': bias_tnn,
        'Ratio Std. Dev.': ratio_sd,
        'RMSE': rmse
    }

elif var_target == 'tasmax':
    metrics_func = {
        'Bias P02': bias_P02,
        'Bias Mean': bias_mean,
        'Bias P98': bias_P98,
        'Bias TXx': bias_tnn,
        'Ratio Std. Dev.': ratio_sd,
        'RMSE': rmse
    }

elif var_target == 'pr':
    metrics_func = {
        'Rel. Bias Mean': bias_rel_mean,
        'Rel. Bias P99': bias_rel_P99,
        'Rel. Bias Rx1day': bias_rel_rx1day,
        'Rel. Bias SDII': bias_rel_SDII,
        'Ratio Interannual Var.': ratio_interannual_var,
        'RMSE (wet days)': rmse_wet
    }

# Define violin limits
metrics_limits = {
    'Bias P02': (-2, 2),
    'Bias Mean': (-2, 2),
    'Bias P98': (-2, 2),
    'Bias TNn': (-2, 2),
    'Bias TXx': (-2, 2),
    'Ratio Std. Dev.': (0.9, 1.1),
    'Rel. Bias Mean': (-75, 75),
    'Rel. Bias P99': (-75, 75),
    'Rel. Bias Rx1day': (-75, 75),
    'Rel. Bias SDII': (-75, 75),
    'Ratio Interannual Var.': (0.4, 1.6),
    'RMSE': (0, 3),
    'RMSE (wet days)': (2, 18)
}

# Compute plot
n_rows, n_cols = 3, 3
fig = plt.figure(figsize=(11, 10))
subplot_idx = 1
pos_idx = 1

for metric in metrics_func.keys():

    ax = fig.add_subplot(n_rows, n_cols, subplot_idx)
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.7)
    ax.set_ylim(metrics_limits[metric][0],
                metrics_limits[metric][1]) 
    ax.set_title(metric, fontsize=14)

    file_name = f'deepESD_{var_target}_ens{num_ensemble}_preds_test.nc'
    data_pred = xr.open_dataset(f'{preds_path}/{file_name}')

    metric_value = metrics_func[metric](target=y_test, pred=data_pred)
    metric_value = metric_value[var_target].values
    metric_value = metric_value.flatten()
    metric_value = metric_value[~np.isnan(metric_value)]

    vc = plt.violinplot(metric_value,
                        positions=[pos_idx],
                        showextrema=False, showmedians=True,
                        quantiles=None, points=1000)

    for pc in vc['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('blue')

    for partAxis in ['cmedians']:
        vc[partAxis].set_colors('blue')

    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_xticklabels([])  # Remove x-axis tick labels

    # Introduce minimal separation between the first and second row of subplots
    if subplot_idx == n_cols:
        plt.subplots_adjust(hspace=0.3)  # Adjust vertical spacing

    subplot_idx = subplot_idx + 1

fig_name = f'violin_original_model_{var_target}.pdf'
plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight')
plt.close()