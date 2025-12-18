# Plot time series for a single station and variable, comparing
# the different training routines used in `fig_violin.py`.

import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_stations_eca = paths['data_stations_eca']
preds_path = paths['data_preds']
figs_path = paths['figs']

##### Configuration #####
# Target variable to plot: 'tasmin', 'tasmax' or 'pr'
var_target = 'pr'
var_target_eca = 'tn' if var_target == 'tasmin' else 'tx' if var_target == 'tasmax' else 'rr'

# Index of the station to plot (0-based).
# Change this index to select a different station.
station_index = 2

# Train / test periods (must match the ones used for training)
years_train = ('1980', '2010')
years_test = ('2011', '2020')

# Training routines to compare (as in `fig_violin.py`)
training_routine_list = ['original', 'pretrained', 'pretrained_finetuning']

# Ensemble member to use for the time series (same for all routines)
ensemble_member = 1
#########################

# Variable units
var_units = {
    'tasmin': '°C',
    'tasmax': '°C',
    'pr': 'mm/day',
}


def load_observations():
    """Load ECA stations dataset and subset the test period."""
    filename = f'{data_stations_eca}/ECA_blend_{var_target_eca}.nc'
    ds = xr.open_dataset(filename).load()

    # Remove auxiliary variables and rename to match target
    drop_vars = [v for v in ('elevation', 'country') if v in ds]
    if drop_vars:
        ds = ds.drop_vars(drop_vars)
    ds = ds.rename({var_target_eca: var_target})

    # Subset to test period
    ds_test = ds.sel(time=slice(*years_test))
    return ds_test


def load_predictions(routine):
    """Load predictions for a given training routine and subset test period."""
    file_name = (
        f'deepESD_stations_eca_{routine}_{var_target}_ens{ensemble_member}_preds_test.nc'
    )
    ds_pred = xr.open_dataset(f'{preds_path}/{file_name}').load()
    ds_pred = ds_pred.sel(time=slice(*years_test))
    return ds_pred


def main():
    # Load observations (ECA stations)
    obs = load_observations()

    # Extract station metadata
    try:
        station_coord = obs['station'].isel(station=station_index).item()
    except (KeyError, ValueError):
        raise RuntimeError(
            "Could not find coordinate 'station' in observations. "
            "Check the structure of the ECA dataset."
        )

    lat = float(obs['lat'].isel(station=station_index).item())
    lon = float(obs['lon'].isel(station=station_index).item())

    station_name = None
    if 'station_name' in obs.data_vars:
        station_name = str(obs['station_name'].isel(station=station_index).item())

    # Observed time series for selected station
    obs_ts = obs[var_target].isel(station=station_index)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot observations
    ax.plot(
        obs_ts['time'].values,
        obs_ts.values,
        label='Observations',
        color='black',
        linewidth=1.5,
    )

    # Colors for different training routines
    routine_colors = {
        'original': 'tab:blue',
        'pretrained': 'tab:orange',
        'pretrained_finetuning': 'tab:green',
    }

    # Human-readable names (same style as in `fig_violin.py`)
    routines_names = {
        'original': 'No pre-training',
        'pretrained': 'Pre-training',
        'pretrained_finetuning': 'Fine-tuning',
    }

    # Plot predictions for each training routine
    for routine in training_routine_list:
        ds_pred = load_predictions(routine)

        if var_target not in ds_pred:
            raise RuntimeError(
                f"Variable '{var_target}' not found in predictions for routine '{routine}'."
            )

        pred_ts = ds_pred[var_target].isel(station=station_index)

        ax.plot(
            pred_ts['time'].values,
            pred_ts.values,
            label=routines_names.get(routine, routine),
            color=routine_colors.get(routine, None),
            linewidth=1.2,
            alpha=0.9,
        )

    # Axes labels and title
    units = var_units.get(var_target, '')
    ylabel = f'{var_target} ({units})' if units else var_target

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Time', fontsize=12)

    # Build station label for the title
    station_label_parts = [f'Station {station_coord}']
    if station_name is not None and station_name.strip():
        station_label_parts.append(f'({station_name})')
    station_label = ' '.join(station_label_parts)

    ax.set_title(
        f'{var_target.upper()} time series - {station_label}\n'
        f'Test period: {years_test[0]}-{years_test[1]} (lat={lat:.2f}, lon={lon:.2f})',
        fontsize=14,
    )

    # Legend and layout
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    # Save figure
    fig_name = (
        f'timeseries_{var_target}_station{station_coord}_ens{ensemble_member}_test.pdf'
    )
    plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

