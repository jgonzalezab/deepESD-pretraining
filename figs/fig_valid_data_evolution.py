import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/')
import src.utils as utils

sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/deep4downscaling')
import deep4downscaling.trans as trans

# Load paths
paths = json.load(open('/gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/configs/paths.json'))
data_path = paths['data']
figs_path = paths['figs']
data_stations_eca = paths['data_stations_eca']

##### Configuration #####
var_target = 'tasmin'
var_target_eca = 'tn' if var_target == 'tasmin' else 'tx' if var_target == 'tasmax' else 'rr'
#########################

# Periods to plot
periods = {'TRAIN': ('2010', '2020'),
           'TEST': ('2021', '2023')}

# Load stations dataset (full period)
stations_filename = f'{data_stations_eca}/ECA_blend_{var_target_eca}.nc'
stations = xr.open_dataset(stations_filename)
stations = stations.drop_vars(('elevation', 'country'))  # Remove projection and altitude variables
stations = stations.rename({var_target_eca: var_target})  # Rename variable to match target
stations = stations.load()

# Remove stations with no values in the training period
_, stations = trans.remove_stations_with_nans(stations.sel(time=slice(*periods['TRAIN'])), 
                                              stations)

def plot_evolution(dataset, period_name, period_dates):
    # Subset for the period
    ds_period = dataset.sel(time=slice(*period_dates))
    
    # Compute availability per year for each station
    # We sum the number of non-null values per year and divide by 365 to match fig_available_data.py
    # Grouping by year is more robust than resampling across different xarray/pandas versions
    availability_per_year = ds_period[var_target].notnull().groupby('time.year').sum(dim='time') / 365
    
    # Aggregated availability across all stations
    aggregated_availability = availability_per_year.sum(dim='station')
    
    # Time labels (years)
    years = availability_per_year.year.values
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: Per station evolution
    for i in range(len(availability_per_year.station)):
        ax1.plot(years, availability_per_year.isel(station=i).values, alpha=0.3, linewidth=0.5)
    
    ax1.set_title(f'Data Availability Evolution per Station ({period_name}) - {var_target}', fontsize=14)
    ax1.set_ylabel('Available Data (Years/Year)', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 2: Aggregated evolution
    ax2.plot(years, aggregated_availability.values, color='tab:blue', linewidth=2, marker='o')
    ax2.set_title(f'Aggregated Data Availability Evolution ({period_name}) - {var_target}', fontsize=14)
    ax2.set_ylabel('Total Available Data (Station-Years/Year)', fontsize=12)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure years are shown as integers on x-axis
    ax2.set_xticks(years)
    if len(years) > 15:
        plt.setp(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    fig_name = f'data_evolution_{var_target}_{period_name.lower()}.pdf'
    plt.savefig(f'{figs_path}/{fig_name}', bbox_inches='tight')
    print(f'Saved figure to {figs_path}/{fig_name}')
    plt.close(fig)

for period_name, period_dates in periods.items():
    plot_evolution(stations, period_name, period_dates)
