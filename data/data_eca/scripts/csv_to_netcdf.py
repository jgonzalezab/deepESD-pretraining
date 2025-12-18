#!/usr/bin/env python3
"""
Convert ECA meteorological station CSV data to NetCDF format.

This script reads station metadata from Master.txt and time series data
from CSV files, then creates NetCDF files for each variable containing
all stations and their metadata.
"""

import pandas as pd
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import os

def read_station_metadata(master_file, exclude_stations=None):
    """
    Read station metadata from Master.txt file.
    
    Args:
        master_file: Path to Master.txt
        exclude_stations: List of station IDs to exclude
    
    Returns:
        DataFrame with columns: station_id, name, lon, lat, elevation, country
    """
    if exclude_stations is None:
        exclude_stations = []
    
    stations = []
    
    with open(master_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse the line: station_id, name, lon, lat, elevation, Country=XX
            parts = line.split(',')
            if len(parts) >= 6:
                station_id = parts[0].strip()
                
                # Skip excluded stations
                if station_id in exclude_stations:
                    continue
                
                name = parts[1].strip()
                lon = float(parts[2].strip())
                lat = float(parts[3].strip())
                elevation = float(parts[4].strip())
                country = parts[5].split('=')[1].strip() if '=' in parts[5] else parts[5].strip()
                
                stations.append({
                    'station_id': station_id,
                    'name': name,
                    'lon': lon,
                    'lat': lat,
                    'elevation': elevation,
                    'country': country
                })
    
    return pd.DataFrame(stations)

def read_variable_data(csv_file):
    """
    Read time series data from CSV file.
    
    Returns:
        DataFrame with dates as index and station IDs as columns
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # First column is YYYYMMDD
    df['date'] = pd.to_datetime(df.iloc[:, 0].astype(str), format='%Y%m%d')
    df.set_index('date', inplace=True)
    
    # Drop the original date column name
    df = df.iloc[:, 1:]
    
    # Clean column names (station IDs)
    df.columns = [str(col).strip() for col in df.columns]
    
    # Replace 'NaN' strings with numpy NaN
    df = df.replace('NaN', np.nan)
    
    return df

def create_netcdf(output_file, variable_name, variable_data, stations_metadata, 
                  variable_long_name, variable_units, scale_factor=1.0):
    """
    Create NetCDF file with variable data and station metadata.
    
    Args:
        output_file: Path to output NetCDF file
        variable_name: Short name for the variable (e.g., 'tx', 'rr')
        variable_data: DataFrame with dates as index and station IDs as columns
        stations_metadata: DataFrame with station information
        variable_long_name: Descriptive name for the variable
        variable_units: Units of the variable
        scale_factor: Factor to multiply data by (e.g., 0.1 for tenths to units)
    """
    # Get station IDs from data columns (these are the stations with data)
    data_stations = variable_data.columns.tolist()
    
    # Filter metadata to match data stations and preserve order
    stations_metadata = stations_metadata[stations_metadata['station_id'].isin(data_stations)]
    stations_metadata = stations_metadata.set_index('station_id').loc[data_stations].reset_index()
    
    # Create NetCDF file
    nc = Dataset(output_file, 'w', format='NETCDF4')
    
    # Create dimensions
    time_dim = nc.createDimension('time', len(variable_data))
    station_dim = nc.createDimension('station', len(data_stations))
    string_dim = nc.createDimension('name_strlen', 100)  # For station names
    
    # Create time variable
    times = nc.createVariable('time', 'f8', ('time',))
    times.units = 'days since 1900-01-01 00:00:00'
    times.calendar = 'gregorian'
    times.long_name = 'time'
    times.standard_name = 'time'
    
    # Convert dates to days since 1900-01-01
    reference_date = datetime(1900, 1, 1)
    time_values = [(date - reference_date).days for date in variable_data.index]
    times[:] = time_values
    
    # Create station coordinate variables
    station_ids = nc.createVariable('station_id', str, ('station',))
    station_ids.long_name = 'station identifier'
    station_ids[:] = stations_metadata['station_id'].values
    
    station_names = nc.createVariable('station_name', str, ('station',))
    station_names.long_name = 'station name'
    station_names[:] = stations_metadata['name'].values
    
    lons = nc.createVariable('lon', 'f4', ('station',))
    lons.units = 'degrees_east'
    lons.long_name = 'longitude'
    lons.standard_name = 'longitude'
    lons[:] = stations_metadata['lon'].values
    
    lats = nc.createVariable('lat', 'f4', ('station',))
    lats.units = 'degrees_north'
    lats.long_name = 'latitude'
    lats.standard_name = 'latitude'
    lats[:] = stations_metadata['lat'].values
    
    elevations = nc.createVariable('elevation', 'f4', ('station',))
    elevations.units = 'meters'
    elevations.long_name = 'elevation above sea level'
    elevations.standard_name = 'surface_altitude'
    elevations[:] = stations_metadata['elevation'].values
    
    countries = nc.createVariable('country', str, ('station',))
    countries.long_name = 'country code'
    countries[:] = stations_metadata['country'].values
    
    # Create main data variable
    var = nc.createVariable(variable_name, 'f4', ('time', 'station'), 
                           fill_value=-999.9, zlib=True, complevel=4)
    var.long_name = variable_long_name
    var.units = variable_units
    var.coordinates = 'time lon lat'
    
    # Fill data (convert DataFrame to numpy array and apply scale factor)
    data_array = variable_data.values.astype(np.float32)
    data_array = data_array * scale_factor
    var[:, :] = data_array
    
    # Global attributes
    nc.title = f'ECA meteorological station data - {variable_long_name}'
    nc.institution = 'European Climate Assessment'
    nc.source = 'ECA blend dataset'
    nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    nc.Conventions = 'CF-1.8'
    nc.featureType = 'timeSeries'
    
    # Close file
    nc.close()
    print(f"Created: {output_file}")

def main():
    """Main function to convert all CSV files to NetCDF."""
    
    # Define paths
    data_dir = 'data'
    master_file = os.path.join(data_dir, 'Master.txt')
    
    # Stations to exclude (Caceres - far from the main Catalonia cluster)
    exclude_stations = ['003920']
    
    # Define variables with their metadata
    # Note: ECA data stores values in tenths (0.1 scale)
    variables = {
        'tx': {
            'file': os.path.join(data_dir, 'ECA_blend_tx.csv'),
            'long_name': 'maximum temperature',
            'units': 'degrees_celsius',
            'scale_factor': 0.1
        },
        'tn': {
            'file': os.path.join(data_dir, 'ECA_blend_tn.csv'),
            'long_name': 'minimum temperature',
            'units': 'degrees_celsius',
            'scale_factor': 0.1
        },
        'tg': {
            'file': os.path.join(data_dir, 'ECA_blend_tg.csv'),
            'long_name': 'mean temperature',
            'units': 'degrees_celsius',
            'scale_factor': 0.1
        },
        'rr': {
            'file': os.path.join(data_dir, 'ECA_blend_rr.csv'),
            'long_name': 'precipitation amount',
            'units': 'mm',
            'scale_factor': 0.1
        }
    }
    
    # Read station metadata
    print("Reading station metadata...")
    stations_metadata = read_station_metadata(master_file, exclude_stations=exclude_stations)
    print(f"Found {len(stations_metadata)} stations (excluded {len(exclude_stations)} stations)")
    if exclude_stations:
        print(f"  Excluded stations: {', '.join(exclude_stations)}")
    
    # Process each variable
    for var_name, var_info in variables.items():
        csv_file = var_info['file']
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping...")
            continue
        
        print(f"\nProcessing {var_name}...")
        
        # Read variable data
        print(f"  Reading {csv_file}...")
        var_data = read_variable_data(csv_file)
        
        # Remove excluded stations from data
        for station_id in exclude_stations:
            if station_id in var_data.columns:
                var_data = var_data.drop(columns=[station_id])
                print(f"  Removed station {station_id} from data")
        
        print(f"  Data shape: {var_data.shape} (time x stations)")
        print(f"  Time range: {var_data.index[0]} to {var_data.index[-1]}")
        
        # Create NetCDF file
        output_file = os.path.join(data_dir, f'ECA_blend_{var_name}.nc')
        print(f"  Creating NetCDF file...")
        create_netcdf(
            output_file=output_file,
            variable_name=var_name,
            variable_data=var_data,
            stations_metadata=stations_metadata,
            variable_long_name=var_info['long_name'],
            variable_units=var_info['units'],
            scale_factor=var_info['scale_factor']
        )
    
    print("\nConversion complete!")
    print("\nGenerated files:")
    for var_name in variables.keys():
        nc_file = os.path.join(data_dir, f'ECA_blend_{var_name}.nc')
        if os.path.exists(nc_file):
            size_mb = os.path.getsize(nc_file) / (1024 * 1024)
            print(f"  {nc_file} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    main()

