#!/usr/bin/env python3
"""
Visualize ECA meteorological station data from NetCDF files using Cartopy.

This script creates PDF visualizations showing:
- Station locations on a map
- Sample time series for selected stations
- Spatial patterns of recent data
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def plot_station_locations(ds, variable_name, output_file):
    """
    Create a map showing all station locations with elevation.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent to cover the stations
    lon_min, lon_max = ds.lon.min().values - 0.5, ds.lon.max().values + 0.5
    lat_min, lat_max = ds.lat.min().values - 0.5, ds.lat.max().values + 0.5
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                     alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot stations colored by elevation
    scatter = ax.scatter(ds.lon.values, ds.lat.values, 
                        c=ds.elevation.values,
                        s=50, 
                        cmap='terrain',
                        edgecolors='black',
                        linewidth=0.5,
                        alpha=0.8,
                        transform=ccrs.PlateCarree(),
                        zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8)
    cbar.set_label('Elevation (m)', fontsize=12)
    
    # Title
    var_info = {
        'tx': 'Maximum Temperature',
        'tn': 'Minimum Temperature',
        'tg': 'Mean Temperature',
        'rr': 'Precipitation'
    }
    title = f'ECA Station Locations - {var_info.get(variable_name, variable_name)}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add station count
    plt.text(0.02, 0.98, f'Total stations: {len(ds.station)}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def plot_time_series_samples(ds, variable_name, output_file):
    """
    Plot time series for a selection of stations.
    """
    var_info = {
        'tx': {'name': 'Maximum Temperature', 'unit': '°C', 'color': 'red'},
        'tn': {'name': 'Minimum Temperature', 'unit': '°C', 'color': 'blue'},
        'tg': {'name': 'Mean Temperature', 'unit': '°C', 'color': 'green'},
        'rr': {'name': 'Precipitation', 'unit': 'mm', 'color': 'steelblue'}
    }
    
    info = var_info.get(variable_name, {'name': variable_name, 'unit': '', 'color': 'black'})
    
    # Select a few stations with good data coverage
    # Get the last 10 years of data
    recent_data = ds.sel(time=slice('2015-01-01', None))
    
    # Find stations with most complete recent data
    data_counts = recent_data[variable_name].count(dim='time')
    top_stations_idx = data_counts.argsort()[-6:][::-1].values
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{info["name"]} - Sample Station Time Series (2015-2025)', 
                fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, station_idx in enumerate(top_stations_idx):
        ax = axes[idx]
        
        # Get station data
        station_data = recent_data[variable_name].isel(station=station_idx)
        station_name = ds.station_name.isel(station=station_idx).values
        station_id = ds.station_id.isel(station=station_idx).values
        elevation = ds.elevation.isel(station=station_idx).values
        
        # Plot
        station_data.plot(ax=ax, color=info['color'], linewidth=0.5, alpha=0.7)
        
        # Add title
        ax.set_title(f'{station_name} ({station_id})\nElev: {elevation:.0f}m', 
                    fontsize=9)
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel(f'{info["name"]} ({info["unit"]})', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def plot_recent_spatial_pattern(ds, variable_name, output_file):
    """
    Create a spatial map showing recent mean values at each station.
    """
    var_info = {
        'tx': {'name': 'Maximum Temperature', 'unit': '°C', 'cmap': 'RdYlBu_r'},
        'tn': {'name': 'Minimum Temperature', 'unit': '°C', 'cmap': 'RdYlBu_r'},
        'tg': {'name': 'Mean Temperature', 'unit': '°C', 'cmap': 'RdYlBu_r'},
        'rr': {'name': 'Precipitation', 'unit': 'mm', 'cmap': 'Blues'}
    }
    
    info = var_info.get(variable_name, {'name': variable_name, 'unit': '', 'cmap': 'viridis'})
    
    # Calculate mean for recent year (2024)
    recent_year = ds.sel(time=slice('2024-01-01', '2024-12-31'))
    
    if variable_name == 'rr':
        # For precipitation, show annual total
        values = recent_year[variable_name].sum(dim='time').values
        metric = 'Annual Total'
    else:
        # For temperature, show annual mean
        values = recent_year[variable_name].mean(dim='time').values
        metric = 'Annual Mean'
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent
    lon_min, lon_max = ds.lon.min().values - 0.5, ds.lon.max().values + 0.5
    lat_min, lat_max = ds.lat.min().values - 0.5, ds.lat.max().values + 0.5
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                     alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot stations with values
    # Filter out NaN values
    mask = ~np.isnan(values)
    
    scatter = ax.scatter(ds.lon.values[mask], ds.lat.values[mask], 
                        c=values[mask],
                        s=100, 
                        cmap=info['cmap'],
                        edgecolors='black',
                        linewidth=0.5,
                        alpha=0.9,
                        transform=ccrs.PlateCarree(),
                        zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8)
    cbar.set_label(f'{metric} {info["name"]} ({info["unit"]})', fontsize=12)
    
    # Title
    plt.title(f'{info["name"]} - {metric} 2024', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Add stats box
    stats_text = f'Stations with data: {mask.sum()}\n'
    stats_text += f'Min: {np.nanmin(values):.1f} {info["unit"]}\n'
    stats_text += f'Max: {np.nanmax(values):.1f} {info["unit"]}\n'
    stats_text += f'Mean: {np.nanmean(values):.1f} {info["unit"]}'
    
    plt.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def plot_annual_climatology(ds, variable_name, output_file):
    """
    Plot annual climatology (monthly means) for all stations.
    """
    var_info = {
        'tx': {'name': 'Maximum Temperature', 'unit': '°C', 'color': 'red'},
        'tn': {'name': 'Minimum Temperature', 'unit': '°C', 'color': 'blue'},
        'tg': {'name': 'Mean Temperature', 'unit': '°C', 'color': 'green'},
        'rr': {'name': 'Precipitation', 'unit': 'mm', 'color': 'steelblue'}
    }
    
    info = var_info.get(variable_name, {'name': variable_name, 'unit': '', 'color': 'black'})
    
    # Use data from 1991-2020 (standard climatological period)
    clim_data = ds.sel(time=slice('1991-01-01', '2020-12-31'))
    
    # Calculate monthly climatology
    monthly_clim = clim_data[variable_name].groupby('time.month').mean(dim='time')
    
    # Calculate statistics across stations
    monthly_mean = monthly_clim.mean(dim='station')
    monthly_min = monthly_clim.min(dim='station')
    monthly_max = monthly_clim.max(dim='station')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_nums = range(1, 13)
    
    # Plot range
    ax.fill_between(month_nums, monthly_min.values, monthly_max.values,
                    alpha=0.3, color=info['color'], label='Min-Max Range')
    
    # Plot mean
    ax.plot(month_nums, monthly_mean.values, 
           color=info['color'], linewidth=2, marker='o', 
           markersize=8, label='Mean', zorder=5)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
    ax.set_title(f'{info["name"]} - Annual Climatology (1991-2020)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(month_nums)
    ax.set_xticklabels(months)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def main():
    """Main function to create all visualizations."""
    
    # Define variables
    variables = ['tx', 'tn', 'tg', 'rr']
    
    print("Creating visualizations from NetCDF files...\n")
    
    for var_name in variables:
        nc_file = f'data/ECA_blend_{var_name}.nc'
        
        print(f"Processing {var_name}...")
        
        # Open dataset
        ds = xr.open_dataset(nc_file)
        
        # Create visualizations
        plot_station_locations(ds, var_name, f'{var_name}_stations_map.pdf')
        plot_time_series_samples(ds, var_name, f'{var_name}_time_series.pdf')
        plot_recent_spatial_pattern(ds, var_name, f'{var_name}_spatial_2024.pdf')
        plot_annual_climatology(ds, var_name, f'{var_name}_climatology.pdf')
        
        ds.close()
        print()
    
    print("All visualizations complete!")
    print("\nGenerated PDF files:")
    for var_name in variables:
        print(f"  - {var_name}_stations_map.pdf")
        print(f"  - {var_name}_time_series.pdf")
        print(f"  - {var_name}_spatial_2024.pdf")
        print(f"  - {var_name}_climatology.pdf")

if __name__ == '__main__':
    main()

