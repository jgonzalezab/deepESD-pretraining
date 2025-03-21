import xarray as xr
import numpy as np
import psutil
import torch

def get_process_memory_usage():

    """
    Measures and returns the memory used by the Python
    process
    """

    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 ** 2)
    print(f"Current process memory usage: {memory_mb} MB")


def _has_leap_years(data: xr.Dataset) -> bool:

    """
    Internal function to detect if a xr.Dataset
    has leap years. It does it by counting the
    number of February 29th days.

    Parameters
    ----------
    data : xr.Dataset
        Dataset to count the number of leap years.

    Returns
    -------
    bool
        Whether it has leap years or not
    """

    data_time = data['time']
    is_feb_29 = data['time'].dt.day.isin(29) & data['time'].dt.month.isin(2)
    num_feb_29 = np.sum(is_feb_29).values.item()
    has_leap_years = (num_feb_29 > 0)

    return has_leap_years

def load_gcm(gcm: str, scenario: str, gcm_path: str) -> xr.Dataset:

    """
    Load the GCM provided for the scenario provided. It performs a basic
    preprocessing and returns a xr.Dataset with the GCM in the same format
    as the ERA5 predictors used in this project.

    Parameters
    ----------
    gcm : str
        Name of the GCM to load followed by the run (separated by an underscore).
        For instance, CanESM5_r1i1p1f1.

    scenario : str
        Scenario to load. Must be one of the following: historical, ssp126,
        ssp245, ssp370, ssp585.

    gcm_path : str
        Path containing the.nc files corresponding to the GCM. Notice that
        these must be formatted in a specific way (see the code for more
        details).

    Returns
    -------
    xr.Dataset
        The specified GCM in the format of the ERA5 predictors used in this
        project.
    """

    # Get the GCM and run to load
    gcm_info = gcm.split('_')
    if len(gcm_info) == 2: 
        gcm, gcm_run = gcm_info
    else:
        raise ValueError('Please provide as gcm name a string containing the gcm and the run separated by an underscore.')

    # Select the proper years string
    if scenario == 'historical':
        years = '19500101-20141231'
    else:
        years = '20150101-21001231'

    # These mappings (vars and heights) are defined based on the ERA5
    # predictors used for this specific project
    vars_mapping = {'ta': 't',
                    'hus': 'q',
                    'va': 'v',
                    'ua': 'u',
                    'psl': 'msl'}
    
    heights_mapping = {500.0: '500',
                       700.0: '700',
                       850.0: '850'}

    # Iterate over all variables to load
    var_list = []
    for var in vars_mapping.keys():
        
        # Load the variable
        data = xr.open_dataset(f'{gcm_path}/{var}_{gcm}_{scenario}_{gcm_run}_{years}.nc').load()
        data = data.drop_dims('bnds')

        # Iterate over the heights of the loaded variable
        if var not in ('psl'):
            data['plev'] = data['plev'] / 100 # Transform the units of the height
            for height in heights_mapping.keys():

                # Sometimes, probably due to a precission issue,
                # heights have the following format XXX.00000001
                try:
                    data_aux = data.sel(plev=height)
                except: # In such case we rely on method='nearest'
                    data_aux = data.sel(plev=height, method='nearest')

                data_aux = data_aux.drop_vars('plev')
                data_aux = data_aux.rename({var: f'{vars_mapping[var]}{heights_mapping[height]}'})
                var_list.append(data_aux)

        else: # This is for psl/msl, as it does not have height
            data_aux = data.rename({var: f'{vars_mapping[var]}'})
            var_list.append(data_aux)

    # Merge all loaded variables into an unique xr.Dataset
    predictor_gcm = xr.merge(var_list)

    # Some GCM models have a calendar of 360_day. Before continuing
    # it is necessary to transform them to the standard format
    if gcm in ('UKESM1-0-LL', 'KACE-1-0-G'):
        predictor_gcm = predictor_gcm.convert_calendar(calendar='standard', align_on='date')

    # Reformat the temporal dimension based on the leap years
    if _has_leap_years(predictor_gcm):
        predictor_gcm = predictor_gcm.assign_coords(
                            {'time': predictor_gcm.indexes['time'].normalize()})
    else:
        predictor_gcm = predictor_gcm.assign_coords(
                            {'time': predictor_gcm.indexes['time'].to_datetimeindex(unsafe=True).normalize()})

    return predictor_gcm

def load_surface_gcm(gcm: str, var: str, scenario: str, gcm_path: str) -> xr.Dataset:

    """
    Load the surface variable (var) for the GCM provided for the scenario
    provided. It performs a basic preprocessing and returns a xr.Dataset
    with the GCM in the same format as the ERA5 predictors used in this
    project.

    Parameters
    ----------
    gcm : str
        Name of the GCM to load followed by the run (separated by an underscore).
        For instance, CanESM5_r1i1p1f1.

    var : str
        Surface variable to load. Must be one of the following: tasmin, tasmax
        or pr.

    scenario : str
        Scenario to load. Must be one of the following: historical, ssp126,
        ssp245, ssp370, ssp585.

    gcm_path : str
        Path containing the.nc files corresponding to the GCM. Notice that
        these must be formatted in a specific way (see the code for more
        details).

    Returns
    -------
    xr.Dataset
        The specified GCM in the format of the ERA5 predictors used in this
        project.
    """

    # Get the GCM and run to load
    gcm_info = gcm.split('_')
    if len(gcm_info) == 2: 
        gcm, gcm_run = gcm_info
    else:
        raise ValueError('Please provide as gcm name a string containing the gcm and the run separated by an underscore.')

    # Select the proper years string
    if scenario == 'historical':
        years = '19500101-20141231'
    else:
        years = '20150101-21001231'

    # Load the variable
    data = xr.open_dataset(f'{gcm_path}/{var}_{gcm}_{scenario}_{gcm_run}_{years}.nc').load()
    data = data.drop_dims('bnds')

    # Reformat the temporal dimension base on the leap years
    if _has_leap_years(data):
        data = data.assign_coords(
                    {'time': data.indexes['time'].normalize()})
    else:
        data = data.assign_coords(
                    {'time': data.indexes['time'].to_datetimeindex().normalize()})

    return data

def nan_masking_xarray(data: xr.Dataset, var_target: str,
                       nan_perc: float) -> xr.Dataset:

    '''
    Randomly mask the data xr.Dataset.
    '''

    data_copy = data.copy(deep=True)

    data_shape = data_copy[var_target].values.shape
    random_array = np.random.uniform(low=0.0, high=1.0, size=data_shape)
    mask = np.where(random_array > nan_perc, 1, np.nan)

    data_copy[var_target] = data_copy[var_target] * mask

    return data_copy

def compute_gcm_mask_PNACC(template):
    
    '''
    Maks the GCM to align with the PNACC domain
    '''

    # Set mask
    var_data = list(template.data_vars.keys())[0]

    template = template.sel(lat=slice(33.48, 45.97),
                            lon=slice(-13.175, 6.775))

    template = ~template.isnull()
    template = template.where(template != False, np.nan)

    # Remove regions
    regions_remove = [(33, 46, -13, -10),
                      (44, 46, -13, 7),
                      (33, 36, -13, 7),
                      (33, 46, 4, 7)]

    for region in regions_remove:
        lat_min, lat_max, lon_min, lon_max = region
        mask = ((template.lat >= lat_min) & (template.lat <= lat_max) &
                (template.lon >= lon_min) & (template.lon <= lon_max))
        mask = mask.broadcast_like(template[var_data])
        template[var_data] = template[var_data].where(~mask, other=np.nan)

    # Remove individual points
    points_remove = [(37.478492, -0.395648),
                     (37.487239, 1.610412),
                     (36.907353, 2.899883),
                     (39.938724, 1.247921),
                     (39.015810, 3.088678),
                     (37.307389, -8.540119),
                     (38.995060, -8.759128),
                     (40.477036, -8.627723)]

    for point in points_remove:
        closest_point = template.sel(lat=point[0], lon=point[1],
                                     method='nearest')
        template[var_data].loc[dict(lat=closest_point[var_data].lat,
                                    lon=closest_point[var_data].lon)] = np.nan

    # Add individual points
    points_add = [(36.180864, -5.687983),
                  (39.950786, 4.095588)]

    for point in points_add:
        closest_point = template.sel(lat=point[0], lon=point[1],
                                     method='nearest')
        template[var_data].loc[dict(lat=closest_point[var_data].lat,
                                    lon=closest_point[var_data].lon)] = 1

    return template