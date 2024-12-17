""" Read Aura MLS data.

MLS data Version 5
https://mls.jpl.nasa.gov/eos-aura-mls/data.php

Goal is to be general enough to work for any of the variables available
"""

import h5py
import xarray as xr
from datetime import datetime
import pandas as pd
import numpy as np
from glob import glob

def _open_one_dataset(filename, var_dict):
    '''Read individual v5

    Parameters
    ----------
    filename : str
        Local path to netCDF4 (HDF5) file.
    var_dict : dict

    Returns
    -------
    ds : xr.Dataset    
    '''

    ds = xr.Dataset()
    varnames = list(var_dict.keys())
    
    f = h5py.File(filename)
    # extract latitude, longitude, time, and pressure information
    ds['lon'] = (['time'],f['HDFEOS']['SWATHS'][varnames[0]]['Geolocation Fields']['Longitude'][:])
    
    ds['lat'] = (['time'],f['HDFEOS']['SWATHS'][varnames[0]]['Geolocation Fields']['Latitude'][:])
    time = pd.to_datetime(
        f['HDFEOS']['SWATHS'][varnames[0]]['Geolocation Fields']['Time'][:],
            unit="s",
            origin="1993-01-01 00:00:00",
        )
    start_time = time[0]
    
    ds['time'] = (['time'],time)
    ds['pressure'] = (['z'],f['HDFEOS']['SWATHS'][varnames[0]]['Geolocation Fields']['Pressure'][:])
    ds['pressure'].attrs.update({"units": "hPa"})

    # extract variables
    for key in varnames:
        if f['HDFEOS']['SWATHS'][key]['Data Fields']['L2gpValue'][:].ndim == 2:
            dimset = ['time','z']
        elif f['HDFEOS']['SWATHS'][key]['Data Fields']['L2gpValue'][:].ndim == 1:
            dimset = ['time']
        ds[key] = (dimset,f['HDFEOS']['SWATHS'][key]['Data Fields']['L2gpValue'][:])
        ds[f'{key}_precision'] = (dimset,f['HDFEOS']['SWATHS'][key]['Data Fields']['L2gpPrecision'][:])
    f.close()
    ds = ds.assign_attrs(reference_time_string=start_time)
    return ds

def open_dataset(fnames,vard):
    """Open one or more TROPOMI L2 NO2 files.

    Parameters
    ----------
    fnames : str
        Glob expression for input file paths.

    Returns
    -------
    Dict
        Dict mapping reference time string (date, YYYY-MM-DD)
        to a list of :class:`xarray.Dataset` granules.
    """
    files = sorted(glob(fnames))
    granules = {}
    
    for file in files:
        granule = _open_one_dataset(file,vard)
        key = granule.attrs["reference_time_string"].strftime(r"%Y-%m-%d")
        granules[key] = granule
    return granules