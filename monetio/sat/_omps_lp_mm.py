""" Read OMPS limb profiler O3 data.

OMPS-LP O3 retrieval version: 2.6
DOI: 10.5067/8MO7DEDYTBH7
Data archive: https://disc.gsfc.nasa.gov/datasets/OMPS_NPP_LP_L2_O3_DAILY_2.6/summary

"""

import h5py
import xarray as xr
from datetime import datetime
import pandas as pd
import numpy as np
from glob import glob


def _open_one_dataset(filename):
    '''Read individual v2.6 OMPS LP O3 file. Data should only be valid for the center slit.
    Apply QC flags as recommended in documentation.

    Parameters
    ----------
    fname : str
        Local path to netCDF4 (HDF5) file.
    variable_dict : dict

    Returns
    -------
    ds : xr.Dataset    
    '''
    
    f = h5py.File(filename,'r')
    temperature = f['AncillaryData']['Temperature'][:]
    pressure = f['AncillaryData']['Pressure'][:]
    lat = f['GeolocationFields']['Latitude'][:]
    lon = f['GeolocationFields']['Longitude'][:]
    alt = f['DataFields']['Altitude'][:]
    
    o3 = f['DataFields']['O3Value'][:]*1.38e-19*temperature/pressure/(10**(-6))
    o3_vres = f['DataFields']['VertRes_O3'][:]
    o3_error = f['DataFields']['O3Precision'][:]*1.38e-19*temperature/pressure/(10**(-6))
    o3_qual = f['DataFields']['O3Quality'][:]
    
    # Recommended quality filters
    algiter = f['DataFields']['O3Status'][:]
    algconv = f['DataFields']['O3Convergence'][:]
    qmv_flag = f['DataFields']['QMV'][:]
    pmc_flag = f['DataFields']['ASI_PMCFlag'][:]

    # cloudtop height
    cld = f['DataFields']['CloudHeight'][:]
    
    start_time = datetime.strptime(f.attrs['RangeBeginningDateTime'].decode(),'%Y-%m-%dT%H:%M:%S.%fZ')
    seconds = pd.to_timedelta(f['GeolocationFields']['SecondsInDay'][:],unit='s')
    time = start_time+seconds

    f.close()

    # Apply recommended data filtering
    p_ind = np.where(pmc_flag != 0)
    qmv_ind = np.where(qmv_flag != 0)
    qualind = np.where(o3_qual != 0)
    algit_ind = np.where((algiter < 2) | (algiter > 7))
    algc_ind = np.where(algconv >= 10)

    o3[p_ind] = np.nan #-999.
    o3[qmv_ind] = np.nan #-999.
    o3[qualind] = np.nan #-999.
    o3[algit_ind] = np.nan
    o3[algc_ind] = np.nan

    # Remove data below cloud top
    d1,d2 = np.meshgrid(alt,cld)
    o3[np.where(d1 < d2)] = np.nan
    
    
    data = xr.Dataset({'O3':(('time','z'),o3),},
               coords={'latitude':(('time'),lat),'longitude':(('time'),lon),'time':(('time'),time),'z':(('z'),alt*1000)},
                     attrs={'missing_value':-999,'reference_time_string':start_time},)
    return data

def open_dataset(fnames):
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
        granule = _open_one_dataset(file)
        key = granule.attrs["reference_time_string"].strftime(r"%Y-%m-%d")
        granules[key] = granule
    return granules