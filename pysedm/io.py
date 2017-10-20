#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to I/O the data """

import os
import warnings
import numpy as np
from astropy.io import fits as pf

REDUXPATH   = os.getenv('SEDMREDUXPATH',default="~/redux/")

CUBE_PROD_ROOTS = {"cube":{"root":"e3d",
                          "comment":"Raw Extract cube"},
                   "flat":{"root":"flat_e3d",
                           "comment":"Flat fielded cube (from Raw Extract cube)"},
                   "flux_calibrated":{"root":"fluxcal_e3d",
                                      "comment":"Flux calibrated flat fielded cube"}
                    }
    
def get_datapath(YYYYMMDD):
    """ Return the full path of the current date """
    return REDUXPATH+"/%s/"%YYYYMMDD


def filename_to_background_name(filename):
    """ predefined structure for background naming """
    last = filename.split("/")[-1]
    return "".join([filename.split(last)[0],"bkgd_"+last])


def get_night_schedule(YYYYMMDD):
    """ Return the list of observations (the what.list) """
    from glob import glob
    schedule_file = glob(get_datapath(YYYYMMDD)+"what*")
    if len(schedule_file)==0:
        warnings.warn("No 'what list' for the given night ")
        return None
    return open(schedule_file[0]).read().splitlines()

def get_night_ccdfiles(YYYYMMDD, skip_calib=False, starts_with="crr_b_", contains="*"):
    """ Return the ccdfile associated to the given night (ccr_b_....fits)

    Parameters
    ----------
    YYYYMMDD: [string]
        date e.g. 20170222
    
    skip_calib: [bool] -optional-
        shall the calibration ccd files (crr_b_.. corresponding the Xe, Hg, Cd or dome)
        be skiped?
        
    starts_with: [string] -optional-
        How the ccdfiles be identified? 

    Returns
    -------
    list of fullpathes
    """
    from glob import glob
    
    timedir = get_datapath(YYYYMMDD)
    basefile = timedir+"%s*%s*.fits*"%(starts_with,contains)
    return glob(basefile) if not skip_calib else\
      [f_ for f_ in glob(basefile) if "Calib" not in pf.getval(f_,"Name") ]
    
def get_night_cubes(YYYYMMDD, kind, target="*"):
    """ Return the ccdfile associated to the given night (ccr_b_....fits)

    Parameters
    ----------
    YYYYMMDD: [string]
        date e.g. 20170222
    
    kind: [string] 
        The kind of cube you want:
        - cube: for wavelength calibrated cube
        - flat: for wavelength calibrated cube after 3d flat correction
        - flux_calibrated: flux calibrated cube
        
    starts_with: [string] -optional-
        How the ccdfiles be identified? 

    Returns
    -------
    list of fullpathes
    """
    from glob import glob
    if kind not in CUBE_PROD_ROOTS.keys():
        raise ValueError("Unknown kind (%s). Available kinds -> "%kind+ ", ".join(CUBE_PROD_ROOTS.keys()))
    root = CUBE_PROD_ROOTS[kind]["root"]

    timedir = get_datapath(YYYYMMDD)
    return glob(timedir+"%s*%s*.fits*"%(root,target))

def get_file_tracematch(YYYYMMDD, contains):
    """ """
    from glob import glob
    tracefile = glob(get_datapath(YYYYMMDD)+"tracematch_*%s*"%contains)
    if len(tracefile) == 0:
        return None
    from .spectralmatching import load_tracematcher
    return load_tracematcher(tracefile[0])

def get_std_cubefiles(date):
    """ """
    ccdfiles  = get_night_ccdfiles(date)
    filestd   = [f_ for f_ in ccdfiles if is_file_stdstars(f_)]
    return [get_night_cubes(date,"flat", f_.split("/")[-1].split(".fits")[0].split("ifu")[-1])[0]
                for f_ in filestd]

#########################
#                       #
#   File Information    #
#                       #
#########################
def is_file_stdstars(filename):
    """ Tests if the 'OBJECT' entry of the file header is associated with a Standard star exposure. (True / False)
    None is returned if the header do not contain an 'OBJECT' entry 
    (see `is_stdstars`)
    Returns
    -------
    bool or None
    """
    return is_stdstars(pf.getheader(filename))

def is_stdstars(header):
    """ Tests if the 'OBJECT' of the given header is associated with a Standard star exposure. (True / False)
    None is returned if the header do not contain an 'OBJECT' entry 

    Returns
    -------
    bool or None
    """
    obj = header.get("OBJECT",None)
    if obj is None:
        return None

    stdnames = ["STD","Feige", "Hitlner", "LTT"]
    return np.any([s_ in obj for s_ in stdnames])

#########################
#                       #
#   NIGHT SOLUTION      #
#                       #
#########################
def load_nightly_tracematch(YYYYMMDD, withmask=False):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from .spectralmatching import load_tracematcher
    if not withmask:
        return load_tracematcher(get_datapath(YYYYMMDD)+"%s_TraceMatch.pkl"%(YYYYMMDD))
    else:
        try:
            return load_tracematcher(get_datapath(YYYYMMDD)+"%s_TraceMatch_WithMasks.pkl"%(YYYYMMDD))
        except:
            warnings.warn("No TraceMatch_WithMasks found. returns the usual TraceMatch")
            return load_nightly_tracematch(YYYYMMDD, withmask=False)

def load_nightly_hexagonalgrid(YYYYMMDD):
    """ Load the Grid id <-> QR<->XY position
    This object must have been created. 
    """
    from .utils.hexagrid import load_hexprojection
    return load_hexprojection(get_datapath(YYYYMMDD)+"%s_HexaGrid.pkl"%(YYYYMMDD))

def load_nightly_wavesolution(YYYYMMDD):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from .wavesolution import load_wavesolution
    return load_wavesolution(get_datapath(YYYYMMDD)+"%s_WaveSolution.pkl"%(YYYYMMDD))
