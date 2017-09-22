#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to I/O the data """

import os

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
    from astropy.io import fits as pf
    
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

#########################
#                       #
#   NIGHT SOLUTION      #
#                       #
#########################
def load_nightly_tracematch(YYYYMMDD):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from .spectralmatching import load_tracematcher
    return load_tracematcher(get_datapath(YYYYMMDD)+"%s_TraceMatch.pkl"%(YYYYMMDD))

def load_nightly_spectralmatch(YYYYMMDD):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from .spectralmatching import load_specmatcher
    return load_specmatcher(get_datapath(YYYYMMDD)+"%s_SpectralMatch.pkl"%(YYYYMMDD))

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
