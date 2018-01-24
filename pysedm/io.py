#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to I/O the data """

import os
import re
import warnings
from glob import glob


REDUXPATH   = os.getenv('SEDMREDUXPATH',default="~/redux/")

############################
#                          #
#  PROD/DB STRUCTURE       #
#                          #
############################
PROD_CUBEROOT        = "e3d"
PROD_SPECROOT        = "spec"
PROD_SENSITIVITYROOT = "fluxcal"

PRODSTRUCT_RE = {"ccd":{"lamp":"^(dome|Hg|Cd|Xe)",
                      "crr":'^(crr)',
                      "orig":"^(ifu)"},
               "bkgd":{"lamp":"^(bkgd).*(dome|Hg|Cd|Xe)",
                       "crr":"^(bkgd_crr)"},
               "cube": {"basic":"^(%s).((?!cal))"%PROD_CUBEROOT, # starts w/ e3d & not containing cal
                        "calibrated":"^(%s_cal)"%PROD_CUBEROOT,
                        "defaultcalibrated":"^(%s_defcal)"%PROD_CUBEROOT},
                "spec": {"basic":"^(%s).((?!cal))"%PROD_SPECROOT, # starts w/ e3d & not containing cal
                         "auto":"^(%sauto).((?!cal))"%PROD_SPECROOT, # starts w/ e3d & not containing cal
                         "calibrated":"^(%s_cal)"%PROD_SPECROOT,
                         "defaultcalibrated":"^(%s_defcal)"%PROD_SPECROOT,
                         "invsensitivity":"^(%s)"%PROD_SENSITIVITYROOT}
              }

__all__ = ["get_night_files",
               "load_nightly_tracematch","load_nightly_hexagonalgrid",
               "load_nightly_wavesolution","load_nightly_flat"]

############################
#                          #
#  Data Access             #
#                          #
############################
def get_night_files(date, kind, target=None, extention=".fits"):
    """ GENERIC IO FUNCTION
    
    Parameters
    ----------
    date: [string] 
        date for which you want file. Format: YYYYMMDD
    
    kind: [string]
        Which kind of file do you want. 
        Format for kind:
         - You can directly provide a regular expression by starting by re:
            - e.g. all the lamp or dome associated data: kind="re:(dome|Hg|Xe|Cd)"
         - Ortherwise, you give provide a predefined format using 'type.subtype' ;
            - e.g. the basic cubes            : kind='cube.basic'
            - e.g. final flux calibrated cubes: kind='cube.fluxcalibrated' ;
            - e.g. quick flux calibrated cubes: kind='cube.defaultcalibrated' ;
            - e.g. the lamp or dome ccd files : kind='ccd.lamp'
            - e.g. cosmic ray corrected ccds  : kind='ccd.crr' ;
            - e.g. lamp and dome ccd files    : kind='ccd.lamp';

        (see all the predifined format in pysedm.io.PRODSTRUCT_RE) 
       
    target: [string/None] -optional-
        Additional selection. The file should also contain the string defined in target.
        target supports regex. e.g. requesting dome or Hg files: target="(dome|Hg)"

    Returns
    -------
    list of string (FullPaths
    """
    if kind.startswith('re:'):
        regex = kind.replace("re:","")
    else:
        if "." not in kind: kind = "%s.*"%kind
        
        kinds = kind.split(".")
        if len(kinds)==2:
            type_, subtype_ = kinds
            # allowing some user flexibility
            if subtype_ in ["cal"]:
                subtype_ = "calibrated"
            elif subtype_ in ["defcal"]:
                subtype_ = "defaultcalibrated"
            elif subtype_ in ["fluxcal"]:
                subtype_ = "invsensitivity"
                
            # parsing the input                
            if subtype_ !="*":
                regex  = PRODSTRUCT_RE[type_][subtype_]
            else:
                regex  = "|".join([subsreg
                    for subsreg in PRODSTRUCT_RE[type_].values()])
                
        else:
            raise TypeError("Enable to parse the given kind: %s"%kind)

    path = get_datapath(date)
    if target in ["*"]:
        target = None
    if extention in ["*",".*"]:
        extention = None

    # - Parsing the files
    return [path+f for f in os.listdir(get_datapath(date))
               if re.search(r'%s'%regex, f) and
                 (target is None or re.search(r'%s'%target, f)) and
                 (extention is None or re.search(r'%s'%extention, f))]

# ---------------- #
#  Reading the DB  #
# ---------------- #
def get_datapath(YYYYMMDD):
    """ Return the full path of the current date """
    return REDUXPATH+"/%s/"%YYYYMMDD

def filename_to_background_name(filename):
    """ predefined structure for background naming """
    last = filename.split("/")[-1]
    return "".join([filename.split(last)[0],"bkgd_"+last])

def get_night_schedule(YYYYMMDD):
    """ Return the list of observations (the what.list) """
    schedule_file = glob(get_datapath(YYYYMMDD)+"what*")
    if len(schedule_file)==0:
        warnings.warn("No 'what list' for the given night ")
        return None
    return open(schedule_file[0]).read().splitlines()
        
def is_file_stdstars(filename):
    """ Tests if the 'OBJECT' entry of the file header is associated with a Standard star exposure. (True / False)
    None is returned if the header do not contain an 'OBJECT' entry 
    (see `is_stdstars`)
    Returns
    -------
    bool or None
    """
    from astropy.io.fits import getheader
    return is_stdstars( getheader(filename) )

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
    return any([s_ in obj for s_ in stdnames])


#########################
#                       #
#   NIGHT SOLUTION      #
#                       #
#########################
# - TraceMatch
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

# - HexaGrid
def load_nightly_hexagonalgrid(YYYYMMDD):
    """ Load the Grid id <-> QR<->XY position
    This object must have been created. 
    """
    from .utils.hexagrid import load_hexprojection
    return load_hexprojection(get_datapath(YYYYMMDD)+"%s_HexaGrid.pkl"%(YYYYMMDD))

# - WaveSolution
def load_nightly_wavesolution(YYYYMMDD):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from .wavesolution import load_wavesolution
    return load_wavesolution(get_datapath(YYYYMMDD)+"%s_WaveSolution.pkl"%(YYYYMMDD))

# - 3D Flat
def load_nightly_flat(YYYYMMDD):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from pyifu.spectroscopy import load_slice
    return load_slice(get_datapath(YYYYMMDD)+"%s_Flat.fits"%(YYYYMMDD))
