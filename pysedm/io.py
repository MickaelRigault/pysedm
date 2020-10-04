#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to I/O the data """

import os
import re
import warnings
from glob import glob
from astropy.io.fits import getheader, getval
from astropy.time import Time
REDUXPATH   = os.getenv('SEDMREDUXPATH',default="~/redux/")
_PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))+"/"
SEDM_REDUCER     = os.getenv('SEDM_USER',default="auto")

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
                         "bkgd":"^(%s).((?!cal))"%PROD_SPECROOT, # starts w/ e3d & not containing cal
                         "auto":"^(%sauto).((?!cal))"%PROD_SPECROOT, # starts w/ e3d & not containing cal
                         "forcepsf":"^(%s_forcepsf).((?!cal))"%PROD_SPECROOT, # starts w/ e3d & not containing cal
                         "calibrated":"^(%s_cal)"%PROD_SPECROOT,
                         "defaultcalibrated":"^(%s_defcal)"%PROD_SPECROOT,
                         "invsensitivity":"^(%s)"%PROD_SENSITIVITYROOT},
                "psf": {"param":"^(%s)(?=.*json)"%"psf"} #.json files containing the psf fitvalues and fitted adr.
              }

__all__ = ["get_night_files",
               "load_nightly_mapper",
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
                 (extention is None or f.endswith(extention))]

def get_cleaned_sedmcube(filename):
    """ get sky and flux calibrated cube """

    cube = get_sedmcube(filename)

    fluxcalfile = io.fetch_nearest_fluxcal(date, cube.filename)
    fluxcal = fluxcalibration.load_fluxcal_spectrum(fluxcalfile)

    cube.remove_sky()
    #cube.scale_by(fluxcal.data) # old version
    cube.scale_by( fluxcal.get_inversed_sensitivity(cube.header.get("AIRMASS", 1.1)), onraw=False )

    return cube
  
#########################
#                       #
#   Reading the DB      #
#                       #
#########################
def get_datapath(YYYYMMDD):
    """ Return the full path of the current date """
    return REDUXPATH+"/%s/"%YYYYMMDD

def fetch_header(date, target, kind="ccd.crr", getkey=None):
    """ Look for a night file (using get_night_files()) and returns its header 
    or a value from it stored with the given getkey."""
    datafile = get_night_files(date, kind, target=target)
    if len(datafile)==0:
        return None
    # Get the entire header
    if getkey is None:
        
        if len(datafile)==1:
            return getheader(datafile[0])
        return {d.split("/"):getheader(d) for d in datafile}
    # Or just a key value from it?
    else:
        from astropy.io.fits import getval
        if len(datafile)==1:
            return getval(datafile[0],getkey)

        return {d.split("/")[-1]:getval(d,getkey) for d in datafile}


def fetch_nearest_fluxcal(date=None, file=None, mjd=None, kind="spec.fluxcal"):
    """ Look for the fluxcal_*.fits file the closest (in time) from the given file and returns it. """
    if date is None:
        if mjd is not None:
            date = Time(mjd, format="mjd").datetime.isoformat().split("T")[0].replace("-","")
        elif file is not None:
            date = filename_to_date(file)
        else:
            raise ValueError("file is None, then date and/or mjd must be given. None here")
        
    filefluxcal = get_night_files(date, kind)
    
    if len(filefluxcal)==0:
        warnings.warn("No %s file for the night %s"%(kind, date))
        return None
    if len(filefluxcal)==1:
        warnings.warn("Only 1 file of kind %s for the night %s"%(kind, date))
        return filefluxcal[0]

    import numpy as np
    if mjd is not None:
        target_mjd_obs = mjd
    else:
        try:
            target_mjd_obs  = getval(file,"MJD_OBS")
        except KeyError:
            warnings.warn("No MJD_OBS keyword found, returning most recent file")
            return filefluxcal[-1]
        
    fluxcal_mjd_obs = [getval(f,"MJD_OBS") for f in filefluxcal]

    return filefluxcal[ np.argmin( np.abs( target_mjd_obs - np.asarray(fluxcal_mjd_obs) ) ) ]

def filename_to_id(filename):
    """ """
    return filename.split("/")[-1].split( header_to_date( getheader(filename) ))[-1][1:9]

def header_to_date( header, sep=""):
    """ returns the datetiume YYYYMMDD associated with the 'JD' from the header """
    datetime = Time(header["JD"], format="jd").datetime

    return sep.join(["%4s"%datetime.year, "%02d"%datetime.month, "%02d"%datetime.day])

def filename_to_time(filename):
    """ """
    date, hour, minut, sec = filename.split("_ifu")[-1].split("_")[:4]
    return Time("-".join([date[i:j] for i,j in [[0,4],[4,6],[6,8]]]) +" "+ ":".join([hour, minut, sec]))

def filename_to_date(filename, iso=False):
    """ """
    if iso:
        return filename_to_time(filename).datetime.isoformat().split("T")[0]
    return filename.split("_ifu")[1].split("_")[0]

def fetch_guider(date, filename, astrom=True, extinction=".fits"):
    """ fetch the guider data for the given filename. """
    print("DEPRECATED fetch_guider(date, filename) -> filename_to_guider(filename)")
    return filename_to_guider(filename, astrom=astrom, extinction=extinction)

def filename_to_guider(filename, astrom=True, extinction=".fits", nomd5=True):
    """ """
    date = filename_to_date(filename)
    id_  = filename_to_id(filename)
    guiders =  [l for l in os.listdir( get_datapath(date)) if id_ in l and "guider" in l 
               and extinction in l and (not nomd5 or not l.endswith(".md5"))]
    if astrom:
        return [get_datapath(date)+"/"+l for l in guiders if "astrom" in l and (not nomd5 or not l.endswith(".md5"))]
    return guiders
    


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
# - Mapper
def load_nightly_mapper(YYYYMMDD, within_ccd_contours=True):
    """ High level object to do i,j<->x,y,lbda """
    from .mapping import Mapper
    
    tracematch = load_nightly_tracematch(YYYYMMDD)
    wsol       = load_nightly_wavesolution(YYYYMMDD)
    hgrid      = load_nightly_hexagonalgrid(YYYYMMDD)
    if within_ccd_contours:
        from .sedm import INDEX_CCD_CONTOURS
        indexes = tracematch.get_traces_within_polygon(INDEX_CCD_CONTOURS)
    else:
        indexes = list(wsol.wavesolutions.keys())
    
    mapper = Mapper(tracematch= tracematch, wavesolution = wsol, hexagrid=hgrid)
    mapper.derive_spaxel_mapping( indexes )
    return mapper

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
def load_nightly_wavesolution(YYYYMMDD, subprocesses=False):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from .wavesolution import load_wavesolution
    if not subprocesses:
        return load_wavesolution(get_datapath(YYYYMMDD)+"%s_WaveSolution.pkl"%(YYYYMMDD))
    return [load_wavesolution(subwave) for subwave in glob(get_datapath(YYYYMMDD)+"%s_WaveSolution_range*.pkl"%(YYYYMMDD))]

# - 3D Flat
def load_nightly_flat(YYYYMMDD):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from pyifu.spectroscopy import load_slice
    return load_slice(get_datapath(YYYYMMDD)+"%s_Flat.fits"%(YYYYMMDD))

#########################
#                       #
#   PSF Product         #
#                       #
#########################
def get_psf_parameters(date, target=None, filepath=False):
    """ """
    path = get_datapath(date)
    if target == "*":
        target = None
    json_files =  [path+f for f in os.listdir(get_datapath(date))
                       if re.search(r'%s'%PRODSTRUCT_RE["psf"]["param"], f)
                       and (target is None or re.search(r'%s'%target, f))
                       and ("defcal") not in f]
    if filepath:
        return json_files
    import json
    return {f.split("/")[-1]:json.load( open(f) ) for f in json_files}
                    
def load_psf_param(date, keys_to_add=None):
    """ """
    from pyifu.adr import ten
    psfdata = get_psf_parameters(date)
    for s, d in psfdata.items():
        d["adr"]["delta_parangle"] = (d["adr"]["parangle"] - d["adr"]["parangle_ref"])%360
        d["adr"]["delta_parangle.err"] = d["adr"]["parangle.err"]
        sour =  s.split(date)[-1][:9]
        if keys_to_add is not None:
            d["header"] = {}
            for k in keys_to_add:
                if "DEC" in k or "RA" in k:
                    val = ten(fetch_header(date, sour, getkey=k))
                else:
                    val = fetch_header(date, sour, getkey=k)
                d["header"][k.lower()] = val
                
    return psfdata

#########################
#                       #
#   References          #
#                       #
#########################
def load_telluric_line(filter=None):
    """ return a TelluricSpectrum (child of pyifu Spectrum) containing the telluric emission line
    from Kitt Peak National Observatory.
    
    Source:
       Buton et al. 2013 (SNIFS, Buton, C., Copin, Y., Aldering, G., et al. 2013, A&A, 549, A8) 
       using data from
       Hinkle, K. H., Wallace, L., & Livingston, W. 2003, BAAS, 35, 1260.
       _Please cite both._
       
    Returns
    -------
    TelluricSpectrum (child of pyifu Spectrum)
    """
    from .utils.atmosphere import load_telluric_spectrum
    return load_telluric_spectrum(_PACKAGE_ROOT+"data/KPNO_lines.fits", filter=filter)




#########################3
#
#  OUTPUT PROD          #
#
#########################

def _saveout_forcepsf_(filecube, cube, cuberes=None, cubemodel=None,
                      cubefitted=None, spec=None, bkgd=None, extraction_type="Force 3DPSF extraction: Spectral Model",
                    mode="auto", spec_info="", fluxcal=True, nofig=False):
    # Cube Model
    if cubemodel is not None:
        cubemodel.set_header(cube.header)
        cubemodel.header["SOURCE"]   = (filecube.split("/")[-1], "This object has been derived from this file")
        cubemodel.header["PYSEDMT"]  = ("Force 3DPSF extraction: Model Cube", "This is the model cube of the PSF extract")
        cubemodel.header["EXTRTYPE"]  = (mode, "Kind of PSF extraction")
        cubemodel.writeto(filecube.replace(PROD_CUBEROOT,"forcepsfmodel_%s_"%mode+PROD_CUBEROOT))

    if cubefitted is not None:
        cubefitted.set_header(cube.header)
        cubefitted.header["SOURCE"]   = (filecube.split("/")[-1], "This object has been derived from this file")
        cubefitted.header["PYSEDMT"]  = ("Force 3DPSF extraction: Fitted Cube", "This is the model cube of the PSF extract")
        cubefitted.header["EXTRTYPE"]  = (mode, "Kind of PSF extraction")
        cubefitted.writeto(filecube.replace(PROD_CUBEROOT,"forcepsf_fitted_%s_"%mode+PROD_CUBEROOT))
        
    if cuberes is not None:
        # Cube Residual                
        cuberes.set_header(cube.header)
        cuberes.header["SOURCE"]   = (filecube.split("/")[-1], "This object has been derived from this file")
        cuberes.header["PYSEDMT"]  = ("Force 3DPSF extraction: Residual Cube", "This is the residual cube of the PSF extract")
        cuberes.header["EXTRTYPE"]  = (mode, "Kind of PSF extraction")
        cuberes.writeto(filecube.replace(PROD_CUBEROOT,"psfres_%s_"%mode+PROD_CUBEROOT))
    
    # ----------------- #
    # Save the Spectrum #
    # ----------------- #
    kind_key =""
    # - build the spectrum
    if spec is not None:
        for k,v in cube.header.items():
            if k not in spec.header:
                spec.header.set(k,v)

        spec.header["SOURCE"]   = (filecube.split("/")[-1], "This object has been derived from this file")
        spec.header["PYSEDMT"]  = (extraction_type, "This is the fitted flux spectrum")
        spec.header["EXTRTYPE"]  = (mode, "Kind of extraction")

        fileout = filecube.replace(PROD_CUBEROOT,PROD_SPECROOT+"%s_%s_"%(kind_key, mode+spec_info))
        spec.writeto(fileout)
        spec.writeto(fileout.replace(".fits",".txt"), ascii=True)
    
        spec._side_properties["filename"] = fileout

        if not nofig:
            from pyifu import get_spectrum
            spec_to_plot = get_spectrum(spec.lbda, spec.data,
                                        variance=spec.variance if spec.has_variance() else None,
                                        header=spec.header)
            spec_to_plot.show(savefile=spec.filename.replace(".fits", ".pdf"),
                              show_zero=fluxcal, show=False)
            spec_to_plot.show(savefile=spec.filename.replace(".fits", ".png"),
                              show_zero=fluxcal, show=False)
        
    # - background
    if bkgd is not None:
        bkgd.set_header(cube.header)
        bkgd.header["SOURCE"]   = (filecube.split("/")[-1], "This object has been derived from this file")
        bkgd.header["PYSEDMT"]  = (extraction_type, "This is the fitted flux spectrum")
        bkgd.header["EXTRTYPE"]  = (mode, "Kind of extraction")
    
        fileout = filecube.replace(PROD_CUBEROOT,PROD_SPECROOT+"%s_%s_bkgd"%(kind_key,mode+spec_info))
        bkgd.writeto(fileout)
        bkgd.writeto(fileout.replace(".fits",".txt"), ascii=True)
