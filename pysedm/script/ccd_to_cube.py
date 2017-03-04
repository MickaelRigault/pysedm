#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
from astrobject.utils.tools import dump_pkl
from .. io import get_datapath
from ..ccd import get_lamp
from ..spectralmatching import get_specmatcher, illustrate_traces
from ..wavesolution import get_cubesolution


_EDGES_Y = 20
_EDGES_X = 100
INDEX_CCD_CONTOURS = [[_EDGES_X,_EDGES_Y],[_EDGES_X,1700],
                      [300,2040-_EDGES_Y],[2040-_EDGES_X,2040-_EDGES_Y],
                        [2040-_EDGES_X,_EDGES_Y]]



############################
#                          #
#  Spectral Matcher        #
#                          #
############################
def build_spectmatcher(date, lamps=["Hg","Cd","Xe"], verbose=True):
    
    """ Create Spaxel trace Solution 
    This enable to know which pixel belong to which spaxel

    Parameters
    ----------
    date: [string]
        YYYMMDD 
    lamps: [list of lamp name] -optional-
        Which lamp will be used to improve the spectral matching

    Returns
    -------
    Void.  (Creates the file DATE_SpectralMatch.pkl)
    """
    from ..ccd import LampCCD
    timedir = get_datapath(date)
    try:
        os.mkdir(timedir+"ProdPlots/")
    except:
        warnings.warn("No Plot directory created. Most likely it already exists.")
        
    if verbose:
        print "Directory affected by Spectral Matcher : %s"%timedir

    smap = get_specmatcher(timedir+"dome.fits")
    # - Mercury
    if "Hg" in lamps:
        if verbose: print("loading Hg")
        Hg   = LampCCD(timedir+"Hg.fits", background=0)
        Hg.sep_extract(thresh=np.nanstd(Hg.rawdata))
        smap.add_arclamp(Hg, match=True)
    else:
        Hg = None
    # - Xenon        
    if "Xe" in lamps:
        if verbose: print("loading Xe")
        Xe   = LampCCD(timedir+"Xe.fits", background=0)
        Xe.sep_extract(thresh=np.nanstd(Hg.rawdata)*2)
        smap.add_arclamp(Xe, match=True)
    else:
        Xe = None
    # - Cadnium        
    if "Cd" in lamps:
        if verbose: print("loading Cd")
        Cd   = LampCCD(timedir+"Cd.fits", background=0)
        Cd.sep_extract(thresh=np.nanstd(Hg.rawdata))
        smap.add_arclamp(Cd, match=True)
    else:
        Cd = None

    if len(smap.arclamps.keys()) > 0:
        smap.set_arcbased_specmatch()
        
    smap.writeto(timedir+"%s_SpectralMatch.pkl"%date)

    # ======= Plot
    for case in [Hg, Xe, Cd]:
        if case is None: continue
        illustrate_traces(case, smap, savefile=timedir+"ProdPlots/%s_SpectralMatch_on_%s.pdf"%(date, case.objname))
        
############################
#                          #
# Spaxel Spacial Position  #
#                          #
############################


############################
#                          #
#  Wavelength Solution     #
#                          #
############################
def build_wavesolution(date, verbose=False, ntest=None,
                       lamps=["Hg","Cd","Xe"],
                       xybounds=None):
    """ Create the wavelength solution for the given night.
    The core of the solution fitting is made in pysedm.wavesolution.

    Parameters
    ----------

    Returns
    -------

    """
    from .. io import get_spectralmatch
    
    timedir = get_datapath(date)
    try:
        os.mkdir(timedir+"ProdPlots/")
    except:
        warnings.warn("No Plot directory created. Most likely it already exists.")
        
    if verbose:
        print "Directory affected by Wavelength Calibration: %s"%timedir

    # ----------------
    # - Load the Data
    # - SpectralMatch using domes
    #   Built by build_spectmatcher
    smap = get_spectralmatch(date)
        
    # - lamps 
    lamps = [get_lamp(timedir+"%s.fits"%s_, specmatch=smap) for s_ in lamps]
    if verbose: print "Cd, Hg and Xe lamp loaded"
    # - The CubeSolution
    csolution = get_cubesolution(*lamps)

    # ----------------
    # - Spaxel Selection
    if xybounds is None: xybounds=INDEX_CCD_CONTOURS
        
    if np.shape(xybounds)[0] ==2:
        idxall = smap.get_idx_within_bounds(*xybounds)
    else:
        from shapely.geometry import Polygon
        idxall =smap.get_idx_within_polygon(Polygon(xybounds))
        
    idx = idxall if ntest is None else np.random.choice(idxall,ntest, replace=False) 

    # - Do The loop and map it thanks to astropy
    from astropy.utils.console import ProgressBar
    def fitsolution(idx_):
        csolution.fit_wavelesolution(specid=idx_, saveplot=timedir+"ProdPlots/%s_wavesolution_spec%d.pdf"%(date,idx_),
                                         contdegree=2, plotprop={"show_guesses":True})
    ProgressBar.map(fitsolution, idx)

    dump_pkl(csolution.wavesolutions, timedir+"%s_WaveSolution.pkl"%date)
    
    
    
