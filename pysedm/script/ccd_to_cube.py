#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
from astrobject.utils.tools import dump_pkl
from .. io import get_datapath
from ..ccd import get_ccd
from ..spectralmatching import get_tracematcher, illustrate_traces
from ..wavesolution import get_wavesolution
import matplotlib.pyplot as mpl

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
def build_tracematcher(date, verbose=True, 
                        build_finetuned_tracematcher=True):
    
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
    Void.  (Creates the file TraceMatch.pkl)
    """
    from .. import io
    from glob import glob
    
    
    timedir = get_datapath(date)
    try:
        os.mkdir(timedir+"ProdPlots/")
    except:
        warnings.warn("No Plot directory created. Most likely it already exists.")
        
    if verbose:
        print "Directory affected by Spectral Matcher : %s"%timedir
    
    # - Load the Spectral Matcher
    smap = get_tracematcher(timedir+"dome.fits")
    smap.writeto(timedir+"%s_TraceMatch.pkl"%date)

    # ----------------- #
    #   Output          #
    # ----------------- #
    if not build_finetuned_tracematcher:
        return
    
    ccdfiles = io.get_night_ccdfiles(date, skip_calib=True) + glob(timedir+"Hg.fits") + glob(timedir+"Cd.fits") + glob(timedir+"Xe.fits")
    def build_finetubed_trace(ccdfile_):
        """ """
        ccd_ = get_ccd(ccdfile_, tracematch=smap) # takes about 2s
        coef = 1 if ccd_.objname not in ["Hg","Xe"] else 2. if ccd_.objname not in ["Hg"] else 0.5
        ccd_.sep_extract(thresh=np.nanstd(ccd_.rawdata)*coef) # takes about 1min
        ccd_.match_trace_and_sep()
        
        tmap = ccd_.get_finetuned_tracematch( ccd_.tracematch.get_traces_within_polygon(INDEX_CCD_CONTOURS))
        tmap.writeto(timedir+"tracematch_%s.pkl"%(ccdfile_.split("/")[-1].replace(".fits","")))
        del ccd_
        del tmap
        
    # - Progress Bar
    from astropy.utils.console import ProgressBar
    ProgressBar.map(build_finetubed_trace, ccdfiles)
        
############################
#                          #
# Spaxel Spacial Position  #
#                          #
############################
def build_hexagonalgrid(date, xybounds=None):
    """ """
    from ..io import load_nightly_spectralmatch
    smap  = load_nightly_spectralmatch(date)
    # ----------------
    # - Spaxel Selection
    if xybounds is None: xybounds=INDEX_CCD_CONTOURS
        
    if np.shape(xybounds)[0] ==2:
        idxall = smap.get_idx_within_bounds(*xybounds)
    else:
        from shapely.geometry import Polygon
        idxall = smap.get_idx_within_polygon(Polygon(xybounds))

    hgrid = smap.extract_hexgrid(idxall)

    timedir = get_datapath(date)
    hgrid.writeto(timedir+"%s_HexaGrid.pkl"%date)
    
############################
#                          #
#  Wavelength Solution     #
#                          #
############################
def build_wavesolution(date, verbose=False, ntest=None,
                       lamps=["Hg","Cd","Xe"], savefig=True,
                       xybounds=None):
    """ Create the wavelength solution for the given night.
    The core of the solution fitting is made in pysedm.wavesolution.

    Parameters
    ----------

    Returns
    -------

    """
    from ..io import load_nightly_spectralmatch
    
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
    smap = load_nightly_spectralmatch(date)
        
    # - lamps 
    lamps = [get_ccd(timedir+"%s.fits"%s_, specmatch=smap) for s_ in lamps]
    if verbose: print "Cd, Hg and Xe lamp loaded"
    # - The CubeSolution
    csolution = get_wavesolution(*lamps)

    # ----------------
    # - Spaxel Selection
    if xybounds is None: xybounds=INDEX_CCD_CONTOURS
        
    if np.shape(xybounds)[0] ==2:
        idxall = smap.get_idx_within_bounds(*xybounds)
    else:
        from shapely.geometry import Polygon
        idxall = smap.get_idx_within_polygon(Polygon(xybounds))
        
    idx = idxall if ntest is None else np.random.choice(idxall,ntest, replace=False) 

    # - Do The loop and map it thanks to astropy
    from astropy.utils.console import ProgressBar
    def fitsolution(idx_):
        saveplot = None if not savefig else \
          timedir+"ProdPlots/%s_wavesolution_spec%d.pdf"%(date,idx_)
        csolution.fit_wavelesolution(specid=idx_, saveplot=saveplot,
                    contdegree=2, plotprop={"show_guesses":True})
        if saveplot is not None:
            mpl.close("all") # just to be sure
            
    ProgressBar.map(fitsolution, idx)

    dump_pkl(csolution.wavesolutions, timedir+"%s_WaveSolution.pkl"%date)
    
    
    
############################
#                          #
#  Build Cubes             #
#                          #
############################
def build_night_cubes(date, skip_calib=True, finetune_trace=False,
                      lbda_min=4000, lbda_max=9000, lbda_npix=250):
    """ """
    from .. import io
    
    # - The Files
    timedir  = get_datapath(date)
    fileccds = io.get_night_ccdfiles(date, skip_calib=skip_calib)
    # - The tools to build the cubes
    smap     = io.load_nightly_spectralmatch(date)
    hgrid    = io.load_nightly_hexagonalgrid(date)
    wcol     = io.load_nightly_wavesolution(date)

    lbda = np.linspace(lbda_min,lbda_max,lbda_npix)
    root = io.CUBE_PROD_ROOTS["cube"]["root"]
    def build_cube(ccdfile):
        ccd_   = get_ccd(ccdfile, tracematch=smap)
        if finetune_trace:
            ccd_.sep_extract(thresh=np.nanstd(ccd_.rawdata)*2)
            ccd_.match_specmatch_and_sep()
            
        cube_  = ccd_.extract_cube(wcol, lbda, hexagrid=hgrid,
                                    finetune_trace=finetune_trace)
        filout = ccdfile.split("/")[-1].split(".fits")[0]
        cube_.writeto(timedir+"%s_%s_%s.fits"%(root,filout,ccd_.objname))

    from astropy.utils.console import ProgressBar
    ProgressBar.map(build_cube, fileccds)
    
    
    
