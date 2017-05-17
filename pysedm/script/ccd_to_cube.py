#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
from astrobject.utils.tools import dump_pkl
from glob import glob
from .. io import get_datapath
from ..ccd import get_ccd
from ..spectralmatching import get_tracematcher, illustrate_traces
from ..wavesolution import get_wavesolution
import matplotlib.pyplot as mpl
from ..sedm import INDEX_CCD_CONTOURS, TRACE_DISPERSION




############################
#                          #
#  Spectral Matcher        #
#                          #
############################
def build_tracematcher(date, verbose=True, width=None, rebuild_nightly_trace=False,
                        build_finetuned_tracematcher=True, night_trace_only=False):
    
    """ Create Spaxel trace Solution 
    This enable to know which pixel belong to which spaxel

    Parameters
    ----------
    date: [string]
        YYYMMDD 
    lamps: [list of lamp name] -optional-
        Which lamp will be used to improve the spectral matching

    width: [None/float] -optional-
        What should be the trace width (in pixels)
        If None, this will use 2 times whatever is defined as "TRACE_DISPERSION" in sedm.py

    Returns
    -------
    Void.  (Creates the file TraceMatch.pkl)
    """
    from .. import io
    
    
    timedir = get_datapath(date)
    try:
        os.mkdir(timedir+"ProdPlots/")
    except:
        warnings.warn("No Plot directory created. Most likely it already exists.")
        
    if verbose:
        print "Directory affected by Spectral Matcher : %s"%timedir
        
    if width is None:
        width = 2.*TRACE_DISPERSION

    # - Load the Spectral Matcher
    if not rebuild_nightly_trace:
        try:
            smap = io.load_nightly_tracematch(date)
        except:
            rebuild_nightly_trace = True
        
    if rebuild_nightly_trace:
        print("Building Nightly Solution")
        smap = get_tracematcher(timedir+"dome.fits", width=width)
        smap.writeto(timedir+"%s_TraceMatch.pkl"%date)
        smap.writeto(timedir+"tracematch_dome.pkl")
        print("Nightly Solution Saved")
        if night_trace_only:
            return
    # ----------------- #
    #   Output          #
    # ----------------- #
    if not build_finetuned_tracematcher:
        return

    
        
    ccdfiles = io.get_night_ccdfiles(date, skip_calib=True) + glob(timedir+"Hg.fits") + glob(timedir+"Cd.fits") + glob(timedir+"Xe.fits")
    def build_finetubed_trace(ccdfile_):
        """ Buid the individual object. 
        This is called by ProgressBar
        """
        ccd_ = get_ccd(ccdfile_, tracematch=smap) # takes about 2s
        # - Do the FineTuning
        if ccd_.objname in ["Hg","Cd","Xe"]:
            thresh = np.nanpercentile(ccd_.rawdata, 80) # Massive emission lines
        else:
            coef = 1.5 if ccd_.objname not in ["Hg","Xe"] else 2. if ccd_.objname not in ["Hg"] else 0.5
            thresh = np.nanstd(ccd_.rawdata)*coef
            
        ccd_.sep_extract(thresh=thresh) # takes about 1min
        ccd_.match_trace_and_sep()
        
        # - Which Trace to use
        indexes = ccd_.tracematch.get_traces_within_polygon(INDEX_CCD_CONTOURS)
        # - Fine Tuning
        finetuned_trace = ccd_.get_finetuned_tracematch( indexes, width=width)
        # - Build the mas
        finetuned_trace.writeto(timedir+"tracematch_%s.pkl"%(ccdfile_.split("/")[-1].replace(".fits","")))
        del ccd_
        del finetuned_trace
        
    # - Progress Bar
    from astropy.utils.console import ProgressBar
    ProgressBar.map(build_finetubed_trace, ccdfiles, multiprocess=False)
        
############################
#                          #
# Spaxel Spacial Position  #
#                          #
############################
def build_hexagonalgrid(date, xybounds=None):
    """ """
    from ..io import load_nightly_tracematch
    smap  = load_nightly_tracematch(date)
    # ----------------
    # - Spaxel Selection
    if xybounds is None: xybounds=INDEX_CCD_CONTOURS
    idxall = smap.get_traces_within_polygon(INDEX_CCD_CONTOURS)

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
    from ..io import load_nightly_tracematch, get_file_tracematch
    
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
    smap = load_nightly_tracematch(date)
        
    # - lamps 
    lamps = [get_ccd(timedir+"%s.fits"%s_, tracematch= get_file_tracematch(date, s_) )
                 for s_ in lamps ]
    if verbose: print "Cd, Hg and Xe lamp loaded"
    # - The CubeSolution
    csolution = get_wavesolution(*lamps)

    # ----------------
    # - Spaxel Selection
    if xybounds is None:
        xybounds = INDEX_CCD_CONTOURS
        
    idxall = smap.get_traces_within_polygon(xybounds)
        
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
def build_night_cubes(date, finetune_trace=False,
                      lbda_min=3700, lbda_max=9200, lbda_npix=260,
                      only_calib=False, no_calib=False,
                      contains=None,test=None):
    """ """
    from .. import io
    # - The Files
    timedir  = get_datapath(date)
    calibkeys = ["Hg.fits","Cd.fits","Xe.fits","dome.fits"]
    calib_files = glob(timedir+"Hg.fits") + glob(timedir+"Cd.fits") + glob(timedir+"Xe.fits") + glob(timedir+"dome.fits")
    if not only_calib:
        fileccds = io.get_night_ccdfiles(date, skip_calib=True)
        if contains is not None:
            fileccds = [f for f in fileccds if contains in f]
        if not no_calib:
            fileccds += calib_files
    else:
        fileccds = calib_files
        
    # - The tools to build the cubes

    hgrid    = io.load_nightly_hexagonalgrid(date)
    wcol     = io.load_nightly_wavesolution(date)
    lbda     = np.linspace(lbda_min,lbda_max,lbda_npix)
    root     = io.CUBE_PROD_ROOTS["cube"]["root"]

    def build_cube(ccdfile):
        if np.any([calibkey_ in ccdfile for calibkey_ in calibkeys]):
            ccd_    = get_ccd(ccdfile, tracematch= io.get_file_tracematch(date, ccdfile.split("/")[-1].split(".")[0]))
        else:
            ccd_    = get_ccd(ccdfile, tracematch= io.get_file_tracematch(date, ccdfile.split(date)[-1].split(".")[0]))
            
        indexes = ccd_.tracematch.get_traces_within_polygon(INDEX_CCD_CONTOURS)
        cube_  = ccd_.extract_cube(wcol, lbda, hexagrid=hgrid, show_progress=True)
        if np.any([calibkey_ in ccdfile for calibkey_ in calibkeys]):
            filout = "%s"%(ccdfile.split("/")[-1].split(".fits")[0])
        else:
            filout = "%s_%s"%(ccdfile.split("/")[-1].split(".fits")[0], ccd_.objname)
        cube_.writeto(timedir+"%s_%s.fits"%(root,filout))

    from astropy.utils.console import ProgressBar
    ProgressBar.map(build_cube, [fileccds[test]] if test is not None else fileccds)

    
    
    
