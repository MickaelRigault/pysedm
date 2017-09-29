#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
from astrobject.utils.tools import dump_pkl
from glob import glob
from .. import io

from ..ccd import get_ccd
from ..spectralmatching import get_tracematcher, illustrate_traces, load_trace_masks
from ..wavesolution import get_wavesolution
import matplotlib.pyplot as mpl
from ..sedm import INDEX_CCD_CONTOURS, TRACE_DISPERSION



CALIBFILES = ["Hg.fits","Cd.fits","Xe.fits","dome.fits"]
############################
#                          #
#  Spectral Matcher        #
#                          #
############################
def build_tracematcher(date, verbose=True, width=None,
                        night_trace_only=True, save_masks=False,
                        rebuild_nightly_trace=False, notebook=False):
    
    """ Create Spaxel trace Solution 
    This enable to know which pixel belong to which spaxel

    Parameters
    ----------
    date: [string]
        YYYMMDD 

    width: [None/float] -optional-
        What should be the trace width (in pixels)
        If None, this will use 2 times whatever is defined as "TRACE_DISPERSION" in sedm.py
        
    save_masks: [bool] -optional-
        Shall this measures all the individual masks and save them?
        This mask building is the most cpu expensive part of the pipeline, 
        it takes about 30min/number_of_core. If this is saved (~150Mo) 
        wavelength-solution and cube-building will be faster.
        
    Returns
    -------
    Void.  (Creates the file TraceMatch.pkl and TraceMatch_WithMasks.pkl if save_masks)
    """
    
    
    timedir = io.get_datapath(date)
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
        print("Nightly Solution Saved")
        
    if save_masks:
        load_trace_masks(smap, smap.get_traces_within_polygon(sedm.INDEX_CCD_CONTOURS), notebook=notebook)
        smap.writeto(timedir+"%s_TraceMatch_WithMasks.pkl"%date)
    
############################
#                          #
# Spaxel Spacial Position  #
#                          #
############################
def build_hexagonalgrid(date, xybounds=None):
    """ """
    smap  = io.load_nightly_tracematch(date)
    # ----------------
    # - Spaxel Selection
    if xybounds is None: xybounds=INDEX_CCD_CONTOURS
    idxall = smap.get_traces_within_polygon(INDEX_CCD_CONTOURS)

    hgrid = smap.extract_hexgrid(idxall)

    timedir = io.get_datapath(date)
    hgrid.writeto(timedir+"%s_HexaGrid.pkl"%date)

############################
#                          #
#   BackGround             #
#                          #
############################
def build_backgrounds(date, lamps=True, only_lamps=False,
                          skip_calib=True, starts_with="crr_b",contains="*",
                     start=2, jump=10, multiprocess=True,
                     savefig=True, notebook=False, ):
    """ """
    timedir  = io.get_datapath(date)
    try:
        os.mkdir(timedir+"ProdPlots/")
    except:
        pass
    
    # - which files ?
    fileccds = []
    if lamps:
        lamp_files = glob(timedir+"Hg.fits") + glob(timedir+"Cd.fits") + glob(timedir+"Xe.fits") + glob(timedir+"dome.fits")
        fileccds  += lamp_files

    if not only_lamps:
        fileccds_ = io.get_night_ccdfiles(date, skip_calib=skip_calib, starts_with=starts_with, contains=contains)
        fileccds  += fileccds_

        
        
    tmap = io.load_nightly_tracematch(date)
    nfiles = len(fileccds)
    print("%d files to go..."%nfiles)
    for i,file_ in enumerate(fileccds):
        ccd_ = get_ccd(file_, tracematch=tmap, background=0)
        ccd_.fit_background(start=start, jump=jump, multiprocess=multiprocess, notebook=notebook,
                                set_it=False, is_std= io.is_stdstars(ccd_.header) )
        ccd_._background.writeto(io.filename_to_background_name(file_))
        if savefig:
            ccd_._background.show(savefile=timedir+"ProdPlots/bkgd_%s.pdf"%(file_.split('/')[-1].replace(".fits","")))
            mpl.close("all")
        
        
    

    
############################
#                          #
#  Wavelength Solution     #
#                          #
############################
def build_wavesolution(date, verbose=False, ntest=None, use_fine_tuned_traces=False,
                       lamps=["Hg","Cd","Xe"], savefig=True,
                       xybounds=None):
    """ Create the wavelength solution for the given night.
    The core of the solution fitting is made in pysedm.wavesolution.

    Parameters
    ----------
    
    Returns
    -------

    """
    timedir = io.get_datapath(date)
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
    smap = io.load_nightly_tracematch(date, withmask=True)
        
    # - lamps 
    lamps = [get_ccd(timedir+"%s.fits"%s_, tracematch= io.get_file_tracematch(date, s_) if use_fine_tuned_traces else smap)
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
        csolution.fit_wavelesolution(traceindex=idx_, saveplot=saveplot,
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
def build_night_cubes(date, 
                      lbda_min=3700, lbda_max=9200, lbda_npix=260,
                      lamps=True, only_lamps=False, skip_calib=True, no_bkgd_sub=False,
                      test=None, notebook=False, **kwargs):
    """ 
    **kwargs goes to get_night_ccdfiles()
    """
    
    timedir  = io.get_datapath(date)
    
    # - The Files
    fileccds = []
    if lamps:
        lamp_files = glob(timedir+"Hg.fits") + glob(timedir+"Cd.fits") + glob(timedir+"Xe.fits") + glob(timedir+"dome.fits")
        fileccds  += lamp_files

    if not only_lamps:
        fileccds_ = io.get_night_ccdfiles(date, skip_calib=skip_calib, **kwargs)
        fileccds  += fileccds_

    print(fileccds)
    # - The tools to build the cubes
    if test:
        return
    # Traces
    tmatch   = io.load_nightly_tracematch(date, withmask=True)
    indexes  = tmatch.get_traces_within_polygon(INDEX_CCD_CONTOURS)
    
    hgrid    = io.load_nightly_hexagonalgrid(date)

    wcol     = io.load_nightly_wavesolution(date)
    wcol._load_full_solutions_()
    
    lbda     = np.linspace(lbda_min,lbda_max,lbda_npix)
    root     = io.CUBE_PROD_ROOTS["cube"]["root"]

    print("All Roots loaded")
    
    def build_cube(ccdfile):
        ccd_    = get_ccd(ccdfile, tracematch = tmatch, background = 0)
        # - Background
        if not no_bkgd_sub:
            ccd_.fetch_background()
        # - Variance
        if not ccd_.has_var():
            ccd_.set_default_variance()
        
        # Build the cube
        cube_   = ccd_.extract_cube(wcol, lbda, hexagrid=hgrid, show_progress=True)
        
        # Save the cube
        if np.any([calibkey_ in ccdfile for calibkey_ in CALIBFILES]):
            filout = "%s"%(ccdfile.split("/")[-1].split(".fits")[0])
        else:
            filout = "%s_%s"%(ccdfile.split("/")[-1].split(".fits")[0], ccd_.objname)
        if not no_bkgd_sub:
            cube_.writeto(timedir+"%s_%s.fits"%(root,filout))
        else:
            cube_.writeto(timedir+"%s_nobkgdsub_%s.fits"%(root,filout))

    from astropy.utils.console import ProgressBar
    ProgressBar.map(build_cube, [fileccds[test]] if test is not None else fileccds)

    
    
    
