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
from ..sedm import INDEX_CCD_CONTOURS, TRACE_DISPERSION, build_sedmcube






############################
#                          #
#  Spectral Matcher        #
#                          #
############################
def build_tracematcher(date, verbose=True, width=None,
                           save_masks=False,
                           rebuild_nightly_trace=False,
                           notebook=False):
    
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
        smap = get_tracematcher(glob(timedir+"dome.fits.*")[0], width=width)
        smap.writeto(timedir+"%s_TraceMatch.pkl"%date)
        print("Nightly Solution Saved")
        
    if save_masks:
        load_trace_masks(smap, smap.get_traces_within_polygon(INDEX_CCD_CONTOURS), notebook=notebook)
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
# Spaxel Spacial Position  #
#                          #
############################
def build_flatfield(date, lbda_min=7000, lbda_max=9000, ref="dome",
                    kind="median", savefig=True):
    """ """
    from ..sedm import get_sedmcube

    
    reffile  = io.get_night_cubes(date, kind="cube", target=ref)[0]
    refcube  = get_sedmcube(reffile)
    sliceref = refcube.get_slice(lbda_min, lbda_max, usemean=True)
    # - How to normalize the Flat
    if kind in ["med", "median"]:
        norm = np.median(sliceref)
    elif kind in ["mean"]:
        norm = np.mean(sliceref)
    elif kind in refcube.indexes:
        norm = sliceref[np.argwhere(refcube.indexes==kind)]
    else:
        raise ValueError("Unable to parse the given kind: %s"%kind)
    
    # - The Flat
    flat     = sliceref / norm

    # - Figure
    if savefig:
        timedir  = io.get_datapath(date)
        try:
            os.mkdir(timedir+"ProdPlots/")
        except:
            pass
    
        from ..sedm import display_on_hexagrid
        xy = np.asarray(refcube.index_to_xy(refcube.indexes)).T
        display_on_hexagrid(flat, refcube.indexes, xy=xy, vmin="0.5",vmax="99.5",
                        outlier_highlight=5, savefile=timedir+"ProdPlots/%s_flat3d.pdf"%date)

    # - Savefing
    from astropy.io import fits as pf
    hdul = []
    # -- Data saving
    hduP = pf.PrimaryHDU(flat, header=refcube.header)

    hduP.header["CALTYPE"] = "FlatField"
    hduP.header["FLATSRC"]  = ref
    hduP.header["FLATREF"]  = kind
    hdul.append(hduP)
    
    hdul.append(pf.ImageHDU([v for i,v in refcube.spaxel_mapping.items()], name='MAPPING'))
    hdul.append(pf.ImageHDU([i for i,v in refcube.spaxel_mapping.items()], name='SPAX_ID'))
    hdul.append(pf.ImageHDU(refcube.spaxel_vertices, name='SPAX_VERT'))
    
    
    hdulist = pf.HDUList(hdul)
    
    hdulist.writeto(timedir+'%s_Flat.fits'%date)
    
############################
#                          #
#   BackGround             #
#                          #
############################
def build_backgrounds(date, smoothing=[0,2], start=2, jump=10, 
                        target=None, lamps=True, only_lamps=False,
                        skip_calib=True, starts_with="crr_b", contains="*",
                        multiprocess=True,
                        savefig=True, notebook=False):
    """ """
    from ..background import build_background
    timedir  = io.get_datapath(date)
    try:
        os.mkdir(timedir+"ProdPlots/")
    except:
        pass
    
    # - The Files
    fileccds = []
    if target is None:
        if lamps:
            lamp_files = glob(timedir+"Hg.fits.*") + glob(timedir+"Cd.fits.*") + glob(timedir+"Xe.fits.*") + glob(timedir+"dome.fits.*")
            fileccds  += lamp_files

        if not only_lamps:
            fileccds_ = io.get_night_ccdfiles(date, skip_calib=skip_calib, **kwargs)
            fileccds  += fileccds_
    else:
        if target in ["dome","Hg","Cd","Xe"]:
            starts_with = ""
            
        fileccds += [f for f in io.get_night_ccdfiles(date, skip_calib=False, contains=target, **kwargs)
                         if "e3d" not in f and "bkgd" not in f] # to avoid picking cubes or background files
        
        
    tmap = io.load_nightly_tracematch(date)
    nfiles = len(fileccds)
    print("%d files to go..."%nfiles)
    for i,file_ in enumerate(fileccds):
        build_background(get_ccd(file_, tracematch=tmap, background=0),
                        start=start, jump=jump, multiprocess=multiprocess, notebook=notebook,
                        smoothing=smoothing,
            savefile = None if not savefig else timedir+"ProdPlots/bkgd_%s.pdf"%(file_.split('/')[-1].replace(".fits","")))
        
        
        
    

    
############################
#                          #
#  Wavelength Solution     #
#                          #
############################
def build_wavesolution(date, verbose=False, ntest=None, use_fine_tuned_traces=False,
                       lamps=["Hg","Cd","Xe"], savefig=True, saveindividuals=False,
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
        
    if use_fine_tuned_traces:
        #lamps = [get_ccd(timedir+"%s.fits"%s_, tracematch= io.get_file_tracematch(date, s_) if use_fine_tuned_traces else smap)
        #           for s_ in lamps ]
        raise ValueError("use_fine_tuned_traces is not supported anymore")
    lamps = [get_ccd(glob(timedir+"%s.fits.*"%s_)[0], tracematch=smap) for s_ in lamps]
    
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
        saveplot = None if not saveindividuals else \
          timedir+"ProdPlots/%s_wavesolution_spec%d.pdf"%(date,idx_)
        csolution.fit_wavelesolution(traceindex=idx_, saveplot=saveplot,
                    contdegree=2, plotprop={"show_guesses":True})
        if saveplot is not None:
            mpl.close("all") # just to be sure
            
    ProgressBar.map(fitsolution, idx)

    dump_pkl(csolution.wavesolutions, timedir+"%s_WaveSolution.pkl"%date)
    if savefig:
        
        hexagrid = io.load_nightly_hexagonalgrid(date)
        csolution.show_dispersion_map(hexagrid,vmin="0.5",vmax="99.5",
                                      outlier_highlight=5, savefile= timedir+"ProdPlots/%s_wavesolution_dispersionmap.pdf"%date)
        
    
############################
#                          #
#  Build Cubes             #
#                          #
############################
def build_night_cubes(date, lbda=None,
                      target=None, lamps=True, only_lamps=False, skip_calib=True, no_bkgd_sub=False,
                      test=None, notebook=False, **kwargs):
    """ 
    **kwargs goes to get_night_ccdfiles()
    """
    
    timedir  = io.get_datapath(date)
    
    # - The Files
    fileccds = []
    if target is None:
        if lamps:
            lamp_files = glob(timedir+"Hg.fits.*") + glob(timedir+"Cd.fits.*") + glob(timedir+"Xe.fits.*") + glob(timedir+"dome.fits.*")
            fileccds  += lamp_files

        if not only_lamps:
            fileccds_ = io.get_night_ccdfiles(date, skip_calib=skip_calib, **kwargs)
            fileccds  += fileccds_
    else:
        if "starts_with" not in kwargs and target in ["dome","Hg","Cd","Xe"]:
            kwargs['starts_with'] = ""
            
        fileccds += [f for f in io.get_night_ccdfiles(date, skip_calib=False, contains=target, **kwargs)
                         if "e3d" not in f and "bkgd" not in f] # to avoid picking cubes or background files

    print(fileccds)
    # - The tools to build the cubes
    if test:
        return
    # Traces for the CCD
    tmatch   = io.load_nightly_tracematch(date, withmask=True) 
    
    # - HexaGrid
    hgrid    = io.load_nightly_hexagonalgrid(date)

    wcol     = io.load_nightly_wavesolution(date)
    wcol._load_full_solutions_()
    

    print("All Roots loaded")    
    def build_cube(ccdfile):
        ccd_    = get_ccd(ccdfile, tracematch = tmatch, background = 0)
        ccd_.fetch_background(set_it=True, build_if_needed=True)
        # - Variance
        if not ccd_.has_var():
            ccd_.set_default_variance()
            
        # - see pysedm.sedm
        build_sedmcube(ccd_, date, lbda=lbda, wavesolution=wcol, hexagrid=hgrid)

    # - Actual Build (no ProgressBar for only 1 case
    to_be_built = [fileccds[test]] if test is not None else fileccds
    if len(to_be_built)>1:
        from astropy.utils.console import ProgressBar
        ProgressBar.map(build_cube, to_be_built)
    elif len(to_be_built)==1:
        build_cube(to_be_built[0])

    
def save_cubeplot(date, kind):
    """ """
    from ..sedm import get_sedmcube
    timedir  = io.get_datapath(date)
    try:
        os.mkdir(timedir+"ProdPlots/")
    except:
        pass
    
    for cubefile in io.get_night_cubes(date, kind):
        cube = get_sedmcube(cubefile)
        cube.show(savefile= timedir+"ProdPlots/%s.pdf"%(cubefile.split('/')[-1].replace(".fits","")))




#################################
#
#   MAIN 
#
#################################
if  __name__ == "__main__":
    print("see pysedm/bin/ccd_to_cube.py")
