#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
import matplotlib.pyplot as mpl
from glob import glob

from astrobject.utils.tools import dump_pkl
from astropy.io import fits

from .. import io

from ..ccd import get_ccd
from ..spectralmatching import get_tracematcher, illustrate_traces, load_trace_masks
from ..wavesolution import get_wavesolution

from ..sedm import INDEX_CCD_CONTOURS, TRACE_DISPERSION, build_sedmcube, build_calibrated_sedmcube, SEDM_LBDA






############################
#                          #
#  Spectral Matcher        #
#                          #
############################
def build_tracematcher(date, verbose=True, width=None,
                           save_masks=False,
                           rebuild=False,
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
        print("Directory affected by Spectral Matcher : %s"%timedir)
        
    if width is None:
        width = 2.*TRACE_DISPERSION

    # - Load the Spectral Matcher
    if not rebuild:
        try:
            smap = io.load_nightly_tracematch(date)
        except:
            rebuild = True
        
    if rebuild:
        print("Building Nightly Solution")
        smap = get_tracematcher(glob(timedir+"dome.fits*")[0], width=width)
        smap.writeto(timedir+"%s_TraceMatch.pkl"%date)
        print("Nightly Solution Saved")
        
    if save_masks:
        if not rebuild and len(glob(timedir+"%s_TraceMatch_WithMasks.pkl"%date))>0:
            warnings.warn("TraceMatch_WithMasks already exists for %s. rebuild is False, so nothing is happening"%date)
            return
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
def build_flatfield(date, lbda_min=7000, lbda_max=9000,
                    ref="dome", build_ref=True,
                    kind="median", savefig=True):
    """ """
    from ..sedm import get_sedmcube
    from pyifu.spectroscopy  import get_slice

    reffile  = io.get_night_files(date, kind="cube.basic", target=ref)

    # - If the reference if not there yet.
    if len(reffile)==0:
        warnings.warn("The reference cube %s does not exist "%ref)
        if build_ref:
            warnings.warn("build_flatfield is building it!")
        else:
            raise IOError("No reference cube to build the flatfield (build_ref was set to False)")
        # --------------------- #
        # Build the reference   #
        # --------------------- #
        tmatch   = io.load_nightly_tracematch(date, withmask=True) 
        # - The CCD
        ccdreffile = io.get_night_files(date, kind="ccd.lamp", target=ref)[0]
        ccdref     = get_ccd(ccdreffile, tracematch = tmatch, background = 0)
        ccdref.fetch_background(set_it=True, build_if_needed=True)
        if not ccdref.has_var():
            ccdref.set_default_variance()
        # - HexaGrid
        hgrid    = io.load_nightly_hexagonalgrid(date)
        wcol     = io.load_nightly_wavesolution(date)
        wcol._load_full_solutions_()
        # - Build a cube
        build_sedmcube(ccdref, date, lbda=None, wavesolution=wcol, hexagrid=hgrid,
                        flatfielded=False, build_calibrated_cube=False,atmcorrected=False)
        
    # ---------------------- #
    #  Actual FlatFielding   #
    # ---------------------- #
    reffile  = reffile  = io.get_night_files(date, kind="cube.basic", target=ref)[0]
    refcube  = get_sedmcube(reffile)
    sliceref = refcube.get_slice(lbda_min, lbda_max, usemean=True)
    # - How to normalize the Flat
    if kind in ["med", "median"]:
        norm = np.nanmedian(sliceref)
    elif kind in ["mean"]:
        norm = np.nanmean(sliceref)
    elif kind in refcube.indexes:
        norm = sliceref[np.argwhere(refcube.indexes==kind)]
    else:
        raise ValueError("Unable to parse the given kind: %s"%kind)
    
    # - The Flat
    flat     = sliceref / norm
    slice_ = get_slice(flat, np.asarray(refcube.index_to_xy(refcube.indexes)),
                        refcube.spaxel_vertices,
                        indexes=refcube.indexes, variance=None, lbda=None)
    # - Figure
    timedir  = io.get_datapath(date)
    # - Saving
    slice_.header["CALTYPE"] = "FlatField"
    slice_.header["FLATSRC"]  = ref
    slice_.header["FLATREF"]  = kind
    slice_.writeto(timedir+'%s_Flat.fits'%date)
    
    if savefig:
        print("Saving flat3d Figure")
        slice_.show(savefile=timedir+"%s_flat3d.pdf"%date)

    
############################
#                          #
#   BackGround             #
#                          #
############################
def build_backgrounds(date, smoothing=[0,2], start=2, jump=10, 
                        target=None, lamps=True, only_lamps=False, skip_calib=True,
                        multiprocess=True,
                        savefig=True, notebook=False, **kwargs):
    """ """
    from ..background import build_background
    timedir  = io.get_datapath(date)
    # - The files
    fileccds = []
    if not only_lamps:
        crrfiles  = io.get_night_files(date, "ccd.crr", target=target)
        if skip_calib: fileccds = [f for f in crrfiles if "Calib" not in fits.getval(f,"Name")]            
        fileccds += crrfiles

    # - Building the background
    tmap = io.load_nightly_tracematch(date)
    nfiles = len(fileccds)
    print("%d files to go..."%nfiles)
    for i,file_ in enumerate(fileccds):
        build_background(get_ccd(file_, tracematch=tmap, background=0),
                        start=start, jump=jump, multiprocess=multiprocess, notebook=notebook,
                        smoothing=smoothing,
            savefile = None if not savefig else timedir+"bkgd_%s.pdf"%(file_.split('/')[-1].replace(".fits","")))
        
    
############################
#                          #
#  Wavelength Solution     #
#                          #
############################
def build_wavesolution(date, verbose=False, ntest=None, use_fine_tuned_traces=False,
                        wavedegree=4, contdegree=3,
                       lamps=["Hg","Cd","Xe"], savefig=True, saveindividuals=False,
                       xybounds=None, rebuild=True):
    """ Create the wavelength solution for the given night.
    The core of the solution fitting is made in pysedm.wavesolution.

    Parameters
    ----------
    
    Returns
    -------

    """
    timedir = io.get_datapath(date)
        
    if verbose:
        print("Directory affected by Wavelength Calibration: %s"%timedir)


    if not rebuild and len(glob(timedir+"%s_WaveSolution.pkl"%(date)))>0:
        warnings.warn("WaveSolution already exists for %s. rebuild is False, so nothing is happening"%date)
        return
    
    # ----------------
    # - Load the Data
    # - SpectralMatch using domes
    #   Built by build_spectmatcher
    smap = io.load_nightly_tracematch(date, withmask=True)
        
    if use_fine_tuned_traces:
        raise ValueError("use_fine_tuned_traces is not supported anymore")
    
    fileccd_lamps = io.get_night_files(date, "ccd.lamp", target="|".join(lamps))
    lamps = [get_ccd(f_, tracematch=smap) for f_ in fileccd_lamps]
    
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
        if saveindividuals:
            saveplot = timedir+"%s_wavesolution_trace%d.pdf"%(date,idx_)
        else:
            saveplot = None
        csolution.fit_wavelesolution(traceindex=idx_, saveplot=None,
                    contdegree=contdegree, wavedegree=wavedegree, plotprop={"show_guesses":True})
        if saveplot is not None:
            csolution._wsol.show(show_guesses=True, savefile=saveplot)
            mpl.close("all")
            
    ProgressBar.map(fitsolution, idx)
    dump_pkl(csolution.wavesolutions, timedir+"%s_WaveSolution.pkl"%date)
    if savefig:
        wsol = io.load_nightly_wavesolution(date)
        hexagrid = io.load_nightly_hexagonalgrid(date)
        pl = wsol.show_dispersion_map(hexagrid,vmin="0.5",vmax="99.5",
                                outlier_highlight=5, show=False)
        pl['fig'].savefig(timedir+"%s_wavesolution_dispersionmap.pdf"%date)
        
    
############################
#                          #
#  Build Cubes             #
#                          #
############################
def build_night_cubes(date, target=None, lamps=True, only_lamps=False,
                          skip_calib=True, **kwargs):
    """ 
    """
    fileccds = []
    if lamps:
        fileccds += io.get_night_files(date, "ccd.lamp", target=target)
    if not only_lamps:
        crrfiles  = io.get_night_files(date, "ccd.crr", target=target)
        if skip_calib: crrfiles = [f for f in crrfiles if "Calib" not in fits.getval(f,"Name")]            
        fileccds += crrfiles


    print(fileccds)
    build_cubes(fileccds, date, **kwargs)


# ----------------- #
#  Build Cubes      #
# ----------------- #
def build_cubes(ccdfiles,  date, lbda=None,
                tracematch=None, wavesolution=None, hexagrid=None,
                flatfielded=True, flatfield=None,
                atmcorrected=True, 
                build_calibrated_cube=True, calibration_ref=None):
    """ Build a cube from the an IFU ccd image. This image 
    should be bias corrected and cosmic ray corrected.

    The standard created cube will be 3Dflat field correct 
    (see flatfielded) and corrected for atmosphere extinction 
    (see atmcorrected). A second cube will be flux calibrated 
    if possible (see build_calibrated_cube).

    Parameters
    ----------
    ccd: [CCD]
        A ccd object from which the cube will be extracted.
        
    date: [string]
        date in usual YYYYMMDD format
    
    lbda: [None/array] -optional-
        wavelength array used to build the cube. 
        If not given the default sedm wavelength range will be used. 
        See pysedm.sedm.SEDM_LBDA.

    // Cube Calibrator //
    
    wavesolution: [WaveSolution] -optional-
        The wavelength solution containing the pixel<->wavelength conversion.
        If None, this will be loaded using `date`.

    hexagrid: [HexaGrid] -optional-
        The Hexagonal Grid tools containing the index<->qr<->xy conversions.
        If None, this will be loaded using `date`.

    flatfield: [Slice] -optional-
        Object containing the relative transmission of the IFU spaxels.
        If None, this will be loaded using `date`.

    // Action Selection //
    
    flatfielded: [bool] -optional-
        Shall the cube be flatfielded?
        - This information will be saved in the header-
    
    atmcorrected: [bool] -optional-
        Shall the cube the corrected for atmosphere extinction?
        - This information will be saved in the header-

    // Additional outcome: Flux calibrated cube //

    build_calibrated_cube: [bool] -optional-
        Shall this method build an additionnal flux calibrated cube?
        
    calibration_ref: [None/string] -optional-
        If you want to build a calibrated cube, you can provide the filename
        of the spectrum containing the inverse-sensitivity (fluxcal*)
        If None, this will load the latest fluxcal object of the night.
        If Nothing found, no flux calibrated cube will be created. 

    Returns
    -------
    Void
    """
    # ------------------ #
    # Loading the Inputs #
    # ------------------ #
    if tracematch is None:
        tracematch   = io.load_nightly_tracematch(date, withmask=True) 
        
    if hexagrid is None:
        hexagrid     = io.load_nightly_hexagonalgrid(date)
    
    if wavesolution is None:
        wavesolution = io.load_nightly_wavesolution(date)
        wavesolution._load_full_solutions_()
    
    if lbda is None:
        lbda = SEDM_LBDA

    if flatfielded and flatfield is None:
        flatfield = io.load_nightly_flat(date)

    # ---------------- #
    # Loading the CCDS #
    # ---------------- #
    ccds = []
    for ccdfile in ccdfiles:
        ccd_    = get_ccd(ccdfile, tracematch = tracematch, background = 0)
        ccd_.fetch_background(set_it=True, build_if_needed=True)
        # - Variance
        if not ccd_.has_var():
            ccd_.set_default_variance()
        ccds.append(ccd_)

    # ---------------- #
    # Build the Cubes  #
    # ---------------- #
    # internal routine 
    def _build_cubes_(ccdin):
        build_sedmcube(ccdin,
                        date, lbda=lbda, wavesolution=wavesolution, hexagrid=hexagrid,
                        flatfielded=flatfielded, flatfield=flatfield,
                        atmcorrected=atmcorrected,  
                        build_calibrated_cube=build_calibrated_cube,
                        calibration_ref=calibration_ref)
    # The actual build
    if len(ccds)>1:
        from astropy.utils.console import ProgressBar
        ProgressBar.map(_build_cubes_, ccds)
    else:
        _build_cubes_(ccds[0])
        
# ---------------- #
# Flux Calibration #
# ---------------- #
def calibrate_night_cubes(date, target=None, calibrated_reference=None):
    """ """
    files = io.get_night_files(date, "cube.basic", target=target)
    flux_calibrate_cubes(files, date=date, calibrated_reference=calibrated_reference)
    

    
def calibrate_cubes(cubefiles, date=None, calibrated_reference=None,
                             multiprocess=True):
    """ """
    # internal routine 
    def _build_cal_cubes_(cubefile):
        build_calibrated_sedmcube(cubefile, date=date,
                                calibration_ref=calibrated_reference)
    # - The build
    if len(cubefiles)==1:
        _build_cal_cubes_(cubefiles[0])
    else:
        from astropy.utils.console import ProgressBar
        ProgressBar.map(_build_cal_cubes_, cubefiles, multiprocess=multiprocess, step=2)
        
    
def save_cubeplot(date, kind="cube.basic"):
    """ """
    from ..sedm import get_sedmcube
    timedir  = io.get_datapath(date)
    
    for cubefile in io.get_night_files(date, kind):
        cube = get_sedmcube(cubefile)
        cube.show(savefile= timedir+"%s.pdf"%(cubefile.split('/')[-1].replace(".fits.gz",".pdf").replace(".fits",".pdf")))




#################################
#
#   MAIN 
#
#################################
if  __name__ == "__main__":
    print("see pysedm/bin/ccd_to_cube.py")
