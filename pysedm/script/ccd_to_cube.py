#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob

from astrobject.utils.tools import dump_pkl
from astropy.io import fits

from .. import io

from ..ccd import get_ccd
from ..spectralmatching import get_tracematcher, illustrate_traces, load_trace_masks

from ..sedm import INDEX_CCD_CONTOURS, TRACE_DISPERSION, build_sedmcube, build_calibrated_sedmcube, SEDM_LBDA

############################
#                          #
#  Spectral Matcher        #
#                          #
############################
def build_tracematcher(date, verbose=True, width=None,
                           save_masks=False,
                           rebuild=False,
                           ncore=None):
    
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
        
    if verbose:
        print("Directory affected by Spectral Matcher : %s"%timedir)
        
    if width is None:
        width = TRACE_DISPERSION

    # - Load the Spectral Matcher
    if not rebuild:
        try:
            smap = io.load_nightly_tracematch(date)
        except:
            rebuild = True
        
    if rebuild:
        print("Building Nightly Solution")
        smap = get_tracematcher( glob(timedir+"dome.fits*")[0], width=width)
        smap.writeto(timedir+f"{date}_TraceMatch.pkl")
        print("Nightly Solution Saved")
        
    if save_masks:
        if not rebuild and len(glob(timedir+"%s_TraceMatch_WithMasks.pkl"%date))>0:
            warnings.warn("TraceMatch_WithMasks already exists for %s. rebuild is False, so nothing is happening"%date)
            return
        load_trace_masks(smap, smap.get_traces_within_polygon(INDEX_CCD_CONTOURS), ncore=ncore)
        smap.writeto(timedir+f"{date}_TraceMatch_WithMasks.pkl")
    
############################
#                          #
# Spaxel Spacial Position  #
#                          #
############################
def build_hexagonalgrid(date, xybounds=None, theta=None):
    """ """
    tracematch = io.load_nightly_tracematch(date, withmask=True)
    if xybounds is None:
        xybounds = INDEX_CCD_CONTOURS

    # = Build it
    idxall = tracematch.get_traces_within_polygon(xybounds)
    hexagrid = tracematch.extract_hexgrid(idxall, theta=theta)
    
    # = store it    
    filepath = io._get_hexagrid_filepath(date)
    hexagrid.writeto( filepath )

############################
#                          #
# Spaxel Spacial Position  #
#                          #
############################
def build_flatfield(date, lbda_min=7000, lbda_max=9000,
                    ref="dome", build_ref=True,
                    kind="median", savefig=True, ncore=None):
    """ """
    from ..sedm import get_sedmcube
    from pyifu.spectroscopy  import get_slice

    reffile = io.get_night_files(date, kind="cube.basic", target=ref)

    # - If the reference if not there yet.
    if len(reffile)==0 or build_ref:
        warnings.warn("The reference cube %s does not exist "%ref)
        if build_ref:
            warnings.warn("build_flatfield is building it!")
        else:
            raise IOError("No reference cube to build the flatfield (build_ref was set to False)")


        # --------------------- #
        # Build the reference   #
        # --------------------- #
        # - Nightly Solutions
        wsol = io.load_nightly_wavesolution(date)
        tracematch = io.load_nightly_tracematch(date, withmask=True)
        hexagrid = io.load_nightly_hexagonalgrid(date)

        # - The CCD
        ccdreffile = io.get_night_files(date, kind="ccd.lamp", target=ref)
        if len(ccdreffile)==0:
            raise IOError(f"not {ref} available for {date}")
        else:
            ccdreffile = ccdreffile[0]
            
        ccdref = get_ccd(ccdreffile, tracematch = tracematch, background = 0)
        ccdref.fetch_background(set_it=True, build_if_needed=True, ncore=ncore)
        if not ccdref.has_var():
            ccdref.set_default_variance()

        # - Build a cube
        refcube = ccdref.extract_cube(wsol, lbda=SEDM_LBDA, 
                                    hexagrid=hexagrid,
                                    pixel_shift=0)
    else:
        refcube  = get_sedmcube(reffile[0])
        
    # ---------------------- #
    #  Actual FlatFielding   #
    # ---------------------- #
    sliceref = refcube.get_slice(lbda_min=lbda_min, lbda_max=lbda_max, usemean=True)
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
    lbda_eff = np.mean([lbda_min, lbda_max])
    flat = sliceref / norm
    slice_ = get_slice(flat, np.asarray(refcube.index_to_xy(refcube.indexes)),
                        refcube.spaxel_vertices,
                        indexes=refcube.indexes, variance=None,
                           lbda=lbda_eff)
    # - Figure
    timedir = io.get_datapath(date)
    
    # - Saving
    slice_.header["CALTYPE"] = "FlatField"
    slice_.header["FLATSRC"] = ref
    slice_.header["FLATREF"] = kind
    slice_.to_fits( os.path.join(timedir, f'{date}_Flat.fits') )
    
    if savefig:
        print("Saving flat3d Figure")
        slice_.show(savefile= os.path.join(timedir, f"{date}_flat3d.pdf") )
        slice_.show(savefile= os.path.join(timedir, f"{date}_flat3d.png") )

    
############################
#                          #
#   BackGround             #
#                          #
############################
def build_backgrounds(date, smoothing=[0,5], start=2, jump=10, 
                        target=None, lamps=True, only_lamps=False, skip_calib=True,
                        multiprocess=True,
                        savefig=True, ncore=None):
    """ """
    from tqdm import tqdm
    
    from ..background import build_background
    timedir  = io.get_datapath(date)
    
    # - The files
    fileccds = []
    if lamps:
        crrfiles = io.get_night_files(date, "ccd.lamp")
        fileccds += crrfiles
        
    if not only_lamps:
        crrfiles  = io.get_night_files(date, "ccd.crr", target=target)
        if skip_calib:
                fileccds = [f for f in crrfiles if "Calib" not in fits.getval(f,"Name")]
        fileccds += crrfiles
        

    # - Building the background
    tmap = io.load_nightly_tracematch(date)
    nfiles = len(fileccds)
    print("%d files to go..."%nfiles)
    for i,file_ in tqdm( enumerate(fileccds), total=nfiles):
        build_background(get_ccd(file_, tracematch=tmap, background=0),
                        start=start, jump=jump, multiprocess=multiprocess, 
                        smoothing=smoothing,
                        savefile = None if not savefig else timedir+"bkgd_%s.pdf"%(file_.split('/')[-1].replace(".fits","")),
                        ncore=ncore)
        
    
############################
#                          #
#  Wavelength Solution     #
#                          #
############################
def build_wavesolution(night, client,
                       verbose=False, ntest=None, idxrange=None,
                       wavedegree=5, contdegree=3, 
                       lamps=["Hg","Cd","Xe"], savefig=True,
                       xybounds=None, rebuild=False):
    """ Create the wavelength solution for the given night.
    The core of the solution fitting is made in pysedm.wavesolution.

    Parameters
    ----------
    
    Returns
    -------

    """
    # validated for version 0.40 #
    
    # = Checking if anything to do = #
    outputfile = io._get_wavesolution_filepath(night)    
    if os.path.isfile(outputfile) and not rebuild:
        warnings.warn(f"WaveSolution already exists for {night}. rebuild is False. Nothing happens")
        return


    # Yes ? ok, let's do it:
    import dask
    from .. import wavesolution

    ## Parse input
    ### Spaxel Selection
    if xybounds is None:
        xybounds = INDEX_CCD_CONTOURS
    

    timedir = os.path.dirname(outputfile)
    if verbose:
        print(f"Directory affected by Wavelength Calibration: {timedir}")
    
    ## Grab indeed input
    ### tracematch (where spaxels are)
    tracematch = io.load_nightly_tracematch(night, withmask=True)
    # => indexes of tracers (so spaxels) that should be considered
    used_indexes = tracematch.get_traces_within_polygon(xybounds)
    # => limit to what you need only (test purposes)
    if idxrange is not None:
        used_indexes = [l for l in used_indexes if l>=idxrange[0] and l<idxrange[1]]

    if ntest is not None:
        used_indexes = np.random.choice(used_indexes, ntest, replace=False) 


    
    ### loads the arclampcollection from which the individuals tracers are obtained.
    arclamps = wavesolution.ArcLampCollection.from_night(night)

    ## Perform the wavesolution computation
    if verbose:
        print(" Loading individual spaxel arc-spectra")
        
    from tqdm import tqdm
    spaxel_arcjson = [arclamps.get_trace_arccollection_json(i) for i in tqdm(used_indexes)]
    
    if verbose:
        print("Getting delayed wavelength solution")

    #                                #
    # Perfom the Wavelength solution #
    #                                #
    ## step 1: creates the spaxels fit_wavelengthsolution (delayed)
    wsolutions = []
    for json in spaxel_arcjson: # loop over all spaxels
        spaxel_arcs  = dask.delayed(wavesolution.ArcSpectrumCollection.from_json)(json)
        wsolution = spaxel_arcs.fit_wavelengthsolution(wavesolution_degree=5, 
                                                    inplace=False)
        wsolutions.append(wsolution)

    ## step 2: compute them and wait for it to be over.
    spaxel_wavesolutions_f = client.compute(wsolutions)
    spaxel_wavesolutions = client.gather(spaxel_wavesolutions_f, errors="skip")

    ## step 3: once over check if a few failed and retry them.
    ##         It happens with some network issues, more than 50 spaxels is unseen, so a bug not a glitch.
    failed_spaxels = [f for f in spaxel_wavesolutions_f if f.status == "error"]
    if len(failed_spaxels)<50: # try once more and then move on.
        _ = client.retry(failed_spaxels)
        spaxel_wavesolutions = client.gather(spaxel_wavesolutions_f, errors="skip")

    ## step 4: grab those that worked (should be 100%)
    worked_indexes = [i for i, f in zip(used_indexes, spaxel_wavesolutions_f) if f.status == "finished"]
    if len(worked_indexes)<len(used_indexes):
        warnings.warn(f"{len(used_indexes)-len(worked_indexes)} have failed.")

    ## step 5: creates the wavesolution object:
    wsol = wavesolution.WaveSolution.from_spaxel_wavesolutions(worked_indexes, spaxel_wavesolutions)
    wsol.to_parquet(outputfile)
    
    if savefig:
        hexagrid = io.load_nightly_hexagonalgrid(night)
        pl = wsol.show_dispersion_map(hexagrid, vmin="0.5",vmax="99.5",
                                              outlier_highlight=5, show=False)
        
        pl['fig'].savefig(os.path.join(timedir,f"{night}_wavesolution_dispersionmap.pdf") )
        pl['fig'].savefig(os.path.join(timedir,f"{night}_wavesolution_dispersionmap.png") )

        
        
    
############################
#                          #
#  Build Cubes             #
#                          #
############################
def build_night_cubes(date, target=None, lamps=True, only_lamps=False,
                          skip_calib=True, ncore=None, **kwargs):
    """ 
    """
    fileccds = []
    if lamps:
        fileccds += io.get_night_files(date, "ccd.lamp", target=target)
    if not only_lamps:
        crrfiles  = io.get_night_files(date, "ccd.crr", target=target)
        if skip_calib: crrfiles = [f for f in crrfiles if "Calib" not in fits.getval(f,"Name")]            
        fileccds += crrfiles

    print("fileccds:", fileccds)
    build_cubes(fileccds, date, ncore=ncore, **kwargs)


# ----------------- #
#  Build Cubes      #
# ----------------- #
def build_cubes(ccdfiles,  date, lbda=None,
                tracematch=None, wavesolution=None, hexagrid=None,
                # Background:
                nobackground = False,
                # Flat Field
                flatfielded=True, flatfield=None,
                # Flexure
                traceflexure_corrected=True, flexure_corrected=True,
                # Calibration
                atmcorrected=True, 
                build_calibrated_cube=False, calibration_ref=None,
                # Out
                build_guider=True, solve_wcs=False,
                fileindex=None, show_progress=False,
                savefig=True, verbose=True, 
                ncore=None):
    """ Build a cube from the an IFU ccd image. This image 
    should be bias and cosmic ray corrected.

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

    nobackground: [bool] -option-
        Shall the ccd background be 0 instead of the usual pipeline background?

    flexure_corrected: [bool] -optional-
        Shall the cube be flexure corrected ?
        - Remark, this means the cube will be built twice: 
            - 1 time to estimated the flexure
            - 1 time with flexure correction applied.

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
    if traceflexure_corrected:
        from ..flexure import get_ccd_jflexure
        
    # ------------------ #
    # Loading the Inputs #
    # ------------------ #
    if tracematch is None:
        tracematch = io.load_nightly_tracematch(date, withmask=True) #False if traceflexure_corrected else True) 
        
    if hexagrid is None:
        hexagrid = io.load_nightly_hexagonalgrid(date)
    
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
        flexuresavefile = None if not savefig else [ccdfile.replace("crr", "flexuretrace_crr").replace(".gz", "").replace(".fits",".pdf"),
                                                    ccdfile.replace("crr", "flexuretrace_crr").replace(".gz", "").replace(".fits",".png")]
        ccd_ = get_ccd(ccdfile, tracematch = tracematch, background = 0,
                              correct_traceflexure = traceflexure_corrected,
                              savefile_traceflexure=flexuresavefile)
        if traceflexure_corrected:
            load_trace_masks(ccd_.tracematch,
                                ccd_.tracematch.get_traces_within_polygon(INDEX_CCD_CONTOURS),
                                ncore=ncore)
            
        if not nobackground:
            ccd_.fetch_background(set_it=True, build_if_needed=True, ncore=ncore)
            ccd_.header["CCDBKGD"] = (True, "is the ccd been background subtracted?")
        else:
            ccd_.header["CCDBKGD"] = (False, "is the ccd been background subtracted?")
            
        # - Variance
        if not ccd_.has_var():
            ccd_.set_default_variance()
        ccds.append(ccd_)
            
    # ---------------- #
    # Build the Cubes  #
    # ---------------- #
    # internal routine 
    def _build_cubes_(ccdin):
        print(ccdin.filename)
        prop = dict(lbda=lbda, wavesolution=wavesolution, hexagrid=hexagrid,
                    flexure_corrected=flexure_corrected,
                    flatfielded=flatfielded, flatfield=flatfield,
                    atmcorrected=atmcorrected, 
                    build_calibrated_cube=build_calibrated_cube,
                    calibration_ref=calibration_ref,
                    fileindex=fileindex,
                    savefig=savefig)
        if build_guider:
            from pysedm import rainbowcam
            try:
                print("INFO: building the guider image")
                rainbowcam.build_meta_ifu_guider(ccdin.filename, solve_wcs=solve_wcs)
            except:
                print("WARNING: rainbowcam cannot build the guider image")

        build_sedmcube(ccdin, date,  **prop)
            
            
    # The actual build
    return [_build_cubes_(ccd_) for ccd_ in ccds]
        

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
        return _build_cal_cubes_(cubefiles[0])
    if multiprocess:
        print("multiprocess removed for calibrate_cubes()")
        
    return [_build_cal_cubes_(cubefiles_) for cubefiles_ in cubefiles]
        
    
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
