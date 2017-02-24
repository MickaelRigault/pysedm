#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
from astrobject.utils.tools import dump_pkl
from ..io import get_datapath
from ..ccd import get_specmatcher, get_lamp
from ..wavesolution import get_cubesolution

def build_wavesolution(date, verbose=False, ntest=None,
                       lamps=["Hg","Cd","Xe"],
                       xybounds=[[50,50],[50,1700],[300,2030],[2030,2030],[2030,50]]):
    """ Create the wavelength solution for the given night.
    The core of the solution fitting is made in pysedm.wavesolution.

    Parameters
    ----------

    Returns
    -------

    """
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
    smap = get_specmatcher(timedir+"dome.fits")
    if verbose: print "Spectral Match built"
    # - lamps 
    lamps = [get_lamp(timedir+"%s.fits"%s_, specmatch=smap)
                 for s_ in lamps]
    if verbose: print "Cd, Hg and Xe lamp loaded"
    # - The CubeSolution
    csolution = get_cubesolution(*lamps)

    # ----------------
    # - Spaxel Selection 
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
    
    
    
