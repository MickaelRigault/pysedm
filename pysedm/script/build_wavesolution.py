#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
from astrobject.utils.tools import dump_pkl
from ..io import get_datapath
from ..ccd import get_specmatcher, get_lamp
from ..wavesolution import get_cubesolution

def build_wavesolution(date, verbose=False, ntest=None, lamps=["Hg","Cd","Xe"],
                        xybounds=[[50,50],[50,1700],[300,2030],[2030,2030],[2030,50]]):
    """ """
    from shapely.geometry import Polygon
    poly = Polygon(xybounds)
    
    timedir = get_datapath(date)
    try:
        os.mkdir(timedir+"ProdPlots/")
    except:
        warnings.warn("No Plot directory created. Most likely it already exists.")
        
    if verbose:
        print "Directory affected by Wavelength Calibration: %s"%timedir

    # ----------------
    # - Load the Data
    smap = get_specmatcher(timedir+"dome.fits")
    if verbose:
        print "Spectral Match built"
    lamps = [get_lamp(timedir+"%s.fits"%s_, specmatch=smap)
                 for s_ in lamps]
        
    if verbose:
        print "Cd, Hg and Xe lamp loaded"

    # ----------------
    # - Load the Data
    csolution = get_cubesolution(*lamps)
    if ntest is not None:
        idx = np.random.choice(smap.get_idx_within_polygon(poly),
                                ntest, replace=False)
    else:
        idx = smap.get_idx_within_polygon(poly)
        
    for i in idx:
        print i
        csolution.fit_wavelesolution(specid=i, saveplot=timedir+"ProdPlots/%s_wavesolution_spec%d.pdf"%(date,i),
                                         contdegree=2, plotprop={"show_guesses":True})
        

    dump_pkl(csolution.wavesolutions, timedir+"%s_WaveSolution.pkl"%date)
    
    
    
