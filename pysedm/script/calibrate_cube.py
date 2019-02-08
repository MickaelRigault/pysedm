#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" Scripts to build the calibrated cubes step by step """
from .. import io
from ..sedm import get_sedmcube

def flat_cubes(date, lbda_min=7000, lbda_max=9000, ref="dome"):
    """ """
    baseroot = io.CUBE_PROD_ROOTS["cube"]["root"]
    newroot  = io.CUBE_PROD_ROOTS["flat"]["root"]
    
    # -------------- #
    # The Reference  #
    # -------------- #
    reffile = io.get_night_cubes(date, kind="cube", target=ref)

    if len(reffile)==0:
        raise ValueError("No cube reference for target %s in night %s"%(ref,date))
    
    refcube  = get_sedmcube(reffile[0])
    flatfied = refcube.get_slice(lbda_min=lbda_min, lbda_max=lbda_max, usemean=True)
    print(flatfied.mean())
    # ----------------- #
    # Build flat cubes  #
    # ----------------- #
    def build_flat_cube(cubefile):
        cube_       = get_sedmcube(cubefile)
        print(cubefile)
        cube_.scale_by(flatfied)
        cube_.writeto(cube_.filename.replace(baseroot,newroot))
        
    from astropy.utils.console import ProgressBar
    cubefiles = io.get_night_cubes(date, kind="cube")
    print(cubefiles)
    ProgressBar.map(build_flat_cube, cubefiles)
    
    
    
    
