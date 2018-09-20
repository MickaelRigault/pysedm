#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" This module is made for the joint IFU Rainbow camera astrometry. """

import numpy as np
import warnings
# Astropy
from astropy import units, coordinates
# Others
import pycalspec
from psfcube import fitter
from . import io
from .sedm import SEDM_ASTROM_PARAM
# ======================= #
#   GET LOCATIONS         #
# ======================= #
def get_object_ifu_pos(cube, parameters=SEDM_ASTROM_PARAM):
    """ the expected cube x,y position of the target within the cube """
    return rainbow_coords_to_ifu(get_ccd_coords(cube), parameters)


def get_ccd_coords(cube):
    """ target position in the rainbow camera. 
    Remark that this area should not be visible in the rainbow camera as this area 
    should be behind the mirror sending the light to the IFU
    """
    from astropy import wcs
    from astropy.io.fits import getheader
    if "STD" in cube.header["OBJECT"]:
        import pycalspec
        radec_ = coordinates.SkyCoord(*pycalspec.std_radec(cube.header["OBJECT"].split("STD-")[-1]), 
                                      unit=(units.hourangle, units.deg))
    else:
        radec_ = coordinates.SkyCoord(cube.header["OBJRA"], cube.header["OBJDEC"], 
                                  unit=(units.hourangle, units.deg))

    date = io.header_to_date(cube.header)
    
    gastrom_file = io.fetch_guider(date, cube.filename)
    if len(gastrom_file) == 0:
        warnings.warn("No guider astrom file for %s"%cube.filename)
        return np.asarray([np.NaN, np.NaN])
    
    gastromwcs =     wcs.WCS(header=getheader(gastrom_file[0]))
    return np.asarray(radec_.to_pixel(gastromwcs))


# ======================= #
#   Other tools           #
# ======================= #

def rainbow_coords_to_ifu(ccd_coords, parameters=SEDM_ASTROM_PARAM):
    """ """
    centroid_ccd = [parameters[4],parameters[5]]
    matrix = np.asarray([ [parameters[0], parameters[1]],
                          [parameters[2], parameters[3]] ])
    
    return np.dot(matrix, (ccd_coords-centroid_ccd).T).T

def fit_cube_centroid(cube_, lbdamin=6000, lbdamax=7000):
    """ Use `fit_slice` function from psfcube to estimate the PSF centroid 
    of a point-source containing in the cube. 
    The fit will be made on the metaslice within the boundaries `lbdamin` `lbdamax`

    Parameters
    ----------
    cube_: [pyifu Cube]
        Cube containing the point source
        
    lbdamin, lbdamax: [floats] -optional-
        lower and upper wavelength boundaries defining the metaslice
        [in Angstrom]

    Returns
    -------
    list: [x,y, dx, dy] # (centroids and errors)
    """
    sl_r = cube_.get_slice(lbda_min=6000, lbda_max=7000, slice_object=True)
    slfit = fitter.fit_slice(sl_r, centroids_err=[4,4])
    return [slfit.fitvalues[k] for k in ["xcentroid","ycentroid","xcentroid.err","ycentroid.err"]]


def get_standard_rainbow_coordinates(rainbow_wcs, stdname):
    """ """
    rhms = coordinates.SkyCoord(*pycalspec.std_radec(stdname),
                                unit=(units.hourangle, units.deg))
    return np.asarray(rhms.icrs.to_pixel(rainbow_wcs))


