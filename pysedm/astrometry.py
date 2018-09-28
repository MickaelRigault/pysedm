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
from .sedm import get_sedm_astrom_param


# ======================= #
#   GET LOCATIONS         #
# ======================= #
def get_object_ifu_pos(cube, parameters=None):
    """ the expected cube x,y position of the target within the cube """
    if parameters is None:
        parameters = get_sedm_astrom_param( cube.header.get("OBSDATE",None) )
        
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


def fit_conversion_matrix(cubes_to_fit, guess=None):
    """ """
    from scipy.spatial import distance
    from scipy.optimize import fmin
    
    if guess is None:
        from pysedm import SEDM_ASTROM_PARAM
        GUESS = np.asarray(SEDM_ASTROM_PARAM)+np.asarray([0,0,0,0,-2,-2])
    else:
        guess= np.asarray(guess)

    list_of_ccd_positions = np.asarray([ get_ccd_coords(c_) for c_ in cubes_to_fit])
    list_of_ifu_positions = np.asarray([ fit_cube_centroid(c_)[:2] for c_ in cubes_to_fit])
    
    def to_fit(parameters):
        list_of_ifu_positions_MODEL = rainbow_coords_to_ifu(list_of_ccd_positions, parameters)
        
        return np.sum([distance.euclidean(im_, i_) for im_, i_ in zip(list_of_ifu_positions_MODEL, 
                                                                      list_of_ifu_positions)])
    return fmin(to_fit, GUESS, maxiter=10000)


# ======================= #
#   Other tools           #
# ======================= #
def estimate_default_position(cube, lbdaranges=[5000,7000]):
    """ """
    sl = cube.get_slice(lbda_min=lbdaranges[0], lbda_max=lbdaranges[1], slice_object=True)
    x,y = np.asarray(sl.index_to_xy(sl.indexes)).T # Slice x and y
    argmaxes = np.argwhere(sl.data>np.percentile(sl.data, 99.5)).flatten() # brightest points
    return np.nanmedian(x[argmaxes]),np.nanmedian(y[argmaxes]) # centroid

def rainbow_coords_to_ifu(ccd_coords, parameters):
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
    centroid = estimate_default_position(cube_)
    slfit = fitter.fit_slice(sl_r, centroids=centroid, centroids_err=[4,4])
    return [slfit.fitvalues[k] for k in ["xcentroid","ycentroid","xcentroid.err","ycentroid.err"]]


def get_standard_rainbow_coordinates(rainbow_wcs, stdname):
    """ """
    rhms = coordinates.SkyCoord(*pycalspec.std_radec(stdname),
                                unit=(units.hourangle, units.deg))
    return np.asarray(rhms.icrs.to_pixel(rainbow_wcs))


