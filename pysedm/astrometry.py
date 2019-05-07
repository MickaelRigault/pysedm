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
    from astropy.time import Time
    if 'OBSDATE' in cube.header:
        cube_date = cube.header.get("OBSDATE", None)
    else:
        cube_date = '2001-01-01'
    if parameters is None:
        parameters = get_sedm_astrom_param(cube_date)
        
    if Time(cube_date) > Time("2019-02-20") and Time(cube_date) < Time("2019-04-17"):
        print("TEMPORARY PATCH FOR SHIFTED IFU POS")
        return rainbow_coords_to_ifu(get_ccd_coords(cube), parameters) + np.asarray([11, 1])
    elif Time(cube_date) > Time("2019-04-22"):
        print("TEMPORARY PATCH 2 FOR SHIFTED IFU POS")
        return rainbow_coords_to_ifu(get_ccd_coords(cube),
                                     parameters) + np.asarray([-5, -30])
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
        try:
            radec_ = coordinates.SkyCoord(*pycalspec.std_radec(cube.header["OBJECT"].split("STD-")[-1].split()[0]),
                                      unit=(units.hourangle, units.deg))
        except ValueError:
            print("WARNING: Standard not found in reference list")
            try:
                radec_ = coordinates.SkyCoord(cube.header["OBJRA"],
                                              cube.header["OBJDEC"],
                                              unit=(units.hourangle, units.deg))
            except KeyError:
                radec_ = coordinates.SkyCoord(cube.header["OBRA"],
                                              cube.header["OBDEC"],
                                              unit=(units.hourangle, units.deg))
    else:
        try:
            radec_ = coordinates.SkyCoord(cube.header["OBJRA"], cube.header["OBJDEC"],
                                  unit=(units.hourangle, units.deg))
        except KeyError:
            radec_ = coordinates.SkyCoord(cube.header["OBRA"],
                                          cube.header["OBDEC"],
                                          unit=(units.hourangle, units.deg))

    date = io.header_to_date(cube.header)
    
    gastrom_file = io.fetch_guider(date, cube.filename)
    if len(gastrom_file) == 0:
        warnings.warn("No guider astrom file for %s"%cube.filename)
        return np.asarray([np.NaN, np.NaN])
    
    gastromwcs =     wcs.WCS(header=getheader(gastrom_file[0]))
    try:
        coords = np.asarray(radec_.to_pixel(gastromwcs))
    except wcs.wcs.NoConvergence:
        warnings.warn("Could not converge on coordinates")
        coords = np.asarray([np.NaN, np.NaN])
    return coords


def fit_conversion_matrix(cubes_to_fit, guess=None):
    """ """
    from scipy.spatial import distance
    from scipy.optimize import fmin
    
    if guess is None:
        guess = get_sedm_astrom_param(cube_to_fit)
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
    lim_perc = 99.5
    bright_lim = np.nanpercentile(sl.data, lim_perc)
    while np.isnan(bright_lim):
        lim_perc -= 5.
        bright_lim = np.nanpercentile(sl.data, lim_perc)
    print("Getting spaxels brighter than %f" % bright_lim)
    argmaxes = np.argwhere(sl.data>bright_lim).flatten() # brightest points
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


