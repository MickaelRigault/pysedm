#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Fitting ADR """

import warnings
import numpy            as np

from pyifu.adrfit      import *

from ..sedm    import DEFAULT_REFLBDA

def get_cube_adr_param(cube, lbdaref=DEFAULT_REFLBDA,
                           lbdastep=10, lbdarange=None,
                           show=True, savefile=None):
    """ """
    from shapely import geometry
    from . import extractstar
    
    x, y          = np.asarray( cube.index_to_xy(cube.indexes)).T
    indexref      = np.argmin(np.abs( cube.lbda-lbdaref))

    # = Building SubCube
    # Which Spaxels
    x0, y0, std0  = guess_aperture(x, y, cube.data[indexref])
    used_indexes  = cube.get_spaxels_within_polygon(geometry.Point(x0,y0).buffer(std0*5))
    # Which Wavelength    
    slice_to_fit = range(len(cube.lbda))[::lbdastep]
    if lbdarange is not None and len(lbdarange)==2:
        flagin = (slice_to_fit>lbdarange[0]) & (slice_to_fit<=lbdarange[1])
        slice_to_fit = slice_to_fit[flagin]
    # Building it
    cube_partial = cube.get_partial_cube(used_indexes,slice_to_fit)
    cube_partial.load_adr()

    # = Fitting the Centroid:
    es = extractstar.ExtractStar(cube_partial)
    es.fit_psf("BiNormalCont")
    xc,yc = es.get_fitted_centroid()
    # 
    adrfit = ADRFitter(cube_partial.adr.copy())
    adrfit.set_data(cube_partial.lbda, xc[0], yc[0], xc[1], yc[1])
    adrfit.fit(airmass_guess=cube.header["AIRMASS"], airmass_boundaries=[1,cube.header["AIRMASS"]*1.5])
    if show or savefile is not None:
        pl = adrfit.show(show=show, savefile=savefile, refsedmcube=cube_partial )
        
    return adrfit.fitvalues
    

def guess_aperture(x_, y_, data_):
    """ Get the centroid and symmetrized standard deviation of the data given their x, y position 
    Returns
    -------
    x0, y0, std_mean (floats)
    """
    flagok = ((x_==x_) * (y_==y_) * (data_==data_))
    x = x_[flagok]
    y = y_[flagok]
    data = data_[flagok]
    argmaxes   = np.argwhere( data>np.percentile(data,95) ).flatten()
    x0, stdx   = np.nanmean(x[argmaxes]), np.nanstd(x[argmaxes])
    y0, stdy   = np.nanmean(y[argmaxes]), np.nanstd(y[argmaxes])
    std_mean   = np.mean([stdx,stdy])
    return x0, y0, std_mean
