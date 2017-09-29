#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Internal small toolbox"""

import numpy as np

__all__ = ["kwargs_update","shape_ajustment"]


def kwargs_update(default,**kwargs):
    """
    """
    k = default.copy()
    for key,val in kwargs.iteritems():
        k[key] = val
        
    return k

def load_pkl(filename):
    """
    """
    import cPickle as pkl
    try:
        pkl_file = open(filename,'rb')
    except:
        raise IOError("The given file does not exist %s"%filename)
    
    return pkl.load(pkl_file)


def dump_pkl(data,filename,**kwargs):
    """
    """
    from cPickle import dump
    if len(filename.split("."))>1 and filename.split(".")[-1]=="pkl":
        outfile =  open(filename,"wb")
    else:
        outfile =  open(filename+".pkl","wb")
    
    dump(data, outfile,**kwargs)
    outfile.close()

    
def make_method(obj):
    """Decorator to make the function a method of *obj*.

    In the current context::
      @make_method(Axes)
      def toto(ax, ...):
          ...
    makes *toto* a method of `Axes`, so that one can directly use::
      ax.toto()
    COPYRIGHT: from Yannick Copin
    """

    def decorate(f):
        setattr(obj, f.__name__, f)
        return f

    return decorate

def shape_ajustment(x,y,model_x,**kwargs):
    """ creates a new version of `y` that matches the conversion x->model_x
    - This uses scipy.interpolate.UnivariateSpline -

    Parameters
    ----------
    x,y: [arrays]
        2 array of the same dimension.
        y is the array you want to reshape as x-> model_x

    model_x: [array]
        Model on which x will be projected.
        
    **kwargs goes to UnivariateSpline:
       - k (degree of smoothing <=5) etc.

    Returns
    -------
    array (same size as model_x)
    """
    from scipy.interpolate import UnivariateSpline

    #flagx = (x>=np.nanmin(model_x)) & (x<=np.nanmax(model_x))
    yrebin = UnivariateSpline(x, y,**kwargs)(model_x)
    
    if len(yrebin)==len(model_x):
        return yrebin
    
    warnings.warn("[shape_adjustment] shape does not match automatically. Removed edges")
    yrebinOK = N.empty((len(yrebin)+1),)
    yrebinOK[1:] = yrebin
    yrebinOK[0]  = yrebin[0]
    
    return yrebinOK

def running_from_notebook():
    """ True is the current system runs from ipython notebook 
    (ipykernel detected in the sys.modules) 
    """
    import sys
    return "ipykernel" in sys.modules

################################
#                              #
#    MPL Like                  #
#                              #
################################
def draw_ellipse(image, xy, a, b, facecolor, edgecolor):
    """Improved ellipse drawing function, based on PIL.ImageDraw."""
    from PIL import Image, ImageDraw
    mask = Image.new( size=image.size, mode='RGBA')
    draw = ImageDraw.Draw(mask)
    draw.ellipse([xy[0]-a, xy[0]-b, xy[1]+a, xy[1]+b],
                     fill=tuple(facecolor), outline=tuple(edgecolor))
    mask = mask.resize(image.size, Image.LANCZOS)
    mask = mask.rotate(45)
    image.paste(mask, mask=mask)
    return image
