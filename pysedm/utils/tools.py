#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Internal small toolbox"""

import numpy as np

__all__ = ["kwargs_update","shape_ajustment", "is_arraylike"]


def kwargs_update(default,**kwargs):
    """
    """
    k = default.copy()
    for key,val in kwargs.items():
        k[key] = val
        
    return k

def load_pkl(filename):
    """
    """
    try:
        import cPickle as pkl
    except:
        import pickle
        with open(filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            return u.load()

    pkl_file = open(filename,'rb')
    return pkl.load(pkl_file)


def dump_pkl(data,filename,**kwargs):
    """
    """
    try:
        from cPickle import dump
    except:
        from pickle import dump
        
    if len(filename.split("."))>1 and filename.split(".")[-1]=="pkl":
        outfile =  open(filename,"wb")
    else:
        outfile =  open(filename+".pkl","wb")
    
    dump(data, outfile,**kwargs)
    outfile.close()

def is_arraylike(a):
    """ Tests if 'a' is an array / list / tuple """
    return isinstance(a, (list, tuple, np.ndarray) )


def fig_backend_test(backup='Agg'):
    """ """
    try:
        import matplotlib.pyplot as mpl
        fig = mpl.figure()
    except:
        import matplotlib
        matplotlib.use(backup)

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

def _loading_multiprocess():
    """ Black magic function to load to enable pickle in multiprocess """
    try:
        import copy_reg as copyreg
    except:
        import copyreg
    import types
    
    def _pickle_method(m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)
        
    copyreg.pickle(types.MethodType, _pickle_method)


def vac_to_air_sdss(vac):
    """ converts air wavelength [AA] into vacuum wavelengths. Tested. """
    return vac / (1.0 + 2.735182*10**-4 + 131.4182 / vac**2 + 2.76249*10**8 / vac**4)

def fetch_vac(x):
    from scipy.optimize import fmin
    def _to_be_min_(x_):
        return np.abs(x - tools.vac_to_air_sdss(x_))
    return fmin(_to_be_min_, x, disp=0)



def fit_intrinsic(data, model, errors, dof, intrinsic_guess=None):
    """ Get the most optimal intrinsic dispersion given the current fitted standardization parameters. 
    
    The optimal intrinsic magnitude dispersion is the value that has to be added in quadrature to 
    the errors such that the chi2/dof is 1.
    Returns
    -------
    float (intrinsic dispersion)
    """
    from scipy.optimize import fmin
    def get_intrinsic_chi2dof(intrinsic):
        return np.abs( np.nansum((data-model)**2/(errors**2+intrinsic**2)) / dof - 1)
    
    if intrinsic_guess is None:
        intrinsic_guess = np.nanmedian(errors)
        
    return fmin(get_intrinsic_chi2dof, intrinsic_guess, disp=0)[0]


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
