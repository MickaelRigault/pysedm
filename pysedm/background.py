#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""  """
import warnings
import numpy as np
import matplotlib.pyplot as mpl

from astropy.io import fits as pf
from astropy.utils.console import ProgressBar


from propobject   import BaseObject
from .utils.tools import kwargs_update, load_pkl, dump_pkl
from .sedm        import SEDM_CCD_SIZE

DEGREE   = 5
LEGENDRE = True

NGAUSS = 1
CONTDEGREE_GAUSS = 4
LEGENDRE_GAUSS= False


# ------------------ #
#  Builder           #
# ------------------ #
def build_background(ccd,
                    smoothing=[0,5],
                    start=2, jump=10, multiprocess=True,notebook=False,
                    savefile=None):
    """ """
    from .io import is_stdstars, filename_to_background_name
    ccd.fit_background(start=start, jump=jump, multiprocess=multiprocess, notebook=notebook,
                                set_it=False, is_std= is_stdstars(ccd.header), smoothing=smoothing)
    ccd._background.writeto( filename_to_background_name(ccd.filename).replace('.gz','') )
    if savefile is not None:
        ccd._background.show(savefile=savefile)
        mpl.close("all")


# ------------------ #
#  Main Functions    #
# ------------------ #
def get_background(contvalues, size=SEDM_CCD_SIZE, smoothing=[0,5]):
    """ """
    back = Background()
    back.create(contvalues)
    back.build(size[0],size[1], smoothing=smoothing)
    return back

def load_background(filename):
    """ """
    back = Background()
    back.load(filename)
    return back

# ------------------- #
#   MultiProcessing   #
# ------------------- #
def get_contvalue(spec):
    """ """
    spec.fit_continuum(DEGREE, legendre=LEGENDRE)
    return spec.contmodel.fitvalues

def get_contvalue_sdt(spec):
    """ """
    spec.fit_continuum(CONTDEGREE_GAUSS, legendre=LEGENDRE_GAUSS, ngauss=NGAUSS)
    return spec.contmodel.fitvalues

def fit_background(ccd, start=2, jump=10, multiprocess=True, ncore=None,
                       notebook=True, is_std=False):
    """ calling `get_contvalue` for each ccd column (xslice).
    This uses astropy's ProgressBar.map 

    Returns 
    -------
    dictionary 
    """
    # Running from ipython notebook
    index_column = range(ccd.width)[start::jump]
    bar = ProgressBar( len(index_column), ipython_widget=notebook)

    # - Multiprocessing 
    if multiprocess:
        import multiprocessing
        if ncore is None:
            if multiprocessing.cpu_count()>20:
                ncore = multiprocessing.cpu_count() - 10
            elif multiprocessing.cpu_count()>8:
                ncore = multiprocessing.cpu_count() - 5
            else:
                ncore = multiprocessing.cpu_count() - 2
            if ncore==0:
                ncore = 1
        p = multiprocessing.Pool(ncore)
        res = {}
        for j, result in enumerate( p.imap(get_contvalue if not is_std else get_contvalue_sdt, [ccd.get_xslice(i_) for i_ in index_column])):
            res[index_column[j]] = result
            bar.update(j)
        bar.update(len(index_column))
        return res
    
    # - No multiprocessing 
    return {index_column[i_]: get_contvalue(ccd.get_xslice(spec_)) if not is_std else get_contvalue_sdt(ccd.get_xslice(spec_)) for i_,spec_ in enumerate(bar)}


def _fit_background_notebook_(ccd, start=2, jump=10, multiprocess=True,
                                  is_std=False, ncore=None,
                                  ipython_widget=True):
    """ calling `get_contvalue` for each ccd column (xslice).
    This uses astropy's ProgressBar.map 
s
    ===
    This version of the code is made to be called from fit_background
    when this software detects that ipython is running from an notebook
    ===

    Returns 
    -------
    dictionary 
    """
    index_column = range(ccd.width)[start::jump]
    bar = ProgressBar( len(index_column), ipython_widget=ipython_widget)

    # - Multiprocessing 
    if multiprocess:
        import multiprocessing
        if ncore is None:
            ncore = multiprocessing.cpu_count() - 1
            if ncore==0:
                ncore = 1
        p = multiprocessing.Pool(ncore)
        res = {}
        for j, result in enumerate( p.imap(get_contvalue if not is_std else get_contvalue_sdt,
                                                         [ccd.get_xslice(i_) for i_ in index_column])):
            res[index_column[j]] = result
            bar.update(j)
        bar.update(len(index_column))
        return res
    
    # - No multiprocessing 
    return {index_column[i_]: get_contvalue(ccd.get_xslice(spec_)) if not is_std else get_contvalue_sdt(ccd.get_xslice(spec_)) for i_,spec_ in enumerate(bar)}

def _get_xaxis_polynomial_(xyv, degree=DEGREE, legendre=LEGENDRE,
                         xmodel=None, clipping = [5,5]):
    """ """
    from modefit.basics import get_polyfit
    x,y,v = xyv
    flagin = ((np.nanmean(y) - clipping[0] * np.nanstd(y)) < y) *  (y< (np.nanmean(y) + clipping[1] * np.nanstd(y)))
    
    contmodel = get_polyfit(x[flagin], y[flagin], v[flagin], degree=degree, legendre=legendre)
    contmodel.fit(a0_guess=np.nanmedian(y[flagin]))
    
    if xmodel is not None:
        return contmodel.fitvalues, contmodel.model.get_model(x=xmodel)#, contmodel
    return contmodel.fitvalues#, None, contmodel

        
class Background( BaseObject ):
    """ """
    PROPERTIES      = ["contvalues" ]
    SIDE_PROPERTIES = ['filename',"header","y"]
    DERIVED_PROPERTIES = ['input_columns', "input_background","background"]
    
    def load(self, filename):
        """ """
        data_ = pf.open(filename)
        
        # - background
        self._derived_properties['background'] = data_[0].data
        self._side_properties['header'] = data_[0].header
        # - contvalues
        contheader = data_["POLYVALUES"].header
        params = [int(k.replace("VALUE","")) for k,v in contheader.items() if "VALUE" in k]
        columns = data_["COLUMNS"].data
        convalues = {col:{contheader['VALUE%d'%i]: c_ for i,c_ in zip(params, cval_i)}
                         for col, cval_i in zip(columns, data_["POLYVALUES"].data)}
        self.create(convalues) 
        
    def create(self, contvalues):
        """ setup the instance based on the input 'contvalue' """
        self._properties['contvalues']    = contvalues
        self._derived_properties['input_columns'] = None
        
    def build(self, width, height, smoothing= [0,5]):
        """ """
        self._derived_properties['background'] = self.get_background(height, width, smoothing=smoothing)
        
    def contvalue_to_polynome(self, contvalue_):
        """ """
        from modefit.basics import polynomial_model, normal_and_polynomial_model
        if "mu0" in contvalue_.keys():
            poly = normal_and_polynomial_model( CONTDEGREE_GAUSS, NGAUSS)
            poly.use_legendre=LEGENDRE_GAUSS
        else:
            poly = polynomial_model( DEGREE )
            poly.use_legendre=LEGENDRE
            
        poly.setup([contvalue_[k] for k in poly.FREEPARAMETERS])
        self._side_properties["y"] = np.linspace(0, SEDM_CCD_SIZE[1], self.n_inputcolumns*2)
        model = poly.get_model(self._y)
        del(poly)
        return model


    def get_background(self, width, height, smoothing = [0,5]):
        """ """
        from scipy import ndimage, interpolate
        # get the blured image
        self._filtered = ndimage.filters.gaussian_filter( self.input_background, smoothing)
        # resample
        orig_shape = np.shape(self.input_background)
        x = np.linspace(0,1, orig_shape[0])
        y = np.linspace(0,1, orig_shape[1])
        nk = interpolate.RectBivariateSpline(x,y, self._filtered, kx=3,ky=3)
        
        return nk(np.linspace(0,1,width),np.linspace(0,1,height))


    # ----------------- #
    #   I/O and Plots   #
    # ----------------- #
    def writeto(self, savefile, overwrite=True, **header_kwargs):
        """ save the background as a .fits file
        
        The object will be structured as follows:
        0 PrimaryHDU: Background image 
        1 POLYVALUES: list of values associated to the best fits of column continuum
                      [parameter names in the header]
        2 COLUMNS: list of the fitted columns
        
        Use load() to read a .fits file created by this method.
        
        Parameters
        ----------
        savefile: [string]
            Full path of the .fits file to be created
            
        overwrite: [bool] -optional-
            Shall this overwrite an existing file if any.

        **header_kwargs additional information to be saved in the Primary header.
        
        Returns
        -------
        Void
        """
        img_shape = np.shape(self.background)
        
        self.header["NAXIS"] = len(img_shape)
        self.header['NAXIS1'] = img_shape[0]
        self.header['NAXIS2'] = img_shape[1]
        self.header['TYPE']   = "background"
        for k,v in header_kwargs:
            self.header[k] = v
            
        # --- Build the HDU
        hdu = [pf.PrimaryHDU(self.background, self.header)] # Background
        
        params = np.sort(list(list(self._contvalues.values())[0].keys()))
        header_POLY = pf.Header()
        for i, p in enumerate(params): 
            header_POLY['VALUE%s'%i] = p
            
        hdu.append(pf.ImageHDU( [[self._contvalues[i][p] for p in params] for i in self.input_columns], name='POLYVALUES', header=header_POLY))
        hdu.append(pf.ImageHDU(self.input_columns, name='COLUMNS'))
        
        hdulist = pf.HDUList(hdu)
        hdulist.writeto(savefile, overwrite =overwrite)

        
    def show(self, savefile=None, vmin=None, vmax=None, show=True, **kwargs):
        """ """
        from .utils.mpl import figout
        
        fig = mpl.figure(figsize=[9,4])
        space = 0.02
        width = 0.38
        axs  = fig.add_axes([0.1,0.1,width,0.8])
        ax   = fig.add_axes([0.1+width+space,0.1,width,0.8])
        axc  = fig.add_axes([0.1+(width+space)*2,0.1,0.02,0.8])
        if vmin is None:
            vmin = "2"
        if vmax is None:
            vmax = "98"
        prop = dict(origin="lower", aspect="auto", 
            vmax=np.percentile(self.input_background, float(vmax)) if type(vmax) == str else vmax,
            vmin=np.percentile(self.input_background, float(vmin)) if type(vmin) == str else vmin)
                          
        axs.imshow(self.input_background,        **prop)
        cl = ax.imshow(self.background, **prop)
        
        axs.set_title("column fit [source]")
        ax.set_title("background")
        ax.set_yticks([])
        fig.colorbar(cl, axc)
        
        fig.figout(savefile=savefile, show=show)
        
    # -------------------- #
    #    Properites        #
    # -------------------- #
    @property
    def n_inputcolumns(self):
        """ Number of column for which the background has been estimated"""
        return len(self._contvalues)

    @property
    def input_columns(self):
        """ index of the ccd column where the background has been estimated """
        if self._derived_properties['input_columns'] is None:
            self._derived_properties['input_columns'] = np.sort(list(self._contvalues.keys()))
            
        return self._derived_properties['input_columns']
    
    @property
    def _contvalues(self):
        """ """
        return self._properties['contvalues']

    @property
    def _y(self):
        """ """
        if self._side_properties['y'] is None:
            self._side_properties["y"] = np.linspace(0, 1, self.n_inputcolumns)
        return self._side_properties['y']
    

    # -- Background
    @property
    def background(self):
        """ """
        return self._derived_properties["background"]
    @property
    def input_background(self):
        """ """
        if self._derived_properties["input_background"] is None:
            self._derived_properties["input_background"] = \
              np.asarray([self.contvalue_to_polynome(self._contvalues[i])
                            for i in self.input_columns]).T
        return self._derived_properties["input_background"]
    
    # -- Fits tools
    @property
    def header(self):
        """ """
        if self._side_properties["header"] is None:
            self._side_properties["header"] = pf.Header()
        return self._side_properties["header"]
