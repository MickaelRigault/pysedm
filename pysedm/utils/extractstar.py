#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract Star based on Modefit """

import warnings
import numpy            as np
from pyifu.spectroscopy import Slice, Cube

import shapely
from propobject          import BaseObject
from modefit.baseobjects import BaseFitter, BaseModel
from pyifu.tools     import kwargs_update

from astropy.modeling.functional_models import Moffat2D
from astropy.modeling.models import Polynomial2D


###########################
#                         #
#  Single Slice Fit       #
#                         #
###########################
def fit_slice(slice_, degree=1, fitbuffer=None,
              psfmodel="MoffatPlane0", fitted_indexes=None,
              **kwargs):
    """ 
    fitbuffer ignored if fitted_indexes provided
    """
    # 10ms / 263ms with fitted_indexes
    slpsf = SlicePSF(slice_, psfmodel=psfmodel, fitted_indexes=fitted_indexes)
    
    # - restrict area
    if fitted_indexes is None and fitbuffer is not None:
        # 1ms
        g_ = slpsf.get_guesses()
        # 288ms
        slpsf.set_fit_area(shapely.geometry.Point(g_["x_0_guess"],g_["y_0_guess"]).buffer(fitbuffer))
        
    # 338ms / 900 without buffer
    slpsf.fit( **kwargs_update(slpsf.get_guesses(),**kwargs) )
    return slpsf

###########################
#                         #
#   Extract Star          #
#                         #
###########################

class ExtractStar( BaseObject ):
    """ Object containing the individual SlicePSF fitter """
    PROPERTIES = ["cube"]
    DERIVED_PROPERTIES = ["slicefitvalues"]
    
    # =================== #
    #   Methods           #
    # =================== #
    def __init__(self, cube ):
        """ """
        self.set_cube(cube)
        
    # ------------------- #
    #  Cube -> Spectrum   #
    # ------------------- #
    # PSF SPECTROSCOPY
    def fit_psf(self, fixed_buffer=None, buffer_refindex=100,
                psfmodel="MoffatPlane0",
                **kwargs):
        """ """
        self._derived_properties['slicefitvalues'] = {}
        
        if fixed_buffer is not None:
            slref          = self.cube.get_slice(index=buffer_refindex, slice_object=True)
            psfref         = SlicePSF(slref, psfmodel=psfmodel)
            g_             = psfref.get_guesses()
            polynom_buffer = shapely.geometry.Point(g_["x_0_guess"],g_["y_0_guess"]).buffer(fixed_buffer)
            fitted_indexes = slref.get_spaxels_within_polygon(polynom_buffer)
        else:
            fitted_indexes = None

            
        for i_ in range(len(self.cube.lbda)):
            print(i_)
            self._derived_properties['slicefitvalues'][i_] = \
              fit_slice(self.cube.get_slice(index=i_, slice_object=True, **kwargs),
                            fitbuffer=20, fitted_indexes=fitted_indexes).fitvalues

    def get_slice_psf(self, lbdarange=None, pdfmodel="MoffatPlane0", fitbuffer=20, **kwargs):
        """ """
        return fit_slice( self.cube.get_slice(lbda_min=lbdarange[0], lbda_max=lbdarange[1],
                                                  slice_object=True),
                            fitbuffer=fitbuffer, **kwargs)

    
    # APERTURE SPECTROSCOPY
    def get_auto_aperture_spectroscopy(self, radius=10, bkgd_annulus=[1,1.2],
                                        refindex=100, waverange=20, **kwargs):
        """ """
        sliceref   = self.get_slice_psf([self.cube.lbda[refindex]-20,self.cube.lbda[refindex]+20],**kwargs)
        xref, yref = sliceref.fitvalues["x_0"], sliceref.fitvalues["y_0"]
        return self.cube.get_aperture_spec(xref, yref, radius=radius, bkgd_annulus=bkgd_annulus, refindex=refindex)
        
    def get_aperture_spectroscopy(self, x, y, radius, radius_min=None, **kwargs):
        """ 
        **kwargs goes to cube.get_slice
        """
        nlbda = len(self.cube.lbda)
        # - Formatting 
        if not hasattr(x, "__iter__") or len(x)==1:
            x = np.ones(nlbda)*x
            
        if not hasattr(y, "__iter__") or len(y)==1:
            y = np.ones(nlbda)*y
            
        if not hasattr(radius, "__iter__") or len(radius)==1:
            radius = np.ones(nlbda)*radius
        if radius_min is None:
            radius_min = [None]*nlbda
        elif not hasattr(radius_min, "__iter__") or len(radius_min)==1:
            radius_min = np.ones(nlbda)*radius_min
            
        # - Aperture per slice
        app = []
        for id_,x_,y_,r_,r_min in zip(np.arange(nlbda), x, y, radius, radius_min):
            sl_ = self.cube.get_slice(index=id_, slice_object=True)
            app.append(sl_.get_aperture(x_,y_,r_,radius_min=r_min, **kwargs))
            
        return app
    # --------- #
    #  SETTER   #
    # --------- #
    def set_cube(self, cube):
        """ attach a 3D cube to the instance """
        if Cube not in cube.__class__.__mro__:
            raise TypeError("the given cube is not a pyifu Cube (of Child of)")
        
        self._properties["cube"] = cube

    # =================== #
    #  Properties         #
    # =================== #
    @property
    def cube(self):
        """ pyifu cube (or child of) """
        return self._properties['cube']

    @property
    def slicefitvalues(self):
        """ psf parameter fitted for each slices """
        return self._derived_properties['slicefitvalues']
    
    def has_fit_ran(self):
        """ test if the slicefitvalues has been set. True means yes """
        return not self.slicefitvalues is None
        
###########################
#                         #
#   The Fitter            #
#                         #
###########################
class SlicePSF( BaseFitter ):
    """ """
    PROPERTIES         = ["slice"]
    SIDE_PROPERTIES    = ["fit_area"]
    DERIVED_PROPERTIES = ["fitted_indexes","dataindex",
                            "xfitted","yfitted","datafitted","errorfitted"]
    
    # =================== #
    #   Methods           #
    # =================== #
    def __init__(self, slice_, 
                     psfmodel="MoffatPlane0",
                     fit_area=None, fitted_indexes=None):
        """ The SlicePSF fitter object

        Parameters
        ---------- 
        slice_: [pyifu Slice] 
            The slice object that will be fitted

        psfmodel: [string] -optional-
            Name of the PSF model used to fit the slice. 
            examples: 
            - MoffatPlane`N`:a Moffat2D profile + `N`-degree Polynomial2D background 
        
        """
        self.set_slice(slice_)
        if "MoffatPlane" in psfmodel:
            self.set_model(get_moffatplane( int(psfmodel.replace("MoffatPlane","")) ))
        else:
            raise ValueError("Only the 'MoffatPlane' psfmodel has been implemented")

        if fitted_indexes is not None:
            self.set_fitted_indexes(fitted_indexes)
        elif fit_area is not None:
            self.set_fit_area(fit_area)
        else:
            self._set_fitted_values_()
            
    # --------- #
    #  SETTER   #
    # --------- #
    def set_slice(self, slice_):
        """ set a pyifu slice """
        if Slice not in slice_.__class__.__mro__:
            raise TypeError("the given slice is not a pyifu Slice (of Child of)")
        
        self._properties["slice"] = slice_

    def set_fit_area(self, polygon):
        """ Provide a polygon. Only data within this polygon will be fit 

        Parameters
        ----------
        polygon: [shapely.geometry.Polygon or array]
            The polygon definition. Spaxels within this area will be fitted.
            This could have 2 formats:
            - array: the vertices. The code will create the polygon using shapely.geometry(polygon)
            - Polygon: i.e. the result of shapely.geometry(polygon)
        
        Returns
        -------
        Void
        """
        if type(polygon) in [np.array, np.ndarray, list]:
            polygon = shapely.geometry(polygon)
        
        self._side_properties['fit_area'] = polygon
        self.set_fitted_indexes(self.slice.get_spaxels_within_polygon(polygon))
        
    def set_fitted_indexes(self, indexes):
        """ provide the spaxel indexes that will be fitted """
        self._derived_properties["fitted_indexes"] = indexes
        self._set_fitted_values_()
    # --------- #
    #  GETTER   #
    # --------- #
    def _get_model_args_(self):
        """ see model.get_loglikelihood"""
        self._set_fitted_values_()
        # corresponding data entry:
        return self._xfitted, self._yfitted, self._datafitted, self._errorfitted

    def get_guesses(self):
        """ return a dictionary containing simple best guesses """
        # gamma =  FWHM / (2 * np.sqrt(2**(1 / alpha) - 1))
        ampl = np.nanmax(self._datafitted)
        x0   = self._xfitted[np.argmax(self._datafitted)]
        y0   = self._yfitted[np.argmax(self._datafitted)]
        return dict(amplitude_guess=ampl, x_0_guess=x0, y_0_guess=y0,
                        gamma_guess=2.3, alpha_guess=2.5,alpha_boundaries=[1,5],
                        c0_0_guess=np.percentile(self._datafitted,10))
        
    # --------- #
    # PLOTTER   #
    # --------- #
    def show(self, savefile=None, show=True,
                 show_centroid=False, centroid_prop={},
                 **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        from astrobject.utils.tools     import kwargs_update
        from astrobject.utils.mpladdon  import figout
        # -- Axes Definition
        fig = mpl.figure(figsize=(9, 3))
        left, width, space = 0.075, 0.2, 0.02
        bottom, height = 0.15, 0.7
        axdata  = fig.add_axes([left+0*(width+space), bottom, width, height])
        axerr   = fig.add_axes([left+1*(width+space), bottom, width, height])
        axmodel = fig.add_axes([left+2*(width+space), bottom, width, height])
        axres   = fig.add_axes([left+3*(width+space), bottom, width,height])

        # -- Axes Definition
        slice_    = self.slice.data
        slice_var = self.slice.variance
        x,y       = np.asarray(self.slice.index_to_xy(self.slice.indexes)).T
        model_    = self.model.get_model(x ,y)

        # Plot the data with the best-fit model
        default_prop = dict(marker="h",s=15,
                            vmin=None, vmax=None)
        
        prop = kwargs_update(default_prop, **kwargs)
        
        # - Data
        axdata.scatter(x,y, c=slice_, **prop)
        axdata.set_title("Data")
        # - Error
        axerr.scatter(x,y,c=np.sqrt(slice_var), **prop)
        axerr.set_title("Error")
        
        # - Model
        prop_all = prop.copy()
        prop_all['alpha'] = prop_all.pop("alpha",0.2)/2.
        # - No fitted in light
        axmodel.scatter(x,y,c=model_, **prop_all)
        axmodel.scatter(self._xfitted,self._yfitted,c=self.model.get_model(self._xfitted,self._yfitted),**prop)
        axmodel.set_title("Model")
        
        # - Residual
        sc = axres.scatter(x,y,c=(slice_ - model_)/np.sqrt(slice_var),**prop)
        axres.set_title("Residual")

        if show_centroid:
            print("show centroid option not ready")
            #centroid = self.get_modelcentroid()
            #centroidprop = kwargs_update(dict(marker="x", color="0.7", lw=3, ms=10), **centroid_prop)
            #[ax_.plot(centroid[0][ref_lbdaidx],centroid[1][ref_lbdaidx], **centroidprop)
            #     for ax_ in fig.axes]

        [ax_.set_yticklabels([]) for ax_ in fig.axes[1:]]
        fig.figout(savefile=savefile, show=show)
        
    # =================== #
    #  Properties         #
    # =================== #
    
    @property
    def slice(self):
        """ pyifu slice """
        return self._properties['slice']

    # -------- #
    # Fitted   #
    # -------- #
    def _set_fitted_values_(self):
        """ """
        x, y = np.asarray(self.slice.index_to_xy(self.fitted_indexes)).T
        self._derived_properties['xfitted'] = x
        self._derived_properties['yfitted'] = y
        self._derived_properties['datafitted']  = self.slice.data[self._fit_dataindex]
        self._derived_properties['errorfitted'] = np.sqrt(self.slice.variance[self._fit_dataindex])
            
    @property
    def _xfitted(self):
        """ """
        return self._derived_properties['xfitted']
    @property
    def _yfitted(self):
        """ """
        return self._derived_properties['yfitted']
    @property
    def _datafitted(self):
        """ """
        return self._derived_properties['datafitted']
    @property
    def _errorfitted(self):
        """ """
        return self._derived_properties['errorfitted']

    # - indexes and ids
    @property
    def fit_area(self):
        """ polygon of the restricted fitted area (if any) """
        return self._side_properties['fit_area']

    @property
    def fitted_indexes(self):
        """ list of the fitted indexes """
        if self._derived_properties["fitted_indexes"] is None:
            return self.slice.indexes
        return self._derived_properties["fitted_indexes"]
    
    @property
    def _fit_dataindex(self):
        """ indices associated with the indexes """
        
        if self._derived_properties["fitted_indexes"] is None:
            return np.arange(self.slice.nspaxels)
        # -- Needed to speed up fit
        if self._derived_properties["dataindex"] is None:
            self._derived_properties["dataindex"] = \
              np.in1d( self.slice.indexes, self.fitted_indexes)
              
        return self._derived_properties["dataindex"]
    
###########################
#                         #
#   Model                 #
#                         #
###########################
class PSF3D( BaseModel ):
    """ """
    FREEPARAMETERS = []
    
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        # - Profile
        cls.PROFILE_PARAMETERS = cls._profile.param_names
        cls.FREEPARAMETERS     = list(cls.PROFILE_PARAMETERS)+list(cls._background.param_names)
        return super(PSF3D, cls).__new__(cls)

    def setup(self, parameters):
        """ """
        self.param_profile    = parameters[:len(self.PROFILE_PARAMETERS)]
        self.param_background = parameters[len(self.PROFILE_PARAMETERS):]
        
    def get_profile(self, x, y):
        """ The profile at the given positions """
        return self._profile.evaluate(x, y, *self.param_profile)
    
    def get_background(self, x, y):
        """ The background at the given positions """
        return self._background(x, y, *self.param_background)
    
    # ================= #
    #   Properties      #
    # ================= #
    def get_loglikelihood(self, x, y, z, dz):
        """ Measure the likelihood to find the data given the model's parameters.
        Set pdf to True to have the array prior sum of the logs (array not in log=pdf).
        In the Fitter define _get_model_args_() that should return the input of this
        """
        res = z - self.get_model(x, y)
        chi2 = np.nansum(res.flatten()**2/dz.flatten()**2)
        return -0.5 * chi2

    def get_model(self, x, y):
        """ the profile + background model. """
        return self.get_profile(x,y) + self.get_background(x,y)

#######################
#                     #
#   Moffat + BKGD     #
#                     #
#######################
def get_moffatplane(bkgdegree):
    """ """
    class _ProfileBackground_( PSF3D ):
        _profile     = Moffat2D()
        _background  = Polynomial2D(degree=bkgdegree)

    return _ProfileBackground_()

def get_gaussianplane(bkgdegree):
    """ """
    class _ProfileBackground_( PSF3D ):
        _profile     = Gaussian2D()
        _background  = Polynomial2D(degree=bkgdegree)
        
    return _ProfileBackground_()
