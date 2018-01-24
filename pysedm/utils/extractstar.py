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

from astropy.modeling.functional_models import Moffat2D, Gaussian2D
from astropy.modeling.models import Polynomial2D

from scipy.stats import multivariate_normal, norm
from pysedm.sedm import IFU_SCALE_UNIT

###########################
#                         #
#  Single Slice Fit       #
#                         #
###########################
from pysedm.utils.tools import _loading_multiprocess
_loading_multiprocess()

def fit_slice(slice_, fitbuffer=None,
              psfmodel="BiGaussianCont", fitted_indexes=None,
              lbda=None, **kwargs):
    """ 
    fitbuffer ignored if fitted_indexes provided
    """
    slpsf = SlicePSF(slice_, psfmodel=psfmodel,
                    fitbuffer=fitbuffer, fitted_indexes=fitted_indexes,
                         lbda=lbda)
    return _fit_slicepsf_(slpsf, **kwargs)

# - Internal Method for Multiprocessing
def _fit_slicepsf_(slicepsf, **kwargs):
    """ Run the Fitting method from the slicepsf """
    slicepsf.fit( **kwargs_update( slicepsf.get_guesses(), **kwargs) )
    return slicepsf

def fit_cube_per_slice(cube, notebook=True, psfmodel="BiGaussianCont",
                    fitted_indexes=None, fitbuffer=10, multiprocess=True):
    """ """
    import multiprocessing
    from astropy.utils.console import ProgressBar
    # ================= #
    #  Loading Slices   #
    # ================= #
    nlbda      = len(cube.lbda)
    prop_slice = {"psfmodel":psfmodel, "fitbuffer":fitbuffer, "fitted_indexes":fitted_indexes}
    all_slices = [SlicePSF(cube.get_slice(index=i_, slice_object=True), **prop_slice)
                      for i_ in range(nlbda)]
    
    if multiprocess:
        bar    = ProgressBar(nlbda, ipython_widget=notebook)
        p, res = multiprocessing.Pool(), {}
        for j, result in enumerate( p.imap(_fit_slicepsf_, all_slices)):
            res[j] = result.fitvalues
            bar.update(j)
        bar.update(nlbda)
        return res
    else:
        return ProgressBar.map(_fit_slicepsf_, all_slices, step=2)



# ======================= #
#   Conversion            #
# ======================= #
def sliceparam_to_cube(x, y, lbda, parameters, psfmodel,
                      indexes=None, spaxel_vertices=None):
    """ """
    from pyifu.spectroscopy import get_cube
    
    if len(parameters) != len(lbda):
        raise ValueError("lbda and parameters must have the same length (1 serie of parameter per slice)")
    
    model = read_psfmodel( psfmodel )
    data = []
    for l_index in range(len(lbda)):
        model.setup([parameters[l_index][k_] for k_ in model.freeparameters])
        data.append(model.get_model(x,y))

    if indexes is None: indexes = np.arange(len(x))
    spaxel_mapping = {id_:[x_,y_] for id_,x_,y_ in zip(indexes,x,y)}

    return get_cube(data, lbda=lbda, spaxel_mapping=spaxel_mapping,
                        spaxel_vertices=spaxel_vertices)

    

# = Method to be externalize

def get_elliptical_distance(x, y, x0=0, y0=0, ell=0, theta=0):
    """
    Parameters
    ----------
    x,y: [array]
        Cartesian Coordinates

    x0,y0: [float] -optional-
        Cartesian coordinate of the ellipse center
        
    ell: [float] -optional-
        Ellipticity [0<ell<1[
        
    theta: [float] -optional-
        Angle of the ellipse [radian]
    
    Returns
    -------
    array for float (elliptical distance)
    """
    c, s  = np.cos(theta), np.sin(theta)
    rot   = np.asarray([[c, s], [-s, c]])
    xx,yy = np.dot(rot, np.asarray([x-x0, y-y0]))
    return np.sqrt(xx**2 + (yy/(1-ell))**2)


def guess_aperture(x, y, data):
    """ Get the centroid and symmetrized standard deviation of the data given their x, y position 

    Returns
    -------
    x0, y0, std_mean (floats)
    """
    argmaxes   = np.argwhere( data>np.percentile(data,95) ).flatten()
    x0, stdx   = np.nanmean(x[argmaxes]), np.nanstd(x[argmaxes])
    y0, stdy   = np.nanmean(y[argmaxes]), np.nanstd(y[argmaxes])
    std_mean   = np.mean([stdx,stdy])
    return x0, y0, std_mean

    

###########################
#                         #
#   Extract Star          #
#                         #
###########################

class ExtractStar( BaseObject ):
    """ Object containing the individual SlicePSF fitter """
    PROPERTIES = ["cube"]
    SIDE_PROPERTIES    = ['psfmodel']
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
    def fit_psf(self, psfmodel="BiGaussianCont",
                fixed_buffer=None, buffer_refindex=100,
                notebook=False, **kwargs):
        """ """
        self._derived_properties['slicefitvalues'] = {}
        if fixed_buffer is not None:
            slref          = self.cube.get_slice(index=buffer_refindex, slice_object=True)
            psfref         = SlicePSF(slref, psfmodel=psfmodel)
            g_             = psfref.get_guesses()
            x, y           = psfref.model.centroid_guess
            polynom_buffer = shapely.geometry.Point(x, y).buffer(fixed_buffer)
            fitted_indexes = slref.get_spaxels_within_polygon(polynom_buffer)
        else:
            fitted_indexes = None

        self._side_properties["psfmodel"]          = psfmodel
        self._derived_properties['slicefitvalues'] = \
          fit_cube_per_slice(self.cube, notebook=notebook, psfmodel=psfmodel,
                                 fitted_indexes=fitted_indexes, multiprocess=True)
          
    def get_slice_psf(self, lbdarange=None, psfmodel="BiGaussianCont", fitbuffer=20, **kwargs):
        """ """
        return fit_slice( self.cube.get_slice(lbda_min=lbdarange[0], lbda_max=lbdarange[1],
                                                  slice_object=True),
                            psfmodel=psfmodel, fitbuffer=fitbuffer, lbda=np.mean(lbdarange),**kwargs)


    # ====================== #
    #  Fitted output         #
    # ====================== #
    def get_fitted_amplitudes(self):
        """ get the amplitude (parameter 'amplitude' per slice) for the fitted slices
        
        Returns
        -------
        [amplitudes, amplitude_errors]
        """
        if not self.has_fit_ran():
            raise AttributeError("You need to first run the fit: see fit_psf()")
        
        indexes = np.sort(list(self.slicefitvalues.keys()))
        return np.asarray([[self.slicefitvalues[i]['amplitude'],self.slicefitvalues[i]['amplitude.err']]
                                    for i in indexes]).T

    def get_fitted_centroid(self):
        """ get the amplitude (parameter 'amplitude' per slice) for the fitted slices
        
        Returns
        -------
        [x0, x0err], [y0, y0err]
        """
        if not self.has_fit_ran():
            raise AttributeError("You need to first run the fit: see fit_psf()")
        
        indexes = np.sort(list(self.slicefitvalues.keys()))
        return np.asarray([[self.slicefitvalues[i]['x_mean'],self.slicefitvalues[i]['x_mean.err']]
                               for i in indexes]).T,\
               np.asarray([[self.slicefitvalues[i]['y_mean'],self.slicefitvalues[i]['y_mean.err']]
                               for i in indexes]).T

    def get_fitted_modelcube(self):
        """ get the fitted model (a cube)
        = based on sliceparam_to_cube = 
        
        Returns
        -------
        Cube
        """
        if not self.has_fit_ran():
            raise AttributeError("You need to first run the fit: see fit_psf()")
        x,y  = np.asarray(self.cube.index_to_xy(self.cube.indexes)).T
        return sliceparam_to_cube(x,y, self.cube.lbda, self.slicefitvalues,
                                  self.psfmodel,
                                  indexes=self.cube.indexes,
                                  spaxel_vertices=self.cube.spaxel_vertices)

    def get_fitted_spectrum(self):
        """ Combines fitted amplitudes with wavelength from given cube to build a spectrum """
        from pyifu.spectroscopy import get_spectrum
        flux, error = self.get_fitted_amplitudes()
        spec = get_spectrum(self.cube.lbda, flux, variance=error**2)
        spec.header['TYPE']     = ("PSFSpectroscopy", "Nature of the object")
        spec.header['PSFMODEL'] = (self.psfmodel, "Which model has been used to extract the spectrum")
        spec.header['SOURCE']   = (self.cube.filename.split('/')[-1] if self.cube.filename is not None else None, "Name of the source cube")
        return spec

    
    def save_results(self):
        """ """
        print('To Be Done')


        
    # APERTURE SPECTROSCOPY
    def get_auto_aperture_spectroscopy(self, radius=10, units="spaxels",
                                           bkgd_annulus=[1,1.2],
                                        refindex=100, waverange=20, **kwargs):
        """
        radius: [float]
        
        units: [string]
             For a gaussian distribution, a radius of 
            - 1   FWHM includes 98.14% of the flux
            - 1.5 FWHM includes 99.95% of the flux
        """
        sliceref   = self.get_slice_psf([self.cube.lbda[refindex]-20,self.cube.lbda[refindex]+20],
                                        psfmodel="GaussianPlane0", **kwargs)
        xref, yref = sliceref.model.centroid
        if units in ["fwhm","FWHM"]:
            effradius = sliceref.model.fwhm * radius
        elif units in ['spaxel',"spaxels","spxl"]:
            effradius = radius
        else:
            from astropy import units
            warnings.warn("Unclear Unit %s. using astropy.units converting to spaxels assuming 1spaxels=0.75arcsec"%units)
            effradius = (radius * units.Unit(units).to("arcsec") / 0.75).value #spaxel size
            
        spec =  self.cube.get_aperture_spec(xref, yref, radius=effradius,
                                            bkgd_annulus=bkgd_annulus, refindex=refindex)
        spec.header['TYPE']     = ("ApertureSpectroscopy", "Nature of the object")
        spec.header['APTYPE']   = ("Auto", "What kind of Aperture Spectroscopy")
        spec.header['APRADIUS'] = (radius, "Radius used to extract the spectra")
        spec.header['APRUNITS'] = (units, "Unit of the Aperture radius")
        return spec
    

        
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
    def psfmodel(self):
        """ name of the PSF model used to fit the cube """
        return self._side_properties['psfmodel']
    
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
class PSFFitter( BaseFitter ):
    """ """
    PROPERTIES         = ["spaxelhandler"]
    SIDE_PROPERTIES    = ["fit_area"]
    DERIVED_PROPERTIES = ["fitted_indexes","dataindex",
                          "xfitted","yfitted","datafitted","errorfitted"]
    # -------------- #
    #  SETTER        #
    # -------------- #
    def _set_spaxelhandler_(self, spaxelhandler ) :
        """ """
        self._properties["spaxelhandler"] = spaxelhandler
        
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
        self.set_fitted_indexes(self._spaxelhandler.get_spaxels_within_polygon(polygon))
        
    def set_fitted_indexes(self, indexes):
        """ provide the spaxel indexes that will be fitted """
        self._derived_properties["fitted_indexes"] = indexes
        self._set_fitted_values_()
        
    # --------- #
    #  GETTER   #
    # --------- #    
       
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def _spaxelhandler(self):
        """ """
        return self._properties['spaxelhandler']


    def _set_fitted_values_(self):
        """ """
        x, y = np.asarray(self._spaxelhandler.index_to_xy(self.fitted_indexes)).T
        self._derived_properties['xfitted'] = x
        self._derived_properties['yfitted'] = y
        self._derived_properties['datafitted']  = self._spaxelhandler.data.T[self._fit_dataindex].T
        self._derived_properties['errorfitted'] = np.sqrt(self._spaxelhandler.variance.T[self._fit_dataindex]).T
            
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
            return self._spaxelhandler.indexes
        return self._derived_properties["fitted_indexes"]
    
    @property
    def _fit_dataindex(self):
        """ indices associated with the indexes """
        
        if self._derived_properties["fitted_indexes"] is None:
            return np.arange(self._spaxelhandler.nspaxels)
        # -- Needed to speed up fit
        if self._derived_properties["dataindex"] is None:
            self._derived_properties["dataindex"] = \
              np.in1d( self._spaxelhandler.indexes, self.fitted_indexes)
              
        return self._derived_properties["dataindex"]

    
class SlicePSF( PSFFitter ):
    """ """    
    # =================== #
    #   Methods           #
    # =================== #
    def __init__(self, slice_,
                     fitbuffer=None,fit_area=None,
                     psfmodel="BiGaussianCont",
                      fitted_indexes=None, lbda=5000):
        """ The SlicePSF fitter object

        Parameters
        ---------- 
        slice_: [pyifu Slice] 
            The slice object that will be fitted
            

        fitbuffer: [float] -optional- 
            = Ignored if fit_area or fitted_indexes are given=

        psfmodel: [string] -optional-
            Name of the PSF model used to fit the slice. 
            examples: 
            - MoffatPlane`N`:a Moffat2D profile + `N`-degree Polynomial2D background 
        
        """
        self.set_slice(slice_)
        # - Setting the model
        self.set_model(read_psfmodel(psfmodel))

        if fitted_indexes is not None:
            self.set_fitted_indexes(fitted_indexes)
        elif fit_area is not None:
            self.set_fit_area(fit_area)
        elif fitbuffer is not None:
            self._set_fitted_values_()
            g = self.get_guesses() 
            x,y = self.model.centroid_guess
            self.set_fit_area(shapely.geometry.Point(x,y).buffer(fitbuffer))
        else:
            self._set_fitted_values_()

        self.use_minuit = True

    # --------- #
    #  FITTING  #
    # --------- #
    def _get_model_args_(self):
        """ see model.get_loglikelihood"""
        self._set_fitted_values_()
        # corresponding data entry:
        return self._xfitted, self._yfitted, self._datafitted, self._errorfitted


    def get_guesses(self):
        return self.model.get_guesses(self._xfitted, self._yfitted, self._datafitted)

    # --------- #
    #  SETTER   #
    # --------- #
    def set_slice(self, slice_):
        """ set a pyifu slice """
        if Slice not in slice_.__class__.__mro__:
            raise TypeError("the given slice is not a pyifu Slice (of Child of)")
        self._set_spaxelhandler_(slice_)
        
    # --------- #
    # PLOTTER   #
    # --------- #
    def show(self, savefile=None, show=True,
                 show_centroid=False, centroid_prop={}, logscale=True,
                 vmin="2", vmax="98", **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        from astrobject.utils.tools     import kwargs_update
        from astrobject.utils.mpladdon  import figout
        # -- Axes Definition
        fig = mpl.figure(figsize=(9, 3))
        left, width, space = 0.075, 0.2, 0.02
        bottom, height = 0.15, 0.7
        axdata  = fig.add_axes([left+0*(width+space), bottom, width, height])
        axerr   = fig.add_axes([left+1*(width+space), bottom, width, height],
                                   sharex=axdata, sharey=axdata)
        axmodel = fig.add_axes([left+2*(width+space), bottom, width, height],
                                   sharex=axdata, sharey=axdata)
        axres   = fig.add_axes([left+3*(width+space), bottom, width, height],
                                   sharex=axdata, sharey=axdata)

        # -- Axes Definition
        slice_    = self.slice.data 
        slice_var = self.slice.variance 
        x,y       = np.asarray(self.slice.index_to_xy(self.slice.indexes)).T
        model_    = self.model.get_model(x ,y) 
        res_      = slice_ - model_
        # Plot the data with the best-fit model
        default_prop = dict(marker="h",s=15)
        prop = kwargs_update(default_prop, **kwargs)

        def _display_data_(ax_, data_, min_,max_, title_=None, xy=None, show_colorrange=True, **prop_):
            vmin_,vmax_ = np.percentile(data_[data_==data_], [float(min_),float(max_)])
            if xy is None:
                x_,y_ = x,y
            else:
                x_,y_ = xy

            ax_.scatter(x_,y_, c=data_, vmin=vmin_, vmax=vmax_, **prop_)
            if title_ is not None:
                ax_.set_title(title_)
            if show_colorrange:
                ax_.text(0.99, 0.99, "c-range: [%.1f;%.1f]"%(vmin_,vmax_),
                            transform=ax_.transAxes, va="top", ha="right",
                         fontsize="small")
            
        # - Data
        _display_data_(axdata, slice_ if not logscale else np.log10(slice_),
                           vmin, vmax, "Data", **prop)
        
        # - Error
        _display_data_(axerr, np.sqrt(slice_var) if not logscale else np.log10(np.sqrt(slice_var)),
                           vmin, vmax, "Error", **prop)
        
        # - Model
        _display_data_(axmodel, model_ if not logscale else np.log10(model_),
                           vmin, vmax, show_colorrange=False, 
                           **kwargs_update(prop,**{"alpha":0.2}))

                
        fmodel = self.model.get_model(self._xfitted,self._yfitted)
        _display_data_(axmodel, fmodel if not logscale else np.log10(fmodel),
                           vmin, vmax, "Model",
                           xy=[self._xfitted,self._yfitted],
                           **prop)


        # - Residual
        _display_data_(axres, res_ if not logscale else np.log10(res_),
                           vmin, vmax, "Residual", **prop)




        """
        prop_all = prop.copy()
        prop_all['alpha'] = prop_all.pop("alpha",0.2)/2.

        fmodel = self.model.get_model(self._xfitted,self._yfitted)
        # - No fitted in light
        axmodel.scatter(x,y,c=model_, **prop_all)
        axmodel.scatter(self._xfitted,self._yfitted,
                c= fmodel if not logscale else np.log(fmodel),
                    **prop)
        axmodel.set_title("Model")
        """

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
        return self._spaxelhandler
    
###########################
#                         #
#   Model                 #
#                         #
###########################
def read_psfmodel(psfmodel):
    """ """
    
    if "MoffatPlane" in psfmodel:
        return get_moffatplane( int(psfmodel.replace("MoffatPlane","")) )
    
    elif "GaussianPlane" in psfmodel:
        return get_gaussianplane( int(psfmodel.replace("GaussianPlane","")) )
                
    elif "BiNormalCont" in psfmodel:
        return get_binormalcont()
            
    elif "BiGaussianCont" in psfmodel:
        return get_bigaussiancont()
    
    elif "GaussianCont" in psfmodel:
        return get_gaussiancont()
    
    else:
        raise ValueError("Only the 'MoffatPlane/GaussianPlane/GaussianCont/BiGaussianCont' psfmodel has been implemented")



class PSFSlice( BaseModel ):
    """ """
    def setup(self, parameters):
        """ """
        self.param_profile    = parameters[:len(self.PROFILE_PARAMETERS)]
        self.param_background = parameters[len(self.PROFILE_PARAMETERS):]

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

# - Based on Astropy Modeling    
class PSFSliceAstropy( PSFSlice ):
    """ """
    FREEPARAMETERS = []
    
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        # - Profile
        cls.PROFILE_PARAMETERS = cls._profile.param_names
        cls.FREEPARAMETERS     = list(cls.PROFILE_PARAMETERS)+list(cls._background.param_names)
            
        return super(PSFSliceAstropy, cls).__new__(cls)
        
    def get_profile(self, x, y):
        """ The profile at the given positions """
        return self._profile.evaluate(x, y, *self.param_profile)
    
    def get_background(self, x, y):
        """ The background at the given positions """
        return self._background(x, y, *self.param_background)
    
# - Based on Scipy Modeling
class PSFSliceScipy( PSFSlice ):
    """ """
    PROFILE_PARAMETERS    = [] # TO BE DEFINED
    BACKGROUND_PARAMETERS = [] # TO BE DEFINED
    
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        # - Profile
        cls.FREEPARAMETERS     = list(cls.PROFILE_PARAMETERS)+list(cls.BACKGROUND_PARAMETERS)
        return super(PSFSliceScipy, cls).__new__(cls)
    
    def get_profile(self, x, y):
        """ The profile at the given positions """
        raise NotImplementedError("You must define the get_profile")
    
    def get_background(self, x, y):
        """ The background at the given positions """
        raise NotImplementedError("You must define the get_background")


    
#######################
#                     #
# BiGaussian + Const  #
#     = Scipy =       #
#######################
def get_gaussiancont():
    return GaussianCont()

class GaussianCont( PSFSliceScipy ):
    """ """
    PROFILE_PARAMETERS = ["amplitude",
                          "x_mean","y_mean",
                          "x_stddev","y_stddev","corr_xy"]
        
    BACKGROUND_PARAMETERS = ["bkgd"]
    
    # --------------- #
    # - Guesses     - #
    # --------------- #
    def get_guesses(self, x, y, data):
        """ return a dictionary containing simple best guesses """
        ampl = np.nanmax(data)
        x0   = x[np.argmax(data)]
        y0   = y[np.argmax(data)]
        self._guess = dict(amplitude_guess=ampl * 15,
                           x_mean_guess=x0, y_mean_guess=y0,
                           x_stddev_guess=2., x_stddev_boundaries=[1.,10],
                           y_stddev_guess=2., y_stddev_boundaries=[1.,10],
                           bkgd_guess=np.percentile(data,10),
                           corr_xy_guess = 0, corr_xy_boundaries= [-0.9, 0.9],
                           #amplitude_ratio_guess = 0.5,
                           #amplitude_ratio_boundaries = [0,1],
                           #stddev_ratio_guess = 2.,
                           #stddev_ratio_boundaries = [1,10],
                            )
        return self._guess
    
    # --------------- #
    # - GETTER      - #
    # --------------- #
    def get_profile(self, x, y):
        """ The profile at the given positions """
        # param_profile:
        #["amplitude", "x_mean","y_mean", "x_stddev","y_stddev","corr_xy",
        # "amplitude_ratio","stddev_ratio"]

        # - Amplitudes
        ampl = self.param_profile[0]
        
        # - centroid
        mean = self.param_profile[1], self.param_profile[2]
        
        # - Covariance Matrix
        stdx, stdy, corr_xy = self.param_profile[3],self.param_profile[4],self.param_profile[5]
        
        cov = np.asarray([[stdx**2, corr_xy*stdx*stdy], [corr_xy*stdx*stdy, stdy**2]])
        
        # - The Gaussians
        normal_1 = multivariate_normal.pdf(np.asarray([x,y]).T, mean=mean, cov=cov)
        return ampl*normal_1
    
    def get_background(self, x, y):
        """ The background at the given positions """
        return self.param_background[0]
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def centroid_guess(self):
        """ """
        return self._guess["x_mean_guess"],self._guess["y_mean_guess"]
    
    @property
    def centroid(self):
        """ """
        return self.fitvalues["x_mean"],self.fitvalues["y_mean"]
    
    @property
    def fwhm(self):
        """ """
        return "To Be Done"

#######################
#                     #
#   Gaussian2D        #
#    = Astropy =      #
#######################
def get_bigaussiancont():
    return BiGaussianCont()

class BiGaussianCont( PSFSliceScipy ):
    """ """
    PROFILE_PARAMETERS = ["amplitude",
                          "x_mean","y_mean",
                          "x_stddev","y_stddev","corr_xy",
                          "amplitude_ratio","stddev_ratio"]
        
    BACKGROUND_PARAMETERS = ["bkgd"]
    
    # --------------- #
    # - Guesses     - #
    # --------------- #
    def get_guesses(self, x, y, data):
        """ return a dictionary containing simple best guesses """
        ampl = np.nanmax(data)
        x0   = x[np.argmax(data)]
        y0   = y[np.argmax(data)]
        self._guess = dict(amplitude_guess=ampl * 15,
                           x_mean_guess=x0, y_mean_guess=y0,
                           x_stddev_guess=1., x_stddev_boundaries=[0.5,5.],
                           y_stddev_guess=1., y_stddev_boundaries=[0.5,5.],
                           bkgd_guess=np.percentile(data,10),
                           corr_xy_guess = 0, corr_xy_boundaries= [-0.9, 0.9],
                           amplitude_ratio_guess = 0.2,
                           amplitude_ratio_boundaries = [0,1],
                           stddev_ratio_guess = 5.,
                           stddev_ratio_boundaries = [1,10],
                            )
        return self._guess
    
    # --------------- #
    # - GETTER      - #
    # --------------- #
    def get_profile(self, x, y):
        """ The profile at the given positions """
        # param_profile:
        #["amplitude", "x_mean","y_mean", "x_stddev","y_stddev","corr_xy",
        # "amplitude_ratio","stddev_ratio"]

        ampl, xmean,ymean, xstd, ystd, corrxy, ampl_ratio, stddev_ratio = self.param_profile        
        
        # - Covariance Matrix
        cov = np.asarray([[xstd**2, corrxy*xstd*ystd], [corrxy*xstd*ystd,ystd**2]])
        
        # - The Gaussians
        normal_1 = multivariate_normal.pdf(np.asarray([x,y]).T, mean=[xmean,ymean], cov=cov)
        normal_2 = multivariate_normal.pdf(np.asarray([x,y]).T, mean=[xmean,ymean], cov=cov*stddev_ratio**2)
        return ampl*(normal_1 + normal_2*ampl_ratio)
    
    def get_background(self, x, y):
        """ The background at the given positions """
        return self.param_background[0]
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def centroid_guess(self):
        """ """
        return self._guess["x_mean_guess"],self._guess["y_mean_guess"]
    
    @property
    def centroid(self):
        """ """
        return self.fitvalues["x_mean"],self.fitvalues["y_mean"]
    
    @property
    def fwhm(self):
        """ """
        return "To Be Done"


#######################
#                     #
#   Gaussian Dist     #
#    = Astropy =      #
#######################
def get_binormalcont():
    return BiNormalCont()

class BiNormalCont( PSFSliceScipy ):
    """ """
    PROFILE_PARAMETERS = ["amplitude",
                          "x_mean","y_mean",
                          "stddev","ell","theta",
                          "amplitude_ratio","stddev_ratio",
                          "x_offset","y_offset"]
        
    BACKGROUND_PARAMETERS = ["bkgd"]
    
    # ================== #
    #  Guess             #
    # ================== #
    def get_guesses(self, x, y, data):
        """ return a dictionary containing simple best guesses """
        ampl       = np.nanmax(data)
        argmaxes   = np.argwhere(data>np.percentile(data,95)).flatten()
        x0, stdx   = np.nanmean(x[argmaxes]), np.nanstd(x[argmaxes])
        y0, stdy   = np.nanmean(y[argmaxes]), np.nanstd(y[argmaxes])
        std_mean   = np.mean([stdx,stdy])
        ell        = 1 - np.nanmin([stdx,stdy])/np.nanmax([stdx,stdy])
        theta      = 0 if stdx>stdy else np.pi/2.
        
        self._guess = dict(amplitude_guess=ampl * 10,
                           x_mean_guess=x0, x_mean_boundaries=[x0-10,x0+10],
                           y_mean_guess=y0, y_mean_boundaries=[y0-10,y0+10],
                           # shift of the 2nd gaussian (tails)
                           x_offset_guess=0, x_offset_boundaries=[-2,2], x_offset_fixed=True,
                           y_offset_guess=0, y_offset_boundaries=[-2,2], y_offset_fixed=True,
                           # - STD
                           stddev_guess = std_mean, stddev_boundaries=[0.5,std_mean*3],
                           # - background
                           bkgd_guess=np.percentile(data,10),
                           # Converges faster by allowing degenerated param...
                           theta_guess=theta, theta_boundaries=[-np.pi,np.pi],
                           ell_guess = ell, ell_boundaries= [-0.9, 0.9],
                           amplitude_ratio_guess = 3.5,
                           amplitude_ratio_fixed = True,
                           amplitude_ratio_boundaries = [3,4],
                           stddev_ratio_guess = 2.2,
                           stddev_ratio_boundaries = [1.8,4],
                            )
        return self._guess
    
    # ================== #
    #  Model             #
    # ================== #
    def _get_elliptical_distance_(self, x, y, x_offset=0, y_offset=0):
        """ """
        return get_elliptical_distance(x,y, x0=self.param_profile[1]+x_offset, y0=self.param_profile[2]+y_offset,
                                        ell=self.param_profile[4],theta=self.param_profile[5])

    def get_profile(self, x, y):
        """ """    
        r1 = self._get_elliptical_distance_(x,y, x_offset=0, y_offset=0)
        if self.param_profile[8] == 0 and self.param_profile[9] == 0:
            r2 = r1
        else:
            r2 = self._get_elliptical_distance_(x,y, x_offset=self.param_profile[6], y_offset=self.param_profile[7])
        
        n1 = norm.pdf(r1, loc=0, scale=self.param_profile[3])
        n2 = norm.pdf(r2, loc=0, scale=self.param_profile[3]*self.param_profile[7])
        
        ampl  = self.param_profile[0]
        ratio = self.param_profile[6]
        
        return ampl * ( (ratio/(1.+ratio)) * n1 + (1./(1+ratio)) * n2)

    def get_background(self,x,y):
        """ The background at the given positions """
        return self.param_background[0]
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def centroid_guess(self):
        """ """
        return self._guess["x_mean_guess"], self._guess["y_mean_guess"]
    
    @property
    def centroid(self):
        """ """
        return self.fitvalues["x_mean"], self.fitvalues["y_mean"]
    
    @property
    def fwhm(self):
        """ """
        return "To Be Done"
    
#######################
#                     #
#   Moffat + BKGD     #
#    = Astropy =      #
#######################
def get_moffatplane(bkgdegree):
    """ """
    class MoffatPlane( MoffatPlaneVirtual ):
        _background  = Polynomial2D(degree=bkgdegree)

    return MoffatPlane()

class MoffatPlaneVirtual( PSFSliceAstropy ):
    """ """
    _profile     = Moffat2D()

    # --------------- #
    def get_guesses(self, x, y, data):
        """ return a dictionary containing simple best guesses """
        ampl = np.nanmax(data)
        x0   = x[np.argmax(data)]
        y0   = y[np.argmax(data)]
        self._guess = dict(amplitude_guess=ampl, x_0_guess=x0, y_0_guess=y0,
                        gamma_guess=3., alpha_guess=2.5, alpha_boundaries=[1,5],
                        c0_0_guess=np.percentile(data,10))
        return self._guess
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def centroid_guess(self):
        """ """
        return self._guess["x_0_guess"],self._guess["y_0_guess"]
    
    @property
    def centroid(self):
        """ """
        return self.fitvalues["x_0"],self.fitvalues["y_0"]
    
    @property
    def fwhm(self):
        """ """
        gamma = self.param_profile[3]
        alpha = self.param_profile[4]
        return gamma * np.sqrt(2.**(1. / alpha) - 1.)
        
#######################
#                     #
#  Gaussian + BKGD    #
#    = Astropy =      #
#######################
class GaussianPlaneVirtual( PSFSliceAstropy ):
    """ """
    _profile     = Gaussian2D()
    # --------------- #
    def get_guesses(self, x, y, data):
        """ return a dictionary containing simple best guesses """
        ampl = np.nanmax(data)
        x0   = x[np.argmax(data)]
        y0   = y[np.argmax(data)]
        self._guess = dict(amplitude_guess=ampl,
                        x_mean_guess=x0, y_mean_guess=y0,
                        x_stddev_guess=1, x_stddev_boundaries=[0.2,10],
                        y_stddev_guess=1, y_stddev_boundaries=[0.2,10], 
                        c0_0_guess=np.percentile(data,10))
        return self._guess
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def centroid_guess(self):
        """ """
        return self._guess["x_mean_guess"],self._guess["y_mean_guess"]
    
    @property
    def centroid(self):
        """ """
        return self.param_profile[1],self.param_profile[2]

    @property
    def fwhm(self):
        """ """
        xstd = self.param_profile[3]
        ystd = self.param_profile[4]
        return 2.355 * np.mean([xstd,ystd])

    
def get_gaussianplane(bkgdegree):
    """ """
    class GaussianPlane( GaussianPlaneVirtual ):
        _background  = Polynomial2D(degree=bkgdegree)
        
    return GaussianPlane()




# ================= #
# = NOT READY YET = #


############################
############################
##                        ##
##  Cube 3D BiNormal      ##
##                        ##
############################
############################

########################
#                      #
#     Fitter           #
#                      #
########################
class CubePSF( PSFFitter ):
    """ """
    
    # =================== #
    #   Methods           #
    # =================== #
    def __init__(self, cube_,
                     fitbuffer=None,fit_area=None,
                     psfmodel="BiNormalCont3d", refindex=150,
                      fitted_indexes=None, lbda=5000):
        """ The SlicePSF fitter object

        Parameters
        ---------- 
        cube_: [pyifu Slice] 
            The slice object that will be fitted
            

        fitbuffer: [float] -optional- 
            = Ignored if fit_area or fitted_indexes are given=

        psfmodel: [string] -optional-
            Name of the PSF model used to fit the slice. 
            examples: 
            - MoffatPlane`N`:a Moffat2D profile + `N`-degree Polynomial2D background 
        
        """
        self.set_cube(cube_)
        if "BiNormalCont3d" in psfmodel:
            self.set_model(get_binormalcont3d(self.cube.lbda, refindex, self.cube.adr) )
        else:    
            raise ValueError("Only the 'BiNormalCont3d' psfmodel has been implemented")
        
        if fitted_indexes is not None:
            self.set_fitted_indexes(fitted_indexes)
        elif fit_area is not None:
            self.set_fit_area(fit_area)
        else:
            self._set_fitted_values_()
        # - Which fitter
        self.use_minuit = True


    # --------- #
    #  FITTING  #
    # --------- #
    def _get_model_args_(self):
        """ see model.get_loglikelihood"""
        self._set_fitted_values_()
        # corresponding data entry:
        return self._xfitted, self._yfitted, None, self._datafitted, self._errorfitted


    def get_guesses(self):
        return self.model.get_guesses(self._xfitted, self._yfitted, self._datafitted)

    
    # -------------- #
    #  SETTER        #
    # -------------- #
    def set_cube(self, cube):
        """ """
        self._set_spaxelhandler_(cube)
        
    @property
    def cube(self):
        return self._spaxelhandler
    
########################
#                      #
#     Model            #
#                      #
########################
class PSF3D( BaseModel ):
    """ """
    NLBDA = 0
    PROFILE_PARAMETERS    = [] # TO BE DEFINED
    BACKGROUND_PARAMETERS = [] # TO BE DEFINED

    # ================== #
    #   DEfinitions
    # ================== #
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        # - Profile
        amplitudes  = ["ampl_%d"%d for d in range(cls.NLBDA)]
        background  = np.concatenate([["%s_%d"%(p_,d)
                                           for p_ in cls.BACKGROUND_PARAMETERS]
                                    for d in range(cls.NLBDA)])
        cls.FREEPARAMETERS     = amplitudes+list(cls.PROFILE_PARAMETERS)+list(background.flatten())
        return super(PSF3D, cls).__new__(cls)

    def setup(self, parameters):
        """ """
        self.amplitudes       = parameters[:self.NLBDA]
        self.param_profile    = parameters[self.NLBDA:self.NLBDA+len(self.PROFILE_PARAMETERS)]
        self.param_background = parameters[self.NLBDA+len(self.PROFILE_PARAMETERS):]

    # ================= #
    #   Properties      #
    # ================= #
    # - Mandatory for modefit
    def get_loglikelihood(self, x, y, lbda, z, dz):
        """ Measure the likelihood to find the data given the model's parameters.
        Set pdf to True to have the array prior sum of the logs (array not in log=pdf).
        In the Fitter define _get_model_args_() that should return the input of this
        """
        res = z - self.get_model(x, y, lbda)
        chi2 = np.nansum(res.flatten()**2/dz.flatten()**2)
        return -0.5 * chi2

    # - Structure of the model
    def get_model(self, x, y, lbda=None):
        """ the profile + background model. """
        return self.get_profile(x,y, lbda=None) + self.get_background(x,y, lbda=None)

    def get_profile(self, x, y, lbda):
        """ The profile at the given positions """
        raise NotImplementedError("You must define the get_profile")
    
    def get_background(self, x, y, lbda):
        """ The background at the given positions """
        raise NotImplementedError("You must define the get_background")


# - The Model
def get_binormalcont3d(lbda, refindex, adr):
    """ """
    class _BiNormalCont3D_(BiNormalCont3D):
        NLBDA = len(lbda)

    return _BiNormalCont3D_(lbda, refindex, adr)

        
class BiNormalCont3D( PSF3D ):
    """ """
    # - BaseObject
    PROPERTIES         = ["lbda","refindex", "adr"]
    DERIVED_PROPERTIES = ["fixed_source_shift"] #X,Y

    # - Model
    NLBDA              = -1
    PROFILE_PARAMETERS = ["x_mean","y_mean",
                          "stddev","ell","theta",
                          "amplitude_ratio","stddev_ratio",
                          "pa_angle","spaxel_unit"] # ADR]
        
    BACKGROUND_PARAMETERS = ["bkgd"]

    # ================== #
    #  Definition        #
    # ================== #
    def __init__(self, lbda, refindex, adr ):
        """ """
        self.set_lbda(lbda)
        self.set_adr(adr)
        self.set_refindex(refindex)
        
    # ================== #
    #  Guess             #
    # ================== #
        
    def get_guesses(self, x, y, data, refindex=150):
        """ return a dictionary containing simple best guesses """
        ampl = np.nanmax(data)
        argmaxes = np.argwhere(data>np.percentile(data[refindex],95)).flatten()
        x0   = np.nanmean(x[argmaxes])
        y0   = np.nanmean(y[argmaxes])
        self._guess = dict(x_mean_guess=x0, x_mean_boundaries=[x0-10,x0+10],
                           y_mean_guess=y0, y_mean_boundaries=[y0-10,y0+10],
                           stddev_guess=2., stddev_boundaries=[0.5,5.],
                           # Converges faster by allowing degenerated param...
                           theta_guess=0, theta_boundaries=[-np.pi,np.pi],
                           ell_guess = 0, ell_boundaries= [-0.9, 0.9],
                           amplitude_ratio_guess = 0.2,
                           amplitude_ratio_boundaries = [0,1],
                           stddev_ratio_guess = 5.,
                           stddev_ratio_boundaries = [1,10],
                           # - ADR 
                           pa_angle_guess= self.adr.parangle, pa_angle_fixed=True,
                           # Spaxel Unit
                           spaxel_unit_guess=IFU_SCALE_UNIT, spaxel_unit_fixed=True,
                           spaxel_unit_boundaries=[0.,10],
                           )
        
        spectop = np.percentile(data, 95, axis=1)
        specsky = np.percentile(data,  5, axis=1)
        for i in range(self.NLBDA):
            self._guess["ampl_%d_guess"%i] = spectop[i]
            self._guess["bkgd_%d_guess"%i] = specsky[i]
            
        return self._guess
    
    # ================== #
    #  Model             #
    # ================== #
    # - ADR Model Prop - #
    @property
    def fixed_adr(self):
        """ Are the ADR parameters: spaxel_unit and pa_angle fixed?"""
        return self.param_input["spaxel_unit_fixed"] and self.param_input["pa_angle_fixed"]
    
    @property
    def rotation_matrix(self):
        c, s = np.cos(self.param_profile[-2]), np.sin(self.param_profile[-2])
        return np.asarray([[c, s], [-s, c]])

    @property
    def spaxel_unit(self):
        return self.param_profile[-1]

    @property
    def fixed_source_shift(self):
        """ """
        if self._derived_properties['fixed_source_shift'] is None:
            self._derived_properties['fixed_source_shift'] = self.get_source_shift()
        return self._derived_properties['fixed_source_shift']
        
    # -------------------- #
    def get_psf_centroid(self):
        if self.fixed_adr:
            xshift,yshift = self.fixed_source_shift
        else:
                xshift,yshift = self.get_source_shift()
            
        return self.param_profile[0]+xshift,self.param_profile[1]+yshift


    # --------------- #
    #  Actual Model   #
    # --------------- #
    # Centroid Evolution
    def get_source_shift(self):
        """  """
        x_default, y_default = self.adr.refract(0, 0, self.lbda, unit=self.spaxel_unit)
        return np.dot(self.rotation_matrix, np.asarray([x_default,y_default]))

    # Elliptical distance to the core of the gaussian
    def _get_elliptical_distance_(self, x, y):
        """ """
        x0, y0 = self.get_psf_centroid()
        return get_elliptical_distance(x, y, x0=self.param_profile[0], y0=self.param_profile[1],
                                        ell=self.param_profile[3],theta=self.param_profile[4])
                                        
    # Elliptical distance to the core of the gaussian
    def _kolmogorov_width_(self):
        """ """
        return self.param_profile[2] * (self.lbda / self.adr.lbdaref)**(-1/5.)

    def get_profile(self, x, y, **kwargs):
        """ """
        r = self._get_elliptical_distance_(x,y)
        scale_base = self._kolmogorov_width_()
        n1 = norm.pdf(r, loc=0, scale=np.asarray([scale_base]*len(r)).T  )
        n2 = norm.pdf(r, loc=0, scale=np.asarray([scale_base *self.param_profile[7]]).T)
        return self.param_profile[0] * (n1 + self.param_profile[6] * n2)

    def get_background(self, x, y,  **kwargs):
        """ The background at the given positions """
        return np.asarray([self.param_background]*len(x)).T
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def centroid_guess(self):
        """ """
        return self._guess["x_mean_guess"],self._guess["y_mean_guess"]
    
    @property
    def centroid(self):
        """ """
        return self.fitvalues["x_mean"],self.fitvalues["y_mean"]
    
    @property
    def fwhm(self):
        """ """
        return "To Be Done"
    
    # ------------- #
    #  Core Prop  - #
    # ------------- #
    # - LBDA
    @property
    def lbda(self):
        return self._properties['lbda']
    
    def set_lbda(self, lbda):
        """ """
        if len(lbda) != self.NLBDA:
            raise ValueError("given lbda (array of %d) does not match the NLBDA property (%d)"%(len(lbda),self.NLBDA))
        self._properties['lbda'] = lbda
        if self.adr is not None and self.refindex is not None:
            self.adr.set(lbdaref=self.lbda[self.refindex])

    # - ADR  
    @property
    def adr(self):
        return self._properties['adr']
    
    def set_adr(self,adr, **kwargs):
        
        self._properties['adr'] = adr
        self.adr.set(**kwargs)
        if self.lbda is not None and self.refindex is not None:
            self.adr.set(lbdaref=self.lbda[self.refindex])
            
    # - REFINDEX      
    @property
    def refindex(self):
        """ """
        return self._properties['refindex']
    
    def set_refindex(self,index):
        """ """
        self._properties['refindex'] = index
        if self.lbda is not None and self.adr is not None:
            self.adr.set(lbdaref=self.lbda[self.refindex])
