#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract Star based on Modefit """

import warnings
import numpy                 as np
from scipy.stats         import norm

from propobject          import BaseObject

from pyifu                import adr
from pyifu.spectroscopy   import Slice

from pysedm.sedm          import IFU_SCALE_UNIT # This is the only SEDM part.
from pysedm.utils.tools   import kwargs_update, is_arraylike, make_method

from modefit.baseobjects import BaseFitter, BaseModel


###########################
#                         #
#  Main Functions         #
#                         #
###########################

def get_psfmodel(psfdata):
    """ """
    psf3d = PSF3D_BiNormalCont()
    psf3d.set_psfdata(psfdata)
    return psf3d


# ======================== #
#                          #
#   High Level Function    #
#                          #
# ======================== #
"""
Force PSF spectroscopy is made in 2 steps:

1) Get the PSF parameters by fitting several meta-slices:

`fit_psf_parameters()` is made to extract the psfmodel.
This psf model contain the normalized psf position and shape.
Only the amplitude remains to be fitted. 

2) Perform ForcePSF3D assuming the PSF shape.

`fit_force_spectroscopy()`


"""
def fit_psf_parameters(cube,lbda_range = [4500,7500], nbins=10,
                        savedata=None, savefig=None,
                        return_psfmodel=True):
    """ Extract the PSF shape parameters for the given cube.
    = This function is made to fit signle point source cube. =

    Parameters
    ----------
    cube: [pyifu.Cube] 
        wavelength calibrated euro3d cube (pyifu format) containing the point source

    lbda_range: [float, float] -optional-
        wavelength range between which the slices will be made

    nbins: [int] -optional-
        the number of bins that will devide the lbda_range to make the slices.

    savefig: [string or None] -optional-
        if you wnat to save the PSF-fitting procedure figure, provide the path where to save it.
        
    return_psfmodel: [bool] -optional-
        shall this method return a PSF3D object (True) or the psffitter one (False)?

    Returns
    -------
    PSF3D or PSFFitter (see return_psfmodel option)
    """
    if savefig:
        params  = np.asarray(["position","stddev","stddev_ratio","ell","theta"])
        nparams = len(params)
        # - Figures
        import matplotlib.pyplot as mpl
        fig        = mpl.figure(figsize=[2*nparams, 6.5])
        left, width, span = 0.09, 0.13, 0.05
        axes_step1 = [fig.add_axes( [left+(width+span)*i, 0.7, width,0.25]) for i in range(nparams)]
        axes_adr   = fig.add_axes( [left, 0.1, 0.5 , 0.5])
        axes_stddev= fig.add_axes( [left+ 0.6, 0.1, 0.25, 0.5])
        
    psfcube = FitPSF(cube)
    # ====
    # Step 1, All Free parameters
    psfcube.fit_slices(lbda_range=lbda_range, nbins=nbins, 
                       ell_guess=0.05, ell_boundaries=[0,0.9], ell_fixed=False,
                       theta_guess=1.5, theta_boundaries=[0,np.pi], theta_fixed=False, 
                       stddev_ratio_guess=1.7,
                       stddev_ratio_boundaries=[1.1,4],
                       stddev_ratio_fixed=False)
    
    cont_param = psfcube.get_const_parameters()

    # -- Figure Step 1
    if savefig is not None:
        psfcube.show(axes=axes_step1, params=params, set_labels=False, show=False)
        for k,v in cont_param.items():
            if k not in params: continue
            which_axe = np.argwhere(params==k).flatten()
            axes_step1[which_axe[0]].axhline(v, ls="-", lw=1, color="k", alpha=0.8, zorder=1)
            axes_step1[which_axe[0]].set_title(k)
            
        # - alpha the refitted one
        for ax_ in np.asarray(axes_step1)[[np.argwhere(params==k).flatten()[0] for k in ["position","stddev"]]]:
            for gc in ax_.get_children():
                if hasattr(gc,"set_alpha"):
                    gc.set_alpha(0.5)
                
            ax_.tick_params(color="0.5", labelcolor="0.5")

        fig.tight_layout()
        axes_step1[-1].text(1.1,0.5, "First Fit Iteration", fontsize="large",
                            transform = axes_step1[-1].transAxes, rotation=-90,
                            va="center", ha="left")
    
    # ====
    # Step 2, fix theta, stddev_ratio and ell
    fitprop = {}
    for k,v in cont_param.items():
        fitprop[k+"_guess"] = v
        fitprop[k+"_fixed"] = True

    psfcube.fit_slices(lbda_range=lbda_range, nbins=nbins, **fitprop)
    ntry = 0
    indexes = None
    while ntry<30:
        psfcube.fit_adr_param(indexes=indexes)
        chi2dof = psfcube.adrfitter.fitvalues["chi2"] / psfcube.adrfitter.dof
        if chi2dof > 10:
            indexes = np.random.choice(np.arange(nbins), int(nbins*0.7), replace=False)
            ntry+=1
            print("Warnings - chi2/dof of %.2f -> refit the adr with 30percent out"%chi2dof)
        else:
            break

    psfcube.fit_stddev()

    # -- Figure Step 2
    if savefig is not None:
        psfcube.show_adr_fit(ax=axes_adr, show_colorbar=False, show=False)
        axes_adr.set_xlabel("x-position")
        axes_adr.set_ylabel("y-position")

        psfcube.show_stddev_fit(ax=axes_stddev,ls="-", lw=1, color="k", alpha=0.8, zorder=1, show=False)
        fig.savefig(savefig)
        
    # ====
    # Output
    if savedata is not None:
        psfcube.write_fitted_data(savedata)
        
    if return_psfmodel:
        return get_psfmodel(psfcube.fitted_data)
    return psfcube

def fit_force_spectroscopy(cube, psfmodel, savefig=None):
    """ Provide a cube and a psfmodel (normalized 3D psf) and get spectrum and background
    
    Parameters
    ----------

    Returns
    -------
    Spectrum, Spectrum (source, background)
    """
    forcepsf = ForcePSF(cube, psfmodel)
    spec,bkgd = forcepsf.fit_forcepsf()
    if savefig is not None:
        forcepsf.show(savefile=savefig)
    return spec, bkgd, forcepsf

#############################
#                           #
#  Profiles                 #
#                           #
#############################

"""
# How To build a new profile?

All profile must have the following format:
```python
def name_of_the_profile(x,y,
                        here, define, your, variables,
                        xcentroid=0, ycentroid=0):
     ''' ADD Documentation '''

     do_the_job

     return list_of_values (1darray with the same length as x/y)
```
The profile should be normalize, but this is not mendatory.

"""
def binormal_profile(x, y,
                stddev, stddev_ratio, amplitude_ratio, theta, ell,
                xcentroid=0, ycentroid=0):
    """ """
    r = get_elliptical_distance(x, y, xcentroid=xcentroid, ycentroid=ycentroid,  ell=ell, theta=theta)
    
    n1 = norm.pdf(r, loc=0, scale=stddev)
    n2 = norm.pdf(r, loc=0, scale=stddev*stddev_ratio)
            
    return ( (amplitude_ratio/(1.+amplitude_ratio)) * n1 + (1./(1+amplitude_ratio)) * n2)


# ---------------------- #
#  Interal Tool          #
# ---------------------- #
def get_elliptical_distance(x, y, xcentroid=0, ycentroid=0, ell=0, theta=0):
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
    xx,yy = np.dot(rot, np.asarray([x-xcentroid, y-ycentroid]))
    return np.sqrt(xx**2 + (yy/(1-ell))**2)


#############################
#                           #
#  Slice Fitter             #
#                           #
#############################
def fit_slice(slice_, fitbuffer=None,
              psfmodel="BiNormalCont", fitted_indexes=None,
              lbda=None, **kwargs):
    """ Fit PSF Slice without forcing it's shape

    Parameters
    ----------

    Returns
    -------
    SlicePSF
    """
    slpsf = SlicePSF(slice_, psfmodel=psfmodel,
                    fitbuffer=fitbuffer, fitted_indexes=fitted_indexes,
                    lbda=lbda)
    slpsf.fit( **kwargs_update( slpsf.get_guesses(), **kwargs) )
    return slpsf


def fit_forcepsf_slice( data, psfshape, variance = None,
                        amplitude_guess=None, 
                        amplitude_boundaries=[0,None], amplitude_fixed=False,
                        background_guess=None,
                        background_boundaries=[0,None], background_fixed=False,
                        print_level=0, errordef=1, full_output=False):
    """ Force PSF fitter: only fitting amplitude and background assuming the PSF shape.
    
    Parameters
    ----------
    data: [array]
        nd array containing the data
        
    psfshape: [array]
        nd array containing the normalized model. 
        = This function will fit an amplitude such that amplitude*psfshape => data

    variance: [array] -optional-
        nd array containing the variance (error**2) on the data
        
    // fit 

    {amplitude,background}_guess: [float]
        guess value for the amplitude (or backgorund). If None, an automatic guess will be used

    {amplitude,background}_boundaries: [min, max] -optional-
        boundaries on the fitted parameter (amplitudes or backgorund). None means, no limit.

    {amplitude,background}_fixed: [bool] -optional-
        Is the amplitude (background) parameter fixed to its guess value?


    // Minuit options

    print_level: [0/1] -optional-
        set the print_level for this Minuit. 0 is quiet.
        1 print out at the end of migrad/hesse/minos. 

    errordef: [float] -optional-
        Amount of increase in fcn to be defined
        as 1 :math:`\sigma`. If None is given, it will look at
        `fcn.default_errordef()`. If `fcn.default_errordef()` is not
        defined or
        not callable iminuit will give a warning and set errordef to 1.
        Default None(which means errordef=1 with a warning).

    // return options

    full_output: [bool] -optional-
        Do you want a dictionary containing the basic fit information (False)
        or this dictionary plus minuit.migrad full output (True)?


    Returns
    -------
    dict (or dict + migrad outout, see full_output)
    """
    from iminuit import Minuit
    # = The Data
    variance = variance if variance is not None else np.ones(len(data)) 
    
    # = The Function
    def local_chi2(amplitude, background): 
        return np.nansum( (data - (amplitude * psfshape+ background))**2 / variance)
                
    # = The Guess
    if amplitude_guess is None:
        amplitude_guess = np.nanmax(data)
    if background_guess is None:
        background_guess      = np.median(data)
    # Setting the Fitter
    param_prop = {}
    inparam = locals()
    for k in ["amplitude", "background"]:
        param_prop[k]          = inparam["%s_guess"%k]
        param_prop["limit_"+k] = inparam["%s_boundaries"%k]
        param_prop["fix_"+k]   = inparam["%s_fixed"%k]
    
    minuit = Minuit(local_chi2,
                        print_level=print_level, errordef=errordef,
                        # guess, boundaries and fixed
                        **param_prop)
        
    # Running the fit
    _migrad_output_ = minuit.migrad()
        
    # Load the output
    if not _migrad_output_[0]["is_valid"]:
        print("WARNING: minuit returned 'False' for migrad `is_valid`")
            
    fitvalues= {"amplitude":minuit.values["amplitude"],
                "amplitude.err":minuit.errors["amplitude"],
                "background":minuit.values["background"],
                "background.err":minuit.errors["background"],
                "chi2":minuit.fval,
                "npoints":len(data)}
        
    # = Output
    if full_output:
        return fitvalues, _migrad_output_
    return fitvalues

###########################
#                         #
#  3D PSF Object (Cube)   #
#                         #
###########################
class _PSF3D_( BaseObject ):
    """ The virtual PSF3D is a flexible method enableling to uncapsulate shape and expected position of a point source """
    PROPERTIES = ["adr","refposition", "unit",
                  "profile_param"]
    DERIVED_PROPERTIES = ["psfdata"]
    
    # - DEFINE THIS
    PROFILE_PARAMETERS = ["ToBeDefined"]

    def __new__(cls,*arg,**kwargs):
        """ Upgrade of the New function to enable the
        the _minuit_ black magic
        """
        obj = super(_PSF3D_,cls).__new__(cls)
        exec("@make_method(_PSF3D_)\n"+\
             "def set_profile_param(self,%s): \n"%(", ".join(obj.PROFILE_PARAMETERS))+\
             '    """ Set the profile parameters. They could be functions or floats.\n'+\
             '    If Function they have to be callable as `funct(lbda)` and return lbda-size array or float \n'+\
             '    """\n'+\
             "    self._properties['profile_param'] = locals().copy()")
        return obj
        
    # =============== #
    #  Methods        #
    # =============== #
    # --------- #
    #  I/O      #
    # --------- #
    def writeto(self, savefile):
        """ """
        import json
        if not savefile.endswith(".json"):
            savefile +=".json"
        with open(savefile, 'w') as fp:
            json.dump(self.psfdata, fp)
            
    def load(self, datafile):
        """ 
        file containing a dictionary with:
        {adr:{ADR_PARAMETERS},
        "profile":{PROFILE_PARAMETERS}
        }
        """
        self.set_psfdata( load_dict(datafile) )

    def set_psfdata(self, data):
        """ """
        raise NotImplementedError("You need to implement the `set_psfdata` method.")
        
    # --------- #
    #  GETTER   #
    # --------- #
    def get_psf_param(self, lbda):
        """ """
        psf = {}
        psf["xcentroid"], psf["ycentroid"] = self.adr.refract(self.refposition[0], self.refposition[1],
                                                  lbda, unit=self.unit)
        for k in self.PROFILE_PARAMETERS:            
            psf[k]  = self.profile_param[k](lbda) if callable(self.profile_param[k]) else self.profile_param[k]
        return psf
    
    # --------- #
    #  SETTER   #
    # --------- #            
    def set_adr(self, adr, xref=0, yref=0, unit=1, **kwargs):
        """ Attach to this method the adr and the coordinate of the centroid at the ADR reference wavelength 
        **kwargs is passed to the adr.set method
        """
        self._properties["adr"] = adr
        self._properties["refposition"] = [xref, yref]
        self._properties["unit"] = unit
        self.adr.set(**kwargs)
        
    # =============== #
    #  Properties     #
    # =============== #
    # ---------
    # Profile
    @property
    def profile_param(self):
        """ """
        return self._properties["profile_param"]
    # ---------
    # ADR
    @property
    def adr(self):
        """ Atmospheric Differential Refraction """
        return self._properties["adr"]
    
    @property
    def refposition(self):
        """ Coordinate [x,y] of the centroid at the ADR reference wavelength (adr.lbdaref)"""
        return self._properties["refposition"]

    @property
    def unit(self):
        """ Unit in arcsec of '1' position shift """
        return self._properties["unit"]

    # ------
    # derived
    @property
    def psfdata(self):
        """ """
        return self._derived_properties["psfdata"]

class PSF3D_BiNormalCont( _PSF3D_ ):
    """ """
    PROPERTIES  = ["stddev_ref"]
    PROFILE_PARAMETERS = ["stddev", "stddev_ratio", "amplitude_ratio",
                          "theta", "ell"]
        
    # =============== #
    #  Methods        #
    # =============== #        

        
    def set_psfdata(self, data):
        """ Methods to set the dictionary:
        {adr:{ADR_PARAMETERS},
        "profile":{PROFILE_PARAMETERS}
        }
        """
        # - ADR
        adrmodel  = adr.ADR()
        for k, v in data["adr"].items():
            if k not in ["xcentroid","ycentroid","position","unit","xref","yref","parangle_ref"] and ".err" not in k:
                adrmodel.set(**{k:v})
            
        self.set_adr(adrmodel, xref=data["adr"]["xref"], yref=data["adr"]["yref"], unit=data["adr"]["unit"])
        
        # - Profile
        self.set_stddev_ref(data["profile"]["stddev_ref"])
        self.set_profile_param(stddev         = self.get_stddev,
                              stddev_ratio    = data["profile"]["stddev_ratio"],
                              amplitude_ratio = data["profile"]["amplitude_ratio"],
                              theta           = data["profile"]["theta"],
                              ell             = data["profile"]["ell"])
        
        self._derived_properties["psfdata"] = data
        
    # -------- #
    #  SETTER  #
    # -------- #
    def set_stddev_ref(self, value):
        """ """
        self._properties["stddev_ref"] = value

    # -------- #
    # GETTER   #
    # -------- #
    def get_psf(self, x, y, lbda):
        """ """
        return binormal_profile(x, y, **self.get_psf_param(lbda))
        
        
    def get_stddev(self, lbda, rho=-1/5.):
        """ """
        return self.stddev_ref * (lbda / self.adr.lbdaref)**(rho)
    # ================== #
    #  Properties        #
    # ================== #
    @property
    def stddev_ref(self):
        """ """
        return self._properties["stddev_ref"]
    
###########################
#                         #
#  Force PSF 3D           #
#                         #
###########################

class ForcePSF( BaseObject ):
    """ """
    PROPERTIES = ["cube","psfmodel"]
    DERIVED_PROPERTIES = ["cubemodel","spec_source","spec_bkgd"]
    # ================== #
    #   Methods          #
    # ================== #
    def __init__(self, cube, psfmodel):
        """ """
        self.set_cube(cube)
        self.set_psfmodel(psfmodel)

    # ------------ #
    #  GETTER      #
    # ------------ #
    
    # ------------ #
    #  SETTER      #
    # ------------ #
    def set_cube(self, cube):
        """ set to this instance a point source 3d cube """
        self._properties["cube"] = cube
        
    def set_psfmodel(self, psfmodel):
        """ attach to this instance the PSF3D object containing the position and shape parameter of the PSF"""
        self._properties['psfmodel'] = psfmodel
        
    # ------------ #
    #  PLOTTER     #
    # ------------ #
    def show(self, lbda_range= [[4000,4500],[5500,6000],[7000,7500]],
                 fig=None, 
                 show=True, savefile=None):
        """ """
        import matplotlib.pyplot as mpl
        if fig is None:
            fig  = mpl.figure(figsize=[7,8])
        
        axspec = fig.add_axes([0.1,0.71,0.8,0.25])
        axspec.set_xlabel("Wavelength [AA]", fontsize="large")
        axspec.set_ylabel("Flux", fontsize="large")
        axspec.set_title(self.cube.filename.split("/")[-1], fontsize="small")
        
        # - The Spectrum
        pl = self.spec_source.show(ax=axspec, label="ForcePSF spectrum", show=False)
        _  = self.spec_bkgd.show(ax=axspec, color="C1", label="background")
        axspec.legend(loc="best", fontsize="small")

        # - The cubes
        width, span = 0.24, 0.04
        height = width  /1.5
        for i, lbda in enumerate(lbda_range):
    
            axc = fig.add_axes([0.1+(width+span)*i, 0.03+(height+span)*2, width, height])
            axm = fig.add_axes([0.1+(width+span)*i, 0.03+(height+span)*1, width, height])
            axr = fig.add_axes([0.1+(width+span)*i, 0.03+(height+span)*0, width, height])
            [ax_.set_yticks([]) for ax_ in [axc,axm,axr]]
            [ax_.set_xticks([]) for ax_ in [axc,axm,axr]]
            if i==0:
                axc.set_ylabel("Data")
                axm.set_ylabel("Model")
                axr.set_ylabel("Residual")
        
            axc.set_title("lbda: [%d, %d]"%(lbda[0],lbda[1]), fontsize="small")
            # Data
            slicepf = self.cube.get_slice(lbda[0],lbda[1], slice_object=True)
            slicepf.show(ax=axc, show_colorbar=False, vmin="2", vmax="98")
            # Model
            model  = self.cubemodel.get_slice(lbda[0],lbda[1], slice_object=True)
            model.show(ax=axm, show_colorbar=False, vmin="2", vmax="98")
            # Res
            res  = self.cuberes.get_slice(lbda[0],lbda[1], slice_object=True)
            res.show(ax=axr, show_colorbar=False, vmin="2", vmax="98")

        if savefile is not None:
            fig.savefig(savefile)
        if show:
            fig.show()
            
    # ------------ #
    #  FITTER      #
    # ------------ #
    def fit_forcepsf(self, store_cubemodel=True):
        """ Fit only the amplitude of each slides assuming the psf shape
        given by the psfmodel object.
        
        The fit is made using a simple chi2.
        """
        from pyifu import get_spectrum, get_cube
        x_,y_ = np.asarray(self.cube.index_to_xy(self.cube.indexes)).T
        flagok = ~np.isnan(x_*y_)
        x,y = x_[flagok],y_[flagok]

        flux,errors = [],[]
        bkgd,bkgderrors = [],[]
        for i,lbda_ in enumerate(self.cube.lbda):
            fitvalue = fit_forcepsf_slice(self.cube.data[i], self.psfmodel.get_psf(x, y, lbda_), variance=self.cube.variance[i])
            # recording
            flux.append(fitvalue["amplitude"])
            errors.append(fitvalue["amplitude.err"])
            bkgd.append(fitvalue["background"])
            bkgderrors.append(fitvalue["background.err"])

        self._derived_properties['spec_source']  =  get_spectrum(self.cube.lbda, np.asarray(flux), variance=np.asarray(errors)**2, header=self.cube.header)
        self._derived_properties['spec_bkgd'] =  get_spectrum(self.cube.lbda, np.asarray(bkgd), variance=np.asarray(bkgderrors)**2, header=self.cube.header)
        if store_cubemodel:
            datamodel = [self.psfmodel.get_psf(x, y, lbda_)*self.spec_source.data[i] + self.spec_bkgd.data[i]
                             for i, lbda_ in enumerate(self.cube.lbda)]
            self._derived_properties['cubemodel'] = get_cube(datamodel, header=None, variance=None, lbda=self.cube.lbda,
                                     spaxel_mapping=self.cube.spaxel_mapping, spaxel_vertices=self.cube.spaxel_vertices)
            
            
        return self.spec_source, self.spec_bkgd

    # ================== #
    #  Properties        #
    # ================== #
    @property
    def cube(self):
        """ Data Cube (x,y,lbda) """
        return self._properties["cube"]

    @property
    def psfmodel(self):
        """ PSF3D containing the shape and position parameters of the PSF. """
        return self._properties['psfmodel']
    # - Derived
    @property
    def spec_source(self):
        """ The fitted source spectrum """
        return self._derived_properties['spec_source']
    @property
    def spec_bkgd(self):
        """ The fitted background spectrum """
        return self._derived_properties['spec_bkgd']
    @property
    def cubemodel(self):
        """ The fitted PSF cube """
        return self._derived_properties['cubemodel']

    @property
    def cuberes(self):
        """ The fitted PSF cube """
        if self.cubemodel is None:
            return None
        return self.cube - self.cubemodel

# ========================= #
#                           #
#   PSF Method Fitter       #
#                           #
# ========================= #
# ---------------------- #
#  Fit PSF Shape         #
# ---------------------- #
class FitPSF( BaseObject ):
    """ Object made to extract the PSF shape parameters 
    
    How to fit properly fit the PSF?
    1) Fit independent slices using fit_slices()
    2) Fit the ADR using fit_adr_param()
    3) Fit the stddev using fit_stddev.
    
    Once all of that is done. Data will be avialable as `fitted_data`

    Remark: You might want to do a double pass on step 1 while
            forcing the constante parameters to there mean (clipped) values
            To do so, between 1 and 2 do:
            1-a) get the constant paramters using get_const_parameters()
            1-b) Fit the slices using fit_slices(key_guess=value, key_fixed=True)
                 where 'key' is the constant parameter and value its mean value.

    """
    
    PROPERTIES         = ["cube"]
    SIDE_PROPERTIES    = ["lbdas","profile"]
    DERIVED_PROPERTIES = ["slicefits","adrfitter",
                          "adrmodel","adr_parameters",
                          "stddev_ref"]
    
    def __init__(self, datacube):
        """ """
        self.set_cube(datacube)
        
    def set_cube(self, cube):
        """ set to this instance a point source 3d cube """
        self._properties["cube"] = cube

    # -------------- #
    #  I/O           #
    # -------------- #
    def write_fitted_data(self, datafile):
        """ """
        import json
        if not datafile.endswith(".json"):
            datafile +=".json"
        with open(datafile, 'w') as fp:
            json.dump(self.fitted_data, fp)
        
    # -------------- #
    #  GETTER        #
    # -------------- #
    def get_psf_param(self, index, key):
        """ returns the fitted PSF parameter for the index-th slice. """
        return np.asarray([self.slicefits[index]["fit"].fitvalues[k]  for k in [key,key+".err"]])

    def get_const_parameters(self):
        """ Return the mean values of the fitted profile parameters expecteed to be constant at all wavelengths 
        Returns
        -------
        dict
        """
        from scipy.stats import sigmaclip
        from astropy.stats import sigma_clip
        def get_const_value(k):
            k_base = np.asarray([self.get_psf_param(i, k) for i in self.slicefits.keys()]).T[0]
            # means fixed parameter
            return k_base[0] if len(np.unique(k_base))==1 else np.nanmean(sigma_clip(k_base, 2,2))
            
        theta            = get_const_value("theta")
        stddev_ratio     = get_const_value("stddev_ratio")
        ell              = get_const_value("ell")
        amplitude_ratio  = get_const_value("amplitude_ratio")
        
        
        return {"theta":theta,"stddev_ratio":stddev_ratio,"ell":ell, "amplitude_ratio":amplitude_ratio}
    
    def get_stddev_trend(self, stdref, lbda):
        """ """
        return stdref * (lbda / self.adrmodel.lbdaref)**(-1/6.)
        
    # --------------- #
    #  Fitter         #
    # --------------- #
    def fit_slices(self, lbda_range=[4500,7500], nbins=10, profile="BiNormalCont",**kwargs):
        """ Mother fitting Method.

        use this method to independently fit slices.

        Parameters
        ----------
        lbda_range: [float, float] -optional-
            wavelength range between which the slices will be made

        nbins: [int] -optional-
            the number of bins that will devide the lbda_range to make the slices.

        profile: [string] -optiona-
            The PSF profile used to fit the slices. 
            (so far only 'BiNormalCont' implemented

        **kwargs goes to each individual slice fitting. could give _guess, etc entry
        
        Returns
        -------
        Void
        """
        
        lbdas = np.linspace(lbda_range[0],lbda_range[1],nbins+1)
        self._side_properties["lbdas"]   = np.asarray([lbdas[:-1],lbdas[1:]]).T
        self._side_properties["profile"] = profile
        
        # - Da fit
        for i,l in enumerate(self.lbdas):
            self.slicefits[i] = {"fit": fit_slice( self.cube.get_slice(l[0],l[1], slice_object=True), psfmodel=profile, **kwargs),
                "lbda_range":l}
            
    def fit_stddev(self, indexes=None):
        """ Get the `stddev_ref` entry. 
        e.g. the zeropoint of the wavelength dependency of the `stddev` profile parameter.

        Parameters
        ----------
        indexes: [None, list] -optional-
            List of indexes (slices) to used.
            If None, all will be used.

        Returns
        -------
        float
        """
        from scipy.optimize import fmin
        if indexes is None:
            indexes = range(len(self.lbdas))
            lbdas   = np.mean(self.lbdas, axis=1)
        else:
            lbdas   = np.mean(self.lbdas[indexes], axis=1)
        
        stddev, estddev = np.asarray([self.get_psf_param(i, "stddev") for i in indexes]).T

        def _fmin_(scale_):
            return np.sum(np.sqrt((stddev-self.get_stddev_trend(scale_, lbdas))**2/estddev**2))
        
        self._derived_properties["stddev_ref"] = fmin(_fmin_, np.median(stddev), disp=0)[0]
        return self.stddev_ref    
        
    def fit_adr_param(self, unit_guess=0.5, parangle_guess=70, indexes=None, **kwargs):
        """ Fir the ADR and set the `adrmodel` attribute.

        Parameters
        ----------
        
        """
        from pysedm.utils import adrfit
        
        
        if self.cube.adr is None:
            self.cube.load_adr()

        # - Position Information
        if indexes is None:
            indexes = range(len(self.lbdas))
            lbdas   = np.mean(self.lbdas, axis=1)
        else:
            lbdas   = np.mean(self.lbdas[indexes], axis=1)
        
        xmean,xmeanerr = np.asarray([self.get_psf_param(i, "xcentroid") for i in indexes ]).T
        ymean,ymeanerr = np.asarray([self.get_psf_param(i, "ycentroid") for i in indexes ]).T
        
        # - Load ADRFitter
        self._derived_properties["adrfitter"] = adrfit.ADRFitter(self.cube.adr.copy(), base_parangle=0, unit=IFU_SCALE_UNIT)
        self.adrfitter.set_data(lbdas, xmean, ymean, xmeanerr, ymeanerr)
        
        self.adrfitter.fit(airmass_guess=self.cube.header["AIRMASS"], airmass_boundaries=[1,self.cube.header["AIRMASS"]*1.5],
                            xref_guess= np.mean(xmean), yref_guess= np.mean(ymean),
                            parangle_guess=(self.cube.header["TEL_PA"]+parangle_guess)%360, **kwargs)
        
        self._derived_properties["adr_parameters"] = self.adrfitter.fitvalues
        self.adr_parameters["header_parangle"]     = self.cube.header["TEL_PA"]
        self.adr_parameters["lbdaref"]             = self.adrfitter.model.lbdaref
        self.adr_parameters["unit"]                = IFU_SCALE_UNIT
        
        self._derived_properties["adrmodel"] = adr.ADR()
        self.adrmodel.set(lbdaref    = self.adr_parameters["lbdaref"],
                          parangle   = self.adr_parameters["parangle"],
                          pressure      = self.cube.adr.pressure,
                          relathumidity = self.cube.adr.relathumidity,
                          temperature   = self.cube.adr.temperature,
                          airmass       = self.adr_parameters['airmass'])
        return self.adr_parameters
    
    # --------------- #
    #  PLOTTING       #
    # --------------- #
    def show(self, params=["position","theta","stddev","stddev_ratio","ell"],
                 axes=None, set_labels=True, show=True):
        """ """
        import matplotlib.pyplot as mpl
        
        nparams = len(params)
        lbdas   = np.mean(self.lbdas, axis=1)
        
        # - Axes
        if axes is not None:
            if len(axes) != nparams:
                raise ValueError("axes and params do not have the same size.")
            fig = axes[0].figure
        else:
            fig = mpl.figure(figsize=[2*len(params),2.5])
            axes = [fig.add_subplot(1,nparams, 1+i) for i in range(nparams)]
        
        # - Position
        if "position" in params:
            axpos = axes[0]
            
            xmean,xmeanerr = np.asarray([self.get_psf_param(i, "xcentroid") for i,l in enumerate(lbdas)]).T
            ymean,ymeanerr = np.asarray([self.get_psf_param(i, "ycentroid") for i,l in enumerate(lbdas)]).T
            
            axpos.scatter(xmean, ymean, c=lbdas, zorder=3)
            axpos.errorscatter(xmean, ymean, xmeanerr, ymeanerr, zorder=2)
            if set_labels:
                axpos.set_xlabel("xcentroid",fontsize="large")
                axpos.set_ylabel("ycentroid",fontsize="large")
                axpos.set_title("position",fontsize="large")
            params = [p for p in params if "position" != p]
            
        # - Keys
        def show_fitvalue(ax, key):
            v, dv = np.asarray([self.get_psf_param(i, key) for i,l in enumerate(lbdas)]).T
            ax.scatter(lbdas, v, c=lbdas, zorder=3)
            ax.errorscatter(lbdas, v, dy=dv, zorder=2)
            if set_labels:
                ax.set_xlabel("mean wavelength")
                ax.set_title(key)

        for i,k in enumerate(params):
            show_fitvalue(axes[i+1], k)
            
        if show:
            fig.show()

    # - Show Fit results
    def show_stddev_fit(self, ax=None, show=True, set_labels=True, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[6,4])
            ax  = fig.add_subplot(111)
        else:
            fig = ax.figure

        lbdas          = np.mean(self.lbdas, axis=1)
        v, dv = np.asarray([self.get_psf_param(i, "stddev") for i,l in enumerate(lbdas)]).T
        ax.scatter(lbdas, v, c=lbdas, zorder=3)
        ax.errorscatter(lbdas, v, dy=dv, zorder=2)
        
        lbdas_model = np.linspace(np.nanmin(lbdas)-10,np.nanmax(lbdas)+10,len(lbdas)*100)
        ax.plot(lbdas_model, self.get_stddev_trend(self.stddev_ref, lbdas_model),
                    scalex=False, scaley=False, **kwargs)
        
        if set_labels:
            ax.set_xlabel("wavelength [AA]")
            ax.set_title("stddev")
            
        if show:
            fig.show()

    def show_adr_fit(self, ax=None, savefile=None, show=True, cmap=None, show_colorbar=True,
                         clabel='Wavelength [A]', refsedmcube=None, **kwargs):
        """ """
        return self.adrfitter.show(ax=ax, savefile=savefile, show=show, cmap=cmap,
                                       show_colorbar=show_colorbar,
                         clabel=clabel, refsedmcube=refsedmcube, **kwargs)

        
    # ================== #
    #   Properties       #
    # ================== #
    @property
    def fitted_data(self):
        """ the basic information for the 3D PSF model """
        # - profile
        dict_profile = self.get_const_parameters()
        dict_profile["stddev_ref"] = self.stddev_ref
        dict_profile["name"] = self.profile
        
        # - adr
        adr_profile = self.adrmodel.data.copy()
        adr_profile["xref"],adr_profile["yref"] = [self.adrfitter.fitvalues[k] for k in ["xref","yref"]]
        adr_profile["unit"] = self.adrfitter.fitvalues["unit"]
        adr_profile["unit"] = self.adrfitter.fitvalues["unit"]
        adr_profile["parangle_ref"] = self.adrfitter.fitvalues["header_parangle"]

        return {"adr":adr_profile, "profile":dict_profile}
    
    # - Properties
    @property
    def cube(self):
        """ Data Cube (x,y,lbda) """
        return self._properties["cube"]
    
    # - Side Properties
    @property
    def profile(self):
        """ Which profile has been used """
        return self._side_properties["profile"]
    
    @property
    def lbdas(self):
        """ range of wavelengths use to define the slices """
        return self._side_properties["lbdas"]
    
    # - Derived Properties
    @property
    def slicefits(self):
        """ Slice fitters """
        if self._derived_properties["slicefits"] is None:
            self._derived_properties["slicefits"] = {}
        return self._derived_properties["slicefits"]

    # - STDDEV
    @property
    def stddev_ref(self):
        """ The fitted stddev refence to be used with get_stddev_trend() """
        return self._derived_properties["stddev_ref"]
    # - ADR
    @property
    def adrfitter(self):
        """ """
        return self._derived_properties["adrfitter"]
    
    @property
    def adrmodel(self):
        """ """
        return self._derived_properties["adrmodel"]
    
    @property
    def adr_parameters(self):
        """ """
        return self._derived_properties["adr_parameters"]

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



###########################
#                         #
#   Model                 #
#                         #
###########################
def read_psfmodel(psfmodel):
    """ """
    
    if "BiNormalCont" in psfmodel:
        return BiNormalCont()
    else:
        raise ValueError("Only the 'BiNormalCont' psfmodel has been implemented")

# -------------------- #
#  Slice PSF Fitter    #
# -------------------- #
class SlicePSF( PSFFitter ):
    """ """
    # =================== #
    #   Methods           #
    # =================== #
    def __init__(self, slice_,
                     fitbuffer=None,fit_area=None,
                     psfmodel="BiNormalCont",
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
                 centroid_prop={}, logscale=True,
                 vmin="2", vmax="98", **kwargs):
        """ """
        import matplotlib.pyplot            as mpl
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

        [ax_.set_yticklabels([]) for ax_ in fig.axes[1:]]
        
        fig.figout(savefile=savefile, show=show)
        
    # =================== #
    #  Properties         #
    # =================== #
    @property
    def slice(self):
        """ pyifu slice """
        return self._spaxelhandler

# -------------------- #
#  Slice PSF Fitter    #
# -------------------- #
class _PSFSliceModel_( BaseModel ):
    """ Virtual PSFSlice Model Class. You need to define 
    - get_profile 
    - get_background 
    """
    PROFILE_PARAMETERS    = [] # TO BE DEFINED
    BACKGROUND_PARAMETERS = [] # TO BE DEFINED
    
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        # - Profile
        cls.FREEPARAMETERS     = list(cls.PROFILE_PARAMETERS)+list(cls.BACKGROUND_PARAMETERS)
        return super( _PSFSliceModel_, cls).__new__(cls)

    # ================= #
    #    Method         #
    # ================= #
    # ---------- #
    #  SETTER    #
    # ---------- #
    def setup(self, parameters):
        """ """
        self.param_profile    = {k:v for k,v in zip( self.PROFILE_PARAMETERS, parameters[:len(self.PROFILE_PARAMETERS)] )}
        self.param_background = {k:v for k,v in zip( self.BACKGROUND_PARAMETERS, parameters[len(self.PROFILE_PARAMETERS):] )} 
        
    # ---------- #
    #  GETTER    #
    # ---------- #
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

    # - To Be Defined
    def get_profile(self, x, y):
        """ The profile at the given positions """
        raise NotImplementedError("You must define the get_profile")
    
    def get_background(self, x, y):
        """ The background at the given positions """
        raise NotImplementedError("You must define the get_background")

    
# -------------------- #
#  Actual Model        #
# -------------------- #
    
class BiNormalCont( _PSFSliceModel_ ):
    """ """
    PROFILE_PARAMETERS = ["amplitude",
                          "stddev", "stddev_ratio", "amplitude_ratio",
                          "theta", "ell",
                          "xcentroid", "ycentroid"]
    
    BACKGROUND_PARAMETERS = ["bkgd"]
    
    # ================== #
    #  Guess             #
    # ================== #
    def get_guesses(self, x, y, data):
        """ return a dictionary containing simple best guesses """
        flagok     = ~np.isnan(x*y*data)
        x          = x[flagok]
        y          = y[flagok]
        data       = data[flagok]
        
        ampl       = np.nanmax(data)
        argmaxes   = np.argwhere(data>np.percentile(data,95)).flatten()
        x0, stdx   = np.nanmean(x[argmaxes]), np.nanstd(x[argmaxes])
        y0, stdy   = np.nanmean(y[argmaxes]), np.nanstd(y[argmaxes])
        std_mean   = np.mean([stdx,stdy]) / 2
        ell        = 1 - np.nanmin([stdx,stdy])/np.nanmax([stdx,stdy])
        theta      = 0 if stdx>stdy else np.pi/2.
        
        self._guess = dict(amplitude_guess=ampl * 5,
                           amplitude_boundaries= [ampl/100, ampl*100],
                           xcentroid_guess=x0, xcentroid_boundaries=[x0-3,x0+3],
                           ycentroid_guess=y0, ycentroid_boundaries=[y0-3,y0+3],
                           # - STD
                           stddev_guess = std_mean,
                           stddev_boundaries=[0.8,std_mean*3],
                           # - background
                           bkgd_guess=np.percentile(data,10),
                           # Converges faster by allowing degenerated param...
                           theta_guess=theta, theta_boundaries=[-0.05,1*np.pi],
                           ell_guess = ell, ell_boundaries= [-0.1, 0.8],
                           # amplitude ratio
                           amplitude_ratio_guess = 3.5,
                           amplitude_ratio_fixed = True,
                           amplitude_ratio_boundaries = [3,4],
                           stddev_ratio_guess = 2.,
                           stddev_ratio_boundaries = [1.1,3.5],
                            )
        return self._guess
    
    # ================== #
    #  Model             #
    # ================== #
    def get_profile(self, x, y):
        """ """
        amplitude = self.param_profile.pop("amplitude")
        return amplitude * binormal_profile(x, y, **self.param_profile)

    def get_background(self,x,y):
        """ The background at the given positions """
        return self.param_background["bkgd"]
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def centroid_guess(self):
        """ """
        return self._guess["xcentroid_guess"], self._guess["ycentroid_guess"]
    
    @property
    def centroid(self):
        """ """
        return self.fitvalues["xcentroid"], self.fitvalues["ycentroid"]
    
    @property
    def fwhm(self):
        """ """
        return "To Be Done"
    
