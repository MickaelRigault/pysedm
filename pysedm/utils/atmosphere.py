#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Simple library containing the Atmosphere related object. """

import warnings
import numpy                as np

from pyifu.spectroscopy import Spectrum
from scipy.special import orthogonal


##########################
#                        #
#   Palomar Extinction   #
#                        #
##########################
class ExtinctionSpectrum( Spectrum ):
    """ """
    PROPERTIES = ["interpolation"]
    
    def get_atm_extinction(self, lbda, airmass):
        """ """
        return 10**(self._interpolation(lbda) * airmass/2.5)
    
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def _interpolation(self, kind="cubic"):
        """ """
        if self._properties['interpolation'] is None:
            from scipy.interpolate import interp1d
            self._properties['interpolation'] = \
              interp1d(self.lbda, self.data, kind=kind)

        return self._properties['interpolation']



def fit_telluric(spec):
    """ """
    print("not done yet")
    
##########################
#                        #
#  Telluric Absorption   #
#                        #
##########################
TELLURIC_REGIONS = {
    "O2": [[6270.2,6331.7],[6862.1,6964.6],[7585.8,7703.0]],
    "H2O":[[5880.5,5959.6],[6466.5,6522.2],[6979.3,7067.2],[7143.3,7398.2],[7887.6,8045.8],[8083.9,8420.8],[8916.0,9929.8]]
    }

def load_telluric_spectrum(filename, filter=None):
    """ """
    spec_ = TelluricLines(None)
    spec_.load(filename)
    if filter is not None:
        return spec_.filter(filter)
    return spec_

def get_telluric_spectrum(lbda, flux, variance=None, header=None,
                          filter=None):
    """ """
    spec_ = TelluricLines(None)
    spec_.create(lbda=lbda, data=flux, variance=variance, header=header)
    if filter is not None:
        return spec_.filter(filter)
    return spec_
    
class TelluricLines( Spectrum ):
    """ """
    DERIVED_PROPERTIES = ["dataO2","dataH2O"]
    def set_data(self, data, variance=None, lbda=None, logwave=None):
        """ Set the spectral data 

        Parameters
        ----------
        data: [array]
            The array containing the data
        
        variance: [array] -optional-
            The variance associated to the data. 
            This must have the same shape as data
           
        lbda: [array] -optional-
            Provide the wavelength array associated with the data.
            This is not mendatory if the header contains this information
            (step, size and start values). 
            N.B: You can always use set_lbda() later on.
            
        logwave: [None / bool] -optional-
            If the wavelength given in log of wavelength. 
            If you known set True (= given in log) or False (= given in angstrom)
            If let to None, this will test if the first wavelength is smaller or 
            higher than a default number (50).

        Returns
        -------
        Void
        """
        out = super(TelluricLines, self).set_data(data, variance=variance, lbda=lbda, logwave=logwave)
        self._derived_properties["data"] = self.rawdata.copy()
        self._derived_properties["data"][self.rawdata<0] = 0
        self._derived_properties["dataO2"] = None
        self._derived_properties["dataH2O"] = None
    
    def show(self, ax=None, show_regions=True, **kwargs ):
        import matplotlib.pyplot    as mpl
        pl = super(TelluricLines, self).show(ax=ax, savefile=None, **kwargs)
        ax,fig = pl["ax"], pl["fig"]

        for o2r in TELLURIC_REGIONS["O2"]:
            ax.axvspan(o2r[0],o2r[1], color=mpl.cm.Blues(0.3,0.3))
            
        for h2or in TELLURIC_REGIONS["H2O"]:
            ax.axvspan(h2or[0],h2or[1], color=mpl.cm.binary(0.3,0.2))
            
        return pl
        

    def _get_telluric_data_(self, airmass, coefo2=1, coefh2o=1, rho_o2=0.58, rho_h2o=0.4):
        """ """
        return self.data_o2 ** (airmass**rho_o2 * coefo2) * self.data_h2o ** (airmass**rho_h2o * coefh2o)

    def get_telluric_absorption(self, airmass, coefo2=1, coefh2o=1, rho_o2=0.58, rho_h2o=0.4,
                                    filter=None):
        """ """
        flux = self._get_telluric_data_(airmass, coefo2=coefo2, coefh2o=coefh2o,
                                            rho_o2=rho_o2, rho_h2o=rho_h2o)
        
        return get_telluric_spectrum(self.lbda, flux, variance=None,
                                         header=self.header, filter=filter)
    
    def get_telluric_throughput(self, airmass, coefo2=1, coefh2o=1, rho_o2=0.58, rho_h2o=0.4,
                                    filter=None):
        """ """
        flux = self._get_telluric_data_(airmass, coefo2=coefo2, coefh2o=coefh2o,
                                            rho_o2=rho_o2, rho_h2o=rho_h2o)
        return get_telluric_spectrum(self.lbda, (1-flux), variance=None,
                                         header=self.header, filter=filter)

    
    # =============== #
    #   Property      #
    # =============== #
    # - derived
    @property
    def data_o2(self):
        """ O2 only regions of the absoption spectrum. The rest is set to 1."""
        if self._derived_properties["dataO2"] is None:
            self._derived_properties["dataO2"] = self.data.copy()
            self._derived_properties["dataO2"][~self.flag_ino2] = 1
        return self._derived_properties["dataO2"]

    @property
    def data_h2o(self):
        """ H2O only regions of the absoption spectrum. The rest is set to 1."""
        if self._derived_properties["dataH2O"] is None:
            self._derived_properties["dataH2O"] = self.data.copy()
            self._derived_properties["dataH2O"][~self.flag_inh2o] = 1
        return self._derived_properties["dataH2O"]
        
    # - on the flight
    @property
    def flag_ino2(self):
        """ boolean array returning True for wavelengthes within the O2 lines (see TELLURIC_REGIONS) """
        return np.asarray(np.sum([ (self.lbda>=l[0]) * (self.lbda<l[1]) for l in TELLURIC_REGIONS["O2"]], axis=0), dtype="bool")

    @property
    def flag_inh2o(self):
        """ boolean array returning True for wavelengthes within the H2O lines (see TELLURIC_REGIONS) """
        return np.asarray(np.sum([ (self.lbda>=l[0]) * (self.lbda<l[1]) for l in TELLURIC_REGIONS["H2O"]], axis=0), dtype="bool")






######################################
#                                    #
# Fitting Atmosphere and Tellurics   #
#                                    #
######################################
import modefit
from modefit.basics import PolyModel, PolynomeFit

class TelluricPolynomeFit( PolynomeFit ):
    """ """
    def __init__(self, x, y, dy, degree, tellspectrum, maskin=None,
                 names=None, legendre=True):
        """ """
        self.__build__()
        if maskin is None:
            self.set_data(x, y, dy)
        else:
            self.set_data(x[maskin], y[maskin], dy[maskin])
        self.set_model(telluric_and_polynomial_model(degree, tellspectrum),
                           use_legendre=legendre)
        self.model.set_xsource(x)
        self.model.set_maskin(maskin)
        
        
    def _display_data_(self, ax, ecolor="0.3", **prop):
        """ """
        from modefit.utils import specplot
        return ax.specplot(self.xdata,self.data, var=self.errors**2,
                        bandprop={"color":ecolor},**prop)
        
    def show(self,savefile=None, show=True, ax=None, 
             show_model=True, modelcolor='k', modellw=2,
             show_details=True, contcolor="C1",tellcolor="0.7",
                 mcmc=False, nsample=100, ecolor='0.3',
                 mcmccolor=None,  **kwargs):
        """ """
        import matplotlib.pyplot as mpl 
        from modefit.utils import figout, errorscatter, kwargs_update

        pkwargs = kwargs_update(dict(ls="-", marker="None", zorder=5),**kwargs)
        if ax is None:
            fig = mpl.figure(figsize=[7,4])
            ax  = fig.add_axes([0.12,0.15,0.78,0.75])
        else:
            fig = ax.figure
            
        
        # Data
        self._display_data_(ax, ecolor=ecolor, label="Data", **pkwargs)
        
        # Model
        if show_model:
            model_to_show = self.model.get_model()
            model = ax.plot(self.model.xsource, model_to_show, ls="-", lw=modellw,
                            color=modelcolor, zorder=np.max([pkwargs["zorder"]+1,2]),
                           label="Full Model" if show_details else "Model")
        
        # -- Add telluric
        if show_details:
            ax.plot(self.model.xsource,
                            self.model._get_continuum_(),
                            ls="-", lw=modellw,
                            color=contcolor, scalex=False, scaley=False, 
                            zorder=1, label="calibration response")
                               
            ax.fill_between(self.model.xsource, -self.model.get_telluric_model(), 
                                  facecolor=tellcolor, alpha=0.5)
            ax.plot(self.model.xsource, -self.model.get_telluric_model(), 
                          color=tellcolor, label="Telluric absorption")

        ax.legend(loc="best", fontsize="medium")
        ax.set_ylabel("Flux", fontsize="large")
        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize="large")
        fig.figout(savefile=savefile, show=show)
        return fig


# ==================== #
#                      #
#   Telluric Polynome  #
#                      #
# ==================== #
def telluric_and_polynomial_model(degree, tellspectrum):
    """ 
    Build a model with a continuum that has a `degree` polynomial continuum
    and `ngauss` on top of it.
    
    Returns
    -------
    Child of NormPolyModel
    """
    class N_TelluricPolyModel( TelluricPolyModel ):
        DEGREE = degree
        
    return N_TelluricPolyModel(tellspectrum)


class TelluricPolyModel( PolyModel ):
    DEGREE = 0
    TELL_FREEPARAMETERS = ["airmass","coefo2", "coefh2o", "rho_o2", "rho_h2o","filter", "amplitude"]

    PROPERTIES = ["tellparameters", "tellspectrum","maskin"]
    # parameter inputs
    airmass_guess = 1.1
    airmass_boundaries= [1,3]
    
    filter_guess = 15
    filter_boundaries = [12,18]
    
    amplitude_guess = -1 # negative because absorption
    amplitude_boundaries= [None,0]
    
    coefo2_guess=1 
    coefo2_boundaries=[0.1,2]
    coefh2o_guess=1
    coefh2o_boundaries=[0.5,3] 
    rho_o2_guess=0.58 
    rho_o2_boundaries=[0.3, 3]
    rho_h2o_guess=0.4 
    rho_h2o_boundaries=[0.1,1]

    # Continuum
    a0_guess = 1
    
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        if not hasattr(cls,"FREEPARAMETERS"):
            cls.FREEPARAMETERS = ["a%d"%(i) for i in range(cls.DEGREE)]
        else:
            cls.FREEPARAMETERS += ["a%d"%(i) for i in range(cls.DEGREE)]
        cls.FREEPARAMETERS += [c for c in cls.TELL_FREEPARAMETERS]
        
        return super(PolyModel,cls).__new__(cls)

    def __init__(self, tellspectrum):
        """ """
        self._properties["tellspectrum"] = tellspectrum
        
    def set_maskin(self, maskin):
        """ """
        if maskin is None:
            self._properties["maskin"] = None
        else:
            self._properties["maskin"] = np.asarray(maskin, dtype="bool")
            
    def setup(self, parameters):
        """ read and parse the parameters """
        # good strategy to have 2 names to easily super() the continuum in get_model
        self._properties["parameters"]     = np.asarray(parameters[:self.DEGREE])
        self._properties["tellparameters"] = np.asarray(parameters[self.DEGREE:])

    def set_tellparameters(self, tellparameters):
        """ """
        if len(tellparameters) != len(self.TELL_FREEPARAMETERS):
            raise ValueError("%d parameter given for tellparameters, %d given"%(len(tellparameters), 
                                                                                len(self.TELL_FREEPARAMETERS)))
        self._properties["tellparameters"] = tellparameters
        
    def get_telluric_model(self, tellparam=None, lbda=None):
        """ """
        if tellparam is not None:
            self.set_tellparameters(tellparam)
            
        # Last tellparameter is the amplitude
        return self.tellparameters[-1]*self.tellspectrum.get_telluric_throughput(**{k:v for k,v in 
                                zip(self.TELL_FREEPARAMETERS[:-1], self.tellparameters[:-1])}).reshape(self.xsource if lbda is None else lbda,"linear").data

    def get_model(self, param=None):
        """ return the model for the given data.
        The modelization is based on legendre polynomes that expect x to be between -1 and 1.
        This will create a reshaped copy of x to scale it between -1 and 1 but
        if x is already as such, save time by setting reshapex to False

        Returns
        -------
        array (size of x)
        """
        if param is not None:
            self.setup(param)

        if self.maskin is not None:
            return (self._get_continuum_() + self.get_telluric_model())[self.maskin]
        return self._get_continuum_() + self.get_telluric_model()

    def _get_continuum_(self, x=None):
        """ """
        
        if x is not None:
            self.set_xsource(x)
            
        if self.use_legendre:            
            model = np.asarray([orthogonal.legendre(i)(self.xsource_scaled) for i in range(self.DEGREE)])
            return np.dot(model.T, self.parameters.T).T[self._xsource_extra_:-self._xsource_extra_]
        else:
            return np.dot(np.asarray([self.xfit**i for i in range(self.DEGREE)]).T, self.parameters.T).T

    @property
    def xsource_scaled(self):
        """ """
        if self._derived_properties["xsource_scaled"] is None and self.xsource is not None:
            self._derived_properties["xsource_scaled"] = np.linspace(-1, 1, len(self.xsource)+self._xsource_extra_*2)
        return self._derived_properties["xsource_scaled"]
    
    @property
    def _xsource_extra_(self):
        """ """
        return 30
    
    @property
    def normparameters(self):
        return self._properties["normparameters"]

    @property
    def tellparameters(self):
        return self._properties["tellparameters"]

    @property
    def tellspectrum(self):
        return self._properties["tellspectrum"]

    @property
    def maskin(self):
        return self._properties["maskin"]
