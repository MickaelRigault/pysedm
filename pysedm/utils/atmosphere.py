#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Simple library containing the Atmosphere related object. """

import warnings
import numpy                as np

from pyifu.spectroscopy import Spectrum



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
    
##########################
#                        #
#  Telluric Absorption   #
#                        #
##########################
TELLURIC_REGIONS = {
    "O2": [[6270.2,6331.7],[6862.1,6964.6],[7585.8,7703.0]],
    "H2O":[[5880.5,5959.6],[6466.5,6522.2],[6979.3,7067.2],[7143.3,7398.2],[7887.6,8045.8],[8083.9,8420.8],[8916.0,9929.8]]
    }

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
        
        
    def get_telluric_absorption(self, airmass, coefo2=1, coefh2o=1, rho_o2=0.58, rho_h2o=0.4,
                                    filter=None):
        """ """
        flux = self.data_o2 ** (airmass**rho_o2 * coefo2) * self.data_h2o ** (airmass**rho_h2o * coefh2o)
        return get_telluric_spectrum(self.lbda, flux, variance=None, header=self.header, filter=filter)
    
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
