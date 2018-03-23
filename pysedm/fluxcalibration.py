#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from propobject import BaseObject

""" Library to estimate the flux response based on standard star spectra """

AVOIDANCE_AREA = {"telluric":[[7450,7750],[6850,7050]],
                 "absorption":[[6400,6650],[4810,4910],
                                [4300,4350],[4030,4130],[3900,3980]]}

POLYDEGREE = 20
def get_fluxcalibrator(stdspectrum, polydegree=POLYDEGREE, fullout=False):
    """ fit a smooth polynome through the ratio between observed standard and their expected flux (calspec) """
    from pyifu import get_spectrum
    fl = FluxCalibrator()
    fl.set_std_spectrum(stdspectrum)
    if fullout:
        return get_spectrum(stdspectrum.lbda, fl.fit_inverse_sensitivity(polydegree=polydegree), header=stdspectrum.header), fl
    return get_spectrum(stdspectrum.lbda, fl.fit_inverse_sensitivity(polydegree=polydegree), header=stdspectrum.header)


class FluxCalibrator( BaseObject ):
    """ """
    PROPERTIES = ["spectrum", "calspec"]
    DERIVED_PROPERTIES = ['calspec_spectrum']
    # ==================== #
    #  Method              #
    # ==================== #
    # -------- #
    # GETTER   #
    # -------- #
    def fit_inverse_sensitivity(self, polydegree=POLYDEGREE, maskout_avoidance_area=True, **kwargs):
        """ **kwargs goes to polyfit.fit(**kwargs)"""
        from modefit import get_polyfit
        
        flag_kept = self.get_avoidance_mask(which="both") if maskout_avoidance_area \
          else np.ones(len(self.spectrum.lbda))

        datacal = self.calspec_spectrum.data / self.spectrum.data
        norm = datacal.mean()
        
        calfit = get_polyfit(self.spectrum.lbda[flag_kept], datacal[flag_kept]/norm,
                                 np.ones(len(datacal[flag_kept]))/10,
                                 polydegree,legendre=True)
        calfit.fit(**kwargs)
        return calfit.model.get_model(self.spectrum.lbda) * norm

    def get_avoidance_mask(self, which="both"):
        """ boolean mask.
        Parameters
        ----------
        which: [string] -optional-
            - both: False if wavelength in any Telluric or Absorption intervals
            - telluric: False if wavelength in Telluric intervals
            - absorption: False if wavelength in Absorption intervals
        """
        flagin_tell = np.any([(self.spectrum.lbda>l[0]) * (self.spectrum.lbda<l[1]) for l in AVOIDANCE_AREA["telluric"]], axis=0)
        flagin_abs = np.any([(self.spectrum.lbda>l[0]) * (self.spectrum.lbda<l[1]) for l in AVOIDANCE_AREA["absorption"]], axis=0)
        if which in ["both","Both"]:
            return ~(flagin_tell+flagin_abs)
        if which in ["telluric","Telluric", "tell"]:
            return ~flagin_tell
        if which in ["absorption","Absorption", "Abs"]:
            return ~flagin_abs
        raise ValueError("Unknown masking request %s (use which/telluric/absorption)"%which)

    # -------- #
    #  SETTER  #
    # -------- #
    def set_std_spectrum(self, spectrum, load_reference=True):
        """ """
        self._properties['spectrum'] = spectrum
        if load_reference:
            self.load_calspec_reference()

    def load_calspec_reference(self):
        """ """
        try:
            import pycalspec
        except ImportError:
            raise ImportError("You need pycalspec. Please pip install pycalspec (or grab it on github)")
        
        self._properties['calspec'] = pycalspec.std_spectrum(self.objectname)
        self._derived_properties['calspec_spectrum'] = self._calspec.reshape(self.spectrum.lbda,"linear")

    # --------- #
    #  PLOTTER  #
    # --------- #
    def show(self, savefile=None, show=True, fluxcal=None):
        """ """
        import matplotlib.pyplot as mpl
        from pyifu import get_spectrum

        if fluxcal is None:
            fluxcal= self.fit_inverse_sensitivity(polydegree=POLYDEGREE)

                
        fig = mpl.figure()
        ax   = fig.add_axes([0.15,0.62,0.8,0.3])
        ax2  = fig.add_axes([0.15,0.1,0.8,0.5])
        axt  = ax.twinx()
        self.spectrum.show(ax=ax, show=False)
        self.calspec_spectrum.show(ax=axt, color="C1", show=False)
        for avoidance,c in zip(AVOIDANCE_AREA.values(), [mpl.cm.Blues(0.6),"0.5"]) :
            for l in avoidance: ax.axvspan(l[0],l[1], color=c, alpha=0.2)
        
        spec = get_spectrum(self.spectrum.lbda, self.calspec_spectrum.data / self.spectrum.data )

        speccal = get_spectrum(self.spectrum.lbda, fluxcal )
        spec.show(ax=ax2, color="0.7", lw=2, label="Calspec / SEDM", show=False)
        speccal.show(ax=ax2, color="C2", lw=1.5, label="Used Polynome ", show=False)
        ax2.set_yscale("log")

        ax.set_xticks([])
        ax.set_yticks([])
        axt.set_yticks([])
        ax2.set_xlabel(r"Wavelength [$\AA$]", fontsize="large")
        ax2.set_ylabel("Inversed Sensitivity", fontsize="large")
        ax2.legend(loc="best")
        if savefile is not None:
            fig.savefig(savefile)
        if show:
            fig.show()
            
    # ==================== #
    #  Properties          #
    # ==================== #
    @property
    def spectrum(self):
        """ Standard star spectrum """
        return self._properties['spectrum']
    
    @property
    def objectname(self):
        """ """
        return self.spectrum.header['OBJECT'].replace("STD-","")

    # - CalSpec
    @property
    def _calspec(self):
        """ The CalSpec Spectrum as given by calspec"""
        return self._properties['calspec']
    
    @property
    def calspec_spectrum(self):
        """ CalSpec Spectrum at the `spectrum` wavelength interpolation """
        return self._derived_properties['calspec_spectrum']