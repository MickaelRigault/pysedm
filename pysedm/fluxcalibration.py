#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from propobject import BaseObject
from pyifu.spectroscopy import Spectrum, get_spectrum
from .sedm import get_sedm_version
""" Library to estimate the flux response based on standard star spectra """

AVOIDANCE_AREA = {"telluric":[[7450,7750],[6850,7050]],
                 "absorption":[[6400,6650],[4810,4910],
                                [4300,4350],[4030,4130],[3900,3980]]}

POLYDEGREE = 40

def get_fluxcalibrator(stdspectrum, polydegree=POLYDEGREE, fullout=False, filter=None):
    """ fit a smooth polynome through the ratio between observed standard and their expected flux (calspec) """
    fl = FluxCalibrator()
    fl.set_std_spectrum(stdspectrum)
    return  fl.fit_inverse_sensitivity(polydegree=polydegree, filter=filter), fl

def show_fluxcalibrated_standard(stdspectrum, savefile=None):
    """ """
    import matplotlib.pyplot as mpl
    import pycalspec
    from astropy import units
    from . import io
    ### Data
    objectname = stdspectrum.header['OBJECT'].replace("STD-","").split()[0]
    #
    try:
        specref = pycalspec.std_spectrum(objectname).filter(5).reshape(stdspectrum.lbda,"linear")
    except IOError:
        print("WARNING: ref flux not found so no flux scale plot possible")
        return None
    specres = get_spectrum(stdspectrum.lbda, specref.data / stdspectrum.data )
    scale_ratio = specres.data.mean()
    specres.scale_by(scale_ratio)
    #
    try:
        dtime = io.filename_to_time( stdspectrum.filename) - io.filename_to_time( stdspectrum.header["CALSRC"] )
    except:
        dtime = None
    ###

    fig = mpl.figure(figsize=[6,4])

    ax   = fig.add_axes([0.13,0.43,0.8,0.5])
    axr  = fig.add_axes([0.13,0.13,0.8,0.28])

    stdspectrum.show(ax=ax, label="observation", show=False, yscalefill=True)
    specref.show(ax=ax, color="C1", label="calspec", show=False, yscalefill=True)

    axr.axhline(1, ls="--", color="0.5")
    axr.axhspan(0.9,1.1, color="0.5", alpha=0.1)
    axr.axhspan(0.8,1.2, color="0.5", alpha=0.1)
    # Residual
    specres.show(ax=axr, color="k", lw=1.5, show=False)

    ax.set_yscale("log")
    for l in AVOIDANCE_AREA["telluric"]: 
        ax.axvspan(l[0],l[1], color=mpl.cm.Blues(0.6), alpha=0.2)
        axr.axvspan(l[0],l[1], color=mpl.cm.Blues(0.6), alpha=0.2)
        
    axr.text(0.02,0.92, "scale ratio: %.1f"%scale_ratio, transform=axr.transAxes,
                va="top", ha="left", backgroundcolor= mpl.cm.binary(0.,0.8))

    ax.set_xlabel(["" for l in ax.get_xlabel()])
    ax.set_ylabel(r"Flux [erg s$^{-1}$ cm$^{-2}$ A$^{-1}$]")
    if dtime is not None:
        ax.set_title( "%s"%objectname+ r" | t$_{obs}$ - t$_{fluxcal}$ : %.2f hours"%((dtime.sec * units.second).to("hour")).value)
    else:
        ax.set_title( "%s"%objectname)

    axr.set_ylabel(r"Flux ratio")
    axr.set_xlabel(r"Wavelength [$\AA$]")

    ax.legend(loc="lower left")
    if savefile:
        fig.savefig(savefile)
    else:
        return fig


# =================== #
#  INTERNAL Tools     #
# =================== #
_TELL_HEADERKEY = "TELL"
def _paramkey_to_headerkey_( paramkey):
    """ converts the argument names from TelluricLines into Fits format header entry """
    return _TELL_HEADERKEY+paramkey.replace("coef","c").replace("rho_","r").replace("filter","fltr").replace("amplitude","AMP").upper()
def _headerkey_to_paramkey_( headerkey ):
    """ converts  Fits format header entry into argument names from TelluricLines into """
    return  headerkey.replace(_TELL_HEADERKEY+"C","coef").replace(_TELL_HEADERKEY+"R","rho_").replace(_TELL_HEADERKEY+"FLTR","filter").replace(_TELL_HEADERKEY+"AMP","amplitude").lower()

    
def get_fluxcal_spectrum(lbda, flux, variance=None, header=None, logwave=None):
    """ Create a Flux Calibrator spectrum from the given data
    
    Parameters
    ----------
    lbda: [array]
        wavelength of the spectrum

    flux: [array]
        flux of the spectrum
    
    variance: [array/None] -optional-
        variance associated to the flux.

    header: [fits header / None]
        fits header assoiated to the data
        
    logwave: [None / bool] -optional-
        If the wavelength given in log of wavelength. 
        If you known set True (= given in log) or False (= given in angstrom)
        If let to None, this will test if the first wavelength is smaller or 
        higher than a default number (50).


    Returns
    -------
    Spectrum
    """
    spec = FluxCalSpectrum(None)
    spec.create(data=flux, variance=variance, header=header, lbda=lbda, logwave=logwave)
    return spec


def load_fluxcal_spectrum(filename,**kwargs):
    """ """
    return FluxCalSpectrum(filename, **kwargs)

class FluxCalSpectrum( Spectrum ):
    """ """
    PROPERTIES      = ["tellspec","tellparam"]
    SIDE_PROPERTIES = ["refairmass", "ref_amplitude"]

    # ================= #
    #   Methods         #
    # ================= #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_header(self, header):
        """ 
        Attach a header. 
        If the given header is None, an empty header will be attached.

        If Telluric parameter information are found in the header. They are set.
        """
        super(FluxCalSpectrum, self).set_header(header)
        [self.set_telluric_parameters(**{_headerkey_to_paramkey_(k):v})
             for k,v in self.header.items() if k.startswith(_TELL_HEADERKEY)]
        
    def set_telluric_parameters(self, **kwargs):
        """ Telluric parameter could be:
        - coefo2, 
        - coefh2o, 
        - rho_o2, 
        - rho_h2o, 
        - filter 
        - amplitude """
        VALIDS =["coefo2", "coefh2o",
                 "rho_o2", "rho_h2o",
                "filter","amplitude"]
        
        for k,v in kwargs.items():
            if k not in VALIDS:
                raise ValueError("%s is not a valid telluric parameter"%k +"these are: "+ ", ".join(VALIDS))
            if k in ["amplitude"]:
                self._side_properties["ref_amplitude"] = v
            else:
                self.tellparam[k] = v

    def load_telluric_spectrum(self):
        """ """
        from .io import load_telluric_line
        self._properties['tellspec'] = load_telluric_line()
        
    # -------- #
    #  GETTER  #
    # -------- #
    def get_inversed_sensitivity(self, airmass, amplitude=None, **kwargs):
        """ """
        if self.is_old_format:
            return 1/self.data
        
        if amplitude is None: amplitude = self.ref_amplitude
        self.set_telluric_parameters(**kwargs)
        return self.data + self.get_telluric_throughput(airmass, amplitude=amplitude)
    
    def get_telluric_absorption(self, airmass, amplitude=None,**kwargs):
        """ """
        if self.is_old_format:
            return 0
        if amplitude is None: amplitude = self.ref_amplitude
        self.set_telluric_parameters(**kwargs)
        return amplitude*self.tellspec.get_telluric_absorption(airmass, **self.tellparam).reshape(self.lbda, "linear").data

    def get_telluric_throughput(self, airmass, amplitude=None, **kwargs):
        """ """
        if self.is_old_format:
            return 0
        if amplitude is None: amplitude = self.ref_amplitude
        self.set_telluric_parameters(**kwargs)
        return amplitude*self.tellspec.get_telluric_throughput(airmass, **self.tellparam).reshape(self.lbda, "linear").data

    
    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, ax=None, airmass=1, color="C0", fillcolor="0.7", fillalpha=0.7,
                 inversed=False, show=True, zorder=4, telllabel="_no_legend_", label=None,**kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[7,4])
            ax  = fig.add_axes([0.12,0.15,0.78,0.75])
        else:
            fig = ax.figure
        
        power= 1 if not inversed else -1
        if not self.is_old_format:
            ax.fill_between(self.lbda, self.data**power, self.get_inversed_sensitivity(airmass)**power, 
                            color=fillcolor, alpha=0.7,zorder=zorder-1, label=telllabel)
        ax.plot( self.lbda, self.get_inversed_sensitivity(airmass)**power, color=color, zorder=zorder,
                     label=label,**kwargs)
        if show:
            fig.show()
            
        return fig
        
            
    # ================= #
    #   Properties      #
    # ================= #
    def has_tellspec(self):
        """ """
        return self._properties['tellspec'] is not None
    
    @property
    def tellparam(self):
        """ """
        if self._properties["tellparam"] is None:
            self._properties["tellparam"] = {}
        return self._properties["tellparam"]
    
    @property
    def tellspec(self):
        """ """
        if not self.has_tellspec(): self.load_telluric_spectrum()
        return self._properties['tellspec']

    @property
    def ref_amplitude(self):
        """ amplitude of the telluric line in during the calibration """
        return self._side_properties["ref_amplitude"]

    @property
    def is_old_format(self):
        """ """
        return "TELLFLTR" not in self.header

    
class FluxCalibrator( BaseObject ):
    """ """
    PROPERTIES = ["spectrum", "calspec"]
    DERIVED_PROPERTIES = ['calspec_spectrum', "fluxcalspectrum"]
    # ==================== #
    #  Method              #
    # ==================== #
    # -------- #
    # GETTER   #
    # -------- #
    def fit_inverse_sensitivity(self, polydegree=POLYDEGREE, filter=4, masked_area=None,**kwargs):
        """ **kwargs goes to polyfit.fit(**kwargs)
        
        Parameters
        ----------
        filter: [None, float] -optional-
            if not None, the free_parameter 'filter' will be forced to the given value (hence not fitted)
                This is similar as doing: `filter_guess=XX, filter_fixed=True`
            if None, nothing will be made.
            If you only want to give initial guess and e.g. boundaries:
               - set filter to None: filter=None
               - provide the fit entry: filter_guess=XX, filter_boundaries=[YY,ZZ]
               
        """
        from .utils.atmosphere import TelluricPolynomeFit
        from .io import load_telluric_line
        from modefit import get_polyfit

        if masked_area is not None:
            maskin = self.get_avoidance_mask("absorption")
        else:
            maskin = None
            
        datacal = self.spectrum.data/self.calspec_spectrum.data
        norm = datacal.mean()
        datacal/=norm

        # Based on least sq
        #errcal = datacal * (np.sqrt(self.spectrum.variance)/self.spectrum.data)# + np.sqrt(self.calspec_spectrum.variance)/self.calspec_spectrum.data)
        errcal = datacal/100
        
        
        if filter is not None:
            kwargs["filter_guess"] = filter
            kwargs["filter_fixed"] = True
        else:
            kwargs["filter_guess"] = 18

        
        end_airmass = float(self.spectrum.header["ENDAIR"]) if "ENDAIR" in self.spectrum.header else self.spectrum.header["AIRMASS"]
        start_airmass = float(self.spectrum.header["AIRMASS"])
        airmass = np.mean([start_airmass, end_airmass])
        ##
        # Older procedure
        ##
        ##
        # New procedure
        ##
        # A: 3 Step procedure.
        # - Step 1 fit for the telluric region
        tell_range = (self.spectrum.lbda>6000) & (self.spectrum.lbda<8500)
        POLY_TELL  = 5
        self.tpoly_tell = TelluricPolynomeFit(self.spectrum.lbda[tell_range],
                                              datacal[tell_range],
                                              errcal[tell_range],
                                              POLY_TELL, load_telluric_line(),
                                                  maskin=None)
        self.tpoly_tell.fit(airmass_guess= airmass, airmass_fixed= True, airmass_boundaries=np.sort([start_airmass, end_airmass])*[1, 1.05],
                      filter_guess=15, filter_fixed=False)
        
        # use this to fix telluric info in for the second run
        kwarg_fit = {**{k+"_guess":self.tpoly_tell.fitvalues[k] for k in self.tpoly_tell.model.TELL_FREEPARAMETERS},
                         **{k+"_fixed":True for k in self.tpoly_tell.model.TELL_FREEPARAMETERS}}
        
        print("step 1 Done")
        # - Step 2 fit The blue region
        if get_sedm_version(self.spectrum.header.get("OBSDATE",None)):
            print("SEDM version <3 (pre-Feb 2019): running the old flux calibration procedure")
            POLY_TELL_BLUE = 20
            self.tpoly_blue = TelluricPolynomeFit(self.spectrum.lbda,
                                                  datacal,
                                                  errcal,
                                                  POLY_TELL_BLUE, load_telluric_line(), maskin=maskin)
            self.tpoly_blue.fit(**kwarg_fit)
            self.tpoly_blue.norm = norm
            response_continuum = self.tpoly_blue.model._get_continuum_()
            POLY_TELL = "%d"%POLY_TELL_BLUE
        else:
            print("step 2 Starting")
            blue_range = (self.spectrum.lbda<8600)
            POLY_TELL_BLUE = 20
            self.tpoly_blue = TelluricPolynomeFit(self.spectrum.lbda[blue_range],
                                                  datacal[blue_range],
                                                  errcal[blue_range],
                                                  POLY_TELL_BLUE, load_telluric_line(), maskin=maskin)
            self.tpoly_blue.fit(**kwarg_fit)
            self.tpoly_blue.norm = norm
            print("step 2 Done")
            #
            # - Step 3 fit The red region
            print("step 3 Starting")
            red_range = (self.spectrum.lbda>8400)
            POLY_TELL_RED = 15
            telluric = self.tpoly_blue.model.get_telluric_model(lbda=self.spectrum.lbda)
        
            self.poly_red = get_polyfit(self.spectrum.lbda[red_range], 
                                        (datacal-telluric)[red_range], 
                                        errcal[red_range], POLY_TELL_RED, legendre=True)
            self.poly_red.fit()
            print("step 3 Done")
            #
            # Finally Merge the two continuums
            #
            cont_blue = np.ones(len(self.spectrum.lbda))*np.NaN
            cont_blue[blue_range] = self.tpoly_blue.model._get_continuum_()
        
            cont_red  = np.ones(len(self.spectrum.lbda))*np.NaN
            cont_red[red_range] = self.poly_red.model.get_model()
        
            response_continuum = np.nanmean([cont_blue, cont_red], axis=0)
            POLY_TELL = "%d,%d"%(POLY_TELL_BLUE,POLY_TELL_RED)
        #
        # END
        #

        
        header = self.spectrum.header.copy()
    
        self.tpoly_blue.fitvalues['amplitude'] *= self.tpoly_blue.norm
        for tellk in self.tpoly_blue.model.TELL_FREEPARAMETERS:
            if tellk in ["airmass"]: continue
            header.set(_paramkey_to_headerkey_(tellk), self.tpoly_blue.fitvalues[tellk], "Telluric Parameter")
            
        header.set("CONTDEG", POLY_TELL,  "degree of the continuum polynome (legendre)")
        self._derived_properties["fluxcalspectrum"] = get_fluxcal_spectrum(self.spectrum.lbda, response_continuum*self.tpoly_blue.norm, header=header)
        
        return self.fluxcalspectrum

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
        self._derived_properties['calspec_spectrum'] = self._calspec.filter(5).reshape(self.spectrum.lbda,"linear")

    # --------- #
    #  PLOTTER  #
    # --------- #
    def show(self, savefile=None, show=True, ratiocolor="k", fluxcalcolor="C2",
                 show_vertbands=True, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        from pyifu import get_spectrum
                
        fig = mpl.figure()
        ax   = fig.add_axes([0.15,0.64,0.8,0.3])
        ax2  = fig.add_axes([0.15,0.12,0.8,0.5])
        axt  = ax.twinx()
        self.spectrum.show(ax=ax, show=False)
        self.calspec_spectrum.show(ax=axt, color="C1", show=False)
        
        for avoidance,c in zip(AVOIDANCE_AREA.values(), [mpl.cm.Blues(0.6),"0.5"]) :
            for l in avoidance: ax.axvspan(l[0],l[1], color=c, alpha=0.2)
        
        spec = get_spectrum(self.spectrum.lbda, self.calspec_spectrum.data / self.spectrum.data )

        spec.show(ax=ax2, color=ratiocolor, label="Calspec / SEDM", show=False, zorder=5, **kwargs)
        if self.fluxcalspectrum is not None:
            self.fluxcalspectrum.show(ax=ax2, color=fluxcalcolor, label="Flux Calibration",
                                          telllabel="incl. telluric", show=False, inversed=True, zorder=4)
            
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
        return self.spectrum.header['OBJECT'].replace("STD-","").split()[0]

    # - CalSpec
    @property
    def _calspec(self):
        """ The CalSpec Spectrum as given by calspec"""
        return self._properties['calspec']
    
    @property
    def calspec_spectrum(self):
        """ CalSpec Spectrum at the `spectrum` wavelength interpolation """
        return self._derived_properties['calspec_spectrum']
    
    @property
    def fluxcalspectrum(self):
        """ """
        return self._derived_properties["fluxcalspectrum"]
