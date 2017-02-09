#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" This modules contains the Wavelength solution tools. """

import warnings
import numpy as np
import matplotlib.pyplot as mpl
from scipy.special import orthogonal
from scipy import optimize
# - External Modules
from propobject              import BaseObject
from astrobject.spectroscopy import BaseSpectrum

# Vacuum wavelength
# from KECK https://www2.keck.hawaii.edu/inst/lris/arc_calibrations.html
# Cadmium http://www.avantes.com/images/stories/applications/pagina_93-94_cadmium.jpg
# Xenon from SNIFS 
_REFORIGIN= 69
LINES= {"Hg":{ #5790.66  :  {"ampl":1. ,"mu":202-_REFORIGIN},
               #5769.59  : {"ampl":1. ,"mu":201-_REFORIGIN},
               5778.0   : {"ampl":1. ,"mu":201-_REFORIGIN,
                           "doublet":True,"info":"merge of 5769.59, 5790.66"},
               5460.735 : {"ampl":10.,"mu":187-_REFORIGIN},
               4358.32  : {"ampl":5. ,"mu":127-_REFORIGIN},
               4046.563 : {"ampl":2. ,"mu":102-_REFORIGIN},
               #3663.27:{"ampl":0.1 ,"mu":3},
               #3654.84:{"ampl":0.1 ,"mu":1},
               #3650.153:{"ampl":1. ,"mu":0},
               },
        "Cd": {4678.15  : {"ampl":2. ,"mu":146-_REFORIGIN},
               4799.91  : {"ampl":5. ,"mu":153-_REFORIGIN},
               5085.822 : {"ampl":8. ,"mu":169-_REFORIGIN},
               6438.5   : {"ampl":4. ,"mu":228-_REFORIGIN}, # Exact Value To be confirmed
                },
        "Xe": {8280.    : {"ampl": 1. ,"mu":280-_REFORIGIN,
                            "doublet":True,"info":"merge of 8230, 8341"},
               8818.5   : {"ampl": 0.8,"mu":295-_REFORIGIN},
               8900     : {"ampl": 0.5,"mu":302-_REFORIGIN,
                            "doublet":True,"info":"merge of 8945, 9050"},
               }
        }

###########################
#                         #
#  Generators             #
#                         #
###########################
def get_arcspectrum(x, y, dy=None, name=None):
    """ """
    spec = ArcSpectrum(wave=x, flux=y, errors=dy)
    spec.set_arcname(name)
    return spec


def get_wavesolution(specid, lamps):
    """ """
    wsol= ArcSpectrumCollection()
    
    for lamp in lamps:
        wsol.add_arcspectrum( get_arcspectrum(x=np.arange(len(lamp.extract_spectrum(specid, on="rawdata")[::-1])),
                                    y=lamp.extract_spectrum(specid, on="data")[::-1],
                                    name=lamp.objname))
    return wsol

###########################
#                         #
#  WaveSolution           #
#                         #
###########################
class WaveSolution( BaseObject ):
    """ """
    PROPERTIES = ["wavesolution","inverse_wavesolution"]

    def __init__(self, polycoefs=None):
        """ 
        Parameters: 
        -----------
        polycoef: [array] -optional-
            The polynomial's coefficients, in decreasing powers.
            For example, ``[1, 2, 3]`` returns an object that represents
            :math:`x^2 + 2x + 3`

        Returns
        -------
        """
        self.__build__()
        if polycoefs is not None:
            self.set_solution(polycoefs)

    # ================== #
    #  Main Methods      #
    # ================== #
    def set_solution(self, polycoef ):
        """ Sets the wavelength solution based on numpy's poly1d
        
        Parameters: 
        -----------
        polycoef: [array]
            The polynomial's coefficients, in decreasing powers.
            For example, ``[1, 2, 3]`` returns an object that represents
            :math:`x^2 + 2x + 3`

        Returns
        -------
        Void 
        """
        self._properties["wavesolution"] = np.poly1d(polycoef)

    def writeto(self, filename, **kwargs):
        """ dump the `data` into the given `filename` with the given `format`.
        - This uses numpy.savetxt -
        """
        np.savetxt(filename, self.data, **kwargs)

    def load(self, filename, **kwargs):
        """ Loads a file containing the polynomial coefficients.
        The e.g. reads files created by `writeto`.
        - This uses numpy loadtxt -
        """
        self.set_solution( np.loadtxt(filename, **kwargs))
        
    # ================== #
    #   Properties       #
    # ================== #
    @property
    def _wavesolution(self):
        """ """
        return self._properties["wavesolution"]
    
    @property
    def data(self):
        """ Coefficents of the poly1d parameters.
        do set_solution(self.data) 
        """
        if not self.has_wavesolution():
            return None
        return self._wavesolution.coeffs
    
    # - WaveSolution
    def has_wavesolution(self):
        """ Test if the wavelength solution has been set. """
        return self._wavesolution is not None
    
    @property
    def lbda_to_pixels(self):
        """ numpy.poly1d by the best fitted wavelength solution fit.
        Example:
        --------
        self.wavesolution(6500) provides the pixel 
        corresponding to the 6500 Angstrom wavelength
        """
        return self._properties["wavesolution"]

    @property
    def pixels_to_lbda(self):
        """ Convert Pixels to Lambda 

        Example:
        --------
        self.inverse_wavesolution(1880) provides the wavelength (Angstrom)
        corresponding to the 1880th pixel
        """
        if self._properties["inverse_wavesolution"] is None:
            if not self.has_wavesolution():
                return None
            try:
                from pynverse import inversefunc
                _HASpynverse = True
            except ImportError:
                warnings.warn("You do not have pynverse. Install it to fasten the pixels_to_lbda conversion (pip install pynverse")
                _HASpynverse = False
                
            if _HASpynverse:
                self._properties["inverse_wavesolution"] = inversefunc(self.lbda_to_pixels)
            else:
                def get_root(f):
                    def _min_(p_):
                        return (r(p_)-f)**2
                    return optimize.fmin(_min_,f, disp=False)
                self._properties["inverse_wavesolution"] = get_root

        return self._properties["inverse_wavesolution"]

###########################
#                         #
#  Arc Spectrum           #
#                         #
###########################
class VirtualArcSpectrum( BaseObject ):
    """ Virtual Class Upon Wich ArcSpectrum and ArcSpectrumCollection is built """

    PROPERTIES         = ["wavesolution"]
    DERIVED_PROPERTIES = ["linefitter", "solutionfitter"]
    
    # ================ #
    #  Main Method     #
    # ================ #
    # --------- #
    #  GETTER   #
    # --------- #
    def get_line_shift(self):
        """ Shift of the central line value based on the considered spaxel """
        wavemax = self.wave[self.get_arg_maxflux(1)]
        wavemax_expected = self.arclines[self.expected_brightesline]["mu"]
        return wavemax-wavemax_expected
    
    def get_arg_maxflux(self, nbest, mask=None, order=3, **kwargs):
        """ The argument of the maximum values. 

        Parameters
        ----------
        nbest: [int]
            Number of local maximum you want. 
            
        mask: [boolean array] -optional-
            Where to look. (value True in mask are kept)
            
        order: [int] -optional-
            How many points on each side to use for the comparison
            to consider ``comparator(n, n+x)`` to be True.

        **kwargs goes to scipy.signal.argrelextrema()
        
        Returns
        -------
        nd-array (n=nbest)
        """
        from scipy.signal import argrelextrema
        f = self.flux if mask is None else self.flux[mask]
        args = argrelextrema(f, np.greater,order=order, **kwargs)[0]
        return args[np.argsort(f[args])][-nbest:]

    # --------- #
    #  FITTER   #
    # --------- #
    def fit_wavelengthsolution(self, wavesolution_degree=3, legendre=False,**kwargs):
        """ """
        from modefit import get_polyfit
        
        if not self.has_linefitter():
            self.setup_linefitter(**kwargs)
            
        mus, emus = self._linefit_to_mus_()
        self._derived_properties["solutionfitter"] = get_polyfit( self.usedlines, mus, emus, wavesolution_degree,
                                                                      legendre=legendre)
        guesses = {"a0_guess":self.databounds[0]}
        if wavesolution_degree>0:
            guesses["a1_guess"] = 0.4
        
        self.solutionfitter.fit(**guesses)
        # - Set the best fit solution
        self._derived_properties["wavesolution"] = \
          WaveSolution([self.solutionfitter.fitvalues["a%i"%i]
                        for i in range(self.solutionfitter.model.DEGREE)[::-1]])
        
    def fit_lineposition(self,contdegree=2, line_shift=None,
                             exclude_reddest_part=True,
                             red_buffer=30,
                             exclude_bluest_part=True,
                             blue_buffer=30
                             ):
        """ Fit gaussian profiles of expected arclamp emmisions.
        The list of fitted lines are given in `usedlines`.
        The fitter object is stored as `linefitter`.
        
        - This method uses `modefit` -

        Pamameters
        ----------
        contdegree: [int] -optional-
            Degree of the (Legendre) polynom used as continuum

        line_shift: [float] -optional-
            Force the expected shift of the lines that are used as first guesses 
            for the fit. If not provided, an internal automatic procedure based
            of the expected and observed brightest emission position is used.
            (see `get_line_shift()`)
        
        exclude_reddest_part: [bool] -optional-
            If True, wavelengths that are more than *red_buffer* [wave units] redder
            than the reddest 'usedline' will be excluded from the fit.
            If activated, this option raises a warning.

        red_buffer: [float] -optional-
           How much redder than the reddest emission line should the fit conserve.
           This is ignored if *exclude_reddest_part* is False

        Returns
        -------
        Void (sets linefitter)
        """
        from modefit import get_normpolyfit
        
        flagin = (self.wave>=self.databounds[0])  * (self.wave<=self.databounds[1])

        guesses = {}
        if line_shift is None:
            lines_shift = self.get_line_shift()
        for i,l in enumerate(self.usedlines):
            guesses["ampl%d_guess"%i]      = self.arclines[l]["ampl"]
            guesses["mu%d_guess"%i]        = self.arclines[l]["mu"]+lines_shift
            guesses["mu%d_boundaries"%i]   = [guesses["mu%d_guess"%i]-30,guesses["mu%d_guess"%i]+30]
            guesses["ampl%d_boundaries"%i] = [0, None]
            guesses["sig%d_guess"%i]       = 1.5
            guesses["sig%d_boundaries"%i]  = [0.5,3] if not "doublet" in self.arclines[l] or not self.arclines[l]["doublet"] else [1, 5]

        if exclude_reddest_part:
            warnings.warn("part redder than %d removed"%(guesses["mu%d_guess"%(len(self.usedlines)-1)]+red_buffer))
            flagin *= (self.wave<=guesses["mu%d_guess"%(len(self.usedlines)-1)]+red_buffer)

        if exclude_bluest_part:
            warnings.warn("part bluer than %d removed"%(guesses["mu0_guess"]-blue_buffer))
            flagin *= (self.wave>=guesses["mu0_guess"]-blue_buffer)
            
        norm = np.nanmean(self.flux[flagin])    
        self._derived_properties["linefitter"] = \
          get_normpolyfit(self.wave[flagin],self.flux[flagin]/norm,
                              self.errors[flagin]/norm if self.has_errors() else
                              np.nanstd(self.flux[flagin])/norm/5.,
                              contdegree, ngauss=len(self.usedlines), legendre=True)
            
        self.linefitter.fit(**guesses)

    def _linefit_to_mus_(self):
        """ returns the best fit values ordered as the usedlines 
        Internal tools to go from linefitter -> wavefitter
        """
        return np.asarray([[self.linefitter.fitvalues["mu%d"%i],
                            self.linefitter.fitvalues["mu%d.err"%i]]
                           for i,l in enumerate(self.usedlines)]).T

    # ------------------ #
    #  Apply Solution    #
    # ------------------ #        
    def pixels_to_lbda(self, pixels):
        """ """
        if self.has_wavesolution():
            return self.wavesolution.pixels_to_lbda(pixels)
        
        raise AttributeError("No wavelength solution define. Run fit_wavesolution()")
            
    def lbda_to_pixels(self, lbda):
        """ based on the wavelength solution loaded (see 'wavesolution'), 
        this converts an lbda in angstrom to a pixel 

        Return
        ------
        array (or float, see lbda)
        """
        if self.has_wavesolution():
            return self.wavesolution.lbda_to_pixels(lbda)
        
        raise AttributeError("No wavelength solution define. Run fit_wavesolution()")
    
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def databounds(self):
        """ limits of the effective spectrum """
        return np.min(np.where(self.flux!=0)), np.max(np.where(self.flux!=0))

    # -----------
    # - Lines
    @property
    def usedlines(self):
        """ Wavelengthes used for the wavelength matching """
        return np.sort(self.arclines.keys())
    
    @property
    def expected_brightesline(self):
        """ lines that has the greatest amplitude in the LINES global variable """
        l, amp = np.asarray([[l,v["ampl"]] for l,v in self.arclines.items()]).T
        return l[np.argmax(amp)]

    # - WaveSolution
    @property
    def wavesolution(self):
        """ """
        return self._derived_properties['wavesolution']
    
    def has_wavesolution(self):
        """ """
        return self.wavesolution is not None
    
    # -------------
    #  Fitters
    # - Lines
    @property
    def linefitter(self):
        """ Object used to fit the gaussian lines. 
        The fitted central values are then used for the solutionfitter() """
        return self._derived_properties["linefitter"]

    def has_linefitter(self):
        """ Has the Gaussian line fitter has been set? """
        return self.linefitter is not None
    
    # - Wavelength Solution    
    @property
    def solutionfitter(self):
        """ Object that enables to fit the wavelength solution """
        return self._derived_properties["solutionfitter"]
    
    def has_solutionfitter(self):
        """ Has the wavelength solution fitter has been set?"""
        return self.solutionfitter is not None
    
class ArcSpectrum( BaseSpectrum, VirtualArcSpectrum ):
    """ Arc lamp Spectrum object containing line fitting and wavelength solution fitting """
    PROPERTIES = ["name"]
    DERIVED_PROPERTIES = ["arclines"]
            
    # ================ #
    #  Main Method     #
    # ================ #
    # --------- #
    #  GETTER   #
    # --------- #
    def line_to_fitnumber(self, line):
        """ What is the index of the line within the 'usedlines' array """
        return np.argwhere(self.usedlines==line)[0]
        
    # --------- #
    #  SETTER   #
    # --------- #
    def set_arcname(self, name):
        """ Provide the name of the arclamp. 
        
        Parameters
        ----------
        name: [string]
            The given arcname should be in the list of predefined LINES
            otherwise you will not be able to acess 
            - usedlines
            - arclines
        
        Returns
        -------
        """
        self._properties["name"] = name
        if name not in LINES.keys():
            warnings.warn("Unknown Arc Lamp (%s). No Line Attached"%name)
            self._derived_properties["arclines"] = {}
        else:
            self._derived_properties["arclines"] = LINES[self.arcname]
            
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def arcname(self):
        """ Name of arc lamp"""
        return self._properties["name"]
    
    @property
    def arclines(self):
        """ ArcLines dictionary containing expected amplitude and central values """
        if self._derived_properties["arclines"] is None:
            self._derived_properties["arclines"] = {}
        return self._derived_properties["arclines"]

###########################
#                         #
#  Collection of Arcs     #
#                         #
###########################
class ArcSpectrumCollection( VirtualArcSpectrum ):
    """  """
    PROPERTIES = ["arcspectra"]
    DERIVED_PROPERTIES = ["arclines"]
    
    # ================ #
    #  Main Methods    #
    # ================ #
    def get_line_shift(self):
        """ Shift of the central line value based on the considered spaxel.
        This is based on sequential check Cd then Hd then Xe. 
        First find, first pick.
        
        Returns
        -------
        float
        """
        
        if "Cd" in self.arcspectra:
            return self.arcspectra["Cd"].get_line_shift()
        if "Hg" in self.arcspectra:
            return self.arcspectra["Hg"].get_line_shift()
        if "Xe" in self.arcspectra:
            return self.arcspectra["Xe"].get_line_shift()
        raise AttributeError("known of the pre-defined lines (Cd, Hg and Xe) have been found in the arcspectra."+\
                              " They are requested for the line_shift ")

    # -------- #
    #  FITTER  #
    # -------- #
    def fit_lineposition(self, sequential=True,
                             contdegree=2, line_shift=None,
                             exclude_reddest_part=True,
                             red_buffer=30,
                             exclude_bluest_part=True,
                             blue_buffer=30
                             ):
        """ Fit gaussian profiles of expected arclamp emmisions.
        The list of fitted lines are given in `usedlines`.
        The fitter object is stored as `linefitter`.
        
        - This method uses `modefit` -

        Pamameters
        ----------
        sequential: [bool] -optional-
            Shall the fit be made line by line?

        contdegree: [int] -optional-
            Degree of the (Legendre) polynom used as continuum

        line_shift: [float] -optional-
            Force the expected shift of the lines that are used as first guesses 
            for the fit. If not provided, an internal automatic procedure based
            of the expected and observed brightest emission position is used.
            (see `get_line_shift()`)
        
        exclude_reddest_part: [bool] -optional-
            If True, wavelengths that are more than *red_buffer* [wave units] redder
            than the reddest 'usedline' will be excluded from the fit.
            If activated, this option raises a warning.

        red_buffer: [float] -optional-
           How much redder than the reddest emission line should the fit conserve.
           This is ignored if *exclude_reddest_part* is False

        Returns
        -------
        Void (sets linefitter)
        """
        lineprop = dict(contdegree=contdegree,line_shift=line_shift,
                        exclude_reddest_part=exclude_reddest_part,
                        red_buffer=red_buffer,
                        exclude_bluest_part=exclude_bluest_part,
                        blue_buffer=blue_buffer)
        
        if not sequential:
            super(ArcSpectrumCollection, self).fit_lineposition(**lineprop )
            self._sequentialfit = False
        else:
            [self.arcspectra[s].fit_lineposition(**lineprop) for s in self.arcnames]
            self._derived_properties["linefitter"] = {s:self.arcspectra[s].linefitter for s in self.arcnames}
            self._sequentialfit = True

    def _linefit_to_mus_(self):
        """ returns the best fit values ordered as the usedlines 
        Internal tools to go from linefitter -> wavefitter
        """
        # - Non Sequential
        if not self.is_linefit_sequential():
            return super(ArcSpectrumCollection, self)._linefit_to_mus_()
        
        # - Sequential
        mus, emus = [],[]
        for li in self.usedlines:
            lamp_ = self.line_to_lamp(li)
            line_ = self.arcspectra[lamp_].line_to_fitnumber(li)[0]
            
            mus.append(self.linefitter[lamp_].fitvalues["mu%d"%line_])
            emus.append(self.linefitter[lamp_].fitvalues["mu%d.err"%line_])
        return np.asarray(mus),np.asarray(emus)
    
    # -------- #
    #  IO      #
    # -------- #
    def line_to_lamp(self, line):
        """ """
        for s in self.arcnames:
            if line in self.arcspectra[s].usedlines:
                return s
    
    def add_arcspectrum(self, arcspectrum, name=None):
        """ """
        if ArcSpectrum not in arcspectrum.__class__.__mro__:
            raise TypeError("The given arcspectrum is not a pysedm's ArcSpectrum object")
        if name is None:
            if arcspectrum.arcname is None:
                raise ValueError("You need to provide a name for the arcspectrum")
            name = arcspectrum.arcname
            
        # - Set it.
        self.arcspectra[name] = arcspectrum
        self._derived_properties["arclines"] = None

    def remove_arcspectrum(self, arcname):
        """ Remove an arcspectrum from the collection """
        self.arcspectra.pop(arcname)
        self._derived_properties["arclines"] = None
        
    # ================ #
    #  Properties      #
    # ================ #
    # -----------
    # Fake Spectrum
    @property
    def wave(self):
        """ """
        return self.arcspectra.values()[0].wave \
          if self.has_spectra() else None
    
    @property
    def flux(self):
        return np.sum([s.flux for s in self.arcspectra.values()], axis=0)\
          if self.has_spectra() else None
    
    @property
    def errors(self):
        """ """
        #TO BE DONE
        return None
    
    def has_flux(self):
        return self.flux is not None
    
    def has_errors(self):
        return self.errors is not None

    # -----------
    # Collection
    @property
    def arcspectra(self):
        """ dictionary containing the individual arcspectrum set """
        if self._properties["arcspectra"] is None:
            self._properties["arcspectra"] = {}
        return self._properties["arcspectra"]
    
    @property
    def nspectra(self):
        """ number of arcspectrum attached to this collection """
        return len(self.arcspectra.keys())
    
    def has_spectra(self):
        """ Tests if you already loaded spectra"""
        return self.nspectra>0

    @property
    def arcnames(self):
        return np.sort(self.arcspectra.keys())
    
    # -----------------
    # Collection Tricks
    def is_linefit_sequential(self):
        """ """
        if not self.has_linefitter():
            return None
        return self._sequentialfit
    
    @property
    def arclines(self):
        """ """
        if self._derived_properties["arclines"] is None:
            di={}
            for d in self.arcspectra.values():
                for k,v in d.arclines.items():
                    di[k]=v
            self._derived_properties["arclines"] = di
            
        return self._derived_properties["arclines"]
    


   


    
