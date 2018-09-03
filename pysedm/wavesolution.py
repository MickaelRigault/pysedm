#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" This modules contains the Wavelength solution tools. """

import warnings
import numpy as np
import matplotlib.pyplot as mpl
from scipy         import optimize
from astropy.utils.console import ProgressBar

try:
    from pynverse import inversefunc
    _HASPYNVERSE = True
except ImportError:
    warnings.warn("Cannont Import pynverse, limited wavelength solution tools | pip install pynverse")
    _HASPYNVERSE = False
# - External Modules
from propobject              import BaseObject
from pyifu.spectroscopy      import Spectrum

# - Internal Modules
from .ccd import CCD

# Vacuum wavelength
# from KECK https://www2.keck.hawaii.edu/inst/lris/arc_calibrations.html
# Cadmium http://www.avantes.com/images/stories/applications/pagina_93-94_cadmium.jpg
# Xenon from SNIFS
#
#
# http://www.astrosurf.com/buil/us/spe2/hresol4.htm
#

# From Keck: Hg
HG_LINES = {"air":[3125.67 , 3131.70 , 3341.48 , 3650.153,
                  3654.84 , 3663.279, 4046.563, 4077.831,
                  4358.327, 4916.068, 5460.750, 5769.598,
                  5790.663],
            "vacuum":  [3126.580, 3132.463, 3342.445, 3651.198,
                        3655.883, 3664.327, 4047.708, 4078.988,  
                        4359.560, 4917.440, 5462.268, 5771.210,
                        5792.276],
             "vacuum_blended": [3129.44,3345.04, 3650.87, 3663.18] 
            }
    
CD_LINES = {"air" : [3080.822, 3082.593, 3133.167, 3252.524, 
                     3261.055, 3403.652, 3466.200, 3467.655, 
                     3499.952, 3610.508, 3612.873, 3614.453, 
                     3649.558, 3981.926, 4140.500, 4306.672, 
                     4412.989, 4662.352, 4678.149, 4799.912, 
                     5085.822, 5154.660, 6099.142, 6111.490, 
                     6325.166, 6330.013, 6438.470, 6778.116, 
                     7345.670],
                     
          "vacuum" : [3081.71730, 3083.48870, 3134.07460, 3253.46220, 
                      3261.99510, 3404.62880, 3467.19230, 3468.64780, 
                      3500.95320, 3611.53750, 3613.90330, 3615.48370, 
                      3650.59830, 3983.05200, 4141.46650, 4307.88320, 
                      4414.22880, 4663.65720, 4679.45870, 4801.25400, 
                      5087.23930, 5156.09640, 6100.83050, 6113.18700, 
                      6326.91510, 6331.76360, 6440.24900, 6779.98650, 
                      7347.69410],
           "vacuum_blended": [3466.85, 3611.01, 4679.325] 
                }


XE_LINES = {"air": [ 7119.614 ,  7642.0124,  8231.6305,  8231.6305,  8280.1113,
                     8280.1113,  8346.8112,  8347.2311,  8408.2036,  8409.1843,
                     8819.4029,  8952.2468,  9045.4415,  9162.6396,  9513.3743,
                     9513.3743,  9685.3175,  9718.1586,  9799.6964],
                     
            # vacuum DERIVED inversing var_to_air_sdss, see utils.tools.fetch_vac
            "vacuum": [7121.58, 7644.12, 8233.90, 8233.90,
                       8282.39, 8282.39, 8349.11, 8349.53,
                       8410.52, 8411.50, 8821.83, 8954.71,
                       9047.93, 9165.16, 9515.99, 9515.99,
                       9687.98, 9720.83, 9802.39],
            "vacuum_blended": [8411.0] 
                }


# -------------- #
#  SEDM          #
# -------------- #
REFWAVELENGTH = 7000

# For Information:
#    Sodium skyline fitted by SNIFS is at 5892.346 Angstrom
SODIUM_SKYLINE_LBDA = 5892.3
TELLURIC_REF_LBDA   = 7624.5 # 
_REFORIGIN = 69

LINES= {"Hg": # IN VACUUM
            { 
               np.mean([5771.210, 5792.276])   : {"ampl":13. ,"mu":201-_REFORIGIN,
                           "doublet":False,
                        "info":"merge of 5771.210, 5792.276 blended"},
               5462.268 : {"ampl":62.,"mu":187-_REFORIGIN},
               4359.560   : {"ampl":50. ,"mu":127-_REFORIGIN},
               4047.708 : {"ampl":10. ,"mu":105-_REFORIGIN}, 
              
               },
        "Cd":  # IN VACUUM
              {4679.325 : {"ampl":14. ,"mu":146-_REFORIGIN}, 
               4801.254 : {"ampl":40. ,"mu":153-_REFORIGIN},
               5087.239 : {"ampl":60. ,"mu":169-_REFORIGIN},
               6440.249 : {"ampl":19. ,"mu":227-_REFORIGIN}, 
               # - Cd and Xe seems to have the same lines
               # Almost same wavelength but diffent enough to save the code
               8280.01  : {"ampl": 1. ,"mu":275-_REFORIGIN,
                            "backup":"Xe", # used only if Xe is not there
                            "doublet":True,"info":"merge of 8230, 8341"},
               #8818.51  : {"ampl": 0.8,"mu":295-_REFORIGIN},
               #9000.01  : {"ampl": 0.5,"mu":302-_REFORIGIN,
               #             "doublet":True,"info":"merge of 8945, 9050"},
                },
        # Important for Xe. Keep the Bright line, the First one. or change get_line_shift()
#        "Xe": {8280.    : {"ampl": 1. ,"mu":280-_REFORIGIN,
#                            "doublet":True,"info":"merge of 8230, 8341"},
#               8818.5   : {"ampl": 0.8,"mu":295-_REFORIGIN},
#               9000     : {"ampl": 0.5,"mu":302-_REFORIGIN,
#                            "doublet":True,"info":"merge of 8945, 9050"},
#               }

        "Xe": {np.average([8233.90, 8282.39], weights=[2,1])  : {"ampl": 16. ,"mu":280-_REFORIGIN,
                            "doublet":False},# yes but really close
                            
               np.average([8349.11, 8411.00], weights=[2,3])   : {"ampl": 3. ,"mu":283-_REFORIGIN,
                            "doublet":False}, # yes but really close
                            
                            
               8821.83    : {"ampl": 11.,"mu":294-_REFORIGIN},
                   
               np.average([8954.71,9047.93]) : {"ampl": 11.,"mu":298-_REFORIGIN,
                             "doublet":True , "info": "merge of lines 9854.71,9047.93"},
               9165.16    : {"ampl": 4.5,"mu":303-_REFORIGIN},
               
               # small lines but isolated
               7644.12    : {"ampl": 1.,"mu":264-_REFORIGIN},
               
               #9425.    : {"ampl": 2.,"mu":309-_REFORIGIN,
               #                "doublet":True , "info": "merge of lines 9400 and 9450"},
               
               #9802.8    : {"ampl": 0.4,"mu":317-_REFORIGIN},
               #9938.0    : {"ampl": 0.4,"mu":320-_REFORIGIN},
               
               }
        }

###########################
#                         #
#  Generators             #
#                         #
###########################
def load_wavesolution(filenames):
    """ Load the wavesolution from saved data """
    wsol = WaveSolution()
    wsol.load(filenames)
    return wsol
    
def get_wavesolution(*lamps):
    """ Loads the object that enables to build the WaveSolution. """
    wsol = WaveSolution()
    [wsol.add_lampccd(l) for l in lamps]
    return wsol

def get_arcspectrum(x, y, databound, dy=None, name=None):
    """ """
    spec = ArcSpectrum(None)
    spec.create(lbda=x, data=y, variance=dy**2 if dy is not None else None, logwave=False)
    spec.set_arcname(name)
    spec.set_databounds(*databound)
    return spec

def get_arccollection(traceindex, lamps):
    """ """
    sol_ = ArcSpectrumCollection()
    
    for lamp in lamps:
        spec_ = lamp.get_spectrum(traceindex, on="data")[::-1]
        lbda_ = np.arange(len(spec_))
        sol_.add_arcspectrum( get_arcspectrum(x=lbda_, y=spec_,
                                            databound= np.sort(len(spec_) - lamp.tracematch.get_trace_xbounds(traceindex)),
                                            name=lamp.objname))
    sol_.set_databounds(*np.sort(len(spec_) - lamp.tracematch.get_trace_xbounds(traceindex)))
    return sol_

# ==================== #
#   Subprocessing      #
# ==================== #
def merge_wavesolutions(wavesolutions):
    """ """
    wsol = WaveSolution()
    for w in wavesolutions:
        for i in w.traceindexes:
            wsol.add_trace_wavesolution(i,w.wavesolutions[i])
        
    return wsol

# -------------------- #
#   MultiProcessing    #
# -------------------- #
def fit_spaxel_wavelesolution(arccollection, sequential=True,
                              contdegree=3, wavedegree=5,
                              saveplot=None, show=False,
                        plotprop={}):
    """ Fit the wavelength solution of the given trace.
    
    This matching is made in two steps:
    1) fit the central positions of the loaded arcspectrum emission lines.
       This fit is based on linear combination of gaussian lines + polynomial continuum.
    2) Fit a polynomial degree to modelize the relation between the 
       fitted central positions (c.f. step 1) with the corresponding wavelengths
       of these lines.

    Parameters
    ----------
    
    sequential: [bool] -optional-
        How the gaussian line profiles (central positions) should be fitted:
        - False: All at once (as if it was a unique spectrum)
        - True:  Once at the time (each arcspectrum fitted independently)

    contdegree: [int] -optional-
        The degree of the polynom underlying the gaussian lines
            
    wavedegree: [int] -optional-
        The degree of the polynom modelizing the pixel<->wavelength relation
            
    saveplot: [None or string] -optional-
        Save the wavelength solution figure. If None, nothing will be saved.
            
    show: [bool] -optional-
        Should the wavelength solution figure be shown?
        
    Returns
    -------
    None
    """
    # 3s 
    arccollection.fit_lineposition(contdegree=contdegree, sequential=sequential)
    # 0.01s
    arccollection.fit_wavelengthsolution(wavedegree, legendre=False)
    return arccollection

def fit_wavesolution(lamps, indexes, multiprocess=True, notebook=False):
    """ """
    if notebook:
        print("RUNNING FROM NOTEBOOK")
        try:
            return _fit_background_notebook_(lamps, indexes, multiprocess=True)
        except:
            warnings.warn("FAILING fit_background for notebooks")
            
    # Running from ipython/python    
    arccollections = [get_arccollection(id_, lamps) for id_ in indexes]
    
    wsolutions = ProgressBar.map(fit_spaxel_wavelesolution,
                                arccollections,
                                multiprocess=multiprocess, step=2)
    #  these are the wavesolutions
    return {i_:wsol_.data for i_,wsol_ in zip(indexes, contval)}

def _fit_wavesolution_notebook_(lamps, indexes, multiprocess=True):
    """ """
    arccollections = [get_arccollection(id_, lamps) for id_ in indexes]
    
    # - Multiprocessing 
    if multiprocess:
        bar = ProgressBar( len(indexes), ipython_widget=True)
        import multiprocessing
        p = multiprocessing.Pool()
        res = {}
        for j, result in enumerate( p.imap(fit_spaxel_wavelesolution, arccollections)):
            res[indexes[j]] = result
            bar.update(j)
        bar.update(len(indexes))
        return res
    
    # - No multiprocessing 
    return {indexes[i_]: fit_spaxel_wavelesolution(arccollection)
                for i_,arccollection in enumerate(arccollections)}


###########################
#                         #
#  Flexure Correction     #
#                         #
###########################
class Flexure( BaseObject ):
    """ """
    PROPERTIES = ["cube","mapper"]
    DERIVED_PROPERTIES = ["fitvalues_sodiumlines"]

    def __init__(self, cube, mapper=None):
        """ """
        self.set_cube(cube)
        self.set_mapper(mapper)
    # ================== #
    #   Method           #
    # ================== #
    # --------- #
    #  SETTER   #
    # --------- #
    def set_cube(self, cube_):
        """ attach to this intance a cube """
        self._properties["cube"] = cube_
        
    def set_mapper(self, mapper_):
        """ attach to this intance a mapper object containing the 
        ccd to cube coordinates conversions """
        self._properties["mapper"] = mapper_
        
    # --------- #
    #  GETTER   #
    # --------- #
    def get_i_shift(self, sodium_reference=SODIUM_SKYLINE_LBDA,
                         telluric_reference=TELLURIC_REF_LBDA,
                        as_slice=True):
        """ assuming the flexure is purely a i [x-ccd coord] shift, 
        how much would explaint the difference with respect to the sodium_reference
        
        i_shift such as `observed_position - reference_position`
        Returns
        """
        mu_eff, mu_efferr = self.get_cube_line_wavelength(array=False, which="sodium")
        mu_t_eff, mu_t_efferr = self.get_cube_line_wavelength(array=False, which="telluric")

        # Sodium
        if mu_efferr==mu_efferr:
            ijs_eff    = self.mapper.get_lbda_ij(mu_eff)
            ijs_reff   = self.mapper.get_lbda_ij(sodium_reference)
            delta_i_na  = [ijs_reff[traceindex][0]-ijs_eff[traceindex][0]
                        for traceindex in self.mapper.traceindexes]
        else:
            delta_i_na = None
            
            
        # Telluric
        if mu_t_efferr==mu_t_efferr:
            ijs_eff_t  = self.mapper.get_lbda_ij(mu_t_eff)
            ijs_reff_t = self.mapper.get_lbda_ij(telluric_reference)
            delta_i_t   = [ijs_reff_t[traceindex][0]-ijs_eff_t[traceindex][0]
                        for traceindex in self.mapper.traceindexes]
        else:
            delta_i_t = None

        # Get the shift
        if delta_i_na is not None and delta_i_t is not None:
            delta_i     = np.average([delta_i_na, delta_i_t], weights=[1/mu_efferr**2,1/mu_t_efferr**2], axis=0)
        elif delta_i_t is not None:
            delta_i     = np.average([delta_i_t], weights=[1/mu_t_efferr**2], axis=0)
        elif delta_i_na is not None:
            delta_i     = np.average([delta_i_na], weights=[1/mu_efferr**2], axis=0)
        else:
            raise AttributeError("I cannot build the Flexure correction. Not a single method (telluric or Na) worked !")
        
        if not as_slice:
            return {i:d for i,d in zip(self.mapper.traceindexes, delta_i)}

        from pyifu import get_slice
        return get_slice(delta_i, self.mapper.traceindex_to_xy(self.mapper.traceindexes), 
                            spaxel_vertices=self.cube.spaxel_vertices, 
                           indexes=self.mapper.traceindexes)
    
    def get_cube_line_wavelength(self, array=False, which="sodium"):
        """ get the cube sodium line position.
        If array=True: the fitted sodium lines and there errors are returns
        otherwise, a robust esitmation of the mean sodium (and error) line is returned

        Robustness: cases more than 2 nMAD times away from the median are excluded.
        """
        from astropy.stats import mad_std
        if which in ["sodium"]:
            id_ = "0"
        elif which in ["telluric"]:
            id_ = "1"
        else:
            raise ValueError("which can only be 'sodium' or 'telluric', %s given"%which)
        
        mus, muserr, sigm, sigmerr = np.asarray([[v["mu%s"%id_],v["mu%s.err"%id_],v["sig%s"%id_], v["sig%s.err"%id_]]
                                                      for v in self.fitvalues_sodiumlines]).T
        
        flagin = (mus==mus) * (muserr>1e-3) * (sigmerr>1e-4) * (sigm < 49)
                                                
        if array:
            return mus[flagin], muserr[flagin]
        
        # - robust calc:
        if np.any(flagin):
            median  = np.nanmedian(mus)
            nmad    = mad_std(mus[mus==mus])
            flagin = (np.abs(mus-median) <= nmad*2.) * flagin
            return np.nanmean(mus[flagin]), np.std(mus[flagin])/np.sqrt(len(mus[flagin])-1)
        return np.NaN, np.NaN
    
    
    # --------- #
    #  FITTER   #
    # --------- #        
    def fit_lines(self, spectrum, 
                    sodium_reference=SODIUM_SKYLINE_LBDA, telluric_reference=TELLURIC_REF_LBDA,
                    lbda_buffer=400, show=False):
        """ fit the sodium line on the given spectrum and returns the fitvalue dictionary.
        = This method using modefit.get_normpolyfit() to fit the emission line=

        Remark: See the `fit_cube_lines()` method for the entire cube fit.

        Parameters
        ----------
        spectrum: [pyifu.Spectrum]
            wavelength calibrated spectrum containing the sodium emmision line 
            
        sodium_reference: [float] -optional-
            wavelength of the sodium sky line. 

        lbda_buffer: [float] -optional-
            how many wavelength above and below the sodium_reference will be used for the fit.
        
        show: [bool] -optional-
            Do you want to display the fit result?

        Returns
        -------
        dict (fitvalues)
        """
        from modefit import get_normpolyfit
        flagout  = (spectrum.lbda<SODIUM_SKYLINE_LBDA-lbda_buffer)# + (spectrum.lbda>telluric_reference+lbda_buffer)
        if spectrum.has_variance():
            flagout += np.isnan(spectrum.data*spectrum.variance)
        else:
            flagout += np.isnan(spectrum.data)
            
        norm    = np.nanmean(spectrum.data[~flagout])
        nfit    = get_normpolyfit(spectrum.lbda[~flagout], spectrum.data[~flagout]/norm, 
                                    spectrum.variance[~flagout]/norm**2*2 if spectrum.has_variance() else None, 
                                degree=5, ngauss=2)
        
        nfit.fit(**{# Sodium 
                    "ampl0_guess": 100,
                    "ampl0_boundaries": [1,1000],
                    "mu0_guess":  sodium_reference,"mu0_boundaries":  [sodium_reference-100.,sodium_reference+100],
                    "sig0_guess": 30,"sig0_boundaries": [15,60],
                    # Telluric
                    "ampl1_guess": -100,
                    "ampl1_boundaries": [-1000,-1],
                    "mu1_guess":  telluric_reference,"mu1_boundaries":  [telluric_reference-100.,telluric_reference+100],
                    "sig1_guess": 40,"sig1_boundaries": [20,70], # larger since composition of
                    # Continuum                    
                    "a0_guess": 1, "a1_guess": 0})
        
        if show:
            nfit.show()
            
        return nfit.fitvalues
    
    def fit_cube_lines(self, sodium_reference=SODIUM_SKYLINE_LBDA,
                                 nspaxels=50, averaging=20, show=False):
        """ """
        index_to_fit = np.asarray(self.cube.get_faintest_spaxels(nspaxels*averaging))
        np.random.shuffle(index_to_fit)
        self._indexes = index_to_fit.reshape(nspaxels,averaging)
        self._derived_properties["fitvalues_sodiumlines"] = []
        for index_ in self._indexes:
            fv = self.fit_lines(self.cube.get_spectrum(index_,"data"),
                                          sodium_reference=sodium_reference, show=show)
            self.fitvalues_sodiumlines.append(fv)
            
    # --------- #
    #  PLOTTER  #
    # --------- #
    def get_i_flexure(self):
        """ """
        return np.mean( list(self.get_i_shift(as_slice=False).values()) )
        
    def show(self, savefile=None, show=True,
                 sodium_reference=SODIUM_SKYLINE_LBDA,
                 telluric_reference=TELLURIC_REF_LBDA):
        """" """
        import matplotlib.pyplot as mpl
        fig = mpl.figure(figsize=[5,3])
        ax  = fig.add_axes([0.15,0.2,0.75,0.7])

        # - Data
        mus, mus_err           = self.get_cube_line_wavelength(True, which="sodium")
        tell_mus, tell_mus_err = self.get_cube_line_wavelength(True, which="telluric")
        delta_mus = np.concatenate([mus-sodium_reference, tell_mus-telluric_reference])
        # Centroid
        meanmus, meanmus_err   = self.get_cube_line_wavelength(False)
        meanmus_t, meanmus_t_err   = self.get_cube_line_wavelength(False, which="telluric")
        
        meanmus   -= sodium_reference
        meanmus_t -= telluric_reference

        
        histprop = dict(bins=15, range=[np.floor(np.percentile(delta_mus, 5)),
                                        np.ceil(np.percentile(delta_mus, 95))],
                            histtype="step", normed=False)
        ax.hist(delta_mus[delta_mus==delta_mus],fill=False, **histprop)
        ax.hist(mus[mus==mus] -sodium_reference,fill=True, 
                facecolor=mpl.cm.Blues(0.6,0.4), lw=0, **histprop)
            
        ax.set_xlabel(r"Fitted sky line shift [$\AA$]", fontsize="medium")
        
        ax.axvline( 0 , ls="-", color="C1", label="Expectation (Na: %.1f ; Tell: %.1f)"%(sodium_reference,
                                                                                         telluric_reference))
        # Sodium
        if meanmus_err == meanmus_err:
            ax.axvline( meanmus, ls="--", color="C2")
            ax.axvspan( meanmus-meanmus_err,meanmus+meanmus_err, color="C2", alpha=0.5,
                            label=r"Sodium shift $(%.1f \pm %.1f)$"%(meanmus,meanmus_err))
        # Telluriv
        if meanmus_t_err==meanmus_t_err:
            ax.axvline( meanmus_t, ls="--", color="C3")
            ax.axvspan( meanmus_t-meanmus_t_err,meanmus_t+meanmus_t_err, color="C3", alpha=0.5,
                            label=r"Telluric shift $(%.1f \pm %.1f)$"%(meanmus_t,meanmus_t_err))
            
            
        ax.legend(loc="upper left",fontsize="x-small")
        
        ax.text(0.98,0.98,"%d spectra made by \n averaging over %d spaxels"%(len(self._indexes),len(self._indexes[0])),
                    va="top",ha="right", transform=ax.transAxes, color="0.5", fontsize="x-small")
        
        ax.text(0.5,0.05,"corresponds to a typical ccd 'i'-shift of %.2f pixels"%(self.get_i_flexure()),
                    va="bottom",ha="center", transform=ax.transAxes, color="k",
                    bbox={'facecolor':'w', 'alpha':0.8,'edgecolor':'0.5', "boxstyle":"round"}, fontsize="small")
        
        ax.set_title("obs: %s | airmass: %.2f"%(self.cube.filename.split("/")[-1],self.cube.header["AIRMASS"]),
                fontsize="x-small", color="0.5")

        ax.set_ylim(0, ax.get_ylim()[-1]*1.5 )
        if savefile is not None:
            fig.savefig(savefile)
        if show:
            fig.show()


    # ================== #
    #   Properties       #
    # ================== #
    @property
    def cube(self):
        """ pyifu 3d cube """
        return self._properties["cube"]
    
    @property
    def mapper(self):
        """ object containing the ccd to 3d cube conversion tools """
        return self._properties["mapper"]

    # -- derived
    @property
    def fitvalues_sodiumlines(self):
        """ """
        if self._derived_properties["fitvalues_sodiumlines"] is None:
            raise AttributeError("fitvalues_sodiumlines has not yet be derived. See the `fit_cube_lines()` method")
        return self._derived_properties["fitvalues_sodiumlines"]

###########################
#                         #
#  WaveSolution           #
#                         #
###########################
class WaveSolution( BaseObject ):
    """ """
    PROPERTIES = ["lamps"]
    DERIVED_PROPERTIES = ["wavesolutions","solutions"]

    # ================== #
    #  Main Methods      #
    # ================== #
    # -------- #
    #  I/O     #
    # -------- #
    def writeto(self, filename):
        """ save the object into the given filename (.pkl). 
        Load it the using the load() method.
        """
        from pysedm.utils.tools import dump_pkl
        dump_pkl(self.wavesolutions, filename)

    def load(self, filename):
        """ Load the object from the given filename (.pkl).
        object created by the writeto() method can be opened this way.
        """
        from pysedm.utils.tools import load_pkl
        data = load_pkl(filename)
        if "wavesolution" not in list(data.values())[0]:
            raise TypeError('The given dictionary file does not seem to be a wavelength solution. No "wavesolution" in the first entry')
        
        self.set_wavesolutions(data)
        
    # -------- #
    # BUILDER  #
    # -------- #
    def fit_wavelesolution(self, traceindex, sequential=True, contdegree=4, wavedegree=5,
                            saveplot=None, show=False, plotprop={}):
        """ Fit the wavelength solution of the given trace.

        This matching is made in two steps:
        1) fit the central positions of the loaded arcspectrum emission lines.
           This fit is based on linear combination of gaussian lines + polynomial continuum.
        2) Fit a polynomial degree to modelize the relation between the 
           fitted central positions (c.f. step 1) with the corresponding wavelengths
           of these lines.

        Parameters
        ----------
        traceindex: [int]
            index of the trace for which you want to fit the wavelength solution
            
        sequential: [bool] -optional-
            How the gaussian line profiles (central positions) should be fitted:
            - False: All at once (as if it was a unique spectrum)
            - True:  Once at the time (each arcspectrum fitted independently)

        contdegree: [int] -optional-
            The degree of the polynom underlying the gaussian lines
            
        wavedegree: [int] -optional-
            The degree of the polynom modelizing the pixel<->wavelength relation
            
        saveplot: [None or string] -optional-
            Save the wavelength solution figure. If None, nothing will be saved.
            
        show: [bool] -optional-
            Should the wavelength solution figure be shown?
        
        Returnsemus
        -------
        None
        """
         # <0.1s # with saved masks
        wsol_ = get_arccollection(traceindex, [self.lampccds[i] for i in self.lampnames])
        # 3s 
        wsol_.fit_lineposition(contdegree=contdegree, sequential=sequential)
        # 0.01s
        wsol_.fit_wavelengthsolution(wavedegree, legendre=False)
        self.wavesolutions[traceindex] = wsol_.data

                                         
        self._wsol = wsol_
        if saveplot is not None or show:
            wsol_.show(traceindex=traceindex, xrange=[3600,9500], savefile=saveplot, **plotprop)

    # -------- #
    # GETTER   #
    # -------- #
    def get_spaxel_wavesolution(self, traceindex):
        """ """
        if traceindex not in self.wavesolutions:
            raise ValueError("Unknown wavelength solution for the spaxels #%d"%traceindex)
        
        if traceindex not in self._solution:
            self._solution[traceindex] = SpaxelWaveSolution( self.wavesolutions[traceindex]["wavesolution"] )
            
        return self._solution[traceindex]

    def _load_full_solutions_(self):
        """ """
        for traceindex in self.wavesolutions.keys():
            if traceindex not in self._solution:
                self._solution[traceindex] = SpaxelWaveSolution(self.wavesolutions[traceindex]["wavesolution"])
                
            self._solution[traceindex].load_pixel_to_lbda_solution()
            
    
    def pixels_to_lbda(self, pixel, traceindex):
        """ Pick the requested spaxel and get the wavelength [in angstrom] that goes with the given pixel """
        return self.get_spaxel_wavesolution(traceindex).pixels_to_lbda(pixel)
    
    def lbda_to_pixels(self, lbda, traceindex):
        """ Pick the requested spaxel and get the pixel that goes with the given wavelength [in angstrom] """
        return self.get_spaxel_wavesolution(traceindex).lbda_to_pixels(lbda)
    
    # -------- #
    #  I/O     #
    # -------- #
    def set_wavesolutions(self, wavesolutions):
        """ Load a dictionary containing the wavelength solution of individual spaxels """
        [self.add_trace_wavesolution(traceindex, data) for traceindex,data in wavesolutions.items()]
        
            
    def add_trace_wavesolution(self, traceindex, data, replace=False):
        """ add to the current instance 'data' of a new wavelength solution.
        see SpaxelWaveSolution.data
        """
        if traceindex in self.wavesolutions.keys() and not replace:
            raise ValueError("%d is already loaded. Set `replace=True` to replace the current value."%traceindex)
        
        self.wavesolutions[traceindex] = data
        self._solution[traceindex]     = SpaxelWaveSolution(data["wavesolution"], datafitted=[data["usedlines"], data["fit_linepos"], data['fit_linepos.err']])
        

        
    def add_lampccd(self, lampccd, name=None):
        """ """
        if CCD not in lampccd.__class__.__mro__:
            raise TypeError("The given lampccd is not a pysedm's CCD object")
        
        if name is None:
            if lampccd.objname is None:
                raise ValueError("You need to provide a name for the arcspectrum")
            name = lampccd.objname
            
        # - Set it.
        self.lampccds[name] = lampccd

    def remove_lampccd(self, name):
        """ """
        self.lampccds.pop(name)

    # --------- #
    # PLOTTER   #
    # --------- #
    def show_dispersion_map(self, hexagrid,
                                ax=None, savefile=None, show=True,
                                kind="nMAD", vmin="0.5", vmax="99.5",
                            clabel=r"nMAD [$\AA$]", **kwargs):
        """ """
        from pysedm.sedm import display_on_hexagrid
        traceindexes = list(self._solution.keys())
        value = nmad = [self._solution[i].get_wavesolution_rms(kind="nMAD") for i in traceindexes]
        return display_on_hexagrid(value, traceindexes,hexagrid=hexagrid, 
                                       ax=ax, vmin=vmin, vmax=vmax,
                                       clabel=clabel, savefile=savefile, show=show,
                                       **kwargs)
    # ================== #
    #  Properties        #
    # ================== #
    @property
    def lampnames(self):
        """ Names of the lamps loaded """
        return np.sort(list(self.lampccds.keys()))
    @property
    def lampccds(self):
        """ dictionary containing the ScienceCCD objects used
        to get the wavelength solution of the cube """
        if self._properties["lamps"] is None:
            self._properties["lamps"] = {}
        return self._properties["lamps"]

    @property
    def wavesolutions(self):
        """ dictionary containing the wavelength solution for the loaded spectra """
        if self._derived_properties["wavesolutions"] is None:
            self._derived_properties["wavesolutions"] = {}
        return self._derived_properties["wavesolutions"]
    @property
    def _solution(self):
        """ WaveSolution object. use get_wavesolution() """
        if self._derived_properties["solutions"] is None:
            self._derived_properties["solutions"] = {}
        return self._derived_properties["solutions"]

    @property
    def traceindexes(self):
        """ list of indexes having wavelengh solution loaded. """
        return list(self.wavesolutions.keys())
    
class SpaxelWaveSolution( BaseObject ):
    """ """
    PROPERTIES = ["wavesolution","inverse_wavesolution"]
    SIDE_PROPERTIES = ["datafitted"]
    def __init__(self, polycoefs=None, datafitted=None):
        """ 
        Parameters: 
        -----------
        polycoef: [array] -optional-
            The polynomial's coefficients, in decreasing powers.
            For example, ``[1, 2, 3]`` returns an object that represents
            :math:`x^2 + 2x + 3`

        datafitted: [3 arrays] -optional-
            You could provide x,y and dy.
            The data used to build the SpaxelWaveSolution

        Returns
        -------
        """
        self.__build__()
        if polycoefs is not None:
            self.set_solution(polycoefs)
        if datafitted is not None:
            self.set_datafitted(*datafitted)
            
    # ================== #
    #  Main Methods      #
    # ================== #
    # ---------- #
    #  GETTER    #
    # ---------- #
    def get_wavesolution_rms(self, kind="wrms"):
        """ """
        if self.datafitted is None:
            raise AttributeError("This needs the self.datafitted to work")

        res = self.pixels_to_lbda(self.datafitted['fit_linepos'])-self.datafitted['usedlines']
        known_kind = ["std or rms", "nmad", "wrms"]
        
        if kind in ["std", "rms"]:
            return np.nanstd(res)
        if kind in ['nMAD', 'nmad','mad_std']:
            from astropy.stats import mad_std
            return mad_std(res)
        if kind in ["wrms", "wRMS"]:
            err   = self.datafitted['fit_linepos.err']
            wmean = np.average(np.average(res, weights=1./err**2))
            return np.sqrt(np.average((res-wmean)**2, weights=1./err*2))
        raise ValueError("unknown kind %s."%kind+" use: "+", ".join(known_kind))
    # ---------- #
    #  SETTER    #
    # ---------- #
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

    def set_datafitted(self, usedlines, fit_linepos, fit_lineposerr):
        """ The expected wavelength, the fitted line position in pixels and its associated error """
        self._side_properties['datafitted'] = {"usedlines":usedlines,
                                                "fit_linepos":fit_linepos,
                                                "fit_linepos.err":fit_lineposerr}
        
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

    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, ax=None, savefile=None, show=True, ecolor="0.3",
                 xrange=None, **kwargs):
        """ """
        from astrobject.utils.mpladdon import figout, errorscatter
        from astrobject.utils.tools    import kwargs_update
        
        if self.datafitted is None:
            raise AttributeError('No datafitted to be shown')
        
        # - Non Sequential
        if ax is None:
            fig = mpl.figure(figsize=[6,5])
            ax = fig.add_axes([0.12,0.12,0.78,0.78])
            ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", fontsize="large")
            ax.set_ylabel(r"Pixels (ccd-row)", fontsize="large")
        else:
            fig = ax.figure

        prop = kwargs_update( dict(mfc=mpl.cm.binary(0.2,0.5), mec=mpl.cm.binary(0.8,0.5),
                                       ms=15, ls="None",mew=1.5, marker="o", zorder=5), **kwargs)
        ax.plot(self.datafitted["usedlines"], self.datafitted["fit_linepos"], **prop)
        er = ax.errorscatter(self.datafitted["usedlines"], self.datafitted["fit_linepos"],
                                 dy=self.datafitted["fit_linepos.err"], zorder=prop["zorder"]-1,
                             ecolor=ecolor)

        if xrange is None:
            xrange = [self.datafitted["usedlines"].min()-100, self.datafitted["usedlines"].max()+100]
        x = np.linspace(xrange[0],xrange[1], 1000)
        ml = ax.plot(x, self.lbda_to_pixels(x), lw=2, color="C1")
        
        fig.figout(savefile=savefile, show=show)
        
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
    
    @property
    def datafitted(self):
        """ """
        return self._side_properties["datafitted"]
    
    # - SpaxelWaveSolution
    def has_wavesolution(self):
        """ Test if the wavelength solution has been set. """
        return self._wavesolution is not None
    
    
    def lbda_to_pixels(self, wavelength):
        """ numpy.poly1d by the best fitted wavelength solution fit.

        Parameters
        ----------
        wavelength: [float or array]
            Wavelength in Angstrom

        Returns
        -------
        pixel (or array of)

        Example:
        --------
        self.wavesolution(6500) provides the pixel 
        corresponding to the 6500 Angstrom wavelength
        """
        return self._wavesolution(wavelength-REFWAVELENGTH)

    def load_pixel_to_lbda_solution(self):
        """ """
        if _HASPYNVERSE:
            self._properties["inverse_wavesolution"] = inversefunc(self.lbda_to_pixels)
        else:
            raise ImportError("You need pynverse | pip install pynverse")
            
    @property
    def pixels_to_lbda(self):
        """ Convert Pixels to Lambda 

        Example:
        --------
        self.inverse_wavesolution(1880) provides the wavelength (Angstrom)
        corresponding to the 1880th pixel
        """
        if self._properties["inverse_wavesolution"] is None:
            self.load_pixel_to_lbda_solution()

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
        if self.arcname in ["Hg"]:
            wavemax = np.max(self.lbda[self.get_arg_maxflux(2)])
        elif self.arcname in ["Xe"]:
            wavemax = np.min(self.lbda[self.get_arg_maxflux(2)])
        else:
            wavemax = self.lbda[self.get_arg_maxflux(1)]
          
          
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
        """ fit the 'pixel<->wavelength' solution 
        
        Parameters
        ----------
        wavesolution_degree: [degree] -optional-
            degree of the polynom used to fit the 'pixel<->wavelength' solution

        legendre: [bool] -optional-
            should the fitted polynom be based on Legendre polynoms?

        **kwargs goes to fit_lineposition()
            
            contdegree: [int] -optional-
                Degree of the (Legendre) polynom used as continuum

            line_shift: [float] -optional-
                Force the expected line pixel shift used as first guesses. 
                If not provided an internal procedure based on expected and observed 
                brightest emission position will be used. 
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
        Void
        """
        from modefit import get_polyfit
        
        if not self.has_linefitter():
            self.fit_lineposition(**kwargs)
            
        mus, emus = self._linefit_to_mus_()
        self._derived_properties["solutionfitter"] = get_polyfit( self.usedlines-REFWAVELENGTH, mus, emus+0.2,
                                                                      wavesolution_degree,
                                                                      legendre=legendre)
        
        guesses = {"a0_guess":np.nanmean(mus)}
        if not legendre:
            if wavesolution_degree>=2:
                guesses["a1_guess"]= 0.05
                
        self.guesses = guesses.copy()
        self.solutionfitter.fit(**guesses)
        self.datafitted = [self.usedlines, mus, emus]
        # - Set the best fit solution
        self._derived_properties["wavesolution"] = \
          SpaxelWaveSolution([self.solutionfitter.fitvalues["a%i"%i]
                        for i in range(self.solutionfitter.model.DEGREE)[::-1]],
                            datafitted=self.datafitted)

    def _load_lineposition_(self, contdegree=2, line_shift=None,
                             exclude_reddest_part=True,
                             red_buffer=30,
                             exclude_bluest_part=True,
                             blue_buffer=30, line_to_skip=None
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
            Force the expected line pixel shift used as first guesses. 
            If not provided an internal procedure based on expected and observed 
            brightest emission position will be used. 
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
        
        # where to look at? (~1ms)
        flagin = (self.lbda>=self.databounds[0])  * (self.lbda<=self.databounds[1]) # 1ms

        # Building guess (~1ms)
        self._normguesses = {}
        if line_shift is None:
            lines_shift = self.get_line_shift()
            
        for i,l in enumerate(self.usedlines):
            self._normguesses["ampl%d_guess"%i]      = self.arclines[l]["ampl"]
            self._normguesses["ampl%d_boundaries"%i] = [self.arclines[l]["ampl"]*0.2, self.arclines[l]["ampl"]*3]
            
            self._normguesses["mu%d_guess"%i]        = self.arclines[l]["mu"]+lines_shift
            self._normguesses["mu%d_boundaries"%i]   = [self._normguesses["mu%d_guess"%i]-2, self._normguesses["mu%d_guess"%i]+2]
            
            self._normguesses["sig%d_guess"%i]       = 1.1 if (not "doublet" in self.arclines[l] or not self.arclines[l]["doublet"]) else 1.8
            self._normguesses["sig%d_boundaries"%i]  = [0.9,1.5] if (not "doublet" in self.arclines[l] or not self.arclines[l]["doublet"]) else [1.1, 3]

        # where do you wanna fit? (~1ms)
        if exclude_reddest_part:
            flagin *= (self.lbda<=self._normguesses["mu%d_guess"%(len(self.usedlines)-1)]+red_buffer)
        else:
            warnings.warn("part redder than %d *not* removed"%(self._normguesses["mu%d_guess"%(len(self.usedlines)-1)]+red_buffer))
            
        if exclude_bluest_part:
            flagin *= (self.lbda>=self._normguesses["mu0_guess"]-blue_buffer)
        else:
            warnings.warn("part bluer than %d *not* removed"%(self._normguesses["mu0_guess"]-blue_buffer))

        # Setup the linefitter (3ms)
        norm = np.nanmean(self.flux[flagin])
        lbdas = self.lbda[flagin].copy()
        errors = self.errors[flagin]/norm if self.has_errors() else np.nanstd(self.flux[flagin])/norm/5.
        
        self._derived_properties["linefitter"] = \
          get_normpolyfit(lbdas,self.flux[flagin]/norm,
                              errors,
                              contdegree, ngauss=len(self.usedlines), legendre=True)

    def fit_lineposition(self, contdegree=2, line_shift=None,
                             exclude_reddest_part=True,
                             red_buffer=30,
                             exclude_bluest_part=True,
                             blue_buffer=30, line_to_skip=None
                             ):
        # VirtualArcSpectrum
        """ Fit gaussian profiles of expected arclamp emmisions.
        The list of fitted lines are given in `usedlines`.
        The fitter object is stored as `linefitter`.
        
        - This method uses `modefit` -

        Pamameters
        ----------
        contdegree: [int] -optional-
            Degree of the (Legendre) polynom used as continuum

        line_shift: [float] -optional-
            Force the expected line pixel shift used as first guesses. 
            If not provided an internal procedure based on expected and observed 
            brightest emission position will be used. 
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
        
        if not self.has_linefitter():
            self._load_lineposition_(contdegree=contdegree,
                                         line_shift=line_shift,
                                         exclude_reddest_part=exclude_reddest_part,
                                         red_buffer=red_buffer,
                                         exclude_bluest_part=exclude_bluest_part,
                                         blue_buffer=blue_buffer, line_to_skip=line_to_skip)
        # The actual fit ~4s
        self._normguesses["a0_guess"] = np.percentile(self.linefitter.data, 25)

        self.linefitter.fit( **self._normguesses )

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
    def data(self):
        """ Foundamental Information about the fit """
        mus,emus = self._linefit_to_mus_() if self.has_linefitter() else [None,None]
        if self.has_linefitter():
            fitvalue = self.linefitter.fitvalues if not self._sequentialfit \
              else {lamp:self.linefitter[lamp].fitvalues for lamp in self.arcnames}
        else:
            fitvalue = None
        return {"lampname": self.arcnames,
                "usedlines": self.usedlines,
                "fit_linepos":mus,"fit_linepos.err":emus,
                "wavesolution": self.wavesolution.data if self.has_wavesolution() else None,
                "line_fitvalues":fitvalue
                    }

    def set_databounds(self, xmin, xmax):
        """ """
        self._properties["databounds"] = [xmin, xmax]
        
    @property
    def databounds(self):
        """ limits of the effective spectrum """
        return self._properties["databounds"]

    # -----------
    # - Lines
    @property
    def usedlines(self):
        """ Wavelengthes used for the wavelength matching """
        return np.sort(list(self.arclines.keys()))
    
    @property
    def expected_brightesline(self):
        """ lines that has the greatest amplitude in the LINES global variable """
        l, amp = np.asarray([[l,v["ampl"]] for l,v in self.arclines.items()]).T
        return l[np.argmax(amp)]

    # - SpaxelWaveSolution
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
    
class ArcSpectrum( Spectrum, VirtualArcSpectrum ):
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
            self.reload_lines()

    def remove_line(self, line):
        """ """
        self.arclines.pop(line)

    def reload_lines(self):
        """ """
        self._derived_properties["arclines"] = LINES[self.arcname].copy()
        
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

    @property
    def flux(self):
        """ """
        return self.data
    
    @property
    def errors(self):
        """ """
        return np.sqrt(self.variance) if self.has_variance() else None
    
    def has_errors(self):
        """ """
        return self.errors is not None

###########################
#                         #
#  Collection of Arcs     #
#                         #
###########################
class ArcSpectrumCollection( VirtualArcSpectrum ):
    """  """
    PROPERTIES = ["arcspectra"]
    DERIVED_PROPERTIES = ["arclines", "lbda"]
    
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
        # ArcSpectrumCollection
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
        self._build_arclines_()
        
        lineprop = dict(contdegree=contdegree, line_shift=line_shift,
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
        return self.arclines[line]["arcname"]
    
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


    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, savefile=None, **kwargs):
        """ """
        if self.has_linefitter() and self.has_wavesolution():
            self._show_full_(savefile=savefile, **kwargs)
        elif self.has_wavesolution():
            self._show_wavesolution_(savefile=savefile, **kwargs)
        elif self.has_linefitter():
            self._show_linefit_(savefile=savefile, **kwargs)
        else:
            raise AttributeError("No SpaxelWaveSolution, Not LineFitter. Nothing to show")

    def _show_full_(self, savefile=None, show=True,
                    fig=None, stampsloc="right", traceindex=None,
                        show_guesses=False, **kwargs):
        """ """
        from astrobject.utils.mpladdon import figout
        if fig is None:
            fig = mpl.figure(figsize=[10,5]) if stampsloc in ["left", "right"] \
              else mpl.figure(figsize=[10,10])

        if stampsloc == "right":
            heigth, bottom = 0.82, 0.12
            space = 0.11
            axwave   = fig.add_axes([0.1,bottom,0.58,heigth])
            axstamps = [fig.add_axes([0.7,bottom+((heigth+space/(self.nspectra-1))/self.nspectra)*i,
                                      0.2, (heigth-space)/self.nspectra])
                            for i in range(self.nspectra)]
                
            self._show_linefit_(show=False, savefile=None, ax=axstamps, remove_xticks=False,
                                    show_guesses=show_guesses)
            self._show_wavesolution_(show=False, savefile=None, ax=axwave, **kwargs)

            axwave.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", fontsize="large")
            axwave.set_ylabel(r"Pixels (ccd-row)", fontsize="large")
            axstamps[0].set_xlabel(r"Pixels (ccd-row)", fontsize="medium")
            if traceindex is not None:
                fig.text(0.01,0.99,"Spectrum #%d"%traceindex,
                        va="top",ha="left", fontsize="small")

        fig.figout(savefile=savefile, show=show)
        
    def _show_linefit_(self, savefile=None, show=True, ax=None,
                           show_gaussian=True, show_guesses=False,
                           remove_xticks=False, **kwargs):
        """ """
        from astrobject.utils.mpladdon import figout
        if not self._sequentialfit:
            # - Non Sequential
            if ax is None:
                fig = mpl.figure(figsize=[8,5])
                ax = fig.add_axes([0.12,0.12,0.78,0.78])
                ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", fontsize="large")
                ax.set_ylabel(r"Flux []", fontsize="large")
            else:
                fig = ax.figure

            self.linefitter.show(ax=ax, savefile=None, show=False, show_gaussian=show_gaussian, **kwargs)
            if remove_xticks:
                ax.set_xticks([])
            if show_guesses:
                    _ = [ax.axvline(self.linefitter[lamp].param_input["mu%s_guess"%i], color="0.8", lw=1) 
                         for i,l in enumerate(self.usedlines)]
        else:
            # - Sequential
            if ax is None:
                fig = mpl.figure(figsize=[10,3])
                ax = [fig.add_subplot(1,self.nspectra,i+1) for i in range(self.nspectra)]
            elif len(ax) != self.nspectra:
                raise ValueError("In Sequential fit, ax must be an array with the size of `nspectra`")
            else:
                fig = ax[0].figure
                
            for ax_, lamp in zip(ax, self.arcnames):
                self.linefitter[lamp].show(ax=ax_, savefile=None, show=False,
                                        show_gaussian=show_gaussian,**kwargs)
                color = self._lamp_to_color_(lamp)
                [[s_.set_color(color),s_.set_linewidth(1.)] for s_ in ax_.spines.values()]
                ax_.text(0.04,0.96, lamp, transform=ax_.transAxes,
                         va="top", ha="left", fontsize="medium")
                ax_.set_yticks([])
                if remove_xticks:
                    ax_.set_xticks([])
                if show_guesses:
                    _ = [ax_.axvline(self.linefitter[lamp].param_input["mu%s_guess"%i], color="0.8", lw=1) 
                         for i,l in enumerate(self.arcspectra[lamp].usedlines)]

                
        fig.figout(savefile=savefile, show=show)
            
    def _show_wavesolution_(self, savefile=None, show=True, ax=None,
                                show_legend=True, show_model=True,
                                ecolor="0.3", xrange=None,**kwargs):
        """ """
        from astrobject.utils.mpladdon import figout, errorscatter
        from astrobject.utils.tools    import kwargs_update
        
        self._plot = {}
        if ax is None:
            fig = mpl.figure(figsize=[8,5])
            ax = fig.add_axes([0.12,0.12,0.78,0.78])
            ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", fontsize="large")
            ax.set_ylabel(r"Pixels (ccd-row)", fontsize="large")
        else:
            fig = ax.figure
            
        # -------
        # - Data
        prop = kwargs_update( dict(ms=15, ls="None",mew=1.5, marker="o", zorder=5), **kwargs)
        mu,emus = self._linefit_to_mus_()
        # Fancy coloring based on lamps
        for i,lamp in enumerate(self.arcnames[::-1]):
            flagin = np.asarray([lamp in self.line_to_lamp(l)  for l in self.usedlines], dtype="bool")
            ax.plot(self.usedlines[flagin], mu[flagin],
                        mfc= self._lamp_to_color_(lamp, 0.5),mec= self._lamp_to_color_(lamp, 0.9),
                        label=r"%s"%lamp,**prop)

        er = ax.errorscatter(self.usedlines, mu, dy=emus, zorder=prop["zorder"]-1,
                             ecolor=ecolor)
        # --------
        # - Model
        if show_model:
            if xrange is None:
                xrange = [self.usedlines.min()-100, self.usedlines.max()+100]
            x = np.linspace(xrange[0],xrange[1], 1000)
            ml = ax.plot(x, self.lbda_to_pixels(x), lw=2, color="C1")
        # --------
        # - Output
        if show_legend:
            ax.legend(loc="upper left", frameon=False, fontsize="medium",
                          markerscale=0.6)

        fig.figout(savefile=savefile, show=show)

    def _lamp_to_color_(self, lamp, alpha=1):
        """ Internal Ploting tool to get a consistent color for the lamp.
        Slower to use that, but prettier."""
        from pysedm.utils.mpl import get_lamp_color
        return get_lamp_color(lamp, alpha=alpha)
        
    # ================ #
    #  Properties      #
    # ================ #
    # -----------
    # Fake Spectrum
    @property
    def lbda(self):
        """ """
        if self._derived_properties['lbda'] is None and self.has_spectra():
            self._derived_properties['lbda'] = list(self.arcspectra.values())[0].lbda
        return self._derived_properties['lbda']
    
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
        return np.sort(list(self.arcspectra.keys()))
    
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
            self._build_arclines_()
        return self._derived_properties["arclines"]
    

    def _build_arclines_(self, rebuild=False):
        """ """
        if self._derived_properties["arclines"] is not None and not rebuild:
            return
        di={}
        for lampname,d in list(self.arcspectra.items()):
            for k,v in list(d.arclines.items()):
                if "backup" in v.keys() and v["backup"] in self.arcnames:
                    warnings.warn("line %s skipped since %s loaded"%(k,v["backup"]))
                    d.remove_line(k)
                    continue
                v["arcname"] = lampname
                di[k]=v
        self._derived_properties["arclines"] = di
        

   


    
