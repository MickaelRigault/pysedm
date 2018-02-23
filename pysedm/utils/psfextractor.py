#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as mpl

from pyifu.mplinteractive import InteractiveCube
from matplotlib.widgets import Button

from .. import io, sedm
from .tools import kwargs_update

def get_psf_shape(cube, xref, yref):
    """ """
    

def header_airmass_to_airmass(date):
    """ """
    psfparam = io.load_psf_param(date, ['MJD_OBS','AIRMASS'])

# ======================= #
#   PSF SHAPE ESTIMATOR   #
# ======================= #
from propobject import BaseObject
class PSFBuilder ( BaseObject):
    """ """
    PROPERTIES         = ["psfdatas"]
    DERIVED_PROPERTIES = ["airmass_model"]

    def __init__(self, psfdatas=None):
        """ """
        if psfdatas is not None:
            self.set_psfdatas(psfdatas)
            
    # ============= #
    #   Methods     #
    # ============= #

    # - PSF total
    def get_psf_property(self, header, xref=0, yref=0, lbdaref=None,
                             profile_model = "median"):
        """ Combining get_adr_property() with get_profile_property() """
        return {"adr": self.get_adr_property(header, xref=xref, yref=yref, lbdaref=lbdaref),
                "profile": self.get_profile_property(model=profile_model)}
    
    # -  ADR
    def get_adr_property(self, header, xref, yref, lbdaref=None):
        """ ADR porperty based on header """
        
        header_parangle = header["TEL_PA"]
        return {'airmass': self.headerairmass_to_airmass(header["AIRMASS"])[0],
                 'lbdaref': lbdaref if lbdaref is not None else np.unique(self.get_key("adr","lbdaref"))[0],
                 'parangle': (sedm.MLA_ROTATION_DEG + header_parangle)%360,
                 'parangle_ref':  header_parangle,
                 'pressure':      header["IN_AIR"],
                 'relathumidity': header["IN_HUM"],
                 'temperature':   header["IN_AIR"],
                 'unit': 0.75,
                 'xref': xref,
                 'yref': yref}
    # -  Profile
    def get_profile_property(self, model="median"):
        """ """
        if model in ["median"]:
            return self._get_median_profile_()
        raise NotImplementedError("only model = 'median' has been implemented.")
    
    def _get_median_profile_(self):
        """ """
        from astropy.stats import mad_std
        baseprop = {}
        for key in ["stddev_ratio","stddev_ref", "stddev_rho", "theta", "ell", "amplitude_ratio"]:
            value = self.get_key("profile",key)
            err   = self.get_key("profile",key+".err")
            
            baseprop[key]        = np.nanmedian(value )
            baseprop[key+".err"] = mad_std( value ) if len(value)>1 else err[0] if err is not None else 0
            
        # - stable
        baseprop["name"] = "BiNormalCont"
        return baseprop


    # -------------- #
    #  For Fitter    #
    # -------------- #
    def property_to_guess(self, property_dict, errscale=1.4, exclude=[]):
        """ method made to feed `modefit` based fitter. 
        it inputs property dictionary (adr_property or profile_property) 
        and uses the key values as guesses and the errors (times errscale) as boundaries.
        If no error or error are zero, the value is fixed.
        """
        modefit_guess = {}
        for k,v in property_dict.items():
            if ".err" in k or k in exclude: continue
            modefit_guess["%s_guess"%k] = v
            if k+".err" in property_dict and property_dict[k+".err"]>0:
                err = property_dict[k+".err"] * errscale
                modefit_guess["%s_boundaries"%k] = [v-err,v+err]
            else:
                modefit_guess["%s_fixed"%k] = True
                
        return modefit_guess        
    
    # ---------- #
    # Conversion #
    # ---------- #
    
    def headerairmass_to_airmass(self, header_airmass):
        """ """
        keymodel = self._get_key_model_("header.airmass","adr.airmass", 2)
        return keymodel.model.get_model(np.atleast_1d(header_airmass))
    
    # -------- #
    #  GETTER  #
    # -------- #
    def get_key(self, source, key, index=None):
        """ Get the value of the keys. 
        Source: which origin? (adr / header / profile)
        key: which key for the source?
        """
        if index is None:
            index = self.indexes
            
        try:
            return np.asarray([self.psfdatas[i][source][key] for i in index])
        except:
            return None

    def show_key_v_key(self, key1, key2, ckey=None, index = None, ax=None):
        """ """
        if index is None: index = self.indexes
        # ---------- #
        #   Keys     #
        # ---------- #
        source1,key1_ = key1.split(".")
        source2,key2_ = key2.split(".")
    
        v1,dv1 = self.get_key( source1, key1_, index), self.get_key(source1, key1_+".err", index)
        v2,dv2 = self.get_key( source2, key2_, index), self.get_key(source2, key2_+".err", index)

        if ckey is not None:
            sourcec, ckey_ = ckey.split(".")
            cv     = self.get_key( sourcec, ckey_, index) if ckey is not None else None
        else:
            cv = None

        # ---------- #
        #   PLOT     #
        # ---------- #
        if ax is None:
            fig = mpl.figure()
            ax  = fig.add_subplot(111)
            ax.set_ylabel(key2)
            ax.set_xlabel(key1)
        else:
            fig = ax.figure
        
        sc = ax.scatter(v1, v2, c=cv, zorder=3)
        ax.errorbar(v1, v2, xerr=dv1, yerr=dv2, 
                    ls="None", ms=0,
                    ecolor="0.6", zorder=2)
        if cv is not None:
            cbar = fig.colorbar(sc)
            cbar.set_label(ckey)
        
        return {"ax":ax, "fig":fig}

    # -------- #
    #  SETTER  #
    # -------- #
    def set_psfdatas(self, psfdatas):
        """ """
        self._properties["psfdatas"] = psfdatas

    def add_key(self, source, key, value, index=None):
        """ """
        if index is None: index = self.indexes
        if len(index) != len(value):
            raise ValueError("index and value do not have the same length (%d vs. %d)"%( len(index), len(value) ) )
        
        for index_, value_ in zip(index, value):
            if source not in self.psfdatas[index_]:
                self.psfdatas[index_]["source"] = {key: value_}
            else:
                self.psfdatas[index_][source][key] = value_
        
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def psfdatas(self):
        """ """
        return self._properties["psfdatas"]

    @property
    def indexes(self):
        """ """
        return list(self.psfdatas.keys())
    
    # - DERIVED

    # - airmass
    def _get_key_model_(self, xkey, ykey, degree=2):
        """ """
        from modefit import get_polyfit
        index  = self.indexes
        source1, value1 = xkey.split(".")
        source2, value2 = ykey.split(".")
        
        v1     = self.get_key(source1, value1, index)
        v2,dv2 = self.get_key(source2, value2, index), self.get_key(source2, value2+".err", index)
        
        if dv2 is None or None in dv2:
            warnings.warn("No error on y for key %s.%s"%(source2, value2))
            dv2 = np.ones(len(v2))

        model_adr = get_polyfit(v1, v2, dv2, 2)
        model_adr.fit()
        return model_adr
        
    @property
    def _airmass_model(self):
        """ """
        return self._derived_properties["airmass_model"]
    
# ======================= #
#    FOR PSF PICKER       #
# ======================= #

class ForcePSFPicker( InteractiveCube ):
    
    PROPERTIES = ["axtarget","aximres","axextract",
                  "axpicked", "psfmodel", "psfbuilder"]
        
    SIDE_PROPERTIES = ["_picking"]
    DERIVED_PROPERTIES = ["guess_positions","cuberes"]
    
    NPICKED = 6

    # =================== #
    #   Method            #
    # =================== #
    def reset(self):
        """ Clean the Axes and set things back to there initial values """
        if self._picking is not None and self._picking["ongoing"]:
            self._picking["text"].remove()
        self._side_properties["_picking"] = None
        [self._clear_axpicked_(ax_) for ax_ in self.axpicked]
        
        super(ForcePSFPicker,self).reset()
        
    # ------------- #
    #   Shape       #
    # ------------- #
    def set_psfbuilder(self, psfbuilder):
        """ """
        self._properties["psfbuilder"] = psfbuilder
        
    def set_figure(self, fig=None, **kwargs):
        """ attach a figure to this method. """
        if fig is None:
            figprop = kwargs_update(dict(figsize=[10,6]), **kwargs)
            self._properties["figure"] = mpl.figure(**figprop)
        elif matplotlib.figure.Figure not in fig.__class__.__mro__:
            raise TypeError("The given fig must be a matplotlib.figure.Figure (or child of)")
        else:
            self._properties["figure"] = mpl.figure(**figprop)
            
    def set_axes(self, axes=None, axpicked=None, **kwargs):
        """ """
        if axes is None and not self.has_figure():
            raise AttributeError("If no axes given, please first set the figure.")
        
        if axes is None:
            figsizes = self.fig.get_size_inches()
            width_spec, width_im = 0.42, 0.2
            left, left2 = 0.1, 0.05
            low, low2 = 0.12,0.1
            height = 0.35
            axspec = self.fig.add_axes([left, low+low2+height, width_spec, height])
            axim   = self.fig.add_axes([left+left2+width_spec, low +low2+height,width_im,height])

            axtarget  = self.fig.add_axes([left, low, width_spec,height])
            aximres   = self.fig.add_axes([left+left2+width_spec, low, width_im,height])
            
            for ax_ in [axspec,axtarget]:
                ax_.set_xlabel(r"Wavelength", fontsize="large")
                ax_.set_ylabel(r"Flux", fontsize="large")


            # - Picked
            axpicked = []
            span_picked = 0.02
            height_picked = (low2+height*2 - span_picked*(self.n_picking-1))/self.n_picking
            for i in range(self.n_picking):
                ax_ = self.fig.add_axes( [left+left2+width_spec+width_im+0.05, low+i*(height_picked+span_picked),
                                                     0.1, height_picked] )
                ax_.tick_params(bottom="off", top="off", left="off", right="off",
                                labelbottom="off", labeltop="off", labelleft="off", labelright="off")
                axpicked.append(ax_)
                
            # - Setting
            self.set_axes([axspec,axim, axtarget, aximres], axpicked)
            
        elif len(axes) != 4:
            raise TypeError("you must provide 4 axes [axspec,axim, axtarget, aximres] and both have to be matplotlib.axes(._axes).Axes (or child of)")
        else:
            # - actual setting
            self._properties["axspec"], self._properties["axim"], self._properties["axtarget"], self._properties["aximres"] = axes
            
            if axpicked is not None:
                self._properties["axpicked"] = axpicked
                bbox = self.axpicked[-1].get_position()
                
                self.fig.text(np.mean([bbox.xmin,bbox.xmax]), bbox.ymax, "Picking Thumbs",
                            va="bottom", ha="center")

            self._properties["axextract"] = self.fig.add_axes([0.7, 0.01, 0.08, 0.05])
            badr = Button(self.axextract, 'Extract')

            for ax in self.axpicked:
                ax.autoscale(False, tight=True)
                
            if not self.has_figure():
                self.set_figure(self.axspec.figure)
                
    # ================= #
    #   FITTER          #
    # ================= #
    def _derived_refpos_(self, axout = None):
        """ returns: parangle, airmass, xref, yref """
        from . import adrfit
        adr_property = self.psfbuilder.get_adr_property(self.cube.header, 0,0)

        axused    = [ax_ for ax_ in self.axpicked if "lbdarange" in self.guess_positions[ax_]]
        lbdas     = np.mean([ self.guess_positions[ax_]["lbdarange"] for ax_ in axused], axis=1)
        xpos,ypos = np.asarray([ self.guess_positions[ax_]["position"] for ax_ in axused]).T
        if len(axused) == 1:
            return adr_property["parangle"], adr_property["airmass"], xpos[0], ypos[0]
        
        if self.cube.adr is None:
            self.cube.load_adr()

        adrfitter = adrfit.ADRFitter(self.cube.adr, lbdaref= adr_property["lbdaref"])
        adrfitter.set_data(lbdas, xpos, ypos, np.ones(len(xpos))*0.33, np.ones(len(xpos))*0.33)
        adrfitter.fit(parangle_guess=adr_property["parangle"], parangle_fixed=True, parangle_boundaries= [0,360],
                      airmass_guess=adr_property["airmass"],   airmass_fixed=True, 
                      airmass_boundaries=[adr_property["airmass"],adr_property["airmass"]+0.2],
                      xref_guess = np.mean(xpos), xref_boundaries = [np.mean(xpos)-2,np.mean(xpos)+2],
                      yref_guess = np.mean(ypos), yref_boundaries = [np.mean(ypos)-2,np.mean(ypos)+2]
                    )
        if axout is not None:
            adrfitter.show(ax = axout, show_colorbar=False, labelkey=["parangle","airmass","xref","yref"])
            
        return [adrfitter.fitvalues[k] for k in ["parangle", "airmass", "xref", "yref"]]

    def get_picked_lbdas_and_positions(self):
        """ """
        axused    = [ax_ for ax_ in self.axpicked if "lbdarange" in self.guess_positions[ax_]]
        lbdas     = np.asarray([ self.guess_positions[ax_]["lbdarange"] for ax_ in axused])
        xpos,ypos = np.asarray([ self.guess_positions[ax_]["position"] for ax_ in axused]).T
        return lbdas, [xpos,ypos]
    
    def force_spectroscopy(self, picking_error=2.,
                            fix_airmass=False, fix_parangle=False,
                            parangle_error=60, savefig=True, record_data=True):
        """ """
        from . import extractstar
        # - Figure out
        if savefig:
            savefigure  = self.cube.filename.replace("e3d","psffit_manual_e3d").replace(".fits",".pdf")
        else:
            savefigure = None

        
        lbdas, centroids = self.get_picked_lbdas_and_positions()
        
        if len(self.selected_spaxels)>5:
            print("INFORMATION: fitting only the selected spaxels")
            cube_to_fit = self.cube.get_partial_cube(self.get_selected_idx(), np.arange(len(self.cube.lbda) ) )
        else:
            cube_to_fit = self.cube

        # Input from the psfbuilder
        adr_prop = self.psfbuilder.property_to_guess(self.psfbuilder.get_adr_property(cube_to_fit.header, 0, 0),
                                                         exclude=["xref","yref","pressure","relathumidity",
                                                                      "temperature","unit","lbdaref","parangle_ref"])
        adr_prop["airmass_fixed"] = fix_airmass
        adr_prop["airmass_boundaries"] = [1., 3.]

        adr_prop["parangle_fixed"] = fix_parangle
        adr_prop["parangle_boundaries"] = [adr_prop["parangle_guess"]-parangle_error,
                                               adr_prop["parangle_guess"]+parangle_error]
        
        # Step 1: Get the PSF Profile
        self.psfcube = extractstar.fit_psf_parameters(cube_to_fit, lbdas,
                                        centroid_guesses= np.asarray(centroids).T,
                                        centroid_errors = picking_error,
                                        savedata=None, savefig=savefigure, show=False,
                                        return_psfmodel=False,
                                        # ADR inpout
                                        adr_prop=adr_prop, allow_adr_trials=False)

        psfmodel = extractstar.get_psfmodel(self.psfcube.fitted_data)
        
        # Step 1: Force PSF
        if savefigure is not None:
            savefigure = savefigure.replace("psffit_manual_e3d","forcepsf_manual_e3d")
        self.spec, self.bkgd, self.forcepsf = extractstar.fit_force_spectroscopy(cube_to_fit, psfmodel,
                                                                    savefig=savefigure, show=False)
        
        
        # - Show Spectra
        self.axtarget.cla()
        self.aximres.cla()

        
        pl = self.spec.show(ax=self.axtarget, label="ForcePSF spectrum", show=False)
        _  = self.bkgd.show(ax=self.axtarget, color="C1", label="background", show=False)
        self.axtarget.legend(loc="best", fontsize="small")

        # - Show Residual Cube
        self._derived_properties["cuberes"] = self.forcepsf.cuberes
        self._resspaxels = self.cuberes._display_im_(self.aximres, interactive=True,
                                                      linewidth=0, edgecolor="0.7")
        self.aximres.set_title("Residual Cube")

        if record_data:
            io._saveout_forcepsf_(self.cube.filename, self.cube,
                                self.forcepsf.cuberes, self.forcepsf.cubemodel, self.spec, self.bkgd, mode="manual")
        self._draw_()
        
        
    # ================= #
    #  Changed          #
    # ================= #
    def interact_enter_axis(self, event):
        """ """
        super(ForcePSFPicker, self).interact_enter_axis(event)
        
        if event.inaxes in self.axpicked+[self.axextract] and not self._nofancy:
            # - change axes linewidth
            [s_.set_linewidth(self.property_backup[event.inaxes]["default_axedgewidth"][i]*2)
                 for i,s_ in enumerate(event.inaxes.spines.values())]

            self.fig.canvas.draw()


    def interact_leave_axis(self, event):
        """ """
        super(ForcePSFPicker, self).interact_leave_axis(event)
        
        if event.inaxes in self.axpicked+[self.axextract] and not self._nofancy:
            # - change axes linewidth            
            [s_.set_linewidth(self.property_backup[event.inaxes]["default_axedgewidth"][i])
                 for i,s_ in enumerate(event.inaxes.spines.values())]
            self.fig.canvas.draw()

    def change_axim_color(self, lbdalim):
        """ """
        colors = self.cube._data_to_color_(self.toshow, lbdalim=lbdalim)
        [s.set_facecolor(c) for s,c in zip(self._spaxels, colors)]
        if self.cuberes is not None:
            colors = self.cuberes._data_to_color_(lbdalim=lbdalim)
            [s.set_facecolor(c) for s,c in zip(self._resspaxels, colors)]
            
    # - on Click
    def interact_onclick(self, event):
        """ """
        if self.fig.canvas.manager.toolbar._active is not None:
            return
        
        if self._picking is None or not self._picking["ongoing"]:
            super(ForcePSFPicker, self).interact_onclick(event)

        if event.inaxes == self.axextract:
            self.force_spectroscopy(parangle_error=180, fix_parangle=False)
            
        # - Picking Axes
        if event.inaxes in self.axpicked:
            self._start_picking_process_(event)
            
        # - Getting the wavelength
        elif event.inaxes  == self.axspec:
            self._onclick_axspec_(event)
        elif event.inaxes == self.axim:
            if self._picking["step"] == "lbdapicked":
                self._complet_picking_(event)
            else:
                super(ForcePSFPicker, self).interact_onclick(event)
                

    def interact_onrelease(self, event):
        """ """
        if self.fig.canvas.manager.toolbar._active is not None:
            return
        
        if self._picking is None or not self._picking["ongoing"]:
            super(ForcePSFPicker, self).interact_onrelease(event)
            
        elif self._picking["step"] == "completed":
            self._record_picking_()
            
        # - Picking Axes
        #if event.inaxes in self.axpicked:
        #    pass
            
        # - Getting the wavelength
        elif event.inaxes  == self.axspec:
            self._lbda_picking_process_(event)

        

    # ACTUAL event
    def _clean_axpicked_(self, ax=None):
        """ """
        if ax is None: ax = self.axpicked
        for ax_ in np.atleast_1d(ax):
            [s_.set_color("k") for i,s_ in enumerate(ax_.spines.values()) ]

    def _clear_axpicked_(self, ax):
        """ """
        ax.cla()
        self._clean_axpicked_(ax)
        self.guess_positions[ax] = {}
        
    # Start
    def _start_picking_process_(self, event):
        """ """
        self._clear_axpicked_(event.inaxes)
        [s_.set_color("C2") for i,s_ in enumerate(event.inaxes.spines.values()) ]
        self._side_properties["_picking"]  = \
          {"ongoing": True,
           "ax":      event.inaxes,
           "step":    "axpicked",
           "text":    self.fig.text(0.05,0.985, "Now Pick wavelength", va="top",ha="left")}
        
        
        self._draw_()
    # LBDA picking
    def _lbda_picking_process_(self, event):
        """ """
        self._picking["text"].remove()
        self._picking["step"] = "lbdapicked"
        self._picking["text"] = self.fig.text(0.05,0.985, "Now a position on the imshow",
                                                  va="top",ha="left")
        self._onrelease_axspec_(event, draw=False)
        self._picking["lbdarange"] = self._selected_wavelength
        self._picking["ax"].set_ylabel(r"$\lambda\ [$%.1f ; %.1f]"%(self._selected_wavelength[0],self._selected_wavelength[1]),
                                           fontsize="small")
        self._draw_()
        
    # Complet
    def _complet_picking_(self, event):
        """ """
        self._picking["position"] = [event.xdata, event.ydata]
        slice_ = self.cube.get_slice()
        self.cube._display_im_(self._picking["ax"], lbdalim=self._picking["lbdarange"])
        
        self._picking["ax"].plot(event.xdata, event.ydata, marker="x", color="C1", lw=2, ms=10)
        
        self._picking["ax"].set_xlim(event.xdata-5, event.xdata+5)
        self._picking["ax"].set_ylim(event.ydata-5, event.ydata+5)
        
        self._picking["ax"].set_autoscale_on(True)
        
        self._picking["step"]    = "completed"
        self._picking["text"].remove()
        self._clean_axpicked_([self._picking["ax"]])
        
        self._draw_()

    def _record_picking_(self):
        """ """
        self.guess_positions[self._picking["ax"]] = \
          {"position":self._picking["position"],
            "lbdarange":self._picking["lbdarange"]}
        
        self._side_properties["_picking"] = None
        
        
    # ================= #
    # Properties        #
    # ================= #
    # - AXES
    @property
    def axtarget(self):
        """ """
        return self._properties["axtarget"]

    @property
    def aximres(self):
        """ """
        return self._properties["aximres"]

    # - Button axes
    @property
    def axextract(self):
        """ """
        return self._properties["axextract"]
    
    # - Ax picked
    @property
    def axpicked(self):
        """ """
        return self._properties["axpicked"]

    @property
    def n_picking(self):
        """ """
        return self.NPICKED


    @property
    def _picking(self):
        """ """
        return self._side_properties["_picking"]

    @property
    def guess_positions(self):
        """ """
        if self._derived_properties["guess_positions"] is None:
            self._derived_properties["guess_positions"] = \
              {ax_:{} for ax_ in self.axpicked}
              
        return self._derived_properties["guess_positions"]
    
    # - PSF Profile
    @property
    def psfbuilder(self):
        """ The psfbuilder """
        return self._properties["psfbuilder"]
    
    @property
    def psfmodel(self):
        """ The PSF model """
        return self._properties["psfmodel"]
    
    @property
    def cuberes(self):
        """ """
        return self._derived_properties["cuberes"]
