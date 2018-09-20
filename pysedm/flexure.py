#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import warnings
from propobject import BaseObject


"""
j-offset flexure tool. "J-offsets" (perpendicular to trace dispersion) lead to lower signal to noise (more background, less signal).

j-offset is measured by maximizing the total flux within the randomly selected subsample of traces while shifting the traces up and down.
"""

########################
#                      #
#  Optimize            #
#                      #
########################
def get_ccd_jflexure(ccd, ntraces=100, tracewidth=1,
                         jscan=[-3,3,10], savefile=None, get_object=False):
    """ give a ccd object (with tracematch loaded) ; this estimate the ccd-j trace flexure.
    [takes about ~5s]

    Parameters:
    -----------
    ccd: [pysedm.CCD]
       ccd object 

    ntraces: [int] -optional-
        Number of randomly selected traces used for the estimation.

    tracewidth: [float] -optional-
        Width (buffer) around trace lines. Pixels contained within line position +/- width are considered as within the trace. 

    jscan: [float, float, int] -optional-
        start, end and step to be scanned in j (jscan-> np.linspace(*jscan))

    savefile: [string/None] -optional-
        If you want to save the figure, provide here it's name.
        If could be a list of filename (if you want several extention for instance)

    get_object: [bool] -optional-
        By default the function returns the j-shift (float).
        However, You can get the full TraceFlexureFit object by setting 'get_object' to True

    Returns
    -------
    flaot (ccd-j shift to apply) [or TraceFlexureFit, see get_object option]
    """
    from .sedm import INDEX_CCD_CONTOURS
    smap = ccd.tracematch.get_sub_tracematch( np.random.choice(ccd.tracematch.get_traces_within_polygon( INDEX_CCD_CONTOURS), ntraces) )
    smap.set_buffer(tracewidth)
    jflex = TraceFlexureFit(ccd, smap)
    jflex.build_pseudomag_scan(*jscan)
    if savefile is not None:
        for savefile_ in np.atleast_1d(savefile):
            jflex.show(savefile=savefile_)
        
    return jflex.estimate_jshift() if not get_object else jflex


class TraceFlexureFit( BaseObject ):
    """ """
    PROPERTIES = ["ccd", "tracematch"]
    DERIVED_PROPERTIES = ["pseudomag_scan"]
    def __init__(self, ccd, tracematch):
        """ """
        self._properties["ccd"] = ccd
        self._properties["tracematch"] = tracematch

    # ===================== #
    #   Method              #
    # ===================== #
    def get_total_pseudomag(self, jshift, subpixelization=2):
        """ """
        smap_current = self.tracematch.get_shifted_tracematch(0,jshift)
        weightmap = smap_current.get_traceweight_mask(subpixelization)
        return -np.log10( np.nansum(self.ccd.data*weightmap) )

        

    def fmin_jshift(self, guess):
        return fmin(self.get_total_pseudomag, guess)

    def build_pseudomag_scan(self, start=-3, end=3, step=10):
        """ """
        from scipy.interpolate import interp1d
        self.pseudomag_scan["js"] = np.linspace(start, end, step)
        self.pseudomag_scan["pseudomag"] = [self.get_total_pseudomag(jshift_) for jshift_ in self.pseudomag_scan["js"]]
        self.pseudomag_scan["interpolation"] = interp1d(self.pseudomag_scan["js"], self.pseudomag_scan["pseudomag"],
                                                            kind="cubic")
    def estimate_jshift(self, nbins=100):
        """ """
        xx = np.linspace(self.pseudomag_scan["js"].min(), self.pseudomag_scan["js"].max(), 100)
        return xx[np.argmin(self.pseudomag_scan["interpolation"](xx))]

    # ------------- #
    #  PLOTTER      #
    # ------------- #
    def show(self, savefile=None, ax=None):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[5,3])
            ax  = fig.add_axes([0.15,0.2,0.75,0.7])
        else:
            fig = ax.figure

        ax.plot( self.pseudomag_scan["js"], self.pseudomag_scan["pseudomag"], marker="o", label="data")
        xx = np.linspace(self.pseudomag_scan["js"].min(), self.pseudomag_scan["js"].max(),100)
        ax.plot(xx, self.pseudomag_scan["interpolation"](xx), color="C1", label="interpolation")
        
        # Result
        jshift = self.estimate_jshift()
        ax.axvline( jshift, color="0.5" )
        ax.text(jshift-0.1, np.max(self.pseudomag_scan["pseudomag"]), "j-flexure: %+.1f pixels"%jshift,
                    rotation=90, color="0.5", va="top", ha="right")

        
        # - labels
        ax.set_ylabel("Total CCD pseudo magnitude", fontsize="medium")
        ax.set_xlabel("ccd-j shift", fontsize="medium")
        ax.legend(loc="lower right", fontsize="small")
        if savefile is not None:
            fig.savefig(savefile, dpi=250)
        else:
            return {"ax":ax, "fig":fig}
        
    
    # ===================== #
    #   Properties          #
    # ===================== #
    @property
    def ccd(self):
        """ """
        return self._properties["ccd"]

    @property
    def tracematch(self):
        """ """
        return self._properties["tracematch"]

    @property
    def pseudomag_scan(self):
        """ """
        if self._derived_properties['pseudomag_scan'] is None:
            self._derived_properties['pseudomag_scan'] = {}
        return self._derived_properties['pseudomag_scan']
########################
#                      #
#  Class               #
#                      #
########################

class TraceFlexure( BaseObject ):
    """ """
    PROPERTIES = ["ccd", "mapper"]
    DERIVED_PROPERTIES = ["flagsep_used", "js", "js_obs"]

    def __init__(self, ccd, mapper=None):
        """ """
        print("DEPRECATED USED TraceFlexureFit")
        self.set_ccd(ccd)
        if mapper is not None:
            self.set_mapper(mapper)
        
    # ================== #
    #   Properties       #
    # ================== #
    def derive_j_offset(self, var_threshold=5, elliptical_limit=0.5, rerun_sepextract=False, verbose=False):
        """ """
        if not self.ccd.has_var():
            if verbose: print("INFO: setting the default variance to the ccd")
            self.ccd.set_default_variance()
        if not self.ccd.has_sepobjects() or rerun_sepextract:
            try:
                if verbose: print("INFO: Running  sep_extract")
                self.ccd.sep_extract()
            except:
                tresh = np.nanmean( np.sqrt(self.ccd.var) )*var_threshold
                if verbose: print("INFO: FAILED running sep_extract. Trying now using tresh=%.2f"%tresh)
                self.ccd.sep_extract(tresh)
                
        # - The SEP data
        a,b,x,y = self.ccd.sepobjects.get(["a","b",'x','y']).T
        self._derived_properties['flag_sepused'] = ((1- b/a)<elliptical_limit)

        # - Getting the expected J
        i_obs, j_obs = x, y
        if verbose: print("INFO: mesuring the expected j-coordinates")
        js= []
        for i_,j_ in zip(x, y):
            try:
                js.append( self.mapper.get_expected_j(i_,j_) )
            except:
                js.append(np.NaN)

        
        self._derived_properties['js']     = np.asarray(js)
        self._derived_properties['js_obs'] = np.asarray(j_obs)
        self._derived_properties['flag_sepused'] = ((1- b/a)<elliptical_limit) * ( self.js==self.js )
        
        if verbose: print("INFO: j-offset found: %+.2f pixels"%self.j_offset)
        return self.j_offset
            
    # ----------- #
    #  SETTER     #
    # ----------- #
    def set_ccd(self, ccd_):
        """  """
        self._properties["ccd"] = ccd_
        
    def set_mapper(self, mapper_):
        """ attach to this intance a mapper object containing the 
        ccd to cube coordinates conversions """
        self._properties["mapper"] = mapper_

    # ----------- #
    #  PLOTTER    #
    # ----------- #
    def show_j_flexure_ccd(self, show_sepobjects=False, zoomin=[600, 1800, 1300, 1350],
                             show=True, savefile=None):
        """ """
        import matplotlib.pyplot as mpl
        if self._derived_properties['js'] is None:
            _NO_JS = True
            warnings.warn("j offset not measured yet. see derive_j_offset()")
        else:
            _NO_JS = False

        
        fig    = mpl.figure(figsize=[10,3.5])
        axccd  = fig.add_axes([0.36,0.15,0.61,0.8])
        axhist = fig.add_axes([0.03,0.15,0.25,0.8])
        axccd.set_xlabel("i [ccd coordinates in pixels]", fontsize="large")
        axccd.set_ylabel("j [ccd coordinates in pixels]", fontsize="large")
        
        pl = self.ccd.show(ax=axccd, show=False)
        self.mapper.tracematch.display_traces(axccd, self.mapper.traceindexes, edgecolors="0.7", linestyle="--")
        self.ccd.tracematch.display_traces(axccd, self.mapper.traceindexes, edgecolors="w")
        
        if show_sepobjects:
            ells = self.ccd.sepobjects.get_detected_ellipses(scaleup=2.5, mask=self._flag_sepused,contours=False)
            for ell in ells:
                axccd.add_artist(ell)
                ell.set_clip_box(axccd.bbox)
                ell.set_facecolor("None")
                ell.set_edgecolor("r")
        if not _NO_JS:
            axhist.hist(self.delta_js[self.delta_js==self.delta_js])
            axhist.axvline(self.j_offset, ls="--",color="k")
            axhist.text(self.j_offset, 0.5, "j trace offset %+.2f pixels"%self.j_offset,
                    va="bottom",ha="right", rotation=90)
            axhist.set_xlabel(r"$\Delta$ j for %d ellipses"%(len(self._flag_sepused[self._flag_sepused])),
                         fontsize="large")
        else:
            axhist.text(0.5, 0.5, "j offset not measured yet. \n see derive_j_offset()",
                    va="center",ha="center", rotation=45, transform=axhist.transAxes)
            
        axccd.figure.canvas.draw()
        axccd.set_xlim(*zoomin[:2])
        axccd.set_ylim(*zoomin[2:])
        if show:
            fig.show()
        if savefile:
            fig.savefig(savefile)
    # ================== #
    #   Properties       #
    # ================== #
    @property
    def ccd(self):
        """ the ccd object """
        return self._properties["ccd"]
    @property
    def mapper(self):
        """ object containing the ccd to 3d cube conversion tools """
        return self._properties["mapper"]

    # - Derived properties
    @property
    def js_obs(self):
        """ the observed j positions """
        if self._derived_properties['js_obs'] is None:
            raise AttributeError("This has not been set. see derive_j_offset()")
        return self._derived_properties['js_obs']

    @property
    def js(self):
        """ the expected j positions """
        if self._derived_properties['js'] is None:
            raise AttributeError("This has not been set. see derive_j_offset()")
        return self._derived_properties['js']
    
    @property
    def delta_js(self):
        """ Offset between the expected j position and the observed onces """
        return self.js_obs - self.js
    
    @property
    def j_offset(self):
        """ """
        return np.nanmedian(self.delta_js)
    
    @property
    def _flag_sepused(self):
        """ """
        return self._derived_properties['flag_sepused']
