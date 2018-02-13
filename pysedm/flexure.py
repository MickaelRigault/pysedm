#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from propobject import BaseObject


class TraceFlexure( BaseObject ):
    """ """
    PROPERTIES = ["ccd", "mapper"]
    DERIVED_PROPERTIES = ["flagsep_used", "js", "js_obs"]

    def __init__(self, ccd, mapper=None):
        """ """
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
    def show_lbda_on_ccd(self, show_sepobjects=False, zoomin=[600, 1800, 1300, 1350],
                             show=True, savefile=None):
        """ """
        import matplotlib.pyplot as mpl
        if self._derived_properties['js'] is None:
            raise AttributeError("j offset not measured yet. see derive_j_offset()")
        
        fig    = mpl.figure(figsize=[10,3.5])
        axccd  = fig.add_axes([0.36,0.15,0.61,0.8])
        axhist = fig.add_axes([0.03,0.15,0.25,0.8])
        axccd.set_xlabel("i [ccd coordinates in pixels]", fontsize="large")
        axccd.set_ylabel("j [ccd coordinates in pixels]", fontsize="large")
        axhist.set_xlabel(r"$\Delta$ j for %d ellipses"%(len(self._flag_sepused[self._flag_sepused])),
                         fontsize="large")
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

        axhist.hist(self.delta_js[self.delta_js==self.delta_js])
        axhist.axvline(self.j_offset, ls="--",color="k")
        axhist.text(self.j_offset, 0.5, "j trace offset %+.2f pixels"%self.j_offset,
                   va="bottom",ha="right", rotation=90)
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
