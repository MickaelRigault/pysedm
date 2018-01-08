#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Fitting ADR """

import warnings
import numpy            as np
from scipy import odr
from pyifu.adr      import ADR

from propobject import BaseObject
from modefit.baseobjects import BaseModel, BaseFitter


def get_cube_adr_param(cube, lbdaref=5000, lbdastep=10, lbdarange=None,
                           show=True, savefile=None):
    """ """
    import shapely
    from . import extractstar
    
    x, y          = np.asarray( cube.index_to_xy(cube.indexes)).T
    indexref      = np.argmin(np.abs( cube.lbda-lbdaref))

    # = Building SubCube
    # Which Spaxels
    x0, y0, std0  = extractstar.guess_aperture(x, y, cube.data[indexref])
    used_indexes  = cube.get_spaxels_within_polygon(shapely.geometry.Point(x0,y0).buffer(std0*5))
    # Which Wavelength    
    slice_to_fit = range(len(cube.lbda))[::lbdastep]
    if lbdarange is not None and len(lbdarange)==2:
        flagin = (slice_to_fit>lbdarange[0]) & (slice_to_fit<=lbdarange[1])
        slice_to_fit = slice_to_fit[flagin]
    # Building it
    cube_partial = cube.get_partial_cube(used_indexes,slice_to_fit)
    cube_partial.load_adr()

    # = Fitting the Centroid:
    es = extractstar.ExtractStar(cube_partial)
    es.fit_psf("BiNormalCont")
    xc,yc = es.get_fitted_centroid()
    # 
    adrfit = ADRFitter(cube_partial.adr.copy())
    adrfit.set_data(cube_partial.lbda, xc[0], yc[0], xc[1], yc[1])
    adrfit.fit(unit_guess=0.4, unit_boundaries=[0.3,1])
    if show or savefile is not None:
        pl = adrfit.show(show=show, savefile=savefile, refsedmcube=cube_partial )
        
    return adrfit.fitvalues
    



class ADRFitter( BaseFitter ):
    """ """
    PROPERTIES         = ["adr","lbda",
                          "x", "y", "dx","dy"]
    DERIVED_PROPERTIES = []
    
    def __init__(self, adr, lbdaref=None):
        """ """
        self._properties['adr'] = adr
        if lbdaref is not None:
            self.adr.set(lbdaref=lbdaref)
            
        self.set_model( ADRModel(self.adr) )
    
    def set_data(self, lbda, x, y, dx, dy):
        """ set the fundatemental properties of the object. 
        These that will be used to the fit """
        self._properties['x']    = np.asarray(x)
        self._properties['y']    = np.asarray(y)
        self._properties['dx']   = np.asarray(dx)
        self._properties['dy']   = np.asarray(dy)
        
        self._properties['lbda'] = np.asarray(lbda)

        indexref = np.argmin(np.abs(self.lbda-self.adr.lbdaref))
        self.model.set_reference(self.lbda[indexref], self.x[indexref], self.y[indexref])
        
    def _get_model_args_(self):
        """ see model.get_loglikelihood"""
        return self.x, self.y, self.lbda, self.dx, self.dy

    # ---------- #
    #  PLOTTER   #
    # ---------- #
    def show(self, ax=None, savefile=None, show=True, cmap=None,
                 show_colorbar=True, clabel="Wavelength [A]",
                 refsedmcube=None, **kwargs):
        """ Plotting method for the ADR fit.
        
        Parameters
        ----------

        Returns
        -------
        """
        import matplotlib.pyplot as mpl
        from pyifu.tools import figout, insert_ax, colorbar
        if ax is None:
            fig = mpl.figure(figsize=[8,5])
            ax  = fig.add_subplot(111)
            ax.set_xlabel("spaxels x-axis")
            ax.set_xlabel("spaxels y-axis")
        else:
            fig = ax.figure
            
        # - Colors
        if cmap is None:
            cmap = mpl.cm.viridis
        vmin, vmax = np.nanmin(self.lbda),np.nanmax(self.lbda)
        colors = cmap( (self.lbda-vmin)/(vmax-vmin) )

        # - data
        scd = ax.scatter(self.x, self.y, facecolors=colors, edgecolors="None",
                       lw=1., label="data", **kwargs)
        # - error
        if self.dx is not None or self.dy is not None:
            ax.errorscatter(self.x, self.y, dx=self.dx, dy=self.dy,
                            ecolor="0.7", zorder=0)
        # - model    
        xmodel, ymodel = self.model.get_model(self.lbda)
        scm = ax.scatter(xmodel, ymodel, edgecolors=colors, facecolors="None",
                       lw=2., label="model", **kwargs)
        # - reference [if any]
        if refsedmcube is not None:
            xexp, yexp = refsedmcube.get_source_position(self.lbda, xref=self.fitvalues['xref'],
                                                         yref=self.fitvalues['yref'])
            ax.scatter(xexp, yexp, marker="s", alpha=0.5,
                       edgecolors=colors, facecolors="None",
                       label="reference \n[fixed x,y,lbda ref]")
            
            
        ax.legend(loc="best", frameon=True, ncol=2)
        ax.text(0.5,1.01, " ; ".join(["%s: %.2f"%(k,self.fitvalues[k]) for k in self.model.FREEPARAMETERS]) + " | %s: %.1f"%("lbdaref",self.model.adr.lbdaref),
                    transform=ax.transAxes, va="bottom", ha="center")
        if show_colorbar:
            axc = ax.insert_ax("right", shrunk=0.89)
            axc.colorbar(cmap, vmin=vmin, vmax=vmax,
                        label=clabel, fontsize="large")
            
        fig.figout(savefile=savefile, show=show)
        return {"ax":ax, "fig":fig, "plot":[scd,scm]}
    
            
    
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def adr(self):
        """ """
        return self._properties['adr']
    
    @property
    def x(self):
        """ x-positions """
        return self._properties['x']

    @property
    def y(self):
        """ y-positions """
        return self._properties['y']
    
    @property
    def dx(self):
        """ x-position errors """
        return self._properties['dx']
    
    @property
    def dy(self):
        """ y-position errors """
        return self._properties['dy']

    
    @property
    def lbda(self):
        """ wavelength [A] """
        return self._properties['lbda']
    


    
class ADRModel( BaseModel):
    """ """
    PROPERTIES      = ["adr", "lbdaref"]
    SIDE_PROPERTIES = [] # could be moved to parameters
    FREEPARAMETERS  = ["parangle", "unit", "xref", "yref"]

    parangle_boundaries = [-180, 180]
    
    def __init__(self, adr, xref=0, yref=0, base_parangle=0):
        """ """
        self.set_adr(adr)
        self._side_properties['xref'] = xref
        self._side_properties['yref'] = yref
        self._side_properties['base_parangle'] = base_parangle
        
    def setup(self, parameters):
        """ """
        self._properties["parameters"] = np.asarray(parameters)
        for i,p in enumerate(self.FREEPARAMETERS):
            if p == "unit":
                self._unit = parameters[i]
            elif p== "xref":
                self._side_properties['xref'] = parameters[i]
            elif p== "yref":
                self._side_properties['yref'] = parameters[i]
            else:
                self.adr.set(**{p:parameters[i]})

    def set_reference(self, lbdaref, xref=0, yref=0):
        """ use 'lbdaref=None' to avoid changing lbdaref """
        if lbdaref is not None:
            self.adr.set(lbdaref=lbdaref)
            
        self._side_properties['xref'] = xref
        self._side_properties['yref'] = yref
        
    def get_model(self, lbda):
        """ return the model for the given data.
        The modelization is based on legendre polynomes that expect x to be between -1 and 1.
        This will create a reshaped copy of x to scale it between -1 and 1 but
        if x is already as such, save time by setting reshapex to False

        Returns
        -------
        array (size of x)
        """
        return self.adr.refract(self.xref, self.yref, lbda, unit=self._unit)

    def get_loglikelihood(self, x, y, lbda, dx=None, dy=None):
        """ Measure the likelihood to find the data given the model's parameters.
        Set pdf to True to have the array prior sum of the logs (array not in log=pdf).

        In the Fitter define _get_model_args_() that should return the input of this
        """
        if dx is None: dx = 1
        if dy is None: dy = 1
        xadr, yadr = self.get_model(lbda)
        point_distance = ((x-xadr)/dx)**2 + ((y-yadr)/dy)**2
        return -0.5 * np.sum(point_distance)

    # ================= #
    #   Properties      #
    # ================= #
    def set_adr(self, adr):
        """ """
        if self._properties['lbdaref'] is not None:
            adr.set(lbdaref=lbdaref)
            
        self._properties['adr'] = adr

    @property
    def adr(self):
        """ pyifu ADR object """
        if self._properties['adr'] is None:
            self.set_adr( ADR() )
        return self._properties['adr']
            
        
    @property
    def lbdaref(self):
        """ reference wavelength of the ADR """
        return self._properties['lbdaref'] if self._properties['lbdaref'] is not None\
          else self.adr.lbdaref

    # - side properties
    @property
    def xref(self):
        """ x-position at the reference wavelength (lbdaref)"""
        return self._side_properties['xref']
    
    @property
    def yref(self):
        """ y-position at the reference wavelength (lbdaref)"""
        return self._side_properties['yref']
        


