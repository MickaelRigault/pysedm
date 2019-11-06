#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy         as np
import matplotlib.pyplot as mpl
from .tools import make_method
from pyifu.mplinteractive import InteractiveCube
LAMPS = np.asarray(["Cd","Hg","Xe"])

try:
    CMAP = mpl.cm.viridis
except:
    warnings.warn("You should update your matplotlib. viridis does not exist...")
    CMAP = mpl.cm.PuBu



__all__ = ["get_lamp_color", "figout"]
    
def get_lamp_color(lamp, alpha):
    """ """
    i = float(np.argwhere(LAMPS == lamp)[0][0])/len(LAMPS)
    return CMAP(i, alpha)


@make_method(mpl.Figure)
def figout(fig,savefile=None,show=True,add_thumbnails=False,
           dpi=200):
    """This methods parse the show/savefile to know if the figure
    shall the shown or saved."""
    
    if savefile in ["dont_show","_dont_show_","_do_not_show_"]:
        show = False
        savefile = None

    if savefile is not None:
        if not savefile.endswith(".pdf"):
            extention = ".png" if not savefile.endswith(".png") else ""
            fig.savefig(savefile+extention,dpi=dpi)
            
        if not savefile.endswith(".png"):
            extention = ".pdf" if not savefile.endswith(".pdf") else ""
            fig.savefig(savefile+extention)
            
        if add_thumbnails:
            fig.savefig(savefile+"_thumb"+'.png',dpi=dpi/10.)
            
    elif show:
        fig.canvas.draw()
        fig.show()

@make_method(mpl.Axes)        
def set_axes_edgecolor(ax, color,  ticks=True, labels=False):
    """ """
    import matplotlib
    prop = {}
    if ticks:
        prop["color"] = color
        prop["which"] = "both"
    if labels:
        prop["labelcolor"] = color
    ax.tick_params(**prop)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color(color)

###############################
#                             #
# SEDMachine Plotting special #
#                             #
###############################
class InteractiveCubeandCCD( InteractiveCube ):
    PROPERTIES = ["ccd", "axccd"]

    def launch(self, *args, **kwargs):
        """ """
        
        self._ccdshow = self.ccdimage.show(ax = self.axccd, aspect="auto", show=False)["imshow"]
        super(InteractiveCubeandCCD, self).launch(*args, **kwargs)

    def set_to_origin(self):
        """ Defines the default parameters """
        super(InteractiveCubeandCCD, self).set_to_origin()
        self._trace_patches = []

    def reset(self):
        """ Set back everything to origin """
        super(InteractiveCubeandCCD, self).reset()
        self.clean_axccd(update_limits=True)
        
    # =========== #
    #  SETTER     #
    # =========== #
    def set_ccd(self, ccd):
        """ """
        self._properties["ccd"] = ccd
        
    def set_figure(self, fig=None, **kwargs):
        """ attach a figure to this method. """
        if fig is None:
            figprop = kwargs_update(dict(figsize=[10,7]), **kwargs)
            self._properties["figure"] = mpl.figure(**figprop)
        elif matplotlib.figure.Figure not in fig.__class__.__mro__:
            raise TypeError("The given fig must be a matplotlib.figure.Figure (or child of)")
        else:
            self._properties["figure"] = mpl.figure(**figprop)

    def set_axes(self, axes=None, **kwargs):
        """ """
        if axes is None and not self.has_figure():
            raise AttributeError("If no axes given, please first set the figure.")
        
        if axes is None:
            figsizes = self.fig.get_size_inches()
            axspec = self.fig.add_axes([0.10,0.6,0.5,0.35])
            axim   = self.fig.add_axes([0.65,0.6,0.75*(figsizes[1]*0.5)/float(figsizes[0]),0.35])

            axccd  = self.fig.add_axes([0.10,0.10,axim.get_position().xmax- 0.1, 0.4])
            
            axspec.set_xlabel(r"Wavelength", fontsize="large")
            axspec.set_ylabel(r"Flux", fontsize="large")
            self.set_axes([axspec,axim,axccd])
            
        elif len(axes) != 3:
            raise TypeError("you must provide 2 axes [axspec and axim] and both have to be matplotlib.axes(._axes).Axes (or child of)")
        else:
            # - actual setting
            self._properties["axspec"], self._properties["axim"], self._properties["axccd"] = axes
            if not self.has_figure():
                self.set_figure(self.axspec.figure)

    # ------------- #
    #  Show Things  #
    # ------------- #
    def show_picked_traces(self):
        """ """
        if not self._hold:
            self.clean_axccd()
            
        self._trace_patches = self.ccdimage.display_traces(self.axccd, self.get_selected_idx(),
                                                            facecolors="None", edgecolors=self._active_color)

    def clean_axccd(self, update_limits=False):
        """ """
        self.axccd.patches = []
        if update_limits:
            self.axccd.set_xlim(0, self.ccdimage.shape[0])
            self.axccd.set_ylim(0, self.ccdimage.shape[1])
        
    # ================= #
    # Change For CCD    #
    # ================= #
    def update_figure_fromaxim(self):
        """ What would happen once the spaxels are picked. """
        if len(self.selected_spaxels)>0:
            self.show_picked_spaxels()
            self.show_picked_spectrum()
            self.show_picked_traces()
            self.fig.canvas.draw()

    # ================= #
    # Properties        #
    # ================= #
    @property
    def axccd(self):
        """ """
        return self._properties["axccd"]
    
    @property
    def ccdimage(self):
        """ """
        return self._properties["ccd"]


