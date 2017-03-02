#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy         as np
import matplotlib.pyplot as mpl
from .tools import make_method

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
