#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from PIL import Image
import os
import io
import numpy as np
import matplotlib.pyplot as mpl




def get_image_row(images, interpolation=3, height="min"):
    """ combine PIL images in row from left to right  """
    
    if type(height) is str:
        height   = getattr(np, height)([img_.size[1] for img_ in images])
    elif type(height) in [int, float]:
        height = height
    else:
        raise TypeError("height should be a int or a string (numpy attribute like min/max/median)")

    imgs_comn = np.hstack([img_.resize( np.round(np.asarray(img_.size)*( height / img_.size[1] ) ).astype("int"), interpolation)
                           for img_ in images])
    return Image.fromarray(imgs_comn)

def get_image_column(images, interpolation=3, width="min"):
    """ combine PIL images in column from top to bottol  """
    if type(width) is str:
        width   = getattr(np, width)([img_.size[0] for img_ in images])
    elif type(width) in [int, float]:
        width = width
    else:
        raise TypeError("width should be a int or a string (numpy attribute like min/max/median)")

    imgs_comn = np.vstack([img_.resize( np.round((np.asarray(img_.size)*( width / img_.size[0] ) )).astype("int"), interpolation)
                           for img_ in images])
    return Image.fromarray(imgs_comn)

def get_buffer(size, text="", hline=None, vline=None, fontsize='medium',
               xy=[0.5,0.5], va="center",ha="center", barcolor="k",
               textprop={}, barprop={}, get_figure=False):
    """ Uses matplotlib to create a simple full size axis in which you can add text and horizonthal lines """
    import os
    import matplotlib.pyplot as mpl
    fig = mpl.figure(figsize=size)
    ax  = fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax.text(xy[0],xy[1], text, va=va,ha=ha,fontsize=fontsize, **textprop)
    if hline is not None:
        for hline_ in np.atleast_1d(hline):
            ax.axhline(hline_, color=barcolor, **barprop )
            
    if vline is not None:
        for vline_ in np.atleast_1d(vline):
            ax.axvline(vline_, color=barcolor, **barprop )
    
    if get_figure:
        return fig
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return Image.open(buf)
