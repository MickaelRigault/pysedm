#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""  """

import numpy as np
import matplotlib.pyplot as mpl

from astropy.utils.console import ProgressBar


from propobject            import BaseObject
from .utils.tools import kwargs_update, load_pkl, dump_pkl


DEGREE = 10
LEGENDRE = True

def get_contvalue(spec):
    spec.fit_continuum(DEGREE, legendre=LEGENDRE)
    return spec.contmodel.fitvalues

def fit_background(ccd, jump=50, multiprocess=True):
    """ calling `get_contvalue` for each ccd column (xslice).
    This uses astropy's ProgressBar.map 

    Returns 
    -------
    dictionary 
    """
    index_column = range(ccd.width)[::jump]
    contval =  ProgressBar.map(get_contvalue, [ccd.get_xslice(i) for i in index_column],
                               multiprocess=multiprocess, step=2)
    
    return {i:c for i,c in zip(index_column, contval)}

class Background( BaseObject ):
    """ """
    PROPERTIES      = ["contvalues", "y"]
    SIDE_PROPERTIES = ['filename']
    DERIVED_PROPERTIES = ['input_columns', "input_background"]
    
    def load(self, filename):
        """ """
        self.create(load_pkl(filename))
        self._side_properties['filename'] = filename
        
    def create(self, contvalues):
        """ setup the instance based on the input 'contvalue' """
        self._properties['contvalues']    = contvalues
        self._derived_properties['input_columns'] = None
        
    def build(self, height):
        """ """
        self._properties["y"] = np.linspace(0, 1, height)
        self._derived_properties["input_background"] = \
          np.asarray([self.contvalue_to_polynome(self._contvalues[i], self._y)
                          for i in self.input_columns])

    def contvalue_to_polynome(self, contvalue_, y = None):
        """ """
        from modefit.basics import polynomial_model
        freeparam = np.sort([k_ for k_ in contvalue_.keys() if "a" in k_ and '.err' not in k_])
        poly = polynomial_model( len(freeparam) )
        poly.use_legendre=LEGENDRE
        poly.setup([contvalue_[k] for k in freeparam])
        return poly.get_model(y if y is not None else self._y)

    def show(self):
        """ """
        fig = mpl.figure()
        ax  = fig.add_subplot(111)
        ax.imshow(self.input_background.T, origin="lower", aspect="auto")
        fig.show()
        
    # -------------------- #
    #    Properites        #
    # -------------------- #
    @property
    def n_inputcolumns(self):
        """ Number of column for which the background has been estimated"""
        return len(self._contvalues)

    @property
    def input_columns(self):
        """ index of the ccd column where the background has been estimated """
        if self._derived_properties['input_columns'] is None:
            self._derived_properties['input_columns'] = np.sort(self._contvalues.keys())
            
        return self._derived_properties['input_columns']
    
    @property
    def _contvalues(self):
        """ """
        return self._properties['contvalues']

    @property
    def _y(self):
        """ """
        return self._properties['y']
    

    # -- Background
    @property
    def input_background(self):
        """ """
        return self._derived_properties["input_background"]
