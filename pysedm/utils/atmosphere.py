#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Simple library containing the Atmosphere related object. """

import warnings
import numpy                as np
from pyifu.spectroscopy import Spectrum




class ExtinctionSpectrum( Spectrum ):
    """ """
    PROPERTIES = ["interpolation"]
    
    def get_atm_extinction(self, lbda, airmass):
        """ """
        return 10**(self._interpolation(lbda) * airmass/2.5)
    
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def _interpolation(self, kind="cubic"):
        """ """
        if self._properties['interpolation'] is None:
            from scipy.interpolate import interp1d
            self._properties['interpolation'] = \
              interp1d(self.lbda, self.data, kind=kind)

        return self._properties['interpolation']
