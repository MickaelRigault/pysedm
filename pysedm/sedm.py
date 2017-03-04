#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module containing the part that are directly SEDM oriented. """

import numpy            as np
from pyifu.spectroscopy import Cube


SEDMSPAXELS = np.asarray([[ np.sqrt(3.)/2., 1./2],[0, 1],[-np.sqrt(3.)/2., 1./2],
                          [-np.sqrt(3.)/2.,-1./2],[0,-1],[ np.sqrt(3.)/2.,-1./2]])*2/3.

class SEDMCube( Cube ):
    """ SEDM Cube """
