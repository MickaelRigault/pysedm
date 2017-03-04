#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to I/O the data """

import os

REDUXPATH   = os.getenv('SEDMREDUXPATH',default="~/redux/")


def get_datapath(YYYYMMDD):
    """ Return the full path of the current date """
    return REDUXPATH+"%s/"%YYYYMMDD


#########################
#                       #
#   NIGHT SOLUTION      #
#                       #
#########################
def load_nightly_spectralmatch(YYYYMMDD):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from .spectralmatching import load_specmatcher
    return load_specmatcher(get_datapath(YYYYMMDD)+"%s_SpectralMatch.pkl"%(YYYYMMDD))

def load_nightly_hexagonalgrid(YYYYMMDD):
    """ Load the Grid id <-> QR<->XY position
    This object must have been created. 
    """
    from .utils.hexagrid import load_hexprojection
    return load_hexprojection(get_datapath(YYYYMMDD)+"%s_HexaGrid.pkl"%(YYYYMMDD))

def load_nightly_wavesolution(YYYYMMDD):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from .wavesolution import load_wavesolution
    return load_wavesolution(get_datapath(YYYYMMDD)+"%s_WaveSolution.pkl"%(YYYYMMDD))
