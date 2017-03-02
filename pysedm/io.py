#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to I/O the data """

import os

REDUXPATH   = os.getenv('SEDMREDUXPATH',default="~/redux/")


def get_datapath(YYYYMMDD):
    """ Return the full path of the current date """
    return REDUXPATH+"%s/"%YYYYMMDD




def get_spectralmatch(YYYYMMDD):
    """ Load the spectral matcher.
    This object must have been created. 
    """
    from .spectralmatching import load_specmatcher
    return load_specmatcher(get_datapath(YYYYMMDD)+"%s_SpectralMatch.pkl"%(YYYYMMDD))
