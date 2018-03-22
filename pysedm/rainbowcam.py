#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" Access the rainbow camera images """

#
# This method is an a
#
import os
import numpy as np
from astropy.io import fits

RAINBOW_DATA_SOURCE = "/scr2/sedm/raw/"
READOUT_NOISE       = 4


def get_rainbow_datapath(DATE):
    """ returns the path of the rainbow camera data """
    return RAINBOW_DATA_SOURCE+DATE+"/"

def get_ifu_guider_images(ifu_header):
    """ """
    if ifu_header['IMGTYPE'].lower() not in ['science', "standard"]:
        raise TypeError("ifu_header is not a header of a Science of standard ifu target")

    # = Getting the start and the beginning of the exposure = #
    #In Richard's pipeline, the JD is the beginning of the exposure,
    #in Nick's one is the end.
    if ifu_header['TELESCOP'] == "60":
        jd_ini, jd_end = ifu_header['JD'] - ifu_header['EXPTIME'] / (24*3600), ifu_header['JD']
    else:
        jd_ini, jd_end = ifu_header['JD'], ifu_header['JD'] + ifu_header['EXPTIME'] / (24*3600)
        
    rb_dir = get_rainbow_datapath("".join(ifu_header['OBSDATE'].split('-')))
    return [rb_dir+f for f in os.listdir(rb_dir)
                if f.startswith("rc") and f.endswith(".fits")
                and jd_ini<=fits.getval(rb_dir+f, "JD")<=jd_end]

def stack_images(rainbow_files, method="median", scale="median"):
    """ return a 2D image corresponding of the stack of the given data """
    # - 
    if scale not in ['median']:
        raise NotImplementedError("only median scaling implemented (not %s)"%scale)
    
    if method not in ['median']:
        raise NotImplementedError("only median stacking method implemented (not %s)"%method)
    # Load the normalized data

    datas = []
    for f_ in rainbow_files:
        data_   = fits.getdata(f_)
        # header_ = fits.getheader(f_) all have the same gain and readout noise...
        if scale in ["median"]: 
            scaling = np.median(data_)
        datas.append(data_/scaling)

    return np.median(datas, axis=0)


