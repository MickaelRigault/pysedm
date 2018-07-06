#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" Access the rainbow camera images """

#
# This method is an a
#
import os
import numpy as np
from astropy import time
from astropy.io import fits
from . import io
RAINBOW_DATA_SOURCE = "/scr2/sedm/raw/"
#READOUT_NOISE       = 4

# ================== #
#  Main Function     #
# ================== #
def build_meta_ifu_guider(ifufile, outdir=None, solve_wcs=True, verbose=False):
    """ Higher level function. 
    It:
    1) fetches the guider images from the rainbow camera raw directory
    2) Merges them into one stacked_guider image
    3) Solves the astrometry on the stacked_guider image.

    Steps 1 and 2 are made using `build_stacked_guider()`
    Step 3 is made using `solve_astrometry` 

    Parameters
    ----------
    ifufile: [string]
        Path of a ifu .fits (or derived .fits as long as they contain the basic associated header keywords)
       
    outfir: [string] -optional-
        Where does the guider image should be saved. 
        If None, it will be in the same directory as the `infufile`
        
    solve_wcs: [bool] -optional-
        Shall "do_astro" (based on astrometry.net) be ran on the stacked image?
        
    Returns
    -------
    Void (creates a guider_`ifufile`)
    """
    savefile = build_stacked_guider(ifufile, outdir)
    if verbose:
        print(" guider image built for %s"%ifufile)
        print(savefile)
    if solve_wcs:
        if verbose:
            print(" running astrometry on %s"%savefile)
        run_do_astrom(savefile)
        if not os.path.isfile( savefile.replace( ".fits", "_astrom.fits" ) ):
            print("do_astrom has failed. Let's rerun it")
            run_do_astrom(savefile)
        else:
            print("do_astrom succeeded.")
            
# ================== #
#   Function         #
# ================== #
def build_stacked_guider(ifufile, outdir=None, overwrite=True):
    """ 
    This function:
    1) fetches the guider images from the rainbow camera raw directory
       [using `get_ifu_guider_images`]
    2) Merges them into one stacked_guider image
       [using `get_ifu_guider_images`]


    Parameters
    ----------
    ifufile: [string]
        Path of a ifu .fits (or derived .fits as long as they contain the basic associated header keywords)
       
    outfir: [string] -optional-
        Where does the guider image should be saved. 
        If None, it will be in the same directory as the `infufile`

    """
    guiders = get_ifu_guider_images( ifufile )
    stacked_image = stack_images( guiders )
    # - building the .fits
    date = io.header_to_date( fits.getheader(ifufile) )
    filein = ifufile.split("/")[-1]
    if outdir is None:
        outdir = io.get_datapath(date)
        
    savefile = outdir+"/guider_%s"%filein
    hdulist = fits.HDUList([fits.PrimaryHDU(stacked_image, fits.getheader(guiders[0]))])
    hdulist.writeto(savefile, overwrite=overwrite)
    return savefile

    
def run_do_astrom(guider_filename_fullpath):
    """ 
    """
    import subprocess
    print(["/scr2/sedmdrp/bin/do_astrom",guider_filename_fullpath])
    subprocess.call(["/scr2/sedmdrp/bin/do_astrom",guider_filename_fullpath])
    
    
# ================== #
#   Tools            #
# ================== #

def get_rainbow_datapath(DATE):
    """ returns the path of the rainbow camera data """
    return RAINBOW_DATA_SOURCE+DATE+"/"

def get_ifu_guider_images(ifufile):
    """ """    
    ifu_header = fits.getheader(ifufile)
    
    fileid = io.filename_to_id(ifufile)
    # - starting
    jd_ini = time.Time("%s %s"%(io.header_to_date(ifu_header, sep="-"),
                                    fileid.replace("_",":"))).jd
    # - end
    jd_end = jd_ini +  ifu_header['EXPTIME'] / (24.*3600)
    # - Where are the guider data  ?
    rb_dir = get_rainbow_datapath( io.header_to_date(ifu_header) )
    # - Return them
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


