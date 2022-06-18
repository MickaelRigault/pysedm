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
SEDMPY_CODE_PATH = "/scr2/sedmdrp/sedmpy/"
# READOUT_NOISE       = 4


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
        Path of a ifu .fits (or derived .fits as long as they contain
        the basic associated header keywords)
       
    outdir: [string] -optional-
        Where does the guider image should be saved. 
        If None, it will be in the same directory as the `infufile`
        
    solve_wcs: [bool] -optional-
        Shall "do_astro" (based on astrometry.net) be ran on the stacked image?

    verbose: [bool] -optional-
        Extra output
        
    Returns
    -------
    Void (creates a guider_`ifufile`)
    """
    savefile = build_stacked_guider(ifufile, outdir)
    if savefile:
        if verbose:
            print(" guider image built for %s" % ifufile)
            print(savefile)
        if solve_wcs:
            if verbose:
                print(" running astrometry on %s" % savefile)
            run_do_astrom(savefile)
            if not os.path.isfile(savefile.replace(".fits", "_astrom.fits")):
                print("do_astrom has failed.")
            else:
                print("do_astrom succeeded.")
    else:
        print("ERROR - unable to build guider image")


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
        Path of a ifu .fits (or derived .fits as long as they contain
         the basic associated header keywords)
       
    outdir: [string] -optional-
        Where does the guider image should be saved. 
        If None, it will be in the same directory as the `infufile`

    overwrite: [bool] -optional-
        Set to overwrite existing file

    """
    guiders = get_ifu_guider_images(ifufile)
    stacked_image, nstack, avscl = stack_images(guiders)
    # did we get any?
    if nstack < 1:
        print("ERROR - no guider images found")
        return None
    # - building the .fits
    date = io.header_to_date(fits.getheader(ifufile))
    filein = ifufile.split("/")[-1]
    if outdir is None:
        outdir = io.get_datapath(date)
        
    savefile = outdir+"/guider_%s"%filein
    hdulist = fits.HDUList([fits.PrimaryHDU(stacked_image,
                                            fits.getheader(guiders[0]))])
    hdulist[0].header['NSTACK'] = nstack
    hdulist[0].header['STACKMTH'] = 'median'
    hdulist[0].header['SCALEMTH'] = 'median'
    hdulist[0].header['STACKSCL'] = avscl
    hdulist.writeto(savefile, overwrite=overwrite)
    return savefile

    
def run_do_astrom(guider_filename_fullpath):
    """ Run the do_astrom script in /scr2/sedmdrp/bin
    """
    try:
        do_astrom = os.path.join(os.environ['SEDMPY'], 'bin/do_astrom')
    except KeyError:
        do_astrom = SEDMPY_CODE_PATH + 'bin/do_astrom'
    import subprocess
    cmd = [do_astrom, guider_filename_fullpath]
    print(" ".join(cmd))
    subprocess.call(cmd)
    # Test results
    astrom_output = guider_filename_fullpath.replace(
        ".fits", "_astrom.fits").replace(".gz", "")
    if not os.path.exists(astrom_output):
        print("ERROR - astrometry failed, trying a median subtraction")
        from scipy import ndimage
        ff = fits.open(guider_filename_fullpath, mode='update')
        image = ff[0].data * 1.0    # Ensure we are float
        fsize = 15
        print("making median filter...")
        medfilt = ndimage.median_filter(image, fsize, mode='constant')
        ff[0].data = image - medfilt
        ff[0].header['MEDSUB'] = True, ' Median subtracted: %d px' % fsize
        ff.close()
        print("Done.  Re-doing do_astrom")
        subprocess.call(cmd)
    
    
# ================== #
#   Tools            #
# ================== #

def get_rainbow_datapath(ut_date):
    """ returns the path of the rainbow camera data """
    try:
        source = os.environ['SEDMRAWPATH']
    except KeyError:
        source = RAINBOW_DATA_SOURCE
    return os.path.join(source, ut_date) + "/"


def get_ifu_guider_images(ifufile):
    """ """    
    ifu_header = fits.getheader(ifufile)
    
    fileid = io.filename_to_id(ifufile)
    # - starting
    jd_ini = time.Time("%s %s" % (io.header_to_date(ifu_header, sep="-"),
                                  fileid.replace("_", ":"))).jd
    # - end
    jd_end = jd_ini + ifu_header['EXPTIME'] / (24.*3600)
    # - Where are the guider data  ?
    rb_dir = get_rainbow_datapath(io.header_to_date(ifu_header))
    # - Return them
    flist = os.listdir(rb_dir)
    rb_list = []
    for f in flist:
        # Use only *.fit* files
        if 'fit' not in f.split('/')[-1]:
            continue
        try:
            ff = fits.open(rb_dir+f)
        except OSError:
            print("WARNING - corrupt fits file: %s" % f)
            continue
        if "CAM_NAME" in ff[0].header:
            # Skip IFU images
            if "IFU" in ff[0].header['CAM_NAME']:
                continue
        else:
            if 'ifu' in f.split('/')[-1]:
                continue
        if "JD" in ff[0].header:
            # Images used to guide ifufile
            if jd_ini <= ff[0].header["JD"] <= jd_end:
                rb_list.append(rb_dir+f)
        else:
            print("WARNING - no JD keyword in %s" % f)
    return rb_list


def stack_images(rainbow_files, method="median", scale="median"):
    """ return a 2D image corresponding of the stack of the given data """
    # - 
    if scale not in ['median']:
        raise NotImplementedError("only median scaling implemented (not %s)"
                                  % scale)
    
    if method not in ['median']:
        raise NotImplementedError(
            "only median stacking method implemented (not %s)" % method)
    # Load the normalized data

    datas = []
    scales = []
    scaling = 1.
    avscale = 1.
    for f_ in rainbow_files:
        data_ = fits.getdata(f_)
        # header_= fits.getheader(f_) all have the same gain and readout noise
        if scale in ["median"]: 
            scaling = np.median(data_)
            scales.append(scaling)
        datas.append(data_/scaling)
        avscale = np.mean(scales)

    return np.median(datas, axis=0)*avscale, len(datas), avscale
