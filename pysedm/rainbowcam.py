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

# ================== #
#  Main Function     #
# ================== #
def build_meta_ifu_guider(ifufile, outdir, solve_wcs=True, **kwargs):
    """ """
    savefile = build_stacked_guider(ifufile, outdir)
    if solve_wcs:
        solve_astrometry(savefile, **kwargs)

# ================== #
#   Function         #
# ================== #

def build_stacked_guider(ifufile, outdir=None, overwrite=True):
    """ """    
    guiders = get_ifu_guider_images(fits.getheader(ifufile))
    stacked_image = stack_images(guiders)
    # - building the .fits
    
    filein = ifufile.split("/")[-1]
    if outdir is None: outdir = "".join(ifufile.split("/")[:-1])
        
    savefile = outdir+"/guider_%s"%filein
    hdulist = fits.HDUList([fits.PrimaryHDU(stacked_image, fits.getheader(guiders[0]))])
    hdulist.writeto(savefile, overwrite=overwrite)
    return savefile

    
def solve_astrometry(img, outimage=None, radius=3, with_pix=True, overwrite=True, tweak=3):
    """ author: @nblago

    img: fits image where astrometry should be solved.
    outimage: name for the astrometry solved image. If none is provided, the name will be "a_"img.
    radius: radius of uncertainty on astrometric position in image.
    with_pix: if we want to include the constraint on the pixel size for the RCCam.
    overwrite: wether the astrometrically solved image should go on top of the old one.
    tewak: parameter for astrometry.net
    """

    from astropy.wcs import InconsistentAxisTypesError
    import subprocess    
    curdir = os.getcwd()
    imgdir = os.path.dirname(img)
    
    os.chdir(imgdir)
    
    img = os.path.abspath(img)
    
    ra  = fits.getval(img, 'RA')
    dec = fits.getval(img, 'DEC')
    #logger.info( "Solving astrometry on field with (ra,dec)=%s %s"%(ra, dec))
    
    astro = os.path.join( os.path.dirname(img), "a_" + os.path.basename(img))


    #If astrometry exists, we don't run it again.
    if (os.path.isfile(astro) and not overwrite):
        return astro
        
    cmd = "solve-field --ra %s --dec %s --radius %.4f -p --new-fits %s -W none -B none -P none -M none -R none -S none -t %d --overwrite %s "%(ra, dec, radius, astro, tweak, img)
    if (with_pix):
        cmd = cmd + " --scale-units arcsecperpix  --scale-low 0.375 --scale-high 0.4"
    
    print (cmd)

    subprocess.call(cmd, shell=True)
    
    print ("Finished astrometry")
    
    #Cleaning after astrometry.net
    #if (os.path.isfile(img.replace(".fits", ".axy"))):
    #    os.remove(img.replace(".fits", ".axy"))
    #if (os.path.isfile(img.replace(".fits", "-indx.xyls"))):
    #    os.remove(img.replace(".fits", "-indx.xyls"))
    #if (os.path.isfile("none")):
   #     try:
   #         os.remove("none")
   #     except:
   #         print ("Could not remove file none.")
        
    #os.chdir(curdir)

    #if (not outimage is None and overwrite and os.path.isfile(astro)):
    #    shutil.move(astro, outimage)
    #    return outimage
    #elif (outimage is None and overwrite and os.path.isfile(astro)):
    #    shutil.move(astro, img)
    #    return img
    #else:
    #    return astro
    
# ================== #
#   Tools            #
# ================== #

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


