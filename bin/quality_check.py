#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
CCD_STD_FLUX_RATIO = 10

def check_if_stdobs_worked(ccdfile, tracematch, limit=CCD_STD_FLUX_RATIO):
    """ """
    from pysedm.ccd import get_ccd
    ccd = get_ccd(ccdfile, tracematch=tmap, background=0)
    mask_out = ccd.tracematch.get_notrace_mask()
    vlow_out, vmedian_out, vmax_out = np.percentile(ccd.rawdata[mask_out],[10,50,99.99])
    vlow_in,   vmedian_in,  vmax_in = np.percentile(ccd.rawdata[~mask_out],[10,50,99.99])
    deltas = {"low":    vlow_in-vlow_out,
              "median": vmedian_in-vmedian_out,
              "max":    vmax_in-vmax_out}
    
    return bool(deltas["max"] > limit*(deltas["median"]/deltas["low"]))

############################@
#
#   Main Quality bin       #
#
############################ 
if  __name__ == "__main__":
    
    import argparse
    import pysedm
    
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" Check the pysedm product of input quality
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='The date YYYYMMDD')

    parser.add_argument('--stdobs',  type=str, default=None,
                        help='Check if an given STD observations (ccd.crr) has been successful. [accepting target, target list (csv) or regex] ')
        
    # - Standard Star object
    

    args = parser.parse_args()
        
    # ================= #
    #   The Scripts     #
    # ================= #

    # --------- #
    #  Date     #
    # --------- #
    date   = args.infile

    
    if args.stdobs is not None:
        tmap = pysedm.load_nightly_tracematch(date)
        for target in args.stdobs.split(","):
            for datafile in pysedm.get_night_files(date, "ccd.crr", target):
                if check_if_stdobs_worked(datafile, tmap):
                    print("%s: STD obs ccd.crr test: Successfull"%datafile.split("/")[-1])
                else:
                    print("%s: STD obs ccd.crr test: FAILED | code 4: no star detected."%datafile.split("/")[-1])
                              

                
                
