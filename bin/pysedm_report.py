#! /usr/bin/env python
# -*- coding: utf-8 -*-

#################################
#
#   MAIN 
#
#################################
if  __name__ == "__main__":
    import io
    import argparse
    import numpy as np
    import pysedm
    from pysedm import io
    import datetime
    
    from astropy.io import fits
    from pysedm.utils import pil
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube
            """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')


    parser.add_argument('--contains',  type=str, default="*",
                        help='Provide here part of the filename. This will build the guider images of all crr images of the given night having `contains` in there name')

    # ================ #
    # END of Option    #
    # ================ #
    args = parser.parse_args()

    # Matplotlib
    # ================= #
    #   The Scripts     #
    # ================= #
    
    # --------- #
    #  Date     #
    # --------- #
    date = args.infile
    
    # Loads all the spectra 
    specfiles = io.get_night_files(date, "spec.basic", args.contains)
    for specfile in specfiles:
        # Footer
        now = datetime.datetime.now()
        timestring = "made the %d-%02d-%02d at %02d:%02d:%02d"%(now.year,now.month,now.day, now.hour, now.minute, now.second)
        
        footer = pil.get_buffer([10,1], "pysedm version %s"%pysedm.__version__+ "  |  " + timestring,
                                fontsize="small", textprop=dict(color="0.5"), barcolor="k")

        
        spec_id = io.filename_to_id(specfile)
        try:
            img_spax = pil.Image.open(pysedm.io.get_night_files(date, "re:psfprofile",spec_id, extention=".png")[0])
            img_psf  = pil.Image.open(pysedm.io.get_night_files(date, "re:psfprofile",spec_id, extention=".png")[0])
            img_spec = pil.Image.open(pysedm.io.get_night_files(date, "re:spec",spec_id, extention=".png")[-1])
            
            img_flexj= pil.Image.open(pysedm.io.get_night_files(date, "re:flexuretrace",spec_id, extention=".png")[0])
            img_flexi= pil.Image.open(pysedm.io.get_night_files(date, "re:flex_sodium",spec_id, extention=".png")[0])
            img_adr  = pil.Image.open(pysedm.io.get_night_files(date, "re:adr",spec_id, extention=".png")[0])
        except:
            print("ERROR: cannot load the PNGs for %s. *Skipped*"%specfile)
            continue

        # Combination
        spec_header = fits.getheader(specfile)
        title = "Name: %s"%spec_header["OBJECT"]
        title+= "| exposure time: %.1f s"%spec_header["EXPTIME"]
        title+= "| file ID: %s "%spec_id
        img_combined = pil.get_image_column(
                                        [pil.get_buffer([10,1], title , fontsize="large"),
                                         pil.get_image_column([ pil.get_image_row([img_spax,img_spec]),img_psf]), 
                                         pil.get_buffer([10,1], "Validation plots", hline=[0.3,0.7], barprop=dict(lw=1)),
                                         pil.get_image_row([ pil.get_image_column([img_flexi,img_flexj]), img_adr]),
                                         footer])
        
        
        
            
