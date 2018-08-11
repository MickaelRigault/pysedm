#! /usr/bin/env python
# -*- coding: utf-8 -*-



def build_image_report(specfile):
    """ """
    from pysedm.utils import pil
    now = datetime.datetime.now()
    timestring = "made the %d-%02d-%02d at %02d:%02d:%02d"%(now.year,now.month,now.day, now.hour, now.minute, now.second)
        
    footer = pil.get_buffer([20,1], "pysedm version %s"%pysedm.__version__+ "  |  " + timestring,
                                fontsize=15, textprop=dict(color="0.5"), barcolor="k")

    date    = io.header_to_date(fits.getheader(specfile))
    spec_id = io.filename_to_id(specfile)
    
    prop_missing = dict(fontsize=30, textprop=dict(color="C1"))
    
    try:
        img_spax = pil.Image.open(pysedm.io.get_night_files(date, "re:ifu_spaxels",spec_id, extention=".png")[0])
    except:
        img_spax = pil.get_buffer([7, 7], "Spaxel IFU image missing" , **prop_missing)
        
    try:
        img_psf  = pil.Image.open(pysedm.io.get_night_files(date, "re:psfprofile",spec_id, extention=".png")[0])
    except:
        img_psf = pil.get_buffer([15, 4], "PSF Profile image missing" , **prop_missing)
        
    try:
        img_spec = pil.Image.open(pysedm.io.get_night_files(date, "re:spec",spec_id, extention=".png")[-1])
    except:
        img_spec = pil.get_buffer([13, 7], "Spectra image missing" , **prop_missing)
        
    try:
        img_flexj= pil.Image.open(pysedm.io.get_night_files(date, "re:flexuretrace",spec_id, extention=".png")[0])
    except:
        img_flexj= pil.get_buffer([7, 4], "j-flexure image missing" , **prop_missing)
        
        
    try:
        img_flexi= pil.Image.open(pysedm.io.get_night_files(date, "re:flex_sodium",spec_id, extention=".png")[0])
    except:
        img_flexi= pil.get_buffer([7, 4], "i-flexure image missing" , **prop_missing)
    
    try:
        img_adr  = pil.Image.open(pysedm.io.get_night_files(date, "re:adr",spec_id, extention=".png")[0])
    except:
        img_adr= pil.get_buffer([9, 4], "ADR image missing" , **prop_missing)
        
        
    
    
    # Combination
    spec_header = fits.getheader(specfile)
    title = "%s"%spec_header["OBJECT"]
    title+= " | exposure time: %.1f s"%spec_header["EXPTIME"]
    title+= " | file ID: %s "%spec_id


    title_img = pil.get_buffer([8,1], title , fontsize=16, hline=[0.3,0.7], barprop=dict(lw=1))

    # Info 
    width = 3
    fluxheader = pil.get_buffer([width,0.8], "Flexure" , fontsize=10, hline=None, textprop=dict(color="0.5"))
    
    
    
    img_upperright = pil.get_image_column([title_img,  img_psf])
    img_upperleft  = pil.get_image_row([img_spax, img_adr])
    img_upper      = pil.get_image_row([img_upperleft, img_upperright])

    
    bar = pil.get_buffer([0.5,8], vline=0.5, barprop=dict(lw=0.2))
    img_lowerright = pil.get_image_column([fluxheader,img_flexi, img_flexj])
    img_lower      = pil.get_image_row([img_spec,bar, img_lowerright])
    

    img_combined  = pil.get_image_column([img_upper, img_lower, 
                                         footer])
            
    return img_combined
        
            


#################################
#
#   MAIN 
#
#################################
if  __name__ == "__main__":
    import argparse
    import numpy as np
    import pysedm
    from pysedm import io
    import datetime
    
    from astropy.io import fits

    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube
            """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')


    parser.add_argument('--contains',  type=str, default="*",
                        help='Provide here part of the filename. This will build the guider images of all crr images of the given night having `contains` in there name')

    parser.add_argument('--slack',  action="store_true", default=False,
                        help='Submit the report to slack')

    # ================ #
    # END of Option    #
    # ================ #
    args = parser.parse_args()

    if args.slack:
        try:
            from pysedmpush import slack
        except ImportError:
            raise ImportError("you need to install pysedmpush to be able to push on slack")
        
    # Matplotlib
    # ================= #
    #   The Scripts     #
    # ================= #
    SLACK_CHANNEL = "pysedm-report"
    # --------- #
    #  Date     #
    # --------- #
    date = args.infile
    
    # Loads all the spectra 
    specfiles = io.get_night_files(date, "spec.basic", args.contains)
    print(specfiles)
    for specfile in specfiles:
            
        img_report = build_image_report(specfile)
        report_filename = specfile.replace("spec_","pysedm_report_").replace(".fits",".png")
        img_report.save(report_filename, dpi=(600,500))

        # Slack push report
        if args.slack:
            if not os.path.isfile( report_filename )
                warnings.warn("No file-image created by the pysedm_report.build_image_report(). Nothing to push on slack")
            else:
                print("pushing the report to %s"%SLACK_CHANNEL)
                header  = fits.getheader(specfile)
                file_id = io.filename_to_id(specfile)
                # Title & caption
                title = "pysedm-report: %s | %s (%s)"%(header["OBJECT"], file_id, date)
                caption = ""
                # Push
                slack.push_image(report_filename, caption=caption, title=title,
                                 channel=SLACK_CHANNEL)
        
