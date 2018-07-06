#! /usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
#################################
#
#   MAIN 
#
#################################
if  __name__ == "__main__":
    
    import argparse

    from pysedm import io, rainbowcam
    
    parser = argparse.ArgumentParser(
        description="""Build the guider images | to be run on pharos """,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='The date YYYYMMDD')
    
    parser.add_argument('--contains',  type=str, default="*",
                        help='Provide here part of the filename. This will build the guider images of all crr images of the given night having `contains` in there name')
    
    parser.add_argument('--solvewcs',  action="store_true", default=False,
                        help='Shall the wcs solution of the guider be solved (ignored if --noguider). [part of the --build]')

    parser.add_argument('--quite',  action="store_true", default=False,
                        help='Set verbose to False')

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


    # ---------------- #
    #  Guider loop     #
    # ---------------- #
    files_to_use = io.get_night_files(date, "ccd.crr", args.contains)
    print(" Guider images will be build for :")
    print(", ".join(files_to_use) )

    for filename in files_to_use:
        print( "** Starting %s **"%filename )
        rainbowcam.build_meta_ifu_guider(filename, solve_wcs = args.solvewcs, verbose = False if args.quite else True)
        
