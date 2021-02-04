#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is to remove cosmic ray spaxels in the SEDM datacube.
-v20210204: 1st version.
"""

#################################
#
#   MAIN
#
#################################
if  __name__ == "__main__":

    import argparse
    import numpy as np

    from pysedm import get_sedmcube, io, byecr

    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube""",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default="None",
                        help='cube filepath')

    parser.add_argument('--inputcube',   action="store_true", default=False,
                        help='Is the input a cube ?')

    parser.add_argument('--auto',  type=str, default=None,
                        help='Shall this run an automatic PSF extraction')

    # Cosmic ray removal with byecr.py
    parser.add_argument('--byecr_lbda', type=float, default=None,
                         help="To check cosmic rays at a given wavelength.")

    parser.add_argument('--byecr_cut', type=float, default=5.0,
                         help="Cut criteria for byecr module.")

    parser.add_argument('--byecr_wspectral', action="store_true", default=False,
                         help="Also use a spectral filtering for byecr.")

    parser.add_argument('--byecr_showcube', action="store_true", default=False,
                         help="Show a cube with detected cosmic rays.")

    args = parser.parse_args()

    # ================= #
    #   The Scripts     #
    # ================= #

    # --------- #
    #  Date     #
    # --------- #
    if not args.inputcube:
        date = args.infile

    # ===================== #
    # Running byecr   #
    # ===================== #

    if args.auto is not None and len(args.auto) >0:

        for target in args.auto.split(","):

            filecubes = io.get_night_files(date, "cube.*", target=target.replace(".fits","")) if not args.inputcube else [target]
            print(filecubes)
            print("cube file from which the cosmic-ray will be removed [auto]: "+ ", ".join(filecubes))

            # - loop over the file cube
            for filecube in filecubes:
                print("Automatic removing cosmic-rays of target %s, file: %s"%(target, filecube))

            #
            # DATA IN
            #
                cube = get_sedmcube(filecube)

                print(" Starting byecr cosmic ray removal ".center(50, "-"))
                targetid = io.filename_to_id(filecube)
                byecrclass = byecr.get_cr_spaxels_from_byecr(date, targetid)
                cr_df = byecrclass.get_cr_spaxel_info(lbda_index=args.byecr_lbda,
                                              cut_criteria=args.byecr_cut,
                                              wspectral=args.byecr_wspectral)

                cube.data[cr_df["cr_lbda_index"], cr_df["cr_spaxel_index"]] = np.nan

                cube.header.set("NCR", len(cr_df), "total number of detected cosmic-rays from byecr")
                cube.header.set("NCRSPX", len(np.unique(cr_df["cr_spaxel_index"])), "total number of cosmic-ray affected spaxels")

                print("Number of detected cosmic rays = %i" %len(cr_df))
                print("Number of cosmic ray affected spaxels = %i" %len(np.unique(cr_df["cr_spaxel_index"])))

            # -
            # - SAVING
            # -
                savefile = filecube.replace("e3d_", "e3d_crr_")
                print("savefile = %s" %savefile)
                cube.writeto(savefile=savefile)
