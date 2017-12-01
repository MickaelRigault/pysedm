#! /usr/bin/env python
# -*- coding: utf-8 -*-


#################################
#
#   MAIN 
#
#################################
if  __name__ == "__main__":
    
    import argparse
    import numpy as np
    from pysedm.utils import extractstar
    from pysedm       import get_sedmcube, io
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')
    
    #parser.add_argument('--rmsky',  action="store_true", default=False,
    #                    help='Removes the sky component from the cube')
    #
    #parser.add_argument('--nskyspaxels',  type=int, default=50,
    #                    help='Number of faintest spaxels used to estimate the sky')

    parser.add_argument('--auto',  type=str, default=None,
                        help='Build a e3d cube of the given target or target list (csv) e.g. --build dome or --build dome,Hg,Cd')
    
    parser.add_argument('--autoradius',  type=float, default=10,
                        help='Radius (in spaxels) of the automatic aperture photometric extraction [default=10]')
    
    args = parser.parse_args()
    # ================= #
    #   The Scripts     #
    # ================= #
    # --------- #
    #  Date     #
    # --------- #
    date = args.infile

    # ================= #
    #   Actions         #
    # ================= #
    
    # - Automatic extraction
    if args.auto is not None and len(args.auto) >0:
        print(args.auto)
        for target in args.auto.split(","):
            fileccds = io.get_night_cubes(date, kind="cube", target=target.replace(".fits",""))
            print(fileccds)
            for filecube in fileccds:
                print("Automatic extraction of target %s, file: %s"%(target, filecube))
                cube = get_sedmcube(filecube)
                es   = extractstar.ExtractStar(cube)
                spec = es.get_auto_aperture_spectroscopy(radius=args.autoradius)
                spec.writeto(filecube.replace("e3d","specauto"))
                spec.show(savefile=filecube.replace("e3d","specauto").replace(".fits",".pdf"))
                
    else:
        print("NO  AUTO")
