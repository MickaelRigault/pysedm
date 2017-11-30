#! /usr/bin/env python
# -*- coding: utf-8 -*-


#################################
#
#   MAIN 
#
#################################
if  __name__ == "__main__":
    
    import argparse
    from pysedm import get_sedmcube
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')

    parser.add_argument('--rmsky',  action="store_true", default=False,
                        help='Removes the sky component from the cube')
    
    parser.add_argument('--nskyspaxels',  type=int, default=50,
                        help='Number of faintest spaxels used to estimate the sky')
    
    parser.add_argument('--ccd',  type=str, default=None,
                        help='provide the origin ccd file')
    
    parser.add_argument('--date',  type=str, default=None,
                        help='Set the date')
    args = parser.parse_args()
    # ================= #
    #  The Scripts      #
    # ================= #
    cube = get_sedmcube(args.infile)
    if args.rmsky:
        cube.remove_sky(nspaxels=args.nskyspaxels, usemean=False)
    if args.ccd is not None:
        from pysedm import io, get_ccd
        if args.date is None:
            date = args.ccd.split("b_ifu")[-1].split("_")[0]
            print("used date %s"%date)
            
        ccd = get_ccd(args.ccd, tracematch=io.load_nightly_tracematch(date, False))
    else:
        ccd = None
    # - The plotting
    cube.show(interactive=True, notebook=False, ccd=ccd)
