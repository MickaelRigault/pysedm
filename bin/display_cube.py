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

    args = parser.parse_args()
    # ================= #
    #  The Scripts      #
    # ================= #
    cube = get_sedmcube(args.infile)
    if args.rmsky:
        cube.remove_sky(self, nspaxels=args.nskyspaxels, usemean=False)

    # - The plotting
    cube.show(interactive=True, notebook=False)
