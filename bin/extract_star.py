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
    from pysedm import get_sedmcube
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')

    parser.add_argument('--keepsky',  action="store_true", default=False,
                        help='Removes the sky component from the cube')
    
    parser.add_argument('--nskyspaxels',  type=int, default=50,
                        help='Number of faintest spaxels used to estimate the sky')


    
    args = parser.parse_args()
    # ================= #
    #  The Scripts      #
    # ================= #
    cube = get_sedmcube(args.infile)
    if not args.keepsky:
        cube.remove_sky(nspaxels=args.nskyspaxels, usemean=False)
        cube._sky.writeto(cube.filename.replace("e3d","skyspec").replace(".gz",""))
    
    es = extractstar.get_extractstar(cube)
    es.fit()
    spec = es.get_modelspectrum()
    if not spec.has_variance() or np.all(np.isnan(spec.variance)):
        specbrightest = cube.get_spectrum(cube.get_brightest_spaxels(100))
        spec._properties["variance"] = specbrightest.variance

    spec.writeto(cube.filename.replace("e3d","spec").replace(".gz",""))
        
