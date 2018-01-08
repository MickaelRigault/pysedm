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
    from pysedm import get_sedmcube, io
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')
    
    parser.add_argument('--adr',  type=str, default=None,
                        help='Run the ADR fit and compares it to the expected centroid positions')
    
    args = parser.parse_args()
    
    date = args.infile
    # ================= #
    #   The Scripts     #
    # ================= #
    if args.adr is not None:
        from pysedm.utils import adrfit
        for target in args.adr.split(","):
            filecubes = io.get_night_files(date, "cube.*", target=target.replace(".fits",""))
            for filecube in filecubes:
                file_ = filecube.split('/')[-1]
                sourcedir = "/".join(filecube.split('/')[:-1])
                print("Estimated ADR parameters for %s"%file_)
                cube      = get_sedmcube(filecube)
                # - fit the ADR and save the image
                fitvalues = adrfit.get_cube_adr_param(cube, show=False, savefile= sourcedir+"/adrfit_%s"%file_.replace(".fits",".pdf"))
                # - Save the fitvalues
                fileout = open(sourcedir+"/adrfit_%s"%file_.replace(".fits",".dat"),"w")
                fileout.write("#fitvalues of ADR fit for %s \n"%file_)
                for k,v in fitvalues.items():
                    fileout.write("%s %s\n"%(k,v))
                fileout.close()
                    

    
