#! /usr/bin/env python
# -*- coding: utf-8 -*-


#################################
#
#   MAIN 
#
#################################
if  __name__ == "__main__":
    import subprocess
    import argparse
    import pysedm
    import numpy as np
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description="""tool to subprocess wavelength solution build.
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='The date YYYYMMDD')

    # --------------- #
    #  Wavesoltuion   #
    # --------------- #
    parser.add_argument('--nsub',  type=str, default=2,
                        help='Number of subprocess to launch. [default 2]')
    
    parser.add_argument('--merge',  action="store_true", default=False,
                        help='Set this keyword to perform the Wavesolution merging. The rest will be ignored.')
    # --------------- #
    #  Subprocessing  #
    # --------------- #
    parser.add_argument('--spaxelrange', type=str, default="None",
                        help='Provide a range of spaxel indexe A,B ; only traces with index i>=A and i<B will be loaded. Indicated in saved filename.')
    
    parser.add_argument('--wavesolplots', action="store_true", default=False,
                        help='Set this to save individual wavelength solution fit results')

    parser.add_argument('--nofig', action="store_true", default=False,
                        help='')
    
    parser.add_argument('--rebuild',  action="store_true", default=False,
                        help='If the object you want to build already exists, nothing happens except if this is set')
    
    # - End
    args = parser.parse_args()
    
    # ================= #
    #   The Scripts     #
    # ================= #

    # --------- #
    #  Date     #
    # --------- #
    date = args.infile

    if args.merge:
        from pysedm.wavesolution import merge_wavesolutions
        wsol = merge_wavesolutions( pysedm.load_nightly_wavesolution(date,True))
        wsol.writeto( pysedm.io.get_datapath(date)+"%s_WaveSolution.pkl"%date)
        if not args.nofig:
            hgrid = pysedm.load_nightly_hexagonalgrid(date)
            wsol.show_dispersion_map(hgrid, vmax="98", vmin="2", outlier_highlight=5,
                            savefile= pysedm.io.get_datapath(date)+"%s_wavesolution_dispersionmap.pdf"%date)
        import sys
        sys.exit(0)
    
    nsplit = int(args.nsub)
    # - Wavelength Solution
    
    idxall = pysedm.load_nightly_tracematch(date, withmask=False).get_traces_within_polygon(pysedm.sedm.INDEX_CCD_CONTOURS)
    # Split it in equal parts:
    k, m      = divmod( np.nanmax(idxall)+10, nsplit)
    idxbounds = [[i * k + min(i, m),(i + 1) * k + min(i + 1, m)] for i in range(nsplit)]
    

    options = []
    if args.nofig:
        options += ["--nofig"]
    if args.wavesolplots:
        options += ["--wavesolplots"]
    if args.spaxelrange not in ['None']:
        print(args.spaxelrange)
        options += ["--spaxelrange","%s,%s"%(args.spaxelrange[0],args.spaxelrange[1])]
    if args.rebuild:
        print(args.rebuild)
        options += ["--rebuild"]

        
    for i,bounds in enumerate(idxbounds):          
        command = ["ccd_to_cube.py", date, "--wavesol","--spaxelrange","%s,%s"%(bounds[0],bounds[1])] + options
        print("launching command: "+" ".join(command))
        subprocess.Popen(command, stdout=subprocess.PIPE)

    
    
