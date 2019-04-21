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
    from pysedm.script.ccd_to_cube import *
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description="""pysedm pipeline to build the cubebuilder objects
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='The date YYYYMMDD')

    parser.add_argument('--rebuild',  action="store_true", default=False,
                        help='If the object you want to build already exists, nothing happens except if this is set')
    
    parser.add_argument('--noguider',  action="store_true", default=False,
                        help='Avoid having a guider stack created. [part of the --build]')
    
    parser.add_argument('--solvewcs',  action="store_true", default=False,
                        help='Shall the wcs solution of the guider be solved (ignored if --noguider). [part of the --build]')
    
    # --------------- #
    #  Cube Building  #
    # --------------- #
    parser.add_argument('--build',  type=str, default=None,
                        help='Build a e3d cube of the given target (accepting regex) or target list (csv) e.g. --build dome or --build dome,Hg,Cd')

    parser.add_argument('--quickbuild', action="store_true", default=False,
                        help='Faster cube extraction: No background No j-flexure, no i-flexure. Keyword "quickbuild" added to the cube.')

    parser.add_argument('--nobackground', action="store_true", default=False,
                        help='Shall the ccd background be 0 instead of the usual pipeline background?')
    
    parser.add_argument('--noflexure', action="store_true", default=False,
                        help='build cubes without flexure correction')
    
    parser.add_argument('--notraceflexure', action="store_true", default=False,
                        help='build cubes without trace flexure (j-ccd correction)')
    
    parser.add_argument('--buildbkgd',  type=str, default=None,
                        help='Build a ccd background of the given target or target list (csv) e.g. --build dome or --build dome,Hg,Cd')

    parser.add_argument('--buildcal',  type=str, default=None,
                        help='Build the flux calibrated e3d cubes. Set this to "*" to use the --build arguments.')

    parser.add_argument('--calsource',  type=str, default=None,
                        help='The Inverse sensitivity spectrum used to calibrated the cubes. By default this uses the latest one.')
    
    # - Trace Matching
    parser.add_argument('--tracematch', action="store_true", default=False,
                        help='build the tracematch solution for the given night. This option saves masks (see tracematchnomasks)')
    
    parser.add_argument('--tracematchnomasks', action="store_true", default=False,
                        help='build te tracematch solution for the given night without saved the masks')
    
    # - Hexagonal Grid
    parser.add_argument('--hexagrid', action="store_true", default=False,
                        help='build the hexagonal grid (index<->qr<->xy) for the given night')

    # - Wavelength Solution
    parser.add_argument('--wavesol', action="store_true", default=False,
                        help='build the wavelength solution for the given night.')
    
    parser.add_argument('--spaxelrange', type=str, default="None",
                        help='Provide a range of spaxel indexe A,B ; only traces with index i>=A and i<B will be loaded. Indicated in saved filename.')
    
    parser.add_argument('--wavesoltest', type=str, default="None",
                        help='to be used with --wavesol. By setting --wavesoltest N one N random wavelength solution will be performed.')

    parser.add_argument('--wavesolplots', action="store_true", default=False,
                        help='Set this to save individual wavelength solution fit results')
    
    # ----------------- #
    #  Raw Calibration  #
    # ----------------- #
    parser.add_argument('--flat',    action="store_true", default=False,
                        help='Build the flat fielding for the night [see flatref to the reference object]')

    parser.add_argument('--flatref',  type=str, default="dome",
                        help='Build the flat fielding for the night ')
    
    parser.add_argument('--flatlbda',  type=str, default="5000,8500",
                        help='The wavelength range for the flat field. Format: min,max [in Angstrom] ')

    # ----------------- #
    #  Short Cuts       #
    # ----------------- #
    
    
    parser.add_argument('--allcalibs', action="store_true", default=False,
                        help='')
    
    parser.add_argument('--allscience', action="store_true", default=False,
                        help='')

    parser.add_argument('--nofig',    action="store_true", default=False,
                        help='')

    args = parser.parse_args()

    # Matplotlib
    # ================= #
    #   The Scripts     #
    # ================= #
    
    # --------- #
    #  Date     #
    # --------- #
    date = args.infile

    # ------------ #
    # Short Cuts   #
    # ------------ #
    if args.allcalibs:
        args.tracematch = True
        args.hexagrid   = True
        args.wavesol    = True
        args.build      = "dome"
        args.flat       = True

        
    # ================= #
    #   Actions         #
    # ================= #
            
    # - Builds
    if args.quickbuild:
        args.noflexure      = True
        args.notraceflexure = True
        args.nobackground   = True
        fileindex = "quickbuild"
    else:
        fileindex = ""
        
    if args.build is not None and len(args.build) >0:
        for target in args.build.split(","):
            build_night_cubes(date, target=target,
                             lamps=True, only_lamps=False, skip_calib=True,
                             fileindex=fileindex,
                             nobackground=bool(args.nobackground),
                             # - options
                             build_guider = False if args.noguider else True,
                             solve_wcs = args.solvewcs,
                             savefig = False if args.nofig else True,
                             flexure_corrected = False if args.noflexure else True,
                             traceflexure_corrected = False if args.notraceflexure else True)
            
    if args.buildcal is not None:
        if args.buildcal=="*": args.buildcal=args.build
        if len(args.buildcal) >0:
            for target in args.buildcal.split(","):
                calibrate_night_cubes(date, target=target, 
                                    lamps=True, only_lamps=False, skip_calib=True)
                    
            
    # - Background
    if args.buildbkgd is not None and len(args.buildbkgd) > 0:
        for target in args.buildbkgd.split(","):
            build_backgrounds(date, target=target,
                            lamps=True, only_lamps=True, skip_calib=True, 
                            notebook=False)
            
    # -----------
    # 
    # ----------- 
    # - TraceMatch
    if args.tracematch or args.tracematchnomasks:
        build_tracematcher(date, save_masks= True if not args.tracematchnomasks else False,
                           notebook=False, rebuild=args.rebuild)
        
    # - Hexagonal Grid        
    if args.hexagrid:
        build_hexagonalgrid(date)
        
    # - Wavelength Solution
    if args.wavesol:
        ntest = None if "None" in args.wavesoltest else int(args.wavesoltest)
        spaxelrange = None if "None" in args.spaxelrange else np.asarray(args.spaxelrange.split(","), dtype="int")

        build_wavesolution(date, ntest=ntest, use_fine_tuned_traces=False,
                            idxrange=spaxelrange,
                            lamps=["Hg","Cd","Xe"], saveindividuals=args.wavesolplots,
                            savefig = False if args.nofig else True,
                            rebuild=args.rebuild)

    # - Flat Fielding
    if args.flat:
        lbda_min,lbda_max = np.asarray(args.flatlbda.split(","), dtype="float")
        build_flatfield(date,
                        lbda_min=lbda_min,
                        lbda_max=lbda_max,
                        ref=args.flatref, build_ref=True,
                        savefig=~args.nofig)
        # Now calc stats
        from pysedm import ccd
        from pysedm.io import get_datapath
        import numpy as np
        dome = ccd.get_dome("dome.fits", background=0, load_sep=True)
        a, b = dome.sepobjects.get(["a", "b"]).T
        savefile = get_datapath(date) + "%s_dome_stats.txt" % date
        stat_f = open(savefile, "w")
        stat_f.write("NSpax: %d\n" % len(b))
        stat_f.write("MinWid: %.3f\n" % min(b))
        stat_f.write("MaxWid: %.3f\n" % max(b))
        stat_f.write("MedWid: %.3f\n" % np.nanmedian(b))
        stat_f.write("AvgWid: %.3f\n" % np.nanmean(b))
        stat_f.write("MinLen: %.3f\n" % min(a))
        stat_f.write("MaxLen: %.3f\n" % max(a))
        stat_f.write("MedLen: %.3f\n" % np.nanmedian(a))
        stat_f.write("AvgLen: %.3f\n" % np.nanmean(a))
        stat_f.close()
        print("nspax, min, avg, med, max Wid: %d, %.3f, %.3f, %.3f, %.3f" %
              (len(b), min(b), np.nanmean(b), np.nanmedian(b), max(b)))
