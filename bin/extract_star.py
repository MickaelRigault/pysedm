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
    from shapely      import geometry
    # 
    from psfcube      import script
    from pysedm       import get_sedmcube, io, fluxcalibration, sedm
    from pysedm.sedm  import IFU_SCALE_UNIT
    
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')

    # // AUTOMATIC EXTRACTION
    parser.add_argument('--auto',  type=str, default=None,
                        help='Shall this run an automatic PSF extraction')

    parser.add_argument('--autorange',  type=str, default="4500,7000",
                        help='Wavelength range [in Angstrom] for measuring the metaslice PSF')
    
    parser.add_argument('--autobins',  type=int, default=7,
                        help='Number of bins within the wavelength range (see --autorange)')

    parser.add_argument('--centroid',  type=str, default=None,
                        help='Where is the point source expected to be? using the "x,y" format. If None, it will be guessed.'+
                            "\nGuess works well for isolated sources.")
    
    parser.add_argument('--buffer',  type=float, default=10,
                        help='Radius [in spaxels] of the aperture used for the PSF fit. (see --centroid for aperture center)')

    parser.add_argument('--psfmodel',  type=str, default="NormalMoffatTilted",
                        help='PSF model used for the PSF fit: NormalMoffat{Flat/Tilted/Curved}')

    parser.add_argument('--lstep',  type=int, default=1,
                        help='Slice width in lbda step: default is 1, use 2 for fainter source and maybe 3 for really faint target')
    
    parser.add_argument('--display',  action="store_true", default=False,
                        help='Select the area to fit using the display function.')

    # - Standard Star object
    parser.add_argument('--std',  action="store_true", default=False,
                        help='Set this to True to tell the program you what to build a calibration spectrum from this object')

    
    parser.add_argument('--nofig',    action="store_true", default=False,
                        help='')

        
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
    extracted_objects = []
    # ---------- #
    # Extraction #
    # ---------- #
    # - Automatic extraction
    if args.auto is not None and len(args.auto) >0:
        final_slice_width = int(args.lstep)
        # - Step 1 parameters
        lbdaranges, bins = np.asarray(args.autorange.split(","), dtype="float"), int(args.autobins+1)
        STEP_LBDA_RANGE = np.linspace(lbdaranges[0],lbdaranges[1], bins+1)
        lbda_step1      = np.asarray([STEP_LBDA_RANGE[:-1], STEP_LBDA_RANGE[1:]]).T
        
        for target in args.auto.split(","):
            filecubes = io.get_night_files(date, "cube.*", target=target.replace(".fits",""))
            print("cube file from which the spectra will be extracted: "+ ", ".join(filecubes))
            
            # - loop over the file cube
            for filecube in filecubes:
                # ----------------- #
                #  Cube to Fit?     #
                # ----------------- #
                print("Automatic extraction of target %s, file: %s"%(target, filecube))
                cube_ = get_sedmcube(filecube)
                if args.display:
                    iplot = cube_.show(interactive=True)
                    cube = cube_.get_partial_cube( iplot.get_selected_idx(), np.arange( len(cube_.lbda)) )
                    args.buffer = 20
                else:
                    cube = cube_
                # Centroid ?
                if args.centroid is None:
                    sl = cube.get_slice(lbda_min=lbdaranges[0], lbda_max=lbdaranges[1], slice_object=True)
                    x,y = np.asarray(sl.index_to_xy(sl.indexes)).T # Slice x and y
                    argmaxes = np.argwhere(sl.data>np.percentile(sl.data,95)).flatten() # brightest points
                    xcentroid,ycentroid  = np.nanmean(x[argmaxes]),np.nanmean(y[argmaxes]) # centroid
                else:
                    xcentroid, ycentroid = np.asarray(args.centroid.split(","), dtype="float")
                    
                # Aperture area ?
                point_polygon = geometry.Point(xcentroid, ycentroid).buffer( float(args.buffer) )
                # => Cube to fit
                cube_to_fit = cube.get_partial_cube( cube.get_spaxels_within_polygon(point_polygon),
                                                      np.arange(len(cube.lbda)))
                # --------------
                # Fitting
                # --------------
                spec, cubemodel, psfmodel, bkgdmodel, psffit, slpsf  = \
                  script.extract_star(cube_to_fit, centroids=[xcentroid, ycentroid],
                                          spaxel_unit = IFU_SCALE_UNIT,
                                          final_slice_width = final_slice_width,
                                          lbda_step1=lbda_step1, psfmodel=args.psfmodel)
                # Hack to be removed:
                print("INFO: Temporary variance hacking to be removed ")
                spec._properties['variance'] = np.ones(len(spec.lbda)) * np.median(spec.variance)

                if final_slice_width != 1:
                    spec = spec.reshape(cube.lbda)

                    #cubemodel.reshape(cube_to_fit.lbda)
                    
                #cuberes   = cube_to_fit - cubemodel

                # --------------
                # Flux Calibation
                # --------------
                if not args.std:
                    from pyifu import load_spectrum
                    fluxcal = load_spectrum(io.fetch_nearest_fluxcal(date, cube.filename))
                    spec.scale_by(1/fluxcal.data)
                    spec.header["FLUXCAL"] = ("True","has the spectra been flux calibrated")
                    spec.header["CALSRC"] = (fluxcal.filename.split("/")[-1], "Flux calibrator filename")
                    
                # --------------
                # Recording
                # --------------
                io._saveout_forcepsf_(filecube, cube, cuberes=None, cubemodel=cubemodel,
                                          cubefitted=cube_to_fit, spec=spec)
                
                if not args.nofig:
                    psffit.show_adr(savefile=spec.filename.replace("spec","adr_fit").replace(".fits",".pdf") ) 
                    psffit.slices[2]["slpsf"].show(savefile=spec.filename.replace("spec","psfprofile").replace(".fits",".pdf"))
                    
                    import matplotlib.pyplot as mpl
                    cube_.show(show=False)
                    ax = mpl.gca()
                    x,y = np.asarray(cube_to_fit.index_to_xy(cube_to_fit.indexes)).T
                    ax.plot(x,y, marker=".", ls="None", ms=1, color="k")
                    ax.figure.savefig(spec.filename.replace("spec","spaxels_source").replace(".fits",".pdf"))
                # -----------------
                #  Is that a STD  ?
                # -----------------
                if args.std and cube.header['IMGTYPE'].lower() in ['standard']:
                    spec.header['OBJECT'] = cube.header['OBJECT']
                    speccal, fl = fluxcalibration.get_fluxcalibrator(spec, fullout=True)
                    speccal.header["SOURCE"] = (spec.filename.split("/")[-1], "This object has been derived from this file")
                    speccal.header["PYSEDMT"] = ("Flux Calibration Spectrum", "Object to use to flux calibrate")
                    filename_inv = spec.filename.replace(io.PROD_SPECROOT,io.PROD_SENSITIVITYROOT)
                    speccal._side_properties['filename'] = filename_inv
                    speccal.writeto(filename_inv)
                    if not args.nofig:
                        fl.show(savefile=speccal.filename.replace(".fits",".pdf"), show=False, fluxcal=speccal.data)
                                    
                # - for the record
                extracted_objects.append(spec)
                
    else:
        print("NO  AUTO")
        
