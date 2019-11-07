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
        description=""" run the interactive plotting of a given cube""",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default="None",
                        help='cube filepath')

    # // AUTOMATIC EXTRACTION
    parser.add_argument('--inputcube',   action="store_true", default=False,
                        help='Is the input a cube ?')
    
    parser.add_argument('--reducer',  type=str, default=io.SEDM_REDUCER,
                        help='What is your name? This will be added in the header [default: $SEDM_REDUCER]')

    #  which extraction
    parser.add_argument('--aperture',  type=str, default=None,
                        help='Do you want a simple aperture extraction ?')
    
    parser.add_argument('--ap_bkgdscale',  type=float, default=1.4,
                        help='Default width scale of the outter background ring [default 1.4]')

    # Auto PSF
    parser.add_argument('--auto',  type=str, default=None,
                        help='Shall this run an automatic PSF extraction')

    parser.add_argument('--autorange',  type=str, default="4000,8000",
                        help='Wavelength range [in Angstrom] for measuring the metaslice PSF')
    
    parser.add_argument('--autobins',  type=int, default=7,
                        help='Number of bins within the wavelength range (see --autorange)')

    parser.add_argument('--buffer',  type=float, default=10,
                        help='Radius [in spaxels] of the aperture used for the PSF fit. (see --centroid for aperture center)')

    parser.add_argument('--psfmodel',  type=str, default="NormalMoffatTilted",
                        help='PSF model used for the PSF fit: NormalMoffat{Flat/Tilted/Curved}')

    parser.add_argument('--seeing',  type=float, default=None,
                        help='Expected seeing. This helps the PSF model to converge.')

    # Centroid
    parser.add_argument('--centroid',  type=str, default="auto", nargs=2,
                        help='Where is the point source expected to be ?'+
                            "\nYou have 3 potential cases:"+
                            "\n   * --centroid auto: Use astrometry from guider if possible or falls back to the 'brightest' mode"+
                            "\n   * --centroid brightest: Use the position of the bightest spaxels "+
                            "\n   * --centroid x0 y0: Use the given x0 y0 position."
                            )
    
    parser.add_argument('--centroiderr',  type=str, default="None",nargs=2,
                        help='Where is the point source location guess boundaries ?'+
                        "\nYou have 2 potential cases:"+
                            "\n   * --centroiderr None: [3 3] if position from the guider, [5 5] if from the 'brightest' mode "+
                            "\n   * --centroiderr [X Y]: use the given values")

    # wavelength step
    parser.add_argument('--display',  action="store_true", default=False,
                        help='Select the area to fit using the display function.')

    parser.add_argument('--vmin',  type=str, default="2",
                        help='Data Percentage used for imshow "vmin" when using the --display mode')
    
    parser.add_argument('--vmax',  type=str, default="98",
                        help='Data Percentage used for imshow "vmax" when using the --display mode')

    # Flux calibration
    parser.add_argument('--nofluxcal',  action="store_true", default=False,
                        help='No Flux calibration')
    
    parser.add_argument('--fluxcalsource',  type=str, default="None",
                        help="Path to a 'fluxcal' .fits file. This file will be used for the flux calibration"+
                        "\nIf nothing given, the nearest (in time and within the night) fluxcal will be used.")
    
    # wavelength step
    parser.add_argument('--lstep',  type=int, default=1,
                        help='Slice width in lbda step: default is 1, use 2 for fainter source and maybe 3 for really faint target')

    # I/O options
    parser.add_argument('--tag',  type=str, default="None",
                        help='Add the tag on output filename.')

    parser.add_argument('--nofig',    action="store_true", default=False,
                        help='Do not create any plot')

    parser.add_argument('--notroncation',  action="store_true", default=False,
                        help='Do not apply the blue and red end edge troncation on flux calibrated spectra')

    # - Standard Star object
    parser.add_argument('--std',  action="store_true", default=False,
                        help='Set this to True to tell the program you what to build a calibration spectrum from this object')
    
    
    
    

        
    args = parser.parse_args()




    # ================= #
    #   The Scripts     #
    # ================= #

    # --------- #
    #  Date     #
    # --------- #
    if not args.inputcube:
        date = args.infile



    # ------------- #
    #               #
    # APERTURE      #
    #               #
    # ------------- #
    if args.aperture is not None and len(args.aperture) >0:
        raise NotImplementedError("--aperture option not implemented yet. Use --auto")
        if args.std:
            raise NotImplementedError("--std option cannot be used (yet) with --aperture. Use --auto")

    

    # ------------- #
    #               #
    # 3D PSF        #
    #               #
    # ------------- #
    if args.auto is not None and len(args.auto) >0:
        
        for target in args.auto.split(","):
            
            filecubes = io.get_night_files(date, "cube.*", target=target.replace(".fits","")) if not args.inputcube else [target]
            print("cube file from which the spectra will be extracted [auto]: "+ ", ".join(filecubes))

            # - loop over the file cube
            for filecube in filecubes:
                print("Automatic extraction of target %s, file: %s"%(target, filecube))
                # ===================== #
                # Running ExtractStar   #
                # ===================== #
                #
                # DATA IN
                #
                cube = get_sedmcube(filecube)
                #
                # OPTIONS
                #
                es_options = {
                    # Display mode ?
                    "display":args.display, "displayprop":{"vmin":args.vmin, "vmax":args.vmax},
                    
                    # Metaslices
                    # -- How many slices
                    "step1range": np.asarray(args.autorange.split(","), dtype="float"),
                    "step1bins": args.autobins,
                    # -- Centroid guess                    
                    "centroid": args.centroid, "prop_position":{"centroiderr":args.centroid},
                    # Sub IFU
                    "spaxelbuffer": args.buffer,
                    # PSF Model
                    "psfmodel": args.psfmodel,
                    "spaxels_to_use":None, # not ready yet, provide here directly the spaxels to be used
                    "fwhm_guess": args.seeing,
                    # Spectral options
                    "slice_width": int(args.lstep),
                    }
                
                #
                # SOURCE EXTRACTION
                #
                es_out = cube.extract_pointsource(**es_options)
                # ===================== #
                # Extract star output   #
                # ===================== #                
                es_object = cube.extractstar
                
                # -
                # - PLOTTING
                # - 
                if not args.nofig:
                    es_object.show_adr(            savefile=es_object.basename.replace("{placeholder}","adr_fit"))
                    es_object.show_metaslices(     savefile=es_object.basename.replace("{placeholder}","metaslices"))
                    es_object.show_extracted_spec( savefile=es_object.basename.replace("{placeholder}","spec_extracted"))
                    es_object.show_mla(            savefile=es_object.basename.replace("{placeholder}","ifu_spaxels_source"))
                    es_object.show_psf(            savefile=es_object.basename.replace("{placeholder}","psfprofile"), sliceid=2)

                    if cube.header['IMGTYPE'].lower() in ['standard'] and es_object.is_spectrum_fluxcalibrated():
                        from pysedm.fluxcalibration import show_fluxcalibrated_standard
                        show_fluxcalibrated_standard(es_object.spectrum, savefile=es_object.basename.replace("{placeholder}","adr_fit")+".pdf")
                        show_fluxcalibrated_standard(es_object.spectrum, savefile=es_object.basename.replace("{placeholder}","adr_fit")+".png")

                
                # -
                # - Standard Specific
                # -
                if args.std and cube.header['IMGTYPE'].lower() in ['standard'] and 'AIRMASS' in cube.header:
                    # Spectrum used for flux calibration
                    raw_spec = es_object.get_spectrum("raw", persecond=True, troncate_edges=False)
                    
                    if raw_spec.header['QUALITY'] == 0:
                        from pysed import fluxcalibration
                        # Based on the flux non calibrated spectsra
                        try:
                            speccal, fl = fluxcalibration.get_fluxcalibrator(
                                spec_raw, fullout=True)
                            speccal.header["SOURCE"] = (
                                spec.filename.split("/")[-1],
                                "This object has been derived from this file")
                            speccal.header["PYSEDMT"] = (
                                "Flux Calibration Spectrum",
                                "Object to use to flux calibrate")
                            filename_inv = spec.filename.replace(
                                io.PROD_SPECROOT,
                                io.PROD_SENSITIVITYROOT).replace("notfluxcal",
                                                                 "")
                            speccal._side_properties['filename'] = filename_inv
                            speccal.writeto(filename_inv)
                            if not args.nofig:
                                fl.show(
                                    savefile=speccal.filename.replace(".fits",
                                                                      ".pdf"),
                                    show=False)
                                fl.show(
                                    savefile=speccal.filename.replace(".fits",
                                                                      ".png"),
                                    show=False)
                        except OSError:
                            print("WARNING: no reference spectrum for target, "
                                  "skipping flux calibration")
                    else:
                        print("WARNING: Standard spectrum of low quality, "
                              "skipping fluxcal generation")
                
                # -
                # - SAVING
                # - 
                es_object.writeto(basename=None, add_tag=args.tag, add_info=None)
