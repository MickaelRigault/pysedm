#! /usr/bin/env python
# -*- coding: utf-8 -*-


#################################
#
#   MAIN 
#
#################################
if __name__ == "__main__":
    
    import argparse
    import numpy as np
    import pyifu
    
    from pysedm import get_sedmcube, io

    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" extract the A/B pair of a given cube""",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default="None",
                        help='cube filepath')

    # // AUTOMATIC EXTRACTION
    parser.add_argument('--inputcube',   action="store_true", default=False,
                        help='Is the input a cube ?')
    
    parser.add_argument('--reducer',  type=str, default=io.SEDM_REDUCER,
                        help='What is your name? This will be added in the '
                             'header [default: $SEDM_REDUCER]')

    # Auto PSF
    parser.add_argument('--auto',  type=str, default=None,
                        help='Shall this run an automatic PSF extraction')

    parser.add_argument('--autorange',  type=str, default="4000,8000",
                        help='Wavelength range [in Angstrom] for measuring the '
                             'metaslice PSF')
    
    parser.add_argument('--autobins',  type=int, default=7,
                        help='Number of bins within the wavelength range '
                             '(see --autorange)')

    parser.add_argument('--buffer',  type=float, default=10,
                        help='Radius [in spaxels] of the aperture used for the '
                             'PSF fit. (see --centroid for aperture center)')

    parser.add_argument('--psfmodel',  type=str, default="NormalMoffatTilted",
                        help='PSF model used for the PSF fit: '
                             'NormalMoffat{Flat/Tilted/Curved}')

    parser.add_argument('--seeing',  type=float, default=2,
                        help='NOT READY YET Expected seeing (not quite in '
                             'arcsec). This helps the PSF model to converge.')

    # Centroids
    parser.add_argument('--centroidA',  type=str, default="auto", nargs="+",
                        help='Where is the A point source expected to be?'
                             '\nYou have 3 potential cases:'
                             '\n * --centroid d auto: Use astrometry from '
                             'guider if possible or falls back to the '
                             'brightest mode'
                             '\n * --centroid brightest: Use the position of '
                             'the bightest spaxels'
                             '\n * --centroid x0,y0: : Use the given x0 y0 '
                             'position (separated by a comma).')

    parser.add_argument('--centroidB', type=str, default="auto", nargs="+",
                        help='Where is the B point source expected to be?'
                             '\nYou have 3 potential cases:'
                             '\n * --centroid d auto: Use astrometry from '
                             'guider if possible or falls back to the '
                             'brightest mode'
                             '\n * --centroid brightest: Use the position of '
                             'the bightest spaxels'
                             '\n * --centroid x0,y0: : Use the given x0 y0 '
                             'position (separated by a comma).')
    
    parser.add_argument('--centroiderr',  type=str, default="None", nargs=2,
                        help="Where is the point source location guess "
                             "boundaries?"
                             "\nYou have 2 potential cases:"
                             "\n   * --centroiderr None: [3 3] if position "
                             "from the guider, [5 5] if from the brightest mode"
                             "\n   * --centroiderr [X Y]: use the given values")

    # display options
    parser.add_argument('--display',  action="store_true", default=False,
                        help='Select the area to fit using '
                             'the display function.')

    parser.add_argument('--vmin',  type=str, default="2",
                        help='Data Percentage used for imshow "vmin" '
                             'when using the --display mode')
    
    parser.add_argument('--vmax',  type=str, default="98",
                        help='Data Percentage used for imshow "vmax" '
                             'when using the --display mode')

    # Flux calibration
    parser.add_argument('--nofluxcal',  action="store_true", default=False,
                        help='No Flux calibration')
    
    parser.add_argument('--fluxcalsource',  type=str, default="None",
                        help="Path to a 'fluxcal' *.fits file. This file will "
                             "be used for the flux calibration"
                             "\nIf nothing given, the nearest (in time and "
                             "within the night) fluxcal will be used.")
    
    # wavelength step
    parser.add_argument('--lstep',  type=int, default=1,
                        help='Slice width in lbda step: default is 1, '
                             'use 2 for fainter source and maybe '
                             '3 for really faint target')

    # I/O options
    parser.add_argument('--tag',  type=str, default="None",
                        help='Add the tag on output filename.')

    parser.add_argument('--nofig',    action="store_true", default=False,
                        help='Do not create any plot')

    parser.add_argument('--notroncation',  action="store_true", default=False,
                        help='Do not apply the blue and red end edge '
                             'troncation on flux calibrated spectra')

    # - Standard Star object
    parser.add_argument('--std',  action="store_true", default=False,
                        help='Set this to True to tell the program you want to '
                             'build a calibration spectrum from this object')

    args = parser.parse_args()

    # ================= #
    #   The Scripts     #
    # ================= #

    # --------- #
    #  Date     #
    # --------- #
    if not args.inputcube:
        in_date = args.infile
    else:
        in_date = None

    # --------- #
    #  Parsing  #
    # --------- #
    if len(args.centroidA) == 1:
        args.centroidA = args.centroidA[0]
    elif len(args.centroidA) > 2:
        raise ValueError("--centroidA should be 'auto', 'brightest' or X Y")
    if len(args.centroidB) == 1:
        args.centroidB = args.centroidB[0]
    elif len(args.centroidB) > 2:
        raise ValueError("--centroidB should be 'auto', 'brightest' or X Y")

    # ------------- #
    #               #
    # 3D PSF        #
    #               #
    # ------------- #
    if args.auto is not None and len(args.auto) > 0:
        
        for target in args.auto.split(","):
            
            filecubes = io.get_night_files(
                in_date, "cube.*", target=target.replace(".fits", "")) \
                if not args.inputcube else [target]
            print("cube file from which the spectra will be extracted "
                  "[auto]: " + ", ".join(filecubes))

            # - loop over the file cube
            for filecube in filecubes:
                print("Automatic extraction of target %s, file: %s" %
                      (target, filecube))
                # ===================== #
                # Running ExtractStar   #
                # ===================== #
                #
                # DATA IN
                #
                cube = get_sedmcube(filecube)
                cube.header['REDUCER'] = args.reducer
                #
                # Extract B observation first
                #
                abdata = cube.data
                cube._derived_properties['data'] = 2000. - abdata
                #
                # OPTIONS
                #
                es_options = {
                    # Display mode ?
                    "display": args.display,
                    "displayprop": {"vmin": args.vmin, "vmax": args.vmax},
                    # Metaslices
                    # -- How many slices
                    "step1range": np.asarray(args.autorange.split(","),
                                             dtype="float"),
                    "step1bins": args.autobins,
                    # -- Centroid guess                    
                    "centroid": args.centroidB,
                    "prop_position": {"centroiderr": args.centroiderr},
                    # Sub IFU
                    "spaxelbuffer": args.buffer,
                    # PSF Model
                    "psfmodel": args.psfmodel,
                    "spaxels_to_use": None,  # not ready yet,
                    # provide here directly the spaxels to be used
                    "fwhm_guess": args.seeing,
                    # Spectral options
                    "slice_width": int(args.lstep),
                    #
                    "verbose": True,
                    }
                
                # ===================== #
                # SOURCE EXTRACTION     #
                # ===================== #
                print(" Starting B extract_pointsource ".center(50, "-"))
                try:
                    es_out = cube.extract_pointsource(**es_options)
                except IOError:
                    # Astrometry was not solved
                    print("ERROR: Astrometry file not found, "
                          "using brightest spaxel instead")
                    es_options["centroid"] = "brightest"
                    es_out = cube.extract_pointsource(**es_options)
                except AttributeError:
                    # Astrometry solved, but target outside IFU
                    print("ERROR: Astrometry places target outside IFU, "
                          "using brightest spaxel instead")
                    es_options["centroid"] = "brightest"
                    es_out = cube.extract_pointsource(**es_options)
                    cube.header['QUALITY'] = 3
                # -> The output object
                es_object_B = cube.extractstar
                spec_B = es_object_B.get_spectrum()
                #
                # Extract A observation next
                #
                cube._derived_properties['data'] = 2000. + abdata
                #
                # OPTIONS
                #
                es_options = {
                    # Display mode ?
                    "display": args.display,
                    "displayprop": {"vmin": args.vmin, "vmax": args.vmax},
                    # Metaslices
                    # -- How many slices
                    "step1range": np.asarray(args.autorange.split(","),
                                             dtype="float"),
                    "step1bins": args.autobins,
                    # -- Centroid guess
                    "centroid": args.centroidA,
                    "prop_position": {"centroiderr": args.centroiderr},
                    # Sub IFU
                    "spaxelbuffer": args.buffer,
                    # PSF Model
                    "psfmodel": args.psfmodel,
                    "spaxels_to_use": None,  # not ready yet,
                    # provide here directly the spaxels to be used
                    "fwhm_guess": args.seeing,
                    # Spectral options
                    "slice_width": int(args.lstep),
                    #
                    "verbose": True,
                }

                # ================= #
                # SOURCE EXTRACTION #
                # ================= #
                print(" Starting A extract_pointsource ".center(50, "-"))
                try:
                    es_out = cube.extract_pointsource(**es_options)
                except IOError:
                    # Astrometry was not solved
                    print("ERROR: Astrometry file not found, "
                          "using brightest spaxel instead")
                    es_options["centroid"] = "brightest"
                    es_out = cube.extract_pointsource(**es_options)
                except AttributeError:
                    # Astrometry solved, but target outside IFU
                    print("ERROR: Astrometry places target outside IFU, "
                          "using brightest spaxel instead")
                    es_options["centroid"] = "brightest"
                    es_out = cube.extract_pointsource(**es_options)
                    cube.header['QUALITY'] = 3
                # -> The output object
                es_object = cube.extractstar
                #
                # Generate output file tag
                add_tag = "_%s" % args.tag if args.tag is not None and \
                    args.tag not in ["None", ""] \
                    else ""
                add_info_spec = "_notfluxcal" if \
                    not es_object.is_spectrum_fluxcalibrated() else ""
                spec_info = "_lstep%s" % es_options["slice_width"] + \
                            add_info_spec

                print("Tag added", add_tag)
                # -
                # - PLOTTING: A is reference
                # -
                plot_tag = "_auto" + add_tag + spec_info + "_"
                if not args.nofig:
                    print(" Saving figures ".center(50, "-"))
                    es_object.show_adr(
                        savefile=es_object.basename.replace(
                            "{placeholder}", "adr_fit" + plot_tag))
                    es_object.show_metaslices(
                        savefile=es_object.basename.replace(
                            "{placeholder}", "metaslices" + plot_tag))
                    es_object.show_extracted_spec(
                        savefile=es_object.basename.replace(
                            "{placeholder}", "extracted_spec" + plot_tag))
                    es_object.show_mla(
                        savefile=es_object.basename.replace(
                            "{placeholder}", "ifu_spaxels_source" + plot_tag),
                        bcoords=(spec_B.header['XPOS'], spec_B.header['YPOS']))
                    es_object.show_psf(
                        savefile=es_object.basename.replace(
                            "{placeholder}", "psfprofile" + plot_tag),
                        sliceid=2)
                # Generate A/B spectrum
                spec_A = es_object.get_spectrum()

                spec_AB = pyifu.get_spectrum(spec_A.lbda,
                                             (spec_A.data + spec_B.data) / 2.,
                                             variance=(spec_A.variance +
                                                       spec_B.variance) / 2.,
                                             header=spec_A.header)
                # Update A/B spectrum header
                spec_AB.header['EXPTIME'] = spec_A.header['EXPTIME'] + \
                                            spec_B.header['EXPTIME']
                spec_AB.header['ABPSFWA'] = spec_A.header['PSFFWHM']
                spec_AB.header['ABXPOSA'] = spec_A.header['XPOS']
                spec_AB.header['ABYPOSA'] = spec_A.header['YPOS']
                spec_AB.header['ABPSFWB'] = spec_B.header['PSFFWHM']
                spec_AB.header['ABXPOSB'] = spec_B.header['XPOS']
                spec_AB.header['ABYPOSB'] = spec_B.header['YPOS']
                es_object._derived_properties["spectrum"] = spec_AB
                # Save A/B spectrum
                es_object.writeto(basename=None, add_tag="auto" + add_tag,
                                  add_info=None)
                # Plot A/B spectrum
                es_object.spectrum.show(
                    savefile=es_object.basename.replace(
                        "{placeholder}", "spec" + plot_tag))
