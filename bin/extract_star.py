#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pysedm
from pysedm       import get_sedmcube, io, fluxcalibration, sedm, astrometry
from pysedm.sedm  import IFU_SCALE_UNIT, SEDM_MLA_RADIUS
from pyifu import get_spectrum






def build_aperture_param(stored_picked_poly, default_bkgd_scale=1.4):
    """ Build an aperture spectroscopy parameters

    Parameters
    ----------
    stored_picked_poly: [list]
        list made by InteractivePlot:
        - [ [ "Circle",[[xcentroid, ycentroid], radius]],
            [ "Circle",[[xcentroid_i, ycentroid_i], radius_i]]
          ]
        It should contains 1 or 2 Circle entries:
        - If only 1, then the background will be defined by the annulus made of 1 and `default_bkgd_scale` times the picked circle.
        - If 2, the background will be the annulus defined by the difference between the 2 circles. 
          The centroid will be the average of the 2 picked circles. 
    
    default_bkgd_scale: [float] -optional-
        Default scale used to define the background if only 1 circle provided (see above)

    Returns
    -------
    2x2 numpy array 
        ([xcentroid, ycentroid], [aperture_radius, background_outer_radius])
        
    """
    idx_circles = [i for i,k in enumerate(stored_picked_poly) if k[0] == "Circle"]
    if len(idx_circles)>2:
        raise AttributeError("More than 2 Circle instance currently stored in plot_interactive. Cannot build a Aperture")
        
    elif len(idx_circles)<1:
        raise AttributeError("You need to have defined at least 1 circular aperture.")
    
    centroids = [stored_picked_poly[i][1][0] for i in idx_circles]
    radiuses  = [stored_picked_poly[i][1][1] for i in idx_circles]
    if len(idx_circles) == 2:
        return np.mean(centroids, axis=0), np.sort(radiuses)
    else:
        print("INFO: default background radius scale used (annulus made of [1,%s] times the aperture radius)"%default_bkgd_scale)
        return np.asarray(centroids[0]), radiuses[0]*np.asarray([1, default_bkgd_scale])





# ======================= #
#
#   Pipeline Steps        #
#
# ======================= #


# -------------------- #
#  Flux Calibration    #
# -------------------- #
def flux_calibrate(spec, fluxcalfile=None, nofluxcal=False):
    """ """
    if fluxcalfile in ["None"]:
        fluxcalfile = None

    if not nofluxcal:
        # Which Flux calibration file ?
        if fluxcalfile is None:
            from pysedm import io
            print("INFO: default nearest fluxcal file used")
            date = io.header_to_date(spec.header)
            fluxcalfile = io.fetch_nearest_fluxcal(date, spec.filename)
        else:
            print("INFO: given fluxcal used.")

        # Do I have a flux calibration file ?
        if fluxcalfile is None:
            print("ERROR: No fluxcal for night %s and no alternative fluxcalsource provided. Uncalibrated spectra saved."%date)
            spec.header["CALSRC"] = (None, "Flux calibrator filename")
            flux_calibrated=False
        else:
            from pysedm.fluxcalibration import load_fluxcal_spectrum
            fluxcal = load_fluxcal_spectrum( fluxcalfile ) 
            spec.scale_by( fluxcal.get_inversed_sensitivity(spec.header.get("AIRMASS", 1.1) ), onraw=False)
            spec.header["CALSRC"] = (fluxcal.filename.split("/")[-1], "Flux calibrator filename")
            flux_calibrated=True
            
    else:
        spec.header["FLUXCAL"] = (False,"has the spectra been flux calibrated")
        spec.header["CALSRC"] = (None, "Flux calibrator filename")
        flux_calibrated=False
        
    # Flux Calibration
    if flux_calibrated:
        spec.header["FLUXCAL"] = (True,"has the spectra been flux calibrated")
        spec.header["BUNIT"]  = ("erg/s/A/cm^2","Flux Units")
    else:
        spec.header["FLUXCAL"] = (False,"has the spectra been flux calibrated")
        spec.header["BUNIT"]  = (spec.header.get('BUNIT',""),"Flux Units")
    return spec, flux_calibrated

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
    import psfcube
    from psfcube      import script
    
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

    parser.add_argument('--autorange',  type=str, default="4500,7000",
                        help='Wavelength range [in Angstrom] for measuring the metaslice PSF')
    
    parser.add_argument('--autobins',  type=int, default=7,
                        help='Number of bins within the wavelength range (see --autorange)')

    parser.add_argument('--buffer',  type=float, default=8,
                        help='Radius [in spaxels] of the aperture used for the PSF fit. (see --centroid for aperture center)')

    parser.add_argument('--psfmodel',  type=str, default="NormalMoffatTilted",
                        help='PSF model used for the PSF fit: NormalMoffat{Flat/Tilted/Curved}')
    
    parser.add_argument('--normed',  action="store_true", default=False,
                        help='[DEV USE ONLY]: apply psfcube normalization on the psf profile.')

    # Apperture


    # // Generic information
    parser.add_argument('--centroid',  type=str, default="None",nargs=2,
                        help='Where is the point source expected to be? using the "x,y" format. If None, it will be guessed.'+
                            "\nGuess works well for isolated sources.")
    
    parser.add_argument('--centroiderr',  type=str, default="None",nargs=2,
                        help='What error do you expect on your given centroid.'+
                            '\nIf not provided, it will be 3 3 for general cases and 5 5 for maximum brightnes backup plan')
    parser.add_argument('--maxpos', action="store_true", default=False,
                        help='Set to use brightest spaxel for position')
    
    parser.add_argument('--lstep',  type=int, default=1,
                        help='Slice width in lbda step: default is 1, use 2 for fainter source and maybe 3 for really faint target')
    
    parser.add_argument('--display',  action="store_true", default=False,
                        help='Select the area to fit using the display function.')

    parser.add_argument('--vmin',  type=str, default="2",
                        help='Data Percentage used for imshow "vmin" when using the --display mode')
    
    parser.add_argument('--vmax',  type=str, default="98",
                        help='Data Percentage used for imshow "vmax" when using the --display mode')
    
    parser.add_argument('--tag',  type=str, default="None",
                        help='Add the tag on output filename.')

    # - Standard Star object
    parser.add_argument('--std',  action="store_true", default=False,
                        help='Set this to True to tell the program you what to build a calibration spectrum from this object')
    
    parser.add_argument('--fluxcalsource',  type=str, default="None",
                        help='Path to a "fluxcal" .fits file. This file will be used for the flux calibration. If nothing given, the nearest (in time and within the night) fluxcal will be used.')
    

    parser.add_argument('--nofluxcal',  action="store_true", default=False,
                        help='No Flux calibration')
    
    parser.add_argument('--notroncation',  action="store_true", default=False,
                        help='Do not apply the blue and red end edge troncation on flux calibrated spectra')

    
    parser.add_argument('--nofig',    action="store_true", default=False,
                        help='')

        
    args = parser.parse_args()
        
    # ================= #
    #   The Scripts     #
    # ================= #

    # --------- #
    #  Date     #
    # --------- #
    if not args.inputcube:
        date = args.infile

    
    # ================= #
    #   Actions         #
    # ================= #
    extracted_objects = []
    # ---------- #
    # Extraction #
    # ---------- #

    # wavelength Step

    # ------------- #
    #               #
    # APERTURE      #
    #               #
    # ------------- #
    if args.aperture is not None and len(args.aperture) >0:
        if args.std:
            raise NotImplementedError("--std option cannot be used (yet) with --aperture. Use --auto")
        
        for target in args.aperture.split(","):
            filecubes = io.get_night_files(date, "cube.*", target=target.replace(".fits","")) if not args.inputcube else [target]
            print("cube file from which the spectra will be extracted [aperture]: "+ ", ".join(filecubes))
            
            # - loop over the file cube
            for filecube in filecubes:
                # ----------------- #
                #  Cube to Fit?     #
                # ----------------- #
                print("Automatic extraction of target %s, file: %s"%(target, filecube))
                cube_ = get_sedmcube(filecube)
                [xcentroid, ycentroid], centroids_err, position_type = \
                    astrometry.position_source(cube_, centroid=args.centroid,
                                    centroiderr=args.centroiderr,
                                    maxpos=args.maxpos)
                if args.display:
                    iplot = cube_.show(interactive=True, launch=False)
                    iplot.axim.scatter( xcentroid, ycentroid, **astrometry.MARKER_PROP[position_type] )
                    iplot.launch(vmin=args.vmin, vmax=args.vmax, notebook=False)
                    [aper_xcentroid, aper_ycentroid], [radius,
                                                       bkgd_radius] = build_aperture_param(
                        iplot._stored_picked_poly, args.ap_bkgdscale)
                else:
                    aper_xcentroid = xcentroid
                    aper_ycentroid = ycentroid
                    radius = args.buffer
                    bkgd_radius = radius * args.ap_bkgdscale

                # Actual aperture extraction
                print("INFO: Aperture extraction ongoing...")

                position_type = "aperture"
                cube_.load_adr()
                spec = cube_.get_aperture_spec(aper_xcentroid, aper_ycentroid, radius,
                                                   bkgd_annulus=[1, bkgd_radius/radius],
                                                    adr=cube_.adr)
                # --------------
                # header info passed 
                # --------------
                spec._side_properties["filename"] = cube_.filename
                for k, v in cube_.header.items():
                    if k not in spec.header:
                        spec.header.set(k, v)

                spec.header.set('POSOK', True, "Is the Target centroid inside the MLA?")
                
                spec.header.set('PYSEDMV', pysedm.__version__, "Version of pysedm used")
                spec.header.set('PYSEDMPI', "M. Rigault and D. Neill", "authors of the pysedm pipeline")
                spec.header.set('EXTRACT', "manual" if args.display else "auto", "Was the Extraction manual or automatic")
                
                spec.header.set('XPOS', xcentroid, "x centroid position at reference wavelength (in spaxels)")
                spec.header.set('YPOS', ycentroid, "y centroid position at reference wavelength (in spaxels)")
                spec.header.set('LBDAPOS', -99, "reference wavelength for the centroids (in angstrom) | not defined in apeture")
                spec.header.set('SRCPOS', position_type, "How was the centroid selected ?")

                spec.header.set("QUALITY", sedm.asses_quality(spec), "spectrum extraction quality flag [3,4 means band ; 0=default] ")
                spec.header.set("REDUCER", args.reducer, "Name of the pysedm pipeline reducer [default: auto]") ######### NOT AUTO
                
                # Aperture shape
                #fwhm_arcsec = psffit.slices[2]["slpsf"].model.fwhm * IFU_SCALE_UNIT * 2
                spec.header.set('APRAD', radius, "spaxel radius of extraction aperture")
                spec.header.set('PSFFWHM', -99, "Not defined in Aperture Mode")
                # fwhm & A/B ratio
                spec.header.set('PSFAB', -99, "A/B ratio of the PSF | Not defined in Aperture mode")

                spec.header.set('CRPIX1', 1, "")    # correct CRPIX1 from e3d
                
                spec_raw = spec.copy()
                # --------------
                # Flux Calibation
                # --------------
                # APERTURE EXTRACTION
                spec, flux_calibrated = flux_calibrate(spec, fluxcalfile=args.fluxcalsource, nofluxcal=args.nofluxcal)
                # --------------
                # Recording
                # --------------
                add_tag = "_%s"%args.tag if args.tag is not None and args.tag not in ["None", ""] else ""
                add_info_spec = "_notfluxcal" if not flux_calibrated else ""
                    
                spec_info = add_info_spec
                io._saveout_forcepsf_(filecube, cube_, cuberes=None, cubemodel=None,
                                          mode="aperture"+add_tag,spec_info=spec_info,
                                          fluxcal=flux_calibrated,
                                          cubefitted=None, spec=spec)
                # --------------
                # Recording
                # --------------
                if not args.nofig:                    
                    # SHOWING THE IFU
                    import matplotlib.pyplot as mpl
                    from matplotlib import patches
                    fig = mpl.figure(figsize=[3.5,3.5])
                    ax = fig.add_axes([0.15,0.15,0.75,0.75])
                    _ = cube_._display_im_(ax, vmax=args.vmax, vmin=args.vmin, lbdalim=[6000,9000])
                    ax.scatter(aper_xcentroid, aper_ycentroid, **astrometry.MARKER_PROP[position_type])

                    aper_circle = patches.Circle([aper_xcentroid, aper_ycentroid],
                                                     radius=radius, fc="None", ec="k",lw=1)
                    aper_back   = patches.Wedge([aper_xcentroid, aper_ycentroid], radius, 0, 360, width=radius-bkgd_radius,
                                                facecolor="0.6", edgecolor="k",
                                                linewidth=1,fill=True, alpha=0.3)
                    ax.add_patch(aper_back)
                    ax.add_patch(aper_circle)

                    ax.set_xticks(np.arange(-20,20, 5))
                    ax.set_yticks(np.arange(-20,20, 5))
                    ax.grid(color='0.6', linestyle='-', linewidth=0.5, alpha=0.5)
                    
                    ax.figure.savefig(spec.filename.replace("spec","ifu_spaxels_source").replace(".fits",".pdf"))
                    ax.figure.savefig(spec.filename.replace("spec","ifu_spaxels_source").replace(".fits",".png"), dpi=150)
                    
            # ----------------------- #                    
            # End Aperture Extraction #
            # ----------------------- #
            
    # ------------- #
    #               #
    # AUTO: PSF     #
    #               #
    # ------------- #    
    if args.auto is not None and len(args.auto) >0:
        final_slice_width = int(args.lstep)
        # - Step 1 parameters
        lbdaranges, bins = np.asarray(args.autorange.split(","), dtype="float"), int(args.autobins + 1)
        lbda_step1 = script.lbda_and_bin_to_lbda_step1(lbdaranges, bins)
        
        for target in args.auto.split(","):
            filecubes = io.get_night_files(date, "cube.*", target=target.replace(".fits","")) if not args.inputcube else [target]
            print("cube file from which the spectra will be extracted [auto]: "+ ", ".join(filecubes))
            
            # - loop over the file cube
            for filecube in filecubes:
                # ----------------- #
                #  Cube to Fit?     #
                # ----------------- #
                print("Automatic extraction of target %s, file: %s"%(target, filecube))
                cube_ = get_sedmcube(filecube)
                try:
                    [xcentroid, ycentroid], centroids_err, position_type = \
                        astrometry.position_source(cube_,
                                                   centroid=args.centroid,
                                                   centroiderr=args.centroiderr,
                                                   maxpos=args.maxpos)
                except IOError:
                    print("Astrometry file not found, using maxpos")
                    [xcentroid, ycentroid], centroids_err, position_type = \
                        astrometry.position_source(cube_,
                                                   centroid=args.centroid,
                                                   centroiderr=args.centroiderr,
                                                   maxpos=True)
                if args.display:
                    iplot = cube_.show(interactive=True, launch=False)
                    iplot.axim.scatter( xcentroid, ycentroid, **astrometry.MARKER_PROP[position_type] )
                    iplot.launch(vmin=args.vmin, vmax=args.vmax, notebook=False)
                    cube = cube_.get_partial_cube( iplot.get_selected_idx(), np.arange( len(cube_.lbda)) )
                    args.buffer = SEDM_MLA_RADIUS
                    if iplot.picked_position is not None:
                        print("You picked the position : ", iplot.picked_position )
                        print(" updating the centroid accordingly ")
                        xcentroid, ycentroid = iplot.picked_position
                        centroids_err = [2., 2.]
                        position_type = "manual"
                else:
                    cube = cube_

                    
                # Centroid ?
                print("INFO: PSF centroid (%s)**"%position_type)
                print("centroid: %.1f %.1f"%(xcentroid, ycentroid)+ " error: %.1f %.1f"%(centroids_err[0], centroids_err[1]))
                if not geometry.Point(0,0).buffer(sedm.SEDM_MLA_RADIUS).contains(geometry.Point(xcentroid, ycentroid)):
                    print("WARNING: centroid outside the MLA field of view. NaN spectra returned. * Quality 3. *")
                    spec  = get_spectrum( sedm.SEDM_LBDA, np.ones(len(sedm.SEDM_LBDA))*np.NaN, header=cube.header )
                    # Other
                    cubemodel, psfmodel, bkgdmodel, psffit, slpsf = None, None, None, None, None
                    cube_to_fit = None
                    # Header info
                    POSOK = False
                    lbdaref     = "nan"
                    fwhm_arcsec = "nan"
                    psf_ab     = "nan"
                    psf_pa      = "nan"
                    psf_airmass = "nan"
                    psf_chi2    = "nan"                    
                else:
                    # Aperture area ?
                    point_polygon = geometry.Point(xcentroid, ycentroid).buffer( float(args.buffer) )
                    
                    # => Cube to fit
                    cube_to_fit = cube.get_partial_cube( cube.get_spaxels_within_polygon(point_polygon),
                                                      np.arange(len(cube.lbda)))
                    # --------------
                    # Fitting
                    # --------------
                    print("INFO: Starting MetaSlice fit")
                    spec, cubemodel, psfmodel, bkgdmodel, psffit, slpsf  = \
                      script.extract_star(cube_to_fit, lbda_step1,
                                          centroids=[xcentroid, ycentroid], centroids_err=centroids_err,
                                          spaxel_unit = IFU_SCALE_UNIT,
                                          final_slice_width = final_slice_width,
                                          psfmodel=args.psfmodel, normalized=args.normed)
                    # Hack to be removed:
                    #print("INFO: Temporary variance hacking to be removed ")
                    spec._properties['rawvariance'] = np.ones(len(spec.lbda)) * np.nanmedian( spec.variance[spec.variance>0] )

                    # Divide out exposure time
                    expt = spec.header.get("EXPTIME", 1.0)
                    print("Dividing counts by %s seconds" % expt)
                    spec.scale_by(expt)
                    spec.header.set("CALSCL", True, "Exposure time divided out")

                    if final_slice_width != 1:
                        spec = spec.reshape(cube.lbda)
                    # For header:
                    POSOK       = True
                    lbdaref     = psffit.adrfitter.model.lbdaref
                    fwhm_arcsec = psffit.slices[2]["slpsf"].model.fwhm * IFU_SCALE_UNIT * 2
                    psf_ab     = psffit.slices[2]["slpsf"].fitvalues['ab']
                    psf_pa      = psffit.adrfitter.fitvalues["parangle"]
                    psf_airmass = psffit.adrfitter.fitvalues["airmass"]
                    psf_chi2    = psffit.adrfitter.fitvalues["chi2"]/psffit.adrfitter.dof
                    
                # --------------
                # header info passed
                # --------------
                spec._side_properties["filename"] = cube_.filename
                for k,v in cube.header.items():
                    if k not in spec.header:
                        spec.header.set(k,v)
                # Additional information
                # centroid
                spec.header.set('POSOK', POSOK, "Is the Target centroid inside the MLA?")
                spec.header.set('PYSEDMV', pysedm.__version__, "Version of pysedm used")
                spec.header.set('PYSEDMPI', "M. Rigault and D. Neill", "authors of the pysedm pipeline")
                spec.header.set('PSFV', psfcube.__version__, "Version of psfcube used")
                spec.header.set('PSFPI', "M. Rigault", "authors of the psfcube")
                spec.header.set('PSFMODEL', args.psfmodel, "PSF model used in psfcube")
                spec.header.set('EXTRACT', "manual" if args.display else "auto", "Was the Extraction manual or automatic")
                
                spec.header.set('XPOS', xcentroid, "x centroid position at reference wavelength (in spaxels)")
                spec.header.set('YPOS', ycentroid, "y centroid position at reference wavelength (in spaxels)")
                spec.header.set('LBDAPOS',lbdaref , "reference wavelength for the centroids (in angstrom)")
                spec.header.set('SRCPOS', position_type, "How was the centroid selected ?")
                
                # PSF shape
                
                spec.header.set('PSFFWHM', fwhm_arcsec, "twice the radius needed to reach half of the pick brightness [in arcsec]")
                
                # fwhm & A/B ratio
                spec.header.set('PSFAB', psf_ab , "A/B ratio of the PSF")
                
                # ADR
                spec.header.set('PSFADRPA', psf_pa, "Fitted ADR paralactic angle")
                spec.header.set('PSFADRZ', psf_airmass, "Fitted ADR airmass")
                try:
                    spec.header.set('PSFADRC2',psf_chi2, "ADR chi2/dof")
                except:
                    spec.header.set('PSFADRC2', "nan", "ADR chi2/dof")

                spec.header.set("QUALITY", sedm.asses_quality(spec), "spectrum extraction quality flag [3,4 means band ; 0=default] ")
                spec.header.set("REDUCER", args.reducer, "Name of the pysedm pipeline reducer [default: auto]") ######### NOT AUTO

                spec.header.set('CRPIX1', 1, "")    # correct CRPIX1 from e3d
                
                # Basic quality check ?

                spec_raw = spec.copy()
                # --------------
                # Flux Calibation
                # --------------
                # PSF EXTRACTION
                spec, flux_calibrated  = flux_calibrate(spec, fluxcalfile=args.fluxcalsource, nofluxcal=args.nofluxcal)
                # -------------
                # Cut Edges of fluxcalibrated
                # -------------
                if flux_calibrated and not args.notroncation:
                    from pysedm.sedm import LBDA_PIXEL_CUT
                    spec = get_spectrum(spec.lbda[LBDA_PIXEL_CUT:-LBDA_PIXEL_CUT], spec.data[LBDA_PIXEL_CUT:-LBDA_PIXEL_CUT],
                                        variance=spec.variance[LBDA_PIXEL_CUT:-LBDA_PIXEL_CUT] if spec.has_variance() else None,
                                        header=spec.header, logwave=spec.spec_prop["logwave"])
                    spec._side_properties["filename"] = spec_raw.filename
                    spec.header.set("EDGECUT", True, "have some edge pixels been removed during flux cal")
                    spec.header.set("EDGECUTL", LBDA_PIXEL_CUT, "number of edge pixels removed during flux cal")
                else:
                    spec.header.set("EDGECUT", False, "bluer and redder pixel of the spectra have been removed during flux calibration")
                    spec.header.set("EDGECUTL", None, "number of bluer and redder pixels of the spectra have been removed during flux calibration")
                                        
                # --------------
                # Recording
                # --------------
                add_tag = "_%s"%args.tag if args.tag is not None and args.tag not in ["None", ""] else ""
                add_info_spec = "_notfluxcal" if not flux_calibrated else ""
                spec_info = "_lstep%s"%final_slice_width + add_info_spec
                # MAIN IO
                io._saveout_forcepsf_(filecube, cube, cuberes=None, cubemodel=cubemodel,
                                      mode="auto"+add_tag,spec_info=spec_info, fluxcal=flux_calibrated,
                                      cubefitted=cube_to_fit, spec=spec)
                # Figure
                if not args.nofig:
                    import matplotlib.pyplot as mpl
                    # Pure spaxel
                    fig = mpl.figure(figsize=[3.5,3.5])
                    ax = fig.add_axes([0.15,0.15,0.75,0.75])
                    _ = cube_._display_im_(ax, vmax=args.vmax, vmin=args.vmin, lbdalim=[6000,9000])
                    if POSOK:
                        x,y = np.asarray(cube_to_fit.index_to_xy(cube_to_fit.indexes)).T
                        ax.plot(x, y, marker=".", ls="None", ms=1, color="k")
                        ax.scatter(xcentroid, ycentroid, **astrometry.MARKER_PROP[position_type])
                    else:
                        ax.text(0.5,0.95, "Target outside the MLA \n [%.1f, %.1f] (in spaxels)"%(xcentroid, ycentroid),
                                    fontsize="large", color="k",backgroundcolor=mpl.cm.binary(0.1,0.4),
                                    transform=ax.transAxes, va="top", ha="center")
                        
                    ax.set_xticks(np.arange(-20,20, 5))
                    ax.set_yticks(np.arange(-20,20, 5))

                    ax.grid(color='0.6', linestyle='-', linewidth=0.5, alpha=0.5)
                    
                    ax.figure.savefig(spec.filename.replace("spec","ifu_spaxels_source").replace(".fits",".pdf"))
                    ax.figure.savefig(spec.filename.replace("spec","ifu_spaxels_source").replace(".fits",".png"), dpi=150)
                    
                    if psffit is not None:
                        x,y = np.asarray(cube_to_fit.index_to_xy(cube_to_fit.indexes)).T
                        # SHOWING ADR
                        psffit.show_adr(savefile=spec.filename.replace("spec","adr_fit").replace(".fits",".pdf") )
                        psffit.show_adr(savefile=spec.filename.replace("spec","adr_fit").replace(".fits",".png") )
                        # SHOWING PSFPROFILE (metaslice)
                        psffit.slices[2]["slpsf"].show(savefile=spec.filename.replace("spec","psfprofile").replace(".fits",".pdf"))
                        psffit.slices[2]["slpsf"].show(savefile=spec.filename.replace("spec","psfprofile").replace(".fits",".png"))
                        
                        # SHOWING SPAXEL (IFU)
                        
                        cube_.show(show=False)
                        ax = mpl.gca()
                        ax.plot(x,y, marker=".", ls="None", ms=1, color="k")
                        ax.scatter(xcentroid, ycentroid, **astrometry.MARKER_PROP[position_type])
                        ax.set_xticks(np.arange(-20,20, 5))
                        ax.set_yticks(np.arange(-20,20, 5))
                        ax.grid(color='0.6', linestyle='-', linewidth=0.5, alpha=0.5)
                    
                        ax.figure.savefig(spec.filename.replace("spec","spaxels_source").replace(".fits",".pdf"))
                    
                    
                    # Special Standard
                    if cube.header['IMGTYPE'].lower() in ['standard'] and flux_calibrated:
                        from pysedm.fluxcalibration import show_fluxcalibrated_standard
                        show_fluxcalibrated_standard(spec, savefile=spec.filename.replace("spec","calibcheck_spec").replace(".fits",".pdf"))
                        show_fluxcalibrated_standard(spec, savefile=spec.filename.replace("spec","calibcheck_spec").replace(".fits",".png"))
                        
                # -----------------
                #  Is that a STD  ?
                # -----------------
                if args.std and cube.header['IMGTYPE'].lower() in ['standard'] and 'AIRMASS' in cube.header:
                    if spec_raw.header['QUALITY'] == 0:
                        # Based on the flux non calibrated spectsra
                        spec_raw.header['OBJECT'] = cube.header['OBJECT']
                        for k,v in cube.header.items():
                            if k not in spec_raw.header:
                                spec_raw.header.set(k, v)

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
                # - for the record
                extracted_objects.append(spec)
                
            # -------------------------- #                    
            # End Auto (PSF) Extraction  #
            # -------------------------- #
