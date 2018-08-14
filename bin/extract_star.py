#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pysedm       import get_sedmcube, io, fluxcalibration, sedm
from pysedm.sedm  import IFU_SCALE_UNIT


MARKER_PROP = {"astrom": dict(marker="x", lw=2, s=80, color="C1", zorder=8),
               "manual": dict(marker="+", lw=2, s=100, color="k", zorder=8),
               "aperture": dict(marker="+", lw=2, s=100, color="k", zorder=8),
               "auto":  dict(marker="o", lw=2, s=200, facecolors="None", edgecolors="C3", zorder=8)
                   }


def position_source(cube, centroid=None, centroiderr=None):
    """ How is the source position selected ? """
    if centroiderr is None or centroiderr in ["None"]:
        centroid_err_given = False
        centroids_err = [3,3]
    else:
        centroid_err_given = True
        centroids_err = np.asarray(centroiderr, dtype="float")
        
    if centroid is None or centroid in ["None"]:
        from pysedm.astrometry  import get_object_ifu_pos
        xcentroid,ycentroid = get_object_ifu_pos( cube )
        if np.isnan(xcentroid*ycentroid):
            print("IFU target location based on CCD astrometry failed. centroid guessed based on brightness used instead")
            sl = cube.get_slice(lbda_min=lbdaranges[0], lbda_max=lbdaranges[1], slice_object=True)
            x,y = np.asarray(sl.index_to_xy(sl.indexes)).T # Slice x and y
            argmaxes = np.argwhere(sl.data>np.percentile(sl.data,95)).flatten() # brightest points
            xcentroid,ycentroid  = np.nanmean(x[argmaxes]),np.nanmean(y[argmaxes]) # centroid
            if not centroid_err_given:
                centroids_err = [5,5]
                            
                position_type="auto" 
        else:
            print("IFU position based on CCD wcs solution used : ",xcentroid,ycentroid)
            position_type="astrom" 
    else:
        xcentroid, ycentroid = np.asarray(centroid, dtype="float")
        print("centroid used", centroid)
        position_type="manual"

    return [xcentroid, ycentroid], centroids_err, position_type

# Aperture Spectroscopy

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
        
    notflux_cal=False
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
            spec.header["FLUXCAL"] = ("False","has the spectra been flux calibrated")
            spec.header["CALSRC"] = (None, "Flux calibrator filename")
            notflux_cal=True
        else:
            from pyifu import load_spectrum
            fluxcal = load_spectrum( fluxcalfile ) 
            spec.scale_by(1/fluxcal.data)
            spec.header["FLUXCAL"] = ("True","has the spectra been flux calibrated")
            spec.header["CALSRC"] = (fluxcal.filename.split("/")[-1], "Flux calibrator filename")
            notflux_cal=False
            
    else:
        spec.header["FLUXCAL"] = ("False","has the spectra been flux calibrated")
        spec.header["CALSRC"] = (None, "Flux calibrator filename")
        notflux_cal=True
        
    return spec, ~notflux_cal
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
    
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')

    # // AUTOMATIC EXTRACTION

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
    
    parser.add_argument('--lstep',  type=int, default=1,
                        help='Slice width in lbda step: default is 1, use 2 for fainter source and maybe 3 for really faint target')
    
    parser.add_argument('--display',  action="store_true", default=False,
                        help='Select the area to fit using the display function.')
    
    parser.add_argument('--tag',  type=str, default="None",
                        help='Add the tag on output filename.')

    # - Standard Star object
    parser.add_argument('--std',  action="store_true", default=False,
                        help='Set this to True to tell the program you what to build a calibration spectrum from this object')
    
    parser.add_argument('--fluxcalsource',  type=str, default="None",
                        help='Path to a "fluxcal" .fits file. This file will be used for the flux calibration. If nothing given, the nearest (in time and within the night) fluxcal will be used.')
    

    parser.add_argument('--nofluxcal',  action="store_true", default=False,
                        help='No Flux calibration')
    
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
            filecubes = io.get_night_files(date, "cube.*", target=target.replace(".fits",""))
            print("cube file from which the spectra will be extracted [aperture]: "+ ", ".join(filecubes))
            
            # - loop over the file cube
            for filecube in filecubes:
                # ----------------- #
                #  Cube to Fit?     #
                # ----------------- #
                print("Automatic extraction of target %s, file: %s"%(target, filecube))
                cube_ = get_sedmcube(filecube)
                [xcentroid, ycentroid], centroids_err, position_type = position_source(cube_, centroid = args.centroid, centroiderr= args.centroiderr)
                if args.display:
                    iplot = cube_.show(interactive=True, launch=False)
                    iplot.axim.scatter( xcentroid, ycentroid, **MARKER_PROP[position_type] )
                    iplot.launch(vmin="2", vmax="98", notebook=False)
                else:
                    print("WARNING: aperture spectroscopy currently works solely with --display on")
                    print("WARNING: aperture extraction ignored for: %s "%filecube)
                    continue


                # Actual aperture extraction
                print("INFO: Aperture extraction ongoing...")
                [aper_xcentroid, aper_ycentroid], [radius, bkgd_radius] = build_aperture_param(iplot._stored_picked_poly, args.ap_bkgdscale)
                position_type = "aperture"
                cube_.load_adr()
                spec = cube_.get_aperture_spec(aper_xcentroid, aper_ycentroid, radius, bkgd_annulus=[1, bkgd_radius/radius],
                                                    adr=cube_.adr)
                # --------------
                # header info passed 
                # --------------
                spec._side_properties["filename"] = cube_.filename
                for k,v in cube_.header.items():
                    if k not in spec.header:
                        spec.header.set(k,v)
                        
                spec.header.set('XPOS', xcentroid, "x centroid position at reference wavelength (in spaxels)")
                spec.header.set('YPOS', ycentroid, "y centroid position at reference wavelength (in spaxels)")
                spec.header.set('LBDAPOS', psffit.adrfitter.model.lbdaref, "reference wavelength for the centroids (in angstrom)")
                spec.header.set('SRCPOS', position_type, "How was the centroid selected ?")

                # Aperture shape
                fwhm_arcsec = psffit.slices[2]["slpsf"].model.fwhm * IFU_SCALE_UNIT * 2
                spec.header.set('PSFFWHM', fwhm_arcsec, "twice the radius needed to reach half of the pick brightness [in arcsec]")
                # fwhm & A/B ratio
                spec.header.set('PSFELL', psffit.slices[2]["slpsf"].fitvalues['ell'], "Ellipticity of the PSF")

                
                spec_raw = spec.copy()
                # --------------
                # Flux Calibation
                # --------------
                spec, flux_calibrated = flux_calibrate(spec, fluxcalfile=args.fluxcalsource, nofluxcal=args.nofluxcal)
                # --------------
                # Recording
                # --------------
                add_tag = "_%s"%args.tag if args.tag is not None and args.tag not in ["None", ""] else ""
                add_info_spec = "_notfluxcal" if not flux_calibrated else ""
                    
                spec_info = add_info_spec
                io._saveout_forcepsf_(filecube, cube_, cuberes=None, cubemodel=None,
                                          mode="aperture"+add_tag,spec_info=spec_info,
                                          cubefitted=None, spec=spec)
                # --------------
                # Recording
                # --------------
                if not args.nofig:                    
                    # Pure spaxel
                    import matplotlib.pyplot as mpl
                    from matplotlib import patches
                    fig = mpl.figure(figsize=[3.5,3.5])
                    ax = fig.add_axes([0.15,0.15,0.75,0.75])
                    _ = cube_._display_im_(ax, vmax="98", vmin="2")
                    ax.scatter(aper_xcentroid, aper_ycentroid, **MARKER_PROP[position_type])

                    aper_circle = patches.Circle([aper_xcentroid, aper_ycentroid],
                                                     radius=radius, fc="None", ec="k",lw=1)
                    aper_back   = patches.Wedge([aper_xcentroid, aper_ycentroid], radius, 0, 360, width=radius-bkgd_radius,
                                                facecolor="0.6", edgecolor="k",
                                                linewidth=1,fill=True, alpha=0.3)
                    ax.add_patch(aper_back)
                    ax.add_patch(aper_circle)

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
        lbdaranges, bins = np.asarray(args.autorange.split(","), dtype="float"), int(args.autobins+1)
        STEP_LBDA_RANGE = np.linspace(lbdaranges[0],lbdaranges[1], bins+1)
        lbda_step1      = np.asarray([STEP_LBDA_RANGE[:-1], STEP_LBDA_RANGE[1:]]).T
        
        for target in args.auto.split(","):
            filecubes = io.get_night_files(date, "cube.*", target=target.replace(".fits",""))
            print("cube file from which the spectra will be extracted [auto]: "+ ", ".join(filecubes))
            
            # - loop over the file cube
            for filecube in filecubes:
                # ----------------- #
                #  Cube to Fit?     #
                # ----------------- #
                print("Automatic extraction of target %s, file: %s"%(target, filecube))
                cube_ = get_sedmcube(filecube)
                [xcentroid, ycentroid], centroids_err, position_type = position_source(cube_, centroid = args.centroid, centroiderr= args.centroiderr)
                if args.display:
                    iplot = cube_.show(interactive=True, launch=False)
                    iplot.axim.scatter( xcentroid, ycentroid, **MARKER_PROP[position_type] )
                    iplot.launch(vmin="2", vmax="98", notebook=False)
                    cube = cube_.get_partial_cube( iplot.get_selected_idx(), np.arange( len(cube_.lbda)) )
                    args.buffer = 20
                    if iplot.picked_position is not None:
                        print("You picked the position : ", iplot.picked_position )
                        print(" updating the centroid accordingly ")
                        xcentroid, ycentroid = iplot.picked_position
                        centroids_err = [2. ,2. ]
                        position_type = "manual"
                else:
                    cube = cube_
                    
                # Centroid ?    
                print("INFO: PSF centroid (%s)**"%position_type)
                print("centroid: %.1f %.1f"%(xcentroid, ycentroid)+ " error: %.1f %.1f"%(centroids_err[0], centroids_err[1]))
                
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
                  script.extract_star(cube_to_fit,
                                          centroids=[xcentroid, ycentroid], centroids_err=centroids_err,
                                          spaxel_unit = IFU_SCALE_UNIT,
                                          final_slice_width = final_slice_width,
                                          lbda_step1=lbda_step1, psfmodel=args.psfmodel, normalized=args.normed)
                # Hack to be removed:
                print("INFO: Temporary variance hacking to be removed ")
                spec._properties['variance'] = np.ones(len(spec.lbda)) * np.min([ np.nanmean( spec.variance ), np.nanmedian( spec.variance )]) / 2.

                if final_slice_width != 1:
                    spec = spec.reshape(cube.lbda)

                # --------------
                # header info passed
                # --------------
                spec._side_properties["filename"] = cube_.filename
                for k,v in cube.header.items():
                    if k not in spec.header:
                        spec.header.set(k,v)
                # Additional information
                # centroid
                spec.header.set('XPOS', xcentroid, "x centroid position at reference wavelength (in spaxels)")
                spec.header.set('YPOS', ycentroid, "y centroid position at reference wavelength (in spaxels)")
                spec.header.set('LBDAPOS', psffit.adrfitter.model.lbdaref, "reference wavelength for the centroids (in angstrom)")
                spec.header.set('SRCPOS', position_type, "How was the centroid selected ?")
                # PSF shape
                fwhm_arcsec = psffit.slices[2]["slpsf"].model.fwhm * IFU_SCALE_UNIT * 2
                spec.header.set('PSFFWHM', fwhm_arcsec, "twice the radius needed to reach half of the pick brightness [in arcsec]")
                # fwhm & A/B ratio
                spec.header.set('PSFELL', psffit.slices[2]["slpsf"].fitvalues['ell'], "Ellipticity of the PSF")
                # ADR
                spec.header.set('PSFADRPA', psffit.adrfitter.fitvalues["parangle"], "Fitted ADR paralactic angle")
                spec.header.set('PSFADRZ', psffit.adrfitter.fitvalues["airmass"], "Fitted ADR airmass")
                spec.header.set('PSFADRC2', psffit.adrfitter.fitvalues["chi2"]/psffit.adrfitter.dof, "ADR chi2/dof")
                # Basic quality check ?

                spec_raw = spec.copy()
                # --------------
                # Flux Calibation
                # --------------                        
                spec, flux_calibrated  = flux_calibrate(spec, fluxcalfile=args.fluxcalsource, nofluxcal=args.nofluxcal)
                        
                # --------------
                # Recording
                # --------------
                add_tag = "_%s"%args.tag if args.tag is not None and args.tag not in ["None", ""] else ""
                add_info_spec = "_notfluxcal" if not flux_calibrated else ""
                    
                spec_info = "_lstep%s"%final_slice_width + add_info_spec
                io._saveout_forcepsf_(filecube, cube, cuberes=None, cubemodel=cubemodel,
                                          mode="auto"+add_tag,spec_info=spec_info,
                                          cubefitted=cube_to_fit, spec=spec)
                # Figure
                if not args.nofig:
                    psffit.show_adr(savefile=spec.filename.replace("spec","adr_fit").replace(".fits",".pdf") )
                    psffit.show_adr(savefile=spec.filename.replace("spec","adr_fit").replace(".fits",".png") ) 
                    psffit.slices[2]["slpsf"].show(savefile=spec.filename.replace("spec","psfprofile").replace(".fits",".pdf"))
                    psffit.slices[2]["slpsf"].show(savefile=spec.filename.replace("spec","psfprofile").replace(".fits",".png"))
                    
                    import matplotlib.pyplot as mpl
                    cube_.show(show=False)
                    ax = mpl.gca()
                    x,y = np.asarray(cube_to_fit.index_to_xy(cube_to_fit.indexes)).T
                    ax.plot(x,y, marker=".", ls="None", ms=1, color="k")
                    ax.scatter(xcentroid, ycentroid, **MARKER_PROP[position_type])
                    ax.figure.savefig(spec.filename.replace("spec","spaxels_source").replace(".fits",".pdf"))
                    
                    # Pure spaxel
                    fig = mpl.figure(figsize=[3.5,3.5])
                    ax = fig.add_axes([0.15,0.15,0.75,0.75])
                    _ = cube_._display_im_(ax, vmax="98", vmin="2")
                    ax.plot(x,y, marker=".", ls="None", ms=1, color="k")
                    ax.scatter(xcentroid, ycentroid, **MARKER_PROP[position_type])
                    ax.figure.savefig(spec.filename.replace("spec","ifu_spaxels_source").replace(".fits",".pdf"))
                    ax.figure.savefig(spec.filename.replace("spec","ifu_spaxels_source").replace(".fits",".png"), dpi=150)
                    
                    # Special Standard
                    if cube.header['IMGTYPE'].lower() in ['standard'] and not notflux_cal:
                        from pysedm.fluxcalibration import show_fluxcalibrated_standard
                        show_fluxcalibrated_standard(spec, savefile=spec.filename.replace("spec","calibcheck_spec").replace(".fits",".pdf"))
                        show_fluxcalibrated_standard(spec, savefile=spec.filename.replace("spec","calibcheck_spec").replace(".fits",".png"))
                        
                # -----------------
                #  Is that a STD  ?
                # -----------------
                if args.std and cube.header['IMGTYPE'].lower() in ['standard']:
                    # Based on the flux non calibrated spectra
                    spec_raw.header['OBJECT'] = cube.header['OBJECT']
                    speccal, fl = fluxcalibration.get_fluxcalibrator(spec_raw, fullout=True)
                    for k,v in cube.header.items():
                        if k not in speccal.header:
                            speccal.header.set(k,v)

                    speccal.header["SOURCE"] = (spec.filename.split("/")[-1], "This object has been derived from this file")
                    speccal.header["PYSEDMT"] = ("Flux Calibration Spectrum", "Object to use to flux calibrate")
                    filename_inv = spec.filename.replace(io.PROD_SPECROOT,io.PROD_SENSITIVITYROOT)
                    speccal._side_properties['filename'] = filename_inv
                    speccal.writeto(filename_inv)
                    if not args.nofig:
                        fl.show(savefile=speccal.filename.replace(".fits",".pdf"), show=False, fluxcal=speccal.data)
                        fl.show(savefile=speccal.filename.replace(".fits",".png"), show=False, fluxcal=speccal.data)
                                    
                # - for the record
                extracted_objects.append(spec)
                
            # -------------------------- #                    
            # End Auto (PSF) Extraction  #
            # -------------------------- #
