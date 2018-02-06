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
    from pysedm       import get_sedmcube, io
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube
            """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')
    
    parser.add_argument('--auto',  type=str, default=None,
                        help='Shall this run an automatic PSF extraction')
    
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
        print(args.auto)
        
        for target in args.auto.split(","):
            filecubes = io.get_night_files(date, "cube.*", target=target.replace(".fits",""))
            print("cube file from which the spectra will be extracted: "+ ", ".join(filecubes))
            
            # - loop over the file cube
            for filecube in filecubes:
                # ----------------- #
                #  Cube to Fit?     #
                # ----------------- #
                print("Automatic extraction of target %s, file: %s"%(target, filecube))
                cube = get_sedmcube(filecube)
                # --------------
                # Fitting
                # --------------
                # Step 1 fit the PSF shape:
                output = cube.filename.replace("e3d","psffit_e3d")
                savedata = output.replace(".fits",".json")
                savefig = savedata.replace(".json",".pdf") if not args.nofig else None

                psfmodel = extractstar.fit_psf_parameters(cube,
                                                        lbda_range=[4500,7000], nbins=10,
                                                        savedata=savedata,savefig=savefig,
                                                        return_psfmodel=True)
                # Step 2 ForcePSF spectroscopy:
                
                output = cube.filename.replace("e3d","forcepsf_e3d")
                savefig = output.replace(".fits",".pdf") if not args.nofig else None
                spec, bkgd, forcepsf = extractstar.fit_force_spectroscopy(cube, psfmodel, savefig=savefig)
                cubemodel = forcepsf.cubemodel
                cuberes   = forcepsf.cuberes
                
                
                # --------------
                # Recording
                # --------------
                # Cube Model
                cubemodel.set_header(cube.header)
                cubemodel.header["SOURCE"]   = (filecube.split("/")[-1], "This object has been derived from this file")
                cubemodel.header["PYSEDMT"]  = ("Force 3DPSF extraction: Model Cube", "This is the model cube of the PSF extract")
                cubemodel.header["PSFTYPE"]  = ("auto", "Kind of PSF extraction")
                cubemodel.writeto(filecube.replace(io.PROD_CUBEROOT,"forcepsfmodel_"+io.PROD_CUBEROOT))
                
                # Cube Residual                
                cuberes.set_header(cube.header)
                cuberes.header["SOURCE"]   = (filecube.split("/")[-1], "This object has been derived from this file")
                cuberes.header["PYSEDMT"]  = ("Force 3DPSF extraction: Residual Cube", "This is the residual cube of the PSF extract")
                cuberes.header["PSFTYPE"]  = ("auto", "Kind of PSF extraction")
                cuberes.writeto(filecube.replace(io.PROD_CUBEROOT,"psfres_"+io.PROD_CUBEROOT))
                
                # ----------------- #
                # Save the Spectrum #
                # ----------------- #
                # - build the spectrum
                spec.set_header(cube.header)
                spec.header["SOURCE"]   = (filecube.split("/")[-1], "This object has been derived from this file")
                spec.header["PYSEDMT"]  = ("Force 3DPSF extraction: Spectral Model", "This is the fitted flux spectrum")
                spec.header["PSFTYPE"]  = ("auto", "Kind of PSF extraction")

                fileout = filecube.replace(io.PROD_CUBEROOT,io.PROD_SPECROOT+"_forcepsf")
                spec.writeto(fileout)
                spec.writeto(fileout.replace(".fits",".txt"), ascii=True)
                
                spec._side_properties["filename"] = fileout
                if not args.nofig:
                    spec.show(savefile=spec.filename.replace(".fits",".pdf"))

                # - background
                bkgd.set_header(cube.header)
                bkgd.header["SOURCE"]   = (filecube.split("/")[-1], "This object has been derived from this file")
                bkgd.header["PYSEDMT"]  = ("Force 3DPSF extraction: Spectral Background Model", "This is the fitted flux spectrum")
                bkgd.header["PSFTYPE"]  = ("auto", "Kind of PSF extraction")

                fileout = filecube.replace(io.PROD_CUBEROOT,io.PROD_SPECROOT+"_forcepsf_bkgd")
                bkgd.writeto(fileout)
                bkgd.writeto(fileout.replace(".fits",".txt"), ascii=True)

                # - for the record
                extracted_objects.append(spec)
                
    else:
        print("NO  AUTO")
        
    # ----------- #
    # Calibration #
    # ----------- #
    if args.std:
        import pycalspec
        from pyifu.spectroscopy import get_spectrum
        from pyifu.tools import figout
        for spec in extracted_objects:
            if "STD" not in spec.header['OBJECT']:
                continue
            
            if "FLUXCAL" in spec.header.keys() and spec.header['FLUXCAL']:
                continue
            
            # - Let's go
            stdname = spec.header['OBJECT'].replace("STD-","")
            stdspec = pycalspec.std_spectrum(stdname).reshape(spec.lbda,"linear")
            speccal = get_spectrum(spec.lbda, stdspec.data/spec.data, variance=None, header=spec.header)
            speccal.header["SOURCE"] = (spec.filename.split("/")[-1], "This object has been derived from this file")
            speccal.header["PYSEDMT"] = ("Flux Calibration Spectrum", "Object to use to flux calibrate")
            
            # - Naming
            filename_inv = spec.filename.replace(io.PROD_SPECROOT,io.PROD_SENSITIVITYROOT)
            speccal._side_properties['filename'] = filename_inv
            speccal.writeto(filename_inv)
                                
            # - Checkplot
            pl = speccal.show(color="C1", lw=2, show=False, savefile=None)
            ax = pl["ax"]
            ax.set_ylabel('Inverse Sensitivity')
            ax.set_yscale("log")
            # telluric
            ax.axvspan(7500,7800, color="0.7", alpha=0.4) 
            ax.text(7700,ax.get_ylim()[-1],  "O2 Telluric ", 
                        va="top", ha="center",  rotation=90,color="0.2", zorder=9)
            # reference            
            axrspec = ax.twinx()
            spec.show(ax=axrspec, color="C0",  label="obs.", show=False, savefile=None)
            axrspec.set_yticks([])
            axrspec.legend(loc="upper right")
            axrspec.set_title("Source:%s | Airmass:%.2f"%(spec.filename.split('/')[-1],spec.header['AIRMASS']),
                    fontsize="small", color="0.5")
            ax.figure.figout(savefile=speccal.filename.replace(".fits",".pdf"))
