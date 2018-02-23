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

    # // AUTOMATIC EXTRACTION
    parser.add_argument('--auto',  type=str, default=None,
                        help='Shall this run an automatic PSF extraction')

    parser.add_argument('--autorange',  type=str, default="4500,7000",
                        help='Wavelength range [in Angstrom] for measuring the metaslice PSF')
    
    parser.add_argument('--autobins',  type=int, default=10,
                        help='Number of bins within the wavelength range (see --autorange)')
    
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
                # - wavelengthes used
                lbda_range = np.asarray(args.autorange.split(","), dtype="float")
                lbdas_ = np.linspace(lbda_range[0],lbda_range[1],args.autobins+1)
                lbdas  = np.asarray([lbdas_[:-1],lbdas_[1:]]).T

                # Step 1 fit the PSF shape:
                output = cube.filename.replace("e3d","psffit_e3d")
                savedata = output.replace(".fits",".json")
                savefig = savedata.replace(".json",".pdf") if not args.nofig else None

                psfmodel = extractstar.fit_psf_parameters(cube, lbdas,
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
                io._saveout_forcepsf_(filecube, cube, cuberes, cubemodel, spec, bkgd)
                
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
            if not args.nofig:
                pl = speccal.show(color="C1", lw=2, show=False, savefile=None)
                ax = pl["ax"]
                ax.set_ylabel('Inverse Sensitivity')
                ax.set_yscale("log")
                # telluric
                ax.axvspan(7550,7700, color="0.7", alpha=0.4) 
                ax.text(7700,ax.get_ylim()[-1],  "O2 Telluric ", 
                        va="top", ha="center",  rotation=90,color="0.2",
                        zorder=9, fontsize=11)
                
                ax.axvspan(6850,7000, color="0.7", alpha=0.2) 
                ax.text(6900,ax.get_ylim()[-1],  "O2 Telluric ", 
                        va="top", ha="center",  rotation=90,color="0.1",
                        zorder=9, fontsize=11)
                
                # reference            
                axrspec = ax.twinx()
                spec.show(ax=axrspec, color="C0",  label="obs.", show=False, savefile=None)
                axrspec.set_yticks([])
                axrspec.legend(loc="upper right")
                axrspec.set_title("Source:%s | Airmass:%.2f"%(spec.filename.split('/')[-1],spec.header['AIRMASS']),
                                      fontsize="small", color="0.5")
                ax.figure.figout(savefile=speccal.filename.replace(".fits",".pdf"))
