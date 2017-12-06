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
                        help='Build a e3d cube of the given target or target list (csv) e.g. --build dome or --build dome,Hg,Cd')
    
    parser.add_argument('--radius',  type=float, default=10,
                        help='Radius for the Aperture spectroscopic extraction [see --runit for radius units]')

    parser.add_argument('--runit',  type=str, default="spaxels",
                        help='Units of the radius. could be: fwhm, spaxels or any astropy.units')

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
            fileccds = io.get_night_files(date, "cube.*", target=target.replace(".fits",""))
            print(fileccds)
            for filecube in fileccds:
                print("Automatic extraction of target %s, file: %s"%(target, filecube))
                cube = get_sedmcube(filecube)
                es   = extractstar.ExtractStar(cube)
                
                spec = es.get_auto_aperture_spectroscopy(radius=args.radius, units=args.runit)
                fileout = filecube.replace(io.PROD_CUBEROOT,io.PROD_SPECROOT+"auto")
                spec.writeto(fileout)
                spec._side_properties["filename"] = fileout
                spec.show(savefile=spec.filename.replace(".fits",".pdf"))
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
            spec.show(ax=axrspec, color="C0",  show_background=False, label="obs.", show=False, savefile=None)
            axrspec.set_yticks([])
            axrspec.legend(loc="upper right")
            axrspec.set_title("Source:%s | Airmass:%.2f"%(spec.filename.split('/')[-1],spec.header['AIRMASS']),
                    fontsize="small", color="0.5")
            ax.figure.figout(savefile=speccal.filename.replace(".fits",".pdf"))
