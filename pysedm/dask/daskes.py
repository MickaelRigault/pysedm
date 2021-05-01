""" Dasked version of pysedm/bin/extractstars.py """

import os
from dask import delayed
from .base import DaskCube


from .. import get_sedmcube, fluxcalibration, io
from ..sedm import SEDMExtractStar, flux_calibrate_sedm
from astropy.io import fits


def get_extractstar(cube, step1range=[4500,7000], step1bins=6,
                   centroid="auto", **kwargs):
    """ """
    es = SEDMExtractStar(cube)
    es.set_lbdastep1(lbdarange=step1range, bins=step1bins)
    es.set_centroid(centroid=centroid)
    return es

def get_fluxcalfile(cubefile):
    """ """
    return io.fetch_nearest_fluxcal(mjd=fits.getval(cubefile,"MJD_OBS"))

def run_extractstar(es, spaxelbuffer = 10,
                    spaxels_to_use=None,
                    spaxels_to_avoid=None,
                    psfmodel="NormalMoffatTilted",
                    slice_width = 1, fwhm_guess=None, 
                    verbose=False,
                    store_fig=True, store_data=True,
                    **kwargs):
    """ """
    
    if spaxels_to_use is not None: # You fixed which you want
        if len(spaxels_to_use)<4:
            warnings.warn("ExtractStar: You provided less than 4 spaxel to fit")
        es.set_fitted_spaxels(spaxels_to_use)

    elif es.fitted_spaxels is None: # Automatic selections (with or without spaxels)
        es.get_spaxels_tofit(buffer=spaxelbuffer, update=True,
                             spaxels_to_avoid=spaxels_to_avoid)

    if verbose: print("* Starting extractstar.run")
    es.run(slice_width=slice_width, psfmodel=psfmodel,
           fwhm_guess=fwhm_guess, verbose=verbose, **kwargs)
    
    rawspec = es.get_spectrum("raw", persecond=True)
    rawspec.set_filename(es.cube.filename.replace("e3d", "esspec"))
    #
    # - Output
    if store_fig:
        basename = es.basename
        es.show_adr(savefile = basename.replace("{placeholder}","esadr")+".pdf")
        es.show_mla(savefile = basename.replace("{placeholder}","esmla")+".pdf")
        es.show_psf(savefile = basename.replace("{placeholder}","espsf")+".pdf")
        rawspec.show(savefile = basename.replace("{placeholder}","esspec")+".pdf")
        
    if store_data:
        rawspec.writeto(rawspec.filename)

    return rawspec

def calibrate_spec(spec, fluxcalfile, default_airmass=1.1,
                       store_fig=True, store_data=True):
    """ """
    if fluxcalfile is None:
        fluxcal = None
    else:
        fluxcal = fluxcalibration.load_fluxcal_spectrum(fluxcalfile)
        
    if fluxcal is None:
        spec.header["FLUXCAL"] = (False,"has the spectra been flux calibrated")
        spec.header["CALSRC"] = (None, "Flux calibrator filename")
        spec.header["BUNIT"]  = (spec.header.get('BUNIT',""),"Flux Units")
        specout = spec.copy()
    else:
        spec.scale_by(fluxcal.get_inversed_sensitivity(spec.header.get("AIRMASS",default_airmass)),
                      onraw=False)
        spec.header["FLUXCAL"] = (True, "has the spectra been flux calibrated")
        spec.header["CALSRC"] = (os.path.basename(fluxcalfile), "Flux calibrator filename")
        spec.header["BUNIT"]  = ("erg/s/A/cm^2", "Flux Units")
        specout = spec.copy()

    specout.set_filename( spec.filename.replace("esspec","spec") )
    if store_fig:
        specout.show( savefile=specout.filename.replace(".fits",".pdf") )
                         
    if store_data:
        specout.writeto(specout.filename)
        
    return specout
        
def build_fluxcalibrator(rawspec_std,
                             store_plot=True, store_data=True):
    """ 
    

    Returns
    -------
    speccal filename
    """
    speccal, fl = fluxcalibration.get_fluxcalibrator(rawspec_std, fullout=True)
    speccal.header.set("SOURCE", os.path.basename(rawspec_std.filename),
                           "This object has been derived from this file")
    speccal.header.set("PYSEDMT","Flux Calibration Spectrum", "Object to use to flux calibrate")

    speccal.set_filename( rawspec_std.filename.replace("esspec","fluxcal") )
    if store_plot:
        _ = fl.show( savefile=speccal.filename.replace(".fits",".pdf") )
        
    if store_data:
        speccal.writeto( speccal.filename )

    # Returning the filename because dask has serialization issue 
    return speccal.filename



class DaskES( DaskCube ):
    """
    1. Get Files
    cubefiles

    2. Get Cubes
    cube = get_sedmcube(filecube)
    cube.header['REDUCER'] = args.reducer
    
    3. remove cubes for CR

    4. Extract point source from the cubes 
    es_out = cube.extract_pointsource(**es_options)
    
    5. Get flux calibration file

    6. flux calibrate the results

    7. store the result
    """


    # =============== #
    #    Methods      #
    # =============== #
    def get_std_basename(self):
        """ """
        dcube = self.get_cubefile_dataframe(False)
        return dcube[dcube["is_std"]]["basename"].values

    # -------- #
    # Running  #
    # -------- #
    def compute(self, get_delayed=False):
        """ """
        stdbasenames = self.get_std_basename()
        spectra = [self.stdconnected_extractstars(stdbasename)
                    for stdbasename in stdbasenames]
        if get_delayed:
            return spectra
        
        return self.client.compute(spectra)

    def stdconnected_extractstars(self, std_basename):
        """ """
        #
        # Get the cube paths
        cubefile_df  = self.get_cubefile_dataframe().xs(std_basename).set_index("basename")
        cubefile_std = cubefile_df.loc[std_basename]["filepath"]
        cubefiles    = cubefile_df.drop(std_basename)["filepath"].values

        #
        # Build the graph
        raw_std = self.extract_rawstar(cubefile_std)
        fluxcalfile = self.build_fluxcalibrator(raw_std)
        f_spec = [raw_std]
        for cubefile in cubefiles:
            rawspec = self.extract_rawstar(cubefile)
            spec = self.calibrate_rawspec(rawspec, fluxcalfile=fluxcalfile)
            f_spec.append(spec)
            
        return f_spec
    
    @classmethod
    def single_extractstars(cls, cube_filename, fluxcalfile=None, calibrate=True):
        """ """
        rawspec = cls.extract_rawstar(cube_filename)
        if fluxcalfile is None and calibrate:
            fluxcalfile = delayed(get_fluxcalfile)(cube_filename)
            
        return cls.calibrate_rawspec(rawspec, fluxcalfile)

    # -------- #
    # Static   #
    # -------- #
    @staticmethod
    def extract_rawstar(cube_filename, store_fig=True, store_data=True, **kwargs):
        """ Extract the pointsource from the given filename 
        
        Parameters
        ----------
        cube_filename: [sting]
            Path to the e3d cube.

        Returns
        -------
        Spectrum (not flux calibrated)
        """
        # 1. Get cube
        cube = delayed(get_sedmcube)(cube_filename)
                
        # 3. Get ExtractStar
        estar = delayed(get_extractstar)(cube)
        
        # 4. Run ExtractStar
        rawspec = delayed(run_extractstar)(estar, store_fig=store_fig, store_data=store_data,
                                               **kwargs)
        return rawspec

    @staticmethod
    def calibrate_rawspec(rawspec, fluxcalfile):
        """ """
        return delayed(calibrate_spec)(rawspec, fluxcalfile)

    @staticmethod
    def build_fluxcalibrator(rawspec):
        """ """
        return delayed(build_fluxcalibrator)(rawspec)
        


    # -------- #
    # Checks   #
    # -------- #
    def compute_check(self, futures):
        """ """
        
