""" Dasked version of pysedm/bin/extractstars.py """

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

def get_fluxcal(cubefile):
    """ """
    fluxcal_file = io.fetch_nearest_fluxcal(mjd=fits.getval(cubefile,"MJD_OBS"))
    if fluxcal_file is None:
        return None
    return fluxcalibration.load_fluxcal_spectrum(fluxcalfile)

def run_extractstar(es, spaxelbuffer = 10,
                    spaxels_to_use=None,
                    spaxels_to_avoid=None,
                    psfmodel="NormalMoffatTilted",
                    slice_width = 1, fwhm_guess=None, 
                    verbose=False, **kwargs):
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
    
    return es.get_spectrum("raw", persecond=True)


def calibrate_spec(spec, fluxcal, default_airmass=1.1):
    """ """
    if fluxcal is None:
        spec.header["FLUXCAL"] = (False,"has the spectra been flux calibrated")
        spec.header["CALSRC"] = (None, "Flux calibrator filename")
        spec.header["BUNIT"]  = (spec.header.get('BUNIT',""),"Flux Units")
        specout = spec.copy()
    else:
        spec.scale_by( fluxcal.get_inversed_sensitivity( spec.header.get("AIRMASS", default_airmass) ),
                      onraw=False)
        spec.header["FLUXCAL"] = (True, "has the spectra been flux calibrated")
        spec.header["CALSRC"] = (os.path.basename(fluxcalfile), "Flux calibrator filename")
        spec.header["BUNIT"]  = ("erg/s/A/cm^2", "Flux Units")
        specout = spec.copy()
        
    return specout
        
def build_fluxcalibrator(rawspec_std, store_plot=True):
    """ """
    speccal, fl = fluxcalibration.get_fluxcalibrator(rawspec_std, fullout=True)
#    speccal.header.set("SOURCE", rawspec_std.filename.split("/")[-1], "This object has been derived from this file")
    speccal.header.set("PYSEDMT","Flux Calibration Spectrum", "Object to use to flux calibrate")
    if store_plot:
        _ = fl # 
    return speccal



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
    # -------- #
    # Running  #
    # -------- #
    def compute(self, cubefiles=None):
        """ """
        if cubefiles is None:
            cubefiles = self.cubefiles
            
        delayed_ = [self.single_extractstars(cubefiles_, build_fluxcalibrator=False)[0]
                        for cubefiles_ in cubefiles]
        return self.client.compute(delayed_)


    def stdconnected_extractstars(self, std_basename):
        """ """
        cubefile_df  = self.get_cubefile_dataframe().xs(std_basename).set_index("basename")
        cubefile_std = cubefile_df.loc[std_filename]["filepath"]
        cubefiles    = cubefile_df.drop(std_filename)["filepath"].values
    
        raw_std = self.extract_rawstar(cubefile_std)
        fluxcal = self.build_fluxcalibrator(raw_std)
        f_spec = [raw_std]
        for cubefile in cubefiles:
            rawspec = self.extract_rawstar(cubefile)
            spec = self.calibrate_rawspec(rawspec, fluxcal)
            f_spec.append(spec)
            
        return f_spec
    
    @classmethod
    def single_extractstars(cls, cube_filename, fluxcal=None, calibrate=True):
        """ """
        rawspec = cls.extract_rawstar(cube_filename)
        if fluxcal is None and calibrate:
            fluxcal = delayed(get_fluxcal)(cube_filename)
            
        return cls.calibrate_rawspec(rawspec, fluxcal)

    # -------- #
    # Static   #
    # -------- #
    @staticmethod
    def extract_rawstar(cube_filename, **kwargs):
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
        rawspec = delayed(run_extractstar)(estar)
        return rawspec

    @staticmethod
    def calibrate_rawspec(rawspec, fluxcal):
        """ """
        return delayed(calibrate_spec)(rawspec, fluxcal)

    @staticmethod
    def build_fluxcalibrator(rawspec):
        """ """
        return delayed(build_fluxcalibrator)(rawspec)
        
