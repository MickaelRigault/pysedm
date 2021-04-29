""" Dasked version of pysedm/bin/extractstars.py """

from dask import delayed
from .base import ClientHolder


from .. import get_sedmcube, fluxcalibration, io
from ..sedm import SEDMExtractStar, flux_calibrate_sedm



def get_extractstar(cube, step1range=[4500,7000], step1bins=6,
                   centroid="auto", **kwargs):
    """ """
    es = SEDMExtractStar(cube)
    es.set_lbdastep1(lbdarange=step1range, bins=step1bins)
    es.set_centroid(centroid=centroid)
    return es

def get_fluxcal_file(cube):
    """ """
    return io.fetch_nearest_fluxcal(mjd=cube.header.get("MJD_OBS"))

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
           fwhm_guess=fwhm_guess, **kwargs)
    
    return es.get_spectrum("raw", persecond=True)


def calibrate_spec(spec, fluxcalfile, default_airmass=1.1):
    """ """
    if fluxcalfile is None:
        spec.header["FLUXCAL"] = (False,"has the spectra been flux calibrated")
        spec.header["CALSRC"] = (None, "Flux calibrator filename")
        spec.header["BUNIT"]  = (spec.header.get('BUNIT',""),"Flux Units")
        specout = spec.copy()
    else:
        fluxcal = fluxcalibration.load_fluxcal_spectrum(fluxcalfile)
        spec.scale_by( fluxcal.get_inversed_sensitivity( spec.header.get("AIRMASS", default_airmass) ),
                      onraw=False)
        spec.header["FLUXCAL"] = (True, "has the spectra been flux calibrated")
        spec.header["CALSRC"] = (os.path.basename(fluxcalfile), "Flux calibrator filename")
        spec.header["BUNIT"]  = ("erg/s/A/cm^2", "Flux Units")
        specout = spec.copy()
        
    return specout
        
def build_fluxcalibrator(rawspec_std):
    """ """
    speccal, fl = fluxcalibration.get_fluxcalibrator(rawspec_std, fullout=True)
    speccal.header.set("SOURCE", rawspec_std.filename.split("/")[-1], "This object has been derived from this file")
    speccal.header.set("PYSEDMT","Flux Calibration Spectrum", "Object to use to flux calibrate")
    return speccal, fl



class DaskES( ClientHolder ):
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
    # Initialisation  #
    # =============== #
    def __init__(self, client=None, cubefiles=None):
        """ """
        _ = super().__init__(client=client)
        if cubefiles is not None:
            self.set_cubefiles(cubefiles)
            
    @classmethod
    def from_name(cls, name, client):
        """ """
        from ztfquery import sedm
        squery = sedm.SEDMQuery()
        cubefiles = squery.get_target_cubes(name, client=client)
        return cls.from_cubefiles(cubefiles=cubefiles, client=client)

    @classmethod
    def from_date(cls, date, client):
        """ """
        from ztfquery import sedm
        squery = sedm.SEDMQuery()
        cubefiles = squery.get_night_cubes(date, client=client)
        return cls.from_cubefiles(cubefiles=cubefiles, client=client)

    @classmethod
    def from_month(cls, year, month, client):
        """ """
        from calendar import monthrange
        monthdates = [f'{year:04d}{month:02d}{d:02d}' for d in range(1, monthrange(year, month)[1] + 1)]
        return cls.from_date(monthdates, client=client)
    
    @classmethod
    def from_cubefiles(cls, cubefiles, client):
        """ shortcut to __init__ for simplicity """
        return cls(client=client, cubefiles=cubefiles)

    # =============== #
    #    Methods      #
    # =============== #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_cubefiles(self, cubefiles):
        """ """
        self._cubefiles = cubefiles
        
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
    
        
    def single_extractstars(self, cube_filename, build_fluxcalibrator=False,  **kwargs):
        """ """
        # 1. Get cube
        cube = delayed(get_sedmcube)(cube_filename)
        
        # 2. Get flux calibration file (if any)
        fluxcalfile = delayed(get_fluxcal_file)(cube) # could be None
        
        # 3. Get ExtractStar
        estar = delayed(get_extractstar)(cube)
        
        # 4. Run ExtractStar
        specraw = delayed(run_extractstar)(estar)

        # 5. Calibrate the spectra
        speccal = delayed(calibrate_spec)(specraw, fluxcalfile)

        # 6. Build the flux Calibrator
        if build_fluxcalibrator:
            fluxcalibrator = delayed(build_fluxcalibrator)(specraw)
        else:
            fluxcalibrator = None
            
        return speccal, fluxcalibrator
        
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def cubefiles(self):
        """ """
        if not hasattr(self,"_cubefiles"):
            return None
        return self._cubefiles

    def has_cubefiles(self):
        """ """
        return self.cubefiles is not None and len(self.cubefiles)>0

