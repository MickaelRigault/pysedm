

from dask import delayed
from .base import DaskCube


from pysedm import get_sedmcube, io, fluxcalibration


def get_cube(cubefile, apply_bycr=True):
    """ """
    cube = get_sedmcube(cubefile)
    if apply_bycr:
        print("BY CR TO BE IMPLEMENTED")

    return cube

def get_fluxcal_file(cube):
    """ """
    return io.fetch_nearest_fluxcal(mjd=cube.header.get("MJD_OBS"))

def calibrate_cube(cube, fluxcalfile, airmass=None, backup_airmass=1.1):
    """ """
    if airmass is None:
        airmass = cube.header.get("AIRMASS", backup_airmass)
        
    fluxcal = fluxcalibration.load_fluxcal_spectrum(fluxcalfile)
    cube.scale_by( fluxcal.get_inversed_sensitivity( cube.header.get("AIRMASS", default_airmass) ),
                      onraw=False)
    return cube

# // HyperGal - INTRINSEC CUBE
def fetch_cutouts(cubefile, radec=None, source="ps1", size=240,
                      subsample=2, as_cigale=True):
    """ """
    if radec is None:
        ra = fits.getheader.getval(cubefile, "RA")
        dec = fits.getheader.getval(cubefile, "DEC")
    else:
        ra,dec = radec

    if source != "ps1":
        raise NotImplementedError("Only Panstarrs cutout implemented")

    from hypergal import panstarrs_target
    
    ps_target = panstarrs_target.Panstarrs_target(ra,dec)
    ps_target.load_cutout(size=size)
    
    return ps_target.build_geo_dataframe(subsample,as_cigale=as_cigale), ps_target.get_target_size(), ps_target.get_target_pos()
    
# 6
def run_sed(geodataframe, redshift, use_cigale=True, snr=3, tmp_path=None,
                cores=1, init_prop={}, clean_output=False, **kwargs):
    """ """
    from hypergal import sed_fitting
    if not use_cigale:
        raise NotImplementedError("Only CIGALE sed fit implemented")

    cigale = sed_fitting.Cigale_sed(geodataframe)
    cigale.setup(redshift=redshift, snr=snr, path_to_save=tmp_path)
    cigale.initial_cigale(cores=cores, **init_prop)
    
    _ = cigale.run_cigale(TMP_PATH_RESULT,FILDER_OUR)
    _ = cigale.get_sample_spectra()
    if clean_output:
        _ = cigale.clean_output()
        
    return cigale.get_3d_cube()

    
# 7. Fit
def split_into_slices(cube_, nslices):
    """ """
    print("split_into_slices TO BE DONE")
    return "List_of_slices"

def fit_slices(cal_slice, int_slice, pixpol, pixsize, *kwargs):
    """ """
    print("fit_slices TO BE DONE")
    parameters = []
    return parameters

def build_adr_and_psf(metaslice_parameters):
    """ """
    adr=None
    psf=None
    return adr, psf


def fit_cube(calibrated_cube, intrinsec_cube, adr, psf):
    """ """
    cubemodel=None
    return cubemodel

class DaskHyperGal( DaskCube ):

    @staticmethod
    def single_hypergal(cubefile_, redshift_,
                            radec=None, use_cigale=True,
                            apply_br=True,
                            airmass=None, nmetaslices=5):
        """ """
        # ----- #
        # Cube  #
        # ----- # 
        # 1. Get cube
        cube = delayed(get_cube)(cubefile_, apply_br=apply_br)

        # 2. Get flux calibration file (if any)
        fluxcalfile = delayed(get_fluxcal_file)(cube) # could be None

        # 3. Flux calibrating the cube
        calibrated_cube = delayed(calibrate_cube)(cube, fluxcalfile, airmass=airmass)

        # ----- #
        # PS1   #
        # ----- #
        # 4. Get the cut out
        cutout_pixsize_pixpos = delayed(fetch_cutouts)(cubefile_, radec, as_cigale=use_cigale)
        geodf_cutouts = cutout_pixsize_pixpos[0]
        pixsize = cutout_pixsize_pixpos[1]
        pixpol = cutout_pixsize_pixpos[2]

        # 6. Run CEAGALS
        intrinsec_cube = delayed(run_sed)(geodf_cutouts, redshift_, use_cigale=use_cigale)
        

        # 7. Fit
        #    7.1 metaslices
        calibrated_mslices = delayed(split_into_slices)(calibrated_cube, nslices=nmetaslices)
        intrinsec_mslices = delayed(split_into_slices)(intrinsec_cube, nslices=nmetaslices)
        #    7.2 fit_parameters
        meta_params = []
        for i in range(nmetaslices):
            metai = delayed(fit_slices)(calibrated_mslices[i], intrinsec_mslices[i], pixsize=pixsize, pixpol=pixpol)
            meta_params.append(metai)

        # 7.3 build ADR and PSF
        adr_psf = delayed(build_adr_and_psf)(meta_params)
        adr = adr_psf[0] # Dask stuff
        psf = adr_psf[1]

        # 7.4 fit_slices
        modelcube = delayed(fit_cube)(calibrated_cube, intrinsec_cube, adr, psf)
        return modelcube
