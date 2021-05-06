

import pandas
from astropy import table
from astropy.io import fits
from dask import delayed
from .base import DaskCube


from pysedm import get_sedmcube, io, fluxcalibration


def get_cube(cubefile, apply_bycr=True):
    """ """
    # To be 
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
    cube.scale_by( fluxcal.get_inversed_sensitivity( cube.header.get("AIRMASS", backup_airmass) ),
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
    
    return ps_target.build_geo_dataframe(subsample,as_cigale=as_cigale), ps_target.get_pix_size(), ps_target.get_target_coord()
    
# 6
def build_intrinsic_cube(geodataframe, redshift, working_dir,
                         use_cigale=True, snr=3,
                         pixsize=None, targetcoord=None, filename="intrinsic_cube.fits",
                         cores=1, init_prop={}, clean_output=True,
                         store_data=True, store_fig=True, **kwargs):
    """ """
    if not use_cigale:
        raise NotImplementedError("Only CIGALE sed fit implemented")

    from hypergal import sed_fitting
    sedkey = "hgsed"
    
    # Loads the SEDFitter
    sedfitter = sed_fitting.CigaleSED(geodataframe)
    sedfitter.setup(redshift=redshift, snr=snr, path_to_save=working_dir)
    #  Cigale Initiates
    sedfitter.initial_cigale(cores=cores, working_dir=working_dir, **init_prop)
    #  Cigale Run and output results in {workingdir}/out
    _ = sedfitter.run_cigale(path_result=None)

    # reads outputs, reshape the spectra and build the cube.
    cube3d = sedfitter.get_sample_spectra(save_dirout_data=None, as_cube=True)
    if pixsize is not None:
        cube3d.header.set("PIXSIZE",pixsize)
    else:
        cube3d.header.set("PIXSIZE", None )
        
    if targetcoord is not None:
        cube3d.header.set("OBJX", targetcoord[0])
        cube3d.header.set("OBJY", targetcoord[1])
    else:
        cube3d.header.set("OBJX", None)
        cube3d.header.set("OBJY", None)
        
    cube3d.set_filename(filename)
    
    if store_data:
        cube3d.writeto(cube3d.filename)
        resdf = table.Table( os.path.join( sedfitter._working_dir, "out","results.fits") ).to_pandas()
        resdf.to_csv(cube3d.filename.replace("hginte3d",sedkey).replace(".fits", ".csv"))
        
    if store_fig:
        savefile = cube3d.filename.replace(".fits",".pdf") 
        cube3d.show(savefile=savefile)
        sedfitter.show_rms(savefile=savefile.replace("hginte3d",sedkey))

    if clean_output:
        _ = sedfitter.clean_output()

    return cube3d

    
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
    def get_cubeinfo(cubefile_):
        """ """
        header_  = fits.getheader(cubefile_)
        ra = header_.get("OBJRA", None)
        dec = header_.get("OBJDEC", None)
        cubefile_id = ''.join(os.path.basename(cubefile_).split("ifu")[-1].split("_")[:4])
        workingdir = f"tmp_{cubefile_id}"
        return [ra,dec], workingdir
    
    @staticmethod
    def get_calibrated_cube(cubefile_, fluxcalfile=None, apply_bycr=True, **kwargs):
        """ """
         # 1. Get cube
        cube = delayed(get_cube)(cubefile_, apply_bycr=apply_bycr)

        # 2. Get flux calibration file (if any)
        if fluxcalfile is None:
            fluxcalfile = delayed(get_fluxcal_file)(cube) # could be None

        # 3. Flux calibrating the cube
        calibrated_cube = delayed(calibrate_cube)(cube, fluxcalfile, **kwargs)
        return calibrated_cube


    @staticmethod
    def get_intrinsic_cube(radec, redshift, workingdir=None, use_cigale=True, store_fig=True, filename="intrinsic_cube.fits"):
        """ """

        # 4. 
        geodf_cutouts_pix_coord = delayed(fetch_cutouts)(radec, as_cigale=use_cigale)
        geodf_cutouts = geodf_cutouts_pix_coord[0]
        pix = geodf_cutouts_pix_coord[1]
        coord = geodf_cutouts_pix_coord[2]
        
        # 5
        intrinsec_cube = delayed(build_intrinsic_cube)(geodf_cutouts, redshift, workingdir=workingdir,
                                                       use_cigale=use_cigale,
                                                       pixsize=pix, targetcoord=coord, filename=filename,
                                                       store_data=True, store_fig=store_fig, **kwargs)
        
        return intrinsec_cube
        

    @staticmethod
    def single_hypergal(cubefile_, redshift_,
                            fluxcalfile=None, # 
                            radec=None, use_cigale=True,
                            apply_br=True,
                            nmetaslices=5,
                            cubeprop={}, intprop={}):
        """ """
        # 0
        radec_wd = delayed(self.get_cubeinfo)(cubefile_)
        radec = radec_wd[0]
        workingdir = radec_wd[1]
        filename_hgint = cubefile_.replace("e3d_crr","hginte3d_crr")
        # ----- #
        # Cube  #
        # ----- #
        # Branch Cube Generation
        calibrated_cube = self.get_calibrated_cube(cubefile_, fluxcalfile=fluxcalfile, apply_br=apply_br, **cubeprop)
       
        # ----- #
        # PS1   #
        # ----- #
        # Branch Intrinsic Cube
        intrinsic_cube = self.get_intrinsic_cube(radec, redshift_, workingdir=workingdir,
                                                 filename=filename_hgint,
                                                 **intprop)
        
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
