
import os
import pandas
from astropy import table, coordinates, units
from astropy.io import fits
from dask import delayed
from .base import DaskCube
import warnings
import numpy as np
from shapely import geometry

from pysedm import get_sedmcube, io, fluxcalibration, astrometry


SEDM_SCALE = 0.558


# // HyperGal - INTRINSEC CUBE
def fetch_cutouts(cubefile=None, radec=None, source="ps1", size=140,
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
    sedfitter.setup_cigale_df(redshift=redshift, snr=snr, path_to_save=os.path.join(working_dir, "cig_df.txt") )
    #  Cigale Initiates
    sedfitter.initiate_cigale(cores=cores, working_dir=working_dir, **init_prop)
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
        warnings.warn("Storing to {cube3d.filename}; filename={filename}")
        cube3d.writeto(cube3d.filename)
        resdf = table.Table( fits.open( os.path.join( sedfitter._working_dir, "out","results.fits") )[1].data
                            ).to_pandas()
        resdf.to_csv(cube3d.filename.replace("hginte3d",sedkey).replace(".fits", ".csv"))
        
    if store_fig:
        savefile = cube3d.filename.replace(".fits",".pdf") 
        cube3d.show(savefile=savefile)
        sedfitter.show_rms(savefile=savefile.replace("hginte3d",sedkey))

    if clean_output:
        _ = sedfitter.clean_output()

    return cube3d



def get_scene(calcube_filename, intrinsiccube_filename, sedm_targetpos,
                  psfmodel="Gauss_Mof_kernel"):# GaussMofKernel
    """ """
    calcube = get_sedmcube(calcube_filename)
    intrinsiccube = get_sedmcube(intrinsiccube_filename)
    from hypergal import geometry_tool as geotool
    from hypergal import intrinsec_cube as scenemodel
    #from hypergal import scenemodel
    int_geometry  = geometry.MultiPolygon(intrinsiccube.get_spaxel_polygon())

    int_pixel = intrinsiccube.header["PIXSIZE"]
    
    int_targetpos = intrinsiccube.header["OBJX"],intrinsiccube.header["OBJY"]
    
    init_hexgrid = geotool.get_cube_grid(calcube, scale=SEDM_SCALE/int_pixel,
                                         targShift=sedm_targetpos,
                                         x0=int_targetpos[0],
                                         y0=int_targetpos[1])
    #    scenemodel.HostScene( )
    hostscene = scenemodel.Intrinsec_cube(int_geometry, int_pixel, init_hexgrid,
                                  intrinsiccube.data.T, intrinsiccube.lbda,
                                  psf_model=psfmodel)
    hostscene.set_sedm_targetpos(sedm_targetpos)
    hostscene.set_int_targetpos(int_targetpos)    
    
    calcube.load_adr()
    hostscene.load_adr( calcube.adr.copy() )
    return hostscene
    
def get_fitter(calcube_filename, scene):
    """ """
    calcube = get_sedmcube(calcube_filename)
    from hypergal import fitter
    f_ = fitter.Fitter(calcube, scene)
    return f_

# 7. Fit
def fit_slice(fitter, sliceid, lbda_range=[5000, 8000], nslices=5, **kwargs):
    """ """
    fitvalues = fitter.fit_slice(lbda_ranges=lbda_range, metaslices=nslices,
                                                     sliceid=sliceid, **kwargs)
    return fitvalues

def build_results(fit_values):
    """ """
    fitvalues_ = {}
    for d_ in fit_values:
        fitvalues_.update(d_)

    return pandas.DataFrame.from_records(fitvalues_)


def fit_adr(fitter, slice_fit_results, lbdaref=None):
    """ """
    x0, y0, x0err, y0err, lbda = slice_fit_results[["x0_IFU", "y0_IFU", "x0_IFU_err", "y0_IFU_err", "lbda"]].values.T
    adrfitres, adrobj = fitter.get_fitted_adr(x0, y0, x0err, y0err, lbda,
                                                  lbdaref=lbdaref)    
    return {**adrobj.data, **{"x0_IFU":fitter.fit_xref,"y0_IFU":fitter.fit_yref}}

def fit_psf(fitter, slice_fit_results):
    """ """
    # TMP PATCH
    colparam = [k for k in slice_fit_results.columns if k+"_err" in slice_fit_results.columns or k=="lbda"]
    fitvalues = slice_fit_results[colparam].T.to_dict()
    fitvalues_err = slice_fit_results[[k+"_err" if k != "lbda" else k for k in colparam ]].T.to_dict()
    
    return fitter.get_fitted_psf(fitvalues, fitvalues_err)
    
def fit_fullcube(fitter, adr_params, psf_params, store_data=True, get_filename=True):
    """ """
    cubemodel = fitter.evaluate_model_cube(parameters={**adr_params, **psf_params})
    cubemodel.set_filename(fitter.sedm_cube.filename.replace("e3d", "hghostmodel"))
    if store_data:
        cubemodel.writeto(cubemodel.filename)

    if get_filename:
        return cubemodel.filename        
    return cubemodel

class DaskHyperGal( DaskCube ):

    @staticmethod
    def get_cubeinfo(cubefile_):
        """ """
        header_  = fits.getheader(cubefile_)
        co=coordinates.SkyCoord(header_.get("OBJRA"), header_.get("OBJDEC"),
                                    frame='icrs', unit=(units.hourangle, units.deg))
        ra=co.ra.deg
        dec=co.dec.deg
        target_pos = astrometry.position_source( get_sedmcube(cubefile_), warn=False)[0]
        cubefile_id = ''.join(os.path.basename(cubefile_).split("ifu")[-1].split("_")[:4])
        workingdir = os.path.abspath(f"tmp_{cubefile_id}")
        return [ra,dec], workingdir, target_pos

    @staticmethod
    def get_intrinsic_cube(radec, redshift, working_dir=None, use_cigale=True,
                               store_fig=True, filename="intrinsic_cube.fits",
                               as_filename=True, **kwargs):
        """ """

        # 4. 
        geodf_cutouts_pix_coord = delayed(fetch_cutouts)(radec=radec, as_cigale=use_cigale)
        geodf_cutouts = geodf_cutouts_pix_coord[0]
        pix = geodf_cutouts_pix_coord[1]
        coord = geodf_cutouts_pix_coord[2]
        
        # 5
        cube = delayed(build_intrinsic_cube)(geodf_cutouts, redshift, working_dir=working_dir,
                                                       use_cigale=use_cigale,
                                                       pixsize=pix, targetcoord=coord, filename=filename,
                                                       store_data=True, store_fig=store_fig, **kwargs)
        
        return cube if not as_filename else cube.filename 
        
    @staticmethod
    def fit_scene(calibrated_cube_filename, intrinsic_cube_filename, sedm_targetpos,
                    lbda_range=[5000,8000], nslices=5, store_data = True, **kwargs):
        """ """
        hostscene = delayed(get_scene)(calibrated_cube_filename, intrinsic_cube_filename, sedm_targetpos=sedm_targetpos)
        fitter = delayed(get_fitter)(calibrated_cube_filename, hostscene)

        fit_param = []
        for i_ in np.arange(nslices):
            fit_param.append(delayed(fit_slice)(fitter, i_, lbda_range=lbda_range, nslices=nslices, **kwargs))

        residual = delayed(build_results)(fit_param)
        
        adr_param = delayed(fit_adr)(fitter, residual)
        psf_param = delayed(fit_psf)(fitter, residual)
        cubemodel = delayed(fit_fullcube)(fitter, adr_param, psf_param, store_data=True, get_filename=True)

        
        return cubemodel
        
    @staticmethod
    def single_hypergal(cubefile_, redshift_,
                            fluxcalfile=None, # 
                            radec=None, use_cigale=True,
                            apply_br=True,
                            nmetaslices=5,
                            cubeprop={}, intprop={}):
        """ """
        # 0
        radec_wd_pos = delayed(self.get_cubeinfo)(cubefile_)
        radec = radec_wd_pos[0]
        workingdir = radec_wd_pos[1]
        filename_hgint = cubefile_.replace("e3d_crr","hginte3d_crr")
        sedm_targetpos = radec_wd_pos[2]
        
        # ----- #
        # Cube  #
        # ----- #
        
        # Branch Cube Generation
        calibrated_cube = self.get_calibrated_cube(cubefile_, fluxcalfile=fluxcalfile, apply_br=apply_br,
                                                       as_filename=False, **cubeprop)
       
        # Branch Intrinsic Cube
        intrinsic_cube = self.get_intrinsic_cube(radec, redshift_, workingdir=workingdir,
                                                 filename=filename_hgint,
                                                 as_filename=False,
                                                 **intprop)
        
        modecube = self.fit_scene(calibrated_cube, intrinsic_cube, sedm_targetpos)
        return None
