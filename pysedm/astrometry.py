#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" This module is made for the joint IFU Rainbow camera astrometry. """

import pandas
import numpy as np
import warnings
# Astropy
from astropy import units, coordinates
from astropy.io import fits
from astropy.wcs import WCS

# Others

from psfcube import fitter
from . import io
from .sedm import get_sedm_astrom_param



SEDM_SPAXEL_ARCSEC = 0.55
SEDM_LEFT_ROTATION = 2.5 * np.pi/180




# ======================= #
#   FIND POINT SOURCE     #
# ======================= #
MARKER_PROP = {"astrom": dict(marker="x", lw=2, s=80, color="C1", zorder=8),
               "manual": dict(marker="+", lw=2, s=100, color="k", zorder=8),
               "aperture": dict(marker="+", lw=2, s=100, color="k", zorder=8),
               "auto":  dict(marker="o", lw=2, s=200, facecolors="None", edgecolors="C3", zorder=8)
                   }

    
def position_source(cube,
                    centroid=None, centroiderr=None,
                    lbdaranges=[5000,7000], maxpos=False):
    """ How is the source position selected ? """
    
    if centroiderr is None or centroiderr in ["None"]:
        centroid_err_given = False
        centroids_err = [3, 3]
    else:
        centroid_err_given = True
        centroids_err = np.asarray(centroiderr, dtype="float")

    # Let's start...
    position_type = "unknown"
    
    # OPTION 1: Use maximum spaxels to derive position
    if maxpos:
        print("Centroid guessed based on brightness")
        xcentroid, ycentroid = estimate_default_position(cube, lbdaranges=lbdaranges)
        if not centroid_err_given:
            centroids_err = [5, 5]
        position_type="auto"
        
    # OPTION 2: Use astrometry of guider images for position, if possible
    elif centroid is None or centroid in ["None"]:
        xcentroid, ycentroid = get_object_ifu_pos(cube)
        
        if np.isnan(xcentroid*ycentroid):
            # FAILS... go back to OPTION 1
            print("IFU target location based on CCD astrometry failed. "
                  "centroid guessed based on brightness used instead")
            return position_source(cube, lbdaranges=lbdaranges, maxpos=True)
        
        # Works !
        print("IFU position based on CCD wcs solution used : ",
                  xcentroid, ycentroid)
        position_type="astrom"
            
    # OPTION 3: Use input centroid values for position
    else:
        try:
            xcentroid, ycentroid = np.asarray(centroid, dtype="float")
        except:
            raise ValueError("Unable to parse `centroid`. Should be None or a 2d float array `xcentroid, ycentroid = centroids`")
        position_type="manual"

    print("Position type %s: %.2f, %.2f" %
          (position_type, xcentroid, ycentroid))

    return [xcentroid, ycentroid], centroids_err, position_type

# ======================= #
#   High Level            #
# ======================= #
def fit_cube_pos(filename, lbdarange=[6000,8000], guess="guess", show=False, **kwargs):
    """ """
    from pyifu import spectroscopy
    from psfcube import fitter
    import shapely
    sl = spectroscopy.Slice.from_cubefile(filename)
    if guess in ["brightest"]:
        centroid = np.nanmean(np.asarray(sl.index_to_xy(sl.get_brightest_spaxels(5))), axis=0)
    else:
        centroid = guess_target_pos(filename)


    fitted_indexes = sl.get_spaxels_within_polygon( shapely.geometry.Point(*centroid).buffer(10) )
    sl2 = sl.get_subslice(fitted_indexes)
    guess_pos = np.nanmean(np.asarray(sl2.index_to_xy(sl2.get_brightest_spaxels(5))), axis=0)

    
    slf = fitter.fit_slice(sl2, centroids=guess_pos, **kwargs)
    xpos, ypox = [[slf.fitvalues["%scentroid"%ax_],slf.fitvalues["%scentroid.err"%ax_]] for ax_ in ["x","y"]]
    if show:
        fig = slf.show(zorder=3)
        [ax_.scatter(*centroid, marker="x", color="C1", zorder=6) for ax_ in fig.axes[:3]]
        [ax_.scatter(xpos[0], ypox[0], marker=".", color="k", zorder=6) for ax_ in fig.axes[:3]]

    del sl2, fitted_indexes
    del sl,centroid,slf
    return xpos, ypox

def filename_to_dateid(filename):
    """ """
    id_ = io.filename_to_id(filename)
    date = io.filename_to_date(filename)
    return date+"_"+id_

def get_allccd_pos(cube_filenames, verbose=True):
    """ """
    # This way, no "too many open files"
    coords = {}
    for fitsfile in cube_filenames:
        dateid = filename_to_dateid(fitsfile)
        coords[dateid] = np.asarray(get_ccd_pos(fitsfile)).tolist()
            
    return coords

def fit_allcubes_pos(cube_filenames, lbdarange=[6000,8000] ,**kwargs):
    """ """
    coords = {}
    for fitfile in cube_filenames:
        dateid = filename_to_dateid(fitfile)
        xpos, ypos = fit_cube_pos(fitfile,lbdarange=lbdarange ,**kwargs)
        coords[dateid] = [xpos, ypos]
        
    return coords
# ======================= #
#   GET LOCATIONS         #
# ======================= #
def guess_target_pos(filename, parameters=None):
    """ """
    from astropy.time import Time
    # date
    cube_date = io.filename_to_date(filename, iso=True)
    
    # = NEWEST VERSION = #
    if Time(cube_date) > Time("2019-04-17"):
        print("INFO: using new astrometric method")
        wcsifu = WCSIFU.from_filename(filename)
        return wcsifu.get_ifu_guess_pos()[0]

    
    print("WARNING: Soon to be deprecated: old ccd<->ifu structure. *You have nothing to do*")
    
    if parameters is None:
        parameters = get_sedm_astrom_param(cube_date)
        
    # CCD Position
    ccd_xy = get_ccd_pos(filename)
    if Time(cube_date) > Time("2019-02-20") and Time(cube_date) < Time("2019-04-17"):
        # print("TEMPORARY PATCH FOR SHIFTED IFU POS")
        return rainbow_coords_to_ifu(ccd_xy, parameters) + np.asarray([11, 1])
    elif Time(cube_date) > Time("2019-04-22"):
        #print("TEMPORARY PATCH 2 FOR SHIFTED IFU POS")
        return rainbow_coords_to_ifu(ccd_xy, parameters) + np.asarray([-5, -30])
    return rainbow_coords_to_ifu(ccd_xy, parameters)

    
    
def get_ccd_pos(filename, radec=None, verbose=True):
    """ 

    Parameters
    ----------
    
    """
    astrom_file = io.filename_to_guider(filename)
    if len(astrom_file)==0:
        if verbose:
            print("No astrom file found for %s"%filename)
        return [np.NaN,np.NaN]

    astrom_file = astrom_file[0]
    with fits.open(filename) as f:
        # Get target coordinates
        header = f[0].header
        if radec is None:
            try:
                radec = coordinates.SkyCoord(header["OBJRA"],header["OBJDEC"],
                                      unit=(units.hourangle, units.deg))
                # Format changed at some point
            except KeyError:
                radec = coordinates.SkyCoord(header["OBRA"], header["OBDEC"],
                                      unit=(units.hourangle, units.deg))
            del header
        elif type(radec) != coordinates.SkyCoord:
            radec = coordinates.SkyCoord(*radec, unit=units.deg)

    # Convert it into ccd pixels    
    with fits.open(astrom_file) as f:
        header = f[0].header
        wcs = WCS(header)
        xy  = np.asarray(radec.to_pixel(wcs))
        del header
        del wcs
        del radec
        
    return xy

def get_object_ifu_pos(cube, parameters=None):
    """ the expected cube x,y position of the target within the cube """
    print("DEPRECATED WARNING get_object_ifu_pos(cube) -> guess_target_pos(cubefile) ")
    return guess_target_pos(cube.filename)

def get_ccd_coords(cube):
    """ target position in the rainbow camera. 
    Remark that this area should not be visible in the rainbow camera as this area 
    should be behind the mirror sending the light to the IFU
    """
    print("DEPRECATED WARNING get_ccd_coords(cube) -> get_ccd_pos(cubefile)")
    return get_ccd_pos(cube.filename)

    
def fit_conversion_matrix(cubes_to_fit, guess=None):
    """ """
    from scipy.spatial import distance
    from scipy.optimize import fmin
    
    if guess is None:
        guess = get_sedm_astrom_param(cube_to_fit)
    else:
        guess= np.asarray(guess)

    list_of_ccd_positions = np.asarray([ get_ccd_coords(c_) for c_ in cubes_to_fit])
    list_of_ifu_positions = np.asarray([ fit_cube_centroid(c_)[:2] for c_ in cubes_to_fit])
    
    def to_fit(parameters):
        list_of_ifu_positions_MODEL = rainbow_coords_to_ifu(list_of_ccd_positions, parameters)
        
        return np.sum([distance.euclidean(im_, i_) for im_, i_ in zip(list_of_ifu_positions_MODEL, 
                                                                      list_of_ifu_positions)])
    return fmin(to_fit, GUESS, maxiter=10000)


# ======================= #
#   Other tools           #
# ======================= #
def estimate_default_position(cube, lbdaranges=[5000,7000]):
    """ """
    sl = cube.get_slice(lbda_min=lbdaranges[0], lbda_max=lbdaranges[1], slice_object=True)
    x,y = np.asarray(sl.index_to_xy(sl.indexes)).T # Slice x and y
    lim_perc = 99.5
    bright_lim = np.nanpercentile(sl.data, lim_perc)
    while np.isnan(bright_lim):
        lim_perc -= 5.
        bright_lim = np.nanpercentile(sl.data, lim_perc)

    argmaxes = np.argwhere(sl.data>bright_lim).flatten() # brightest points
    return np.nanmedian(x[argmaxes]),np.nanmedian(y[argmaxes]) # centroid

def rainbow_coords_to_ifu(ccd_coords, parameters):
    """ """
    centroid_ccd = np.asarray([parameters[4],parameters[5]])
    matrix = np.asarray([ [parameters[0], parameters[1]],
                          [parameters[2], parameters[3]] ])
    
    return np.dot(matrix, (ccd_coords-centroid_ccd).T).T

def fit_cube_centroid(cube_, lbdamin=6000, lbdamax=7000):
    """ Use `fit_slice` function from psfcube to estimate the PSF centroid 
    of a point-source containing in the cube. 
    The fit will be made on the metaslice within the boundaries `lbdamin` `lbdamax`

    Parameters
    ----------
    cube_: [pyifu Cube]
        Cube containing the point source
        
    lbdamin, lbdamax: [floats] -optional-
        lower and upper wavelength boundaries defining the metaslice
        [in Angstrom]

    Returns
    -------
    list: [x,y, dx, dy] # (centroids and errors)
    """
    sl_r = cube_.get_slice(lbda_min=lbdamin, lbda_max=lbdamax, slice_object=True)
    centroid = estimate_default_position(cube_)
    slfit = fitter.fit_slice(sl_r, centroids=centroid, centroids_err=[4,4])
    return [slfit.fitvalues[k] for k in ["xcentroid","ycentroid","xcentroid.err","ycentroid.err"]]

def get_standard_rainbow_coordinates(rainbow_wcs, stdname):
    """ """
    import pycalspec
    rhms = coordinates.SkyCoord(*pycalspec.std_radec(stdname),
                                unit=(units.hourangle, units.deg))
    return np.asarray(rhms.icrs.to_pixel(rainbow_wcs))





# ======================== #
#                          #
#    Class                 #
#                          #
# ======================== #
class WCSIFU():
    """ """

    def __init__(self, date=None):
        """ """
        if date is not None:
            self.load_transform(date)

    # ============== #
    #   Get          #
    # ============== #
    def coords_to_ccdxy(self, filename=None):
        """ """
        if filename is None:
            if not hasattr(self,"filename"):
                raise ValueError("no filename loaded, you must provide one.")
            
            filename = self.filename

    # ============== #
    #   Loader       #
    # ============== #
    @classmethod
    def from_filename(cls, filename):
        """ """
        date = io.filename_to_date( filename, iso=True)
        this = cls.from_date(date)
        this.filename = filename
        return this
    
    @classmethod
    def from_date(cls, date):
        """ """
        return cls(date)
    
    def load_transform(self, date):
        """ """
        from astropy.time import Time
        from skimage import transform
        if Time(date) < Time("2019-09-06"):
            transform_type = "Affine"
            parameters  = np.asarray([[ 9.40402465e-01, -1.17417175e-01, -8.52095952e+02],
                                  [ 3.03093868e-02, -7.43615349e-01,  6.94043198e+02],
                                  [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
            
        elif Time(date) > Time("2019-04-17"):
            transform_type = "Affine"
            parameters  = np.asarray([[ 8.02889540e-01, -2.30053433e-01, -6.00201420e+02],
                                      [ 1.94282497e-01, -5.06596412e-01,  2.93073783e+02],
                                      [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        else:
            raise ValueError("Transformation ready for date > 2019-04-17")
        
        self.transformation = eval("transform.%sTransform(parameters)"%transform_type)

    # ============== #
    #   Methods      #
    # ============== #
    def ccd_to_ifu(self, ccd_x, ccd_y):
        """ """
        return self.transformation(np.asarray([ccd_x, ccd_y]).T)
    
    def get_ifu_guess_pos(self, cube_filename=None):
        """ """
        if cube_filename is None:
            if not hasattr(self,"filename"):
                raise ValueError("no filename loaded, you must provide one.")
            
            cube_filename = self.filename
            
        return self.ccd_to_ifu( *self.get_ccd_pos(cube_filename) )

    @staticmethod
    def get_ccd_pos(cube_filename):
        """ """
        return get_ccd_pos(cube_filename)



def get_astrometry_fitter(date=None, download_data=False):
    """ 
    See timeline here http://www.astro.caltech.edu/sedm/Hardware.html
    """
    from ztfquery import sedm
    from astropy.time import Time
    if date is None:
        date = "2019-09-10"

    if Time(date) > Time("2019-09-05"):
        timerange = ["2019-09-05",None]
        NON_STDTARGET = ["ZTF19abqyoxj","ZTF19abwnpus","ZTF19abuayqg","ZTF19abuxhqc","ZTF19abujuex",
                     "ZTF19abuszom","ZTF19abyfazm","ZTF19abwscnk","ZTF19abqwtfu","ZTF19abvanim",
                     "ZTF19abrlznf","ZTF19abxvkvc","ZTF19abuzjqa","ZTF19abwaohs","ZTF19abwrzqb",
                     "ZTF18abksgkt","ZTF19abvhduf","ZTF19abupyxe","ZTF19abuhlxk","ZTF19abxivpg",
                     "ZTF19abvdbyx","ZTF19abttrte","ZTF19ablwwox","ZTF19aburnqh","ZTF19abupyxe"]

        NON_STDTARGET += ["ZTF19abrdnty","ZTF19abqykyd"]

    elif Time(date) > Time("2019-05-15"):
        print("Time Period: 2019-04-17, 2019-09-01")
        timerange = ["2019-05-15","2019-09-01"]        
        NON_STDTARGET = ["ZTF19abiahik","ZTF19abhzelh","ZTF19abhztqq","ZTF19abhemuy","ZTF19abiahik",
                         "ZTF19abhzelh","ZTF19abidbqp","ZTF19abgfuhh","ZTF19abgsyrp",
                         "ZTF19abfvpkq","ZTF19aawethv","ZTF19abhdqlq","ZTF19abcvrpm","ZTF19abdrwsz",
                         "ZTF19abdzehg","ZTF19abangub","ZTF19abegruk","ZTF19abdgpuv",
                          "ZTF19abbnamr","ZTF19abdkecj","ZTF19aatyrpj", "ZTF19aayjqpa"]

    elif Time(date) > Time("2019-04-23"):
        timerange = ["2019-04-23","2019-05-15"]
        NON_STDTARGET = ["ZTF19aatesgp","ZTF19aarkpif","ZTF19aatlmbo","ZTF19aatgxjy",
                        "ZTF19aarjfqe", "ZTF19aarstfg","ZTF19aariwpf"]

    else:
        raise ValueError("Astrometry Fitting made for date>2019-04-17")

    # Cosmics: ZTF19abdoznh
    # ztfquery finding target files
    p = sedm.SEDMQuery()
    if download_data:
        _ = [p.download_target_data(name, which="cube") for name in NON_STDTARGET] 
        _ = [p.download_target_data(name, which="astrom") for name in NON_STDTARGET]

    
    target_files = p.get_local_data(NON_STDTARGET,  timerange=timerange)
    return AstrometryFitter(target_files)

"""
20190728
"""



        
class AstrometryFitter():
    def __init__(self, cube_filenames):
        """ """
        self.input_files = {filename_to_dateid(f):f for f in cube_filenames}
        self.load_ccd_pos(cube_filenames)
        self.load_cube_pos(cube_filenames)
        
    def load_ccd_pos(self, cube_filenames=None, validation=False):
        """ """        
        coords = get_allccd_pos(cube_filenames)
        attrname = "ccd_xy%s"%("_validation" if validation else "")
        setattr(self,attrname, coords)
        
    def load_cube_pos(self, cube_filenames=None, validation=False):
        """ """        
        coords = fit_allcubes_pos(cube_filenames)
        attrname = "cube_xy%s"%("_validation" if validation else "")
        setattr(self,attrname, coords)
        
    def _load_dataframe_(self):
        """ """
        import pandas
        dict_out = {}
        for id_ in [i for i in self.cube_xy.keys() if i in self.ccd_xy.keys()]:
            dict_out[id_] = {}
            dict_out[id_]["spaxels_x"],dict_out[id_]["spaxels_y"] = np.asarray(self.cube_xy[id_])[:,0]
            dict_out[id_]["ccd_x"],dict_out[id_]["ccd_y"] = np.asarray(self.ccd_xy[id_])
            
        self._coordinates = pandas.DataFrame(dict_out).T
        delta_x,delta_y = (self.transformation(self.coordinates[["ccd_x","ccd_y"]])- 
                           self.coordinates[["spaxels_x","spaxels_y"]].values).T
        self.coordinates["delta_x"] = delta_x
        self.coordinates["delta_y"] = delta_y
        self.coordinates["delta"] = np.sqrt(delta_x**2 + delta_y**2)
        
    def load_transform(self, ttype="affine", **kwargs):
        """ 
        
        Parameters
        -----------
        ttype : [string] -optional-
            Type of transform.
            could be:
            'euclidean', similarity', 'affine', 'piecewise-affine', 'projective', 'polynomial'
            
        **kwargs
            Function parameters (src, dst, n, angle)::

            NAME / TTYPE        FUNCTION PARAMETERS
            'euclidean'         `src, `dst`
            'similarity'        `src, `dst`
            'affine'            `src, `dst`
            'piecewise-affine'  `src, `dst`
            'projective'        `src, `dst`
            'polynomial'        `src, `dst`, `order` (polynomial order,
                                                      default order is 2)

        """
        from skimage import transform
        self.flagout = np.asarray(np.nansum(np.isnan(self.coordinates).values, axis=1), dtype="bool")
        dd      = self.coordinates[~self.flagout]
        tform   =  transform.estimate_transform(ttype, 
                                                dd[["ccd_x","ccd_y"]].values,
                                                dd[["spaxels_x","spaxels_y"]].values, 
                                                **kwargs)
        
        self._transform = tform
        
    def get_residual_distance(self):
        """ """
        dd = self.coordinates[~self.flagout]
        spaxel_pos = dd[["spaxels_x","spaxels_y"]].values
        ccd_pos    = dd[["ccd_x","ccd_y"]].values
        spaxel_pos-self.transformation(ccd_pos)
        return spaxel_pos
    
    def show(self):
        """ """
        import matplotlib.pyplot as mpl
        fig = mpl.figure(figsize=[9,3])
        axccd   = fig.add_axes([0.08,0.1,0.27,0.8])
        axcube  = fig.add_axes([0.4,0.1,0.27,0.8])
        axmatch = fig.add_axes([0.72,0.1,0.27,0.8])

        dd = self.coordinates[~self.flagout]
        spaxel_pos = dd[["spaxels_x","spaxels_y"]].values
        ccd_pos    = dd[["ccd_x","ccd_y"]].values
        npoints = len(dd)
        colors = mpl.cm.viridis(np.arange(len(dd))/len(dd))
        axccd.scatter(*ccd_pos.T, c=colors)
        axcube.scatter(*spaxel_pos.T, c=colors)

        axmatch.scatter(*(spaxel_pos-self.transformation(ccd_pos)).T, 
                        c=colors, marker="x", s=100)
        # // Titles
        axccd.set_title("RC-Expectation")
        axcube.set_title("IFU Measurement")
        axmatch.set_title(r"$\Delta$  IFU pos-prediction")
    
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def ntargets(self):
        """ """
        return len(self.coordinates)
    
    @property
    def coordinates(self):
        """ """
        if not hasattr(self, "_coordinates") or self._coordinates is None:
            if hasattr(self,"ccd_xy") and hasattr(self,"cube_xy"):
                self._load_dataframe_()
            else:
                self._coordinates = None
        return self._coordinates
    @property
    def transformation(self):
        """ """
        if not hasattr(self,"_transform"):
            self.load_transform()
        return self._transform
    
