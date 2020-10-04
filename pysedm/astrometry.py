#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" This module is made for the joint IFU Rainbow camera astrometry. """

import pandas
import numpy as np
import warnings
from scipy import stats
# Astropy
from astropy import units, coordinates, wcs
from astropy.io import fits
from astropy.time import Time


import iminuit
# Others

from psfcube import fitter
from . import io
from . import sedm





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
def guess_target_pos(filename, parameters=None, oldmethod=False):
    """ """
    if oldmethod:
        return oldversion_get_target_pos(filename, parameters=parameters)
    
    astro = Astrometry(filename)
    return astro.get_target_coordinate()


def oldversion_get_target_pos(filename, parameters=None):
    print
    # date
    cube_date = io.filename_to_date(filename, iso=True)
    
    # = NEWEST VERSION = #
    if Time(cube_date) > Time("2019-04-17"):
        print("INFO: using new astrometric method")
        wcsifu = WCSIFU.from_filename(filename)
        return wcsifu.get_ifu_guess_pos()[0]

    
    print("WARNING: Soon to be deprecated: old ccd<->ifu structure. *You have nothing to do*")
    
    if parameters is None:
        parameters = sedm.get_sedm_astrom_param(cube_date)
        
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
        wcs_ = wcs.WCS(header)
        xy  = np.asarray(radec.to_pixel(wcs_))
        del header
        del wcs_
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
        guess = sedm.get_sedm_astrom_param(cube_to_fit)
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
class Astrometry():
    # Manages connections between RA-Dec | IFU | Guider coordinates
    
    def __init__(self, filename=None, isodate=None):
        """ """
        if filename is not None:
            self.filename = filename
            self._load_parameters_(io.filename_to_date(filename, iso=True))
        if isodate is not None:
            self._load_parameters_(isodate)

    # ================ #
    #  Methods         #
    # ================ #
    def _load_parameters_(self, date):
        """ """
        self.date = date
        if Time(date) >= Time("2020-10-04"):
            position = 1039.5,968.5
            scale=0.55
            rotation=1
            print("THIS ONE 2020-08-28")
        
        elif Time(date) >= Time("2020-08-28"):
            position = 1027.5,950.5
            scale=0.55
            rotation=1
            print("THIS ONE 2020-08-28")

        elif Time(date) >= Time("2020-07-10"):
            position = 1023.5, 958.5
            scale=0.55
            rotation=1
            print("THIS ONE 2020-10-07")

        elif Time(date) > Time("2019-09-06"):
            position = 1026.5, 976
            scale=0.55
            rotation=1
            print("THIS ONE 2019-09-06")
            
        elif Time(date) > Time("2019-05-14"):
            position = 1026.5, 975.3
            scale=0.55
            rotation=1
            print("THIS ONE *2019-05-14*")
            
        elif Time(date) > Time("2019-04-23"):
            print("WARNING TIME RANGE NOT DONE")
            position = 1027, 975
            scale=0.55
            rotation=2            
        elif Time(date) > Time("2019-04-18"):
            print("WARNING TIME RANGE NOT DONE")
            position = 1027, 975
            scale=0.55
            rotation=2            
        elif Time(date) > Time("2019-02-08"):
            print("WARNING TIME RANGE NOT DONE")
            position = 1027, 975
            scale=0.55
            rotation=2            
        elif Time(date) > Time("2019-01-05"):
            print("WARNING THIS DATA SHOULD NOT EXIST.")
            position = 1027, 975
            scale=0.55
            rotation=2            
        else:
            print("WARNING THIS DATA SHOULD NOT EXIST.")
            position = 1027, 975
            scale=0.55
            rotation=2            

        self.parameters= {"position":position, 
                          "scale":scale, 
                          "rotation":rotation}
        
    # -------- #
    #  SETTER  #
    # -------- #
    def set_guider_offset(self, offsetx, offsety):
        """ """
        self._guider_offset = np.asarray([offsetx, offsety])

    def set_ifu_offset(self, spaxelx, spaxely):
        """ """
        self._ifu_offset = np.asarray([spaxelx, spaxely])
        
    # -------- #
    #  LOADER  #
    # -------- #
    def load_cube(self):
        """ """
        self.cube = sedm.get_sedmcube(self.filename)
        self.cube.load_adr()

    def load_slice(self, lbdarange=[5000,6000]):
        """ """
        from pyifu import spectroscopy
        self.slice = spectroscopy.Slice.from_cubefile(self.filename, lbdarange=lbdarange)
        
    def load_guider(self, load_gaiacat=True):
        """ """
        from astrobject import photometry
        self._guider = photometry.get_image(self.astromfile, background=0)
        if load_gaiacat:
            self._guider.download_catalogue("gaia")
            
    # -------- #
    #  GETTER  #
    # -------- #
    def get_target_coordinate(self, where="ifu"):
        """ """
        return self.radec_to(where, *self.target_radec)

    # -------- #
    # CONVERT  #
    # -------- #
    def ifu_to(self, where, spaxelx, spaxely):
        """ """
        return self.radec_to(where, *self.ifu_to_radec(spaxelx, spaxely))

    def guider_to(self, where, x, y):
        """ """
        if where in ["guider","guiderxy","astrom", "astromxy"]:
            return np.asarray([x,y])
        if where in ["coords", "radec", "world", "sky"]:
            return self.wcs_guider.all_pix2world(np.asarray([np.atleast_1d(x), np.atleast_1d(y)]).T, 0)[0]

        return self.radec_to(where, *self.guider_to("radec",x, y))
    
    def radec_to(self, where, ra, dec):
        """ 
        
        Parameters
        ----------
        where: [string]
            The destination, it could be:
            - guider/astrom: guider image ccd (pixels)
            - ifu: IFU (spaxels)
            
        ra, dec: [float, float or list of]
            Input Coordinates 
            
        Returns
        -------
        same format of ra, dec
        """
        # CCD Positions (reference image or guider)
        if where in ["guider","guiderxy","astrom", "astromxy"]:
            return self.wcs_guider.all_world2pix(np.asarray([np.atleast_1d(ra), np.atleast_1d(dec)]).T, 0)[0]
        
        # IFU Position
        if where in ["ifu"]:
            return self.radec_to_ifu(ra, dec)
        
        raise ValueError("Cannot parse the input `where` (%s) could be: 'ref'/'guider'/'ifu'"%where)
        
    def radec_to_ifu(self, ra, dec):
        """ """
        scaling = np.asarray([-self.parameters["scale"]/np.cos(self.ifucentroid_radec[1]*np.pi/180), 
                    self.parameters["scale"]])
                
        offset_spaxels = (np.asarray([ra,dec]).T-self.ifucentroid_radec)*units.deg.to("arcsec") / scaling
        theta = self.parameters["rotation"] * np.pi/180
        rot = np.asarray([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        return (np.dot(rot, offset_spaxels.T).T +  self.ifu_offset).T
    
    def ifu_to_radec(self, spaxelx, spaxely):
        """ """
        scaling = np.asarray([-self.parameters["scale"]/np.cos(self.ifucentroid_radec[1]*np.pi/180), 
                             self.parameters["scale"]])
        
        theta = - self.parameters["rotation"] * np.pi/180
        rot = np.asarray([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        offset_spaxels = np.dot(rot, (np.asarray([spaxelx, spaxely]).T- self.ifu_offset).T) 
        return ((offset_spaxels.T* scaling) / units.deg.to("arcsec") + self.ifucentroid_radec).T

    # -------- #
    # PLOTTER  #
    # -------- #
    def show_guider_wcs(self, **kwargs):
        """ """
        return self.guider.show(**{**dict(show_catalogue=True), **kwargs})

    # ================ #
    #  Properties      #
    # ================ #
    @property
    def astromfile(self):
        """ """
        if not hasattr(self,"_astromfile"):
            try:
                self._astromfile = io.filename_to_guider(self.filename)[0]
            except:
                raise IOError("No ")
        return self._astromfile
    
    @property
    def wcs_guider(self):
        """ """
        if not hasattr(self, "_wcsguider"):
            header = fits.getheader(self.astromfile)
            self._wcsguider = wcs.WCS( header )
            del header
        return self._wcsguider
    
    @property
    def guider(self):
        """ """
        if not hasattr(self,"_guider"):
            self.load_guider()
        return self._guider

    @property
    def target_radec(self):
        """ RA and Dec of the target (as stored in the hearder)"""
        if not hasattr(self, "_target_radec"):
            header = fits.getheader(self.filename)
            _scoord =  coordinates.SkyCoord(header.get("OBJRA"),header.get("OBJDEC"), unit=(units.hour, units.deg))
            del header
            self._target_radec = _scoord.ra.deg, _scoord.dec.deg
            
        return self._target_radec
    
    @property
    def guider_offset(self):
        """ Offset of the guider ccd pixel corresponding to the IFU centroid (spaxel 0,0) """
        if not hasattr(self, "_guider_offset"):
            self.set_guider_offset(0,0)
        return self._guider_offset

    @property
    def ifu_offset(self):
        """ Offset of the ifu centroid (spaxel 0,0 -> ifu_offset) """
        if not hasattr(self, "_ifu_offset"):
            self.set_ifu_offset(0,0)
        return self._ifu_offset

    @property
    def ifucentroid_radec(self):
        """ """
        return self.wcs_guider.all_pix2world([self.parameters["position"] + self.guider_offset], 0)[0]

    
class IFUReference(Astrometry):
    
    def __init__(self, filename=None, isodate=None, load_refimage=True, **kwargs):
        """ """
        _ = super().__init__(filename=filename, isodate=isodate)
        
        self.load_slice([7000,7500])
        
        if load_refimage:
            try:
                self.load_refimage()
                self.reload_refimage_sep(0.1)
            except:
                warnings.warn("Cannot load the reference image")

    @classmethod
    def from_astrometry(cls, astrometry, load_refimage=True):
        """ """
        this = cls(astrometry.filename, load_refimage=False)
        if hasattr(astrometry,"_guider"):
            this._astromfile = astrometry._astromfile
            this._guider = astrometry._guider
        if hasattr(astrometry,"slice"):
            this.slice = astrometry.slice
        if hasattr(astrometry,"cube"):
            this.cube = astrometry.cube
            
        # offset if any
        this.set_ifu_offset(*astrometry.ifu_offset)
        this.set_guider_offset(*astrometry.guider_offset)
        if load_refimage:    
            this.load_refimage()

        return this
        
        
    # ================ #
    #  Methods         #
    # ================ #
    # ------- #
    #  SETTER #
    # ------- #
    # ------- #
    # LOADER  #
    # ------- #
    def load_refimage(self):
        """ """
        from photoifu import photoref
        self.photoref = photoref.PhotoReference.from_coords( self.target_radec )
        self._sources_xy = self.refimage.sepobjects.get(["x","y"]).T
        
    def reload_refimage_sep(self, thresh=1, **kwargs):
        """ """
        self.refimage.sep_extract(thresh)
        self._sources_xy = self.refimage.sepobjects.get(["x","y"]).T
    
    # ------- #
    # GETTER  #
    # ------- #
    def get_iso_contours(self, where="ref", targetmag=None, isomag=None):
        """ """
        if targetmag is not None:
            iref.photoref.build_pointsource_image(targetmag)
        if isomag is None:
            isomag = np.linspace(20,26,13)
            
        contours_ref = self.photoref.get_iso_contours(isomag, asdict=True)
        if where in ["reference","ref","refimage","refccd","refxy"]:
            return contours_ref
        return {iso: [self.reference_to(where, *cc.T).T for cc in conts] for iso, conts in contours_ref.items()}
        
    def get_refimage_sources_coordinates(self, where="ref", inifu=False):
        """ """
        xpos, ypos = self._sources_xy
        if where in ["ref"]:
            x_,y_ = np.asarray([xpos, ypos])
        else:
            ra,dec = self.refimage.pixel_to_coords(xpos, ypos).T
            if where in ["world","coords","radec"]:
                x_,y_ = np.asarray([ra,dec])
            else:
                x_,y_ = np.asarray(self.radec_to(where, ra, dec))
                
        if inifu:
            flagin = self.slice.contains(*self.get_refimage_sources_coordinates("ifu", inifu=False))
            return np.asarray([x_[flagin],y_[flagin]])
        
        return np.asarray([x_,y_])

    def get_fakeslice(self, mag, seeing=None, refseeing=1, slicearcsec=0.55):
        """ """
        self.photoref.build_pointsource_image(mag, refseeing/self.photoref.refimage.pixel_size_arcsec.value)
        if seeing is not None:
            addseeing = np.sqrt(seeing**2 - refseeing**2)
        else:
            addseeing = None
        return self.photoref.project_to_slice(self.slice, slicearcsec, self.get_target_coordinate("ifu"), 
                                              gaussianconvol = addseeing / self.photoref.refimage.pixel_size_arcsec.value)

    def get_slicemodeller(self, mag, seeing=1.1, **kwargs):
        """ """
        from photoifu import slice
        smodel = slice.SliceModeller(self.slice, self.photoref)
        smodel.set_target(self.get_target_coordinate(), mag)
        smodel.load_modelslice(seeing=seeing, **kwargs)
        smodel.load_gridslice(True)
        smodel.load_gridslice(False)
        return smodel
        
    # ------- #
    # PLOTTER #
    # ------- #    
    def show(self, show_title=True, ifuvmax="99", ifuvmin="5"):
        """ """
        import matplotlib.pyplot as mpl
        from matplotlib.patches import Polygon
        fig = mpl.figure(figsize=[6,3])
        axphoto = fig.add_axes([0.08,0.135,0.4,0.7])
        axifu   = fig.add_axes([0.55,0.135,0.35,0.7])
        
        self.show_image(ax=axphoto)
        self.show_slice(ax=axifu, show_colorbar=False, vmax=ifuvmax, vmin=ifuvmin)
        
        xy_edges = self.radec_to("ref", *self.ifu_to_radec(*np.asarray( self.slice.convexhull_vertices)) )
        axphoto.add_patch( Polygon(xy_edges, facecolor="None", edgecolor=mpl.cm.viridis(0), linewidth=2))
        axifu.set_yticks([])
        axifu.set_xticks([])
        
        if show_title:
            title = "%s"%self.slice.header["OBJECT"].split()[0]+" | "+ \
                    "RA: %.4f Dec: %.4f"%(self.target_radec)   +" | "+ \
                    "airmass: %.2f"%self.slice.header["AIRMASS"] +"\n"+ \
                    "AZ %.2f"%self.slice.header["TEL_AZ"] +" | "+ \
                    "EL %.1f"%self.slice.header["TEL_EL"] +" | "+ \
                    "parangle %.1f"%self.slice.header["TEL_PA"]
            
            _ = fig.text(0.5,0.975,title , va="top", ha="center", color="0.7")
            
        return fig
        
    def _show_sources_(self, ax, which, **kwargs):
        """ """
        flagin = self.slice.contains(*self.get_refimage_sources_coordinates("ifu")) 
        sources = self.get_refimage_sources_coordinates(which)
        ax.plot(*sources.T[flagin].T, color="k", **self._marker_prop)
        ax.plot(*sources.T[~flagin].T, color="k", alpha=0.3, **self._marker_prop)
        ax.plot(*self.get_target_coordinate(which), color="C1", **self._marker_prop)
        
    def show_image(self, ax=None, **kwargs):
        """ """
        _ = self.refimage.show(ax=ax, show_target=False, **kwargs)
        if ax is None:
            ax = _["ax"]

        self._show_sources_(ax, "ref", **kwargs)
        
    def show_slice(self, ax=None, **kwargs):
        """ """
        fig = self.slice.show(ax=ax, **{**dict(zorder=4), **kwargs})
        if ax is None:
            ax = fig.axes[0]
            
        self._show_sources_(ax, "ifu", **kwargs)

    def show_segmap(self, **kwargs):
        """ """
        return  self.photoref.show(**kwargs)
    
    @property
    def _marker_prop(self):
        """ """
        return dict( ls="None", ms=8, mew=1.5, marker="x", scalex=False,scaley=False, zorder=5)
    
    # ------- #
    # CONVERT #
    # ------- #
    def reference_to(self, where, x, y):
        """ Generic convertion tool to go from pixels in the  reference images 
        to anywhere.

        Parameters
        ----------
        where: [string]
        
            The destination, it could be:
            - ref/refimage/refxy: reference image ccd (pixels)
            - guider/astrom: guider image ccd (pixels)
            - coords/world/radec: RA and Dec
            - ifu: IFU (spaxels)

        """
        if where in ["reference","ref","refimage","refccd","refxy"]:
            return np.asarray([x,y])
        
        ra, dec = self.refimage.pixel_to_coords(x, y).T
        if where in ["coords", "radec", "world", "sky"]:
            return ra, dec
        
        return self.radec_to(where, ra, dec)
    
    def radec_to(self, where, ra, dec):
        """ 
        
        Parameters
        ----------
        where: [string]
        
            The destination, it could be:
            - ref/refimage/refxy: reference image ccd (pixels)
            - guider/astrom: guider image ccd (pixels)
            - ifu: IFU (spaxels)
            
        ra, dec: [float, float or list of]
            Input Coordinates 
            
        Returns
        -------
        same format of ra, dec
        """
        # CCD Positions (reference image or guider)
        if where in ["reference","ref","refimage","refccd","refxy"]:
            return self.refimage.coords_to_pixel(ra, dec)

        return super().radec_to( where, ra, dec)
        
    def reference_to_ifu(self, x,y):
        """ """
        return self.reference_to("ifu", x,y)
    
    def ifu_to_reference(self, spaxelx, spaxely):
        """ """
        return self.ifu_to("ref", spaxelx, spaxely)
    
    # ================ #
    #  Properties      #
    # ================ #    
    @property
    def refimage(self):
        """ Reference image """
        return self.photoref.refimage
    

    @property
    def refcatdata(self):
        """ """
        return self.photoref.catdata


class SliceAligner():
    """ """
    FREEPARAMETERS = ["theta", "offsetx","offsety"]
    def __init__(self, slice_, source_position):
        """ """
        print("*** SliceAligner is DEPRECATED ***")
        self.slice = slice_
        self.set_sourcepos(*source_position)

    # =========== #
    #  Methods    #
    # =========== #
    def _data_to_slice_(self, data):
        """ """
        return spectroscopy.get_slice(data, 
                                      self.slice.xy.T, 
                                      spaxel_vertices=self.slice.spaxel_vertices, 
                                      indexes=self.slice.indexes, 
                                      lbda=self.slice.lbda )        
    def load_slicegrid(self, size=None):
        """ """
        self._grid = self.get_slicegrid(size=size)
        
    def load_slicesources(self, usepeak=False, **kwargs):
        """ """

        if not hasattr(self,"dfsep"):
            self.extract_slicesource(**kwargs)
            
        keys = ["x","y"] if not usepeak else ["xpeak", "ypeak"]
        self._slicesources = self.dfsep[keys].values.T
        self._slicesources_sky = coordinates.SkyCoord(*self._slicesources, unit=units.arcsec)
        self._ellipses = {"x":self.dfsep[keys[0]].values,"y":self.dfsep[keys[1]].values,
                          "a":self.dfsep["a"].values,"b":self.dfsep["b"].values,
                          "theta":self.dfsep["theta"].values}
        
    def extract_slicesource(self, thresh=1e-4, **kwargs):
        """ 
        Properties
        ----------
        thresh:

        kwargs goes to sep.extract(thresh=thresh, **kwargs)
            # Default (by sep.)
            mask=None, minarea=5,
            filter_kernel=default_kernel, filter_type='matched',
            deblend_nthresh=32, deblend_cont=0.005 (customed), clean=True,
            clean_param=1.0, segmentation_map=False
            maskthresh : float, optional

            # Customisation 
            - deblend_cont=1e-4

            # documentation
            - minarea : int, optional
                Minimum number of pixels required for an object. Default is 5.
            - filter_kernel : `~numpy.ndarray` or None, optional
                Filter kernel used for on-the-fly filtering (used to
                enhance detection). Default is a 3x3 array:
                [[1,2,1], [2,4,2], [1,2,1]]. Set to ``None`` to skip
                convolution.
            - filter_type : {'matched', 'conv'}, optional
                Filter treatment. This affects filtering behavior when a noise
                array is supplied. ``'matched'`` (default) accounts for
                pixel-to-pixel noise in the filter kernel. ``'conv'`` is
                simple convolution of the data array, ignoring pixel-to-pixel
                noise across the kernel.  ``'matched'`` should yield better
                detection of faint sources in areas of rapidly varying noise
                (such as found in coadded images made from semi-overlapping
                exposures).  The two options are equivalent when noise is
                constant.
            - deblend_nthresh : int, optional
                Number of thresholds used for object deblending. Default is 32.
            - deblend_cont : float, optional
                Minimum contrast ratio used for object deblending.
                To entirely disable deblending, set to 1.0.
            - clean : bool, optional
                Perform cleaning? Default is True.
            - clean_param : float, optional
                Cleaning parameter (see SExtractor manual). Default is 1.0.
            - segmentation_map : bool, optional
                If True, also return a "segmentation map" giving the member
                pixels of each object. Default is False.
        """
        import sep
        import pandas
        flagin = self.slice.contains(*self.grid.geodataframe[["x","y"]].values.T, 
                                     buffer=-2)
    
        data = self.grid.geodataframe["data"].values.copy()
        # set nans to things outside the slice
        data[~flagin] = np.NaN
        # Normalize everything
        data = (data - np.percentile(data[data==data],1)) / (np.percentile(data[data==data],99) - np.percentile(data[data==data],1))
        # reshape
        self.imgdata = data.reshape(self.gridsize*2,self.gridsize*2)
        # run sep.extract and convert results in pandas
        extract_prop = {**{"deblend_cont":1e-4},**kwargs}
        self.dfsep = pandas.DataFrame( sep.extract(self.imgdata, thresh, **extract_prop) )
        # Shift the grid to be aligned with slice coordinates
        keys = ["xmin","xmax","ymin","ymax","x","y","xpeak","ypeak"]
        self.dfsep[keys] = self.dfsep[keys].apply(lambda x:x-self.gridsize )
        return self.dfsep
    
    # ------- #
    # FITTER  #
    # ------- #
    # - Distance
    def get_source_matching(self):
        """ """
        return self.slicesources_skycoord.match_to_catalog_sky(
            coordinates.SkyCoord(*self.source_effposition, unit=units.arcsec)
        )
    
    # - Initialisation
    def get_initial_guess(self, asarray=False, **kwargs):
        """ """
        dict_ = {**{t:0 for t in self.freeparameters}, **{"theta":0, "offsetx":0, "offsety":0}, **kwargs}
        if not asarray:
            return dict_
        return [dict_[t] for t in self.freeparameters]
    
    def get_boundaries(self, asarray=False,**kwargs):
        """ """
        dict_  = {**{t:[None,None] for t in self.freeparameters},
                **{"theta":[-np.pi,np.pi], "offsetx":[-5,5], "offsety":[-3,2]},**kwargs}
        if not asarray:
            return dict_
        return [dict_[t] for t in self.freeparameters]
        
    def get_fiterror(self, asarray=False,**kwargs):
        """ """
        dict_  = {**{t:0.1 for t in self.freeparameters},**kwargs}
        if not asarray:
            return dict_
        return [dict_[t] for t in self.freeparameters]
    
    def get_fixed(self, asarray=False,**kwargs):
        """ """
        dict_ =  {**{t:False for t in self.freeparameters},**{"theta":True},**kwargs}
        if not asarray:
            return dict_
        return [dict_[t] for t in self.freeparameters]
        
    # - Prior
    def get_logpriors(self, parameters=None):
        """
        product of all the single parameter priors
        """
        if parameters is not None:
            self.set_parameters(parameters)

        # List of priors
        return 0

    # - Likelihood
    def get_loglikelihood(self, parameters=None):
        """
        returns likelihood (L), such that
        chi2 = -2*np.sum(np.log(L))
        """
        if parameters is not None:
            self.set_parameters(param)

        distance = self.get_source_matching()[1].arcsec
        loglikelihood = stats.norm.logpdf(distance,
                                     loc=0,
                                     scale= 1,
                                     )
        
        return loglikelihood
    
    def get_chi2(self, parameters=None):
        """ """
        if parameters is not None:
            self.set_parameters(parameters)

        chi2 = -2 * np.sum( self.get_loglikelihood())
        return chi2

    def get_priored_chi2(self, parameters=None):
        """ """
        if parameters is not None:
            self.set_parameters(parameters)

        return self.get_chi2() - 2*np.sum( self.get_logpriors())
    
    
    def fit(self, use_prior=True, fixed_prop={}, guess_prop={}, boundaries_prop={}, error_prop={}):
        """
        """
        # 1. Setup iminuit
        # 2. call minuit
        # 3. return and store value

        # need to be updated!!!
        #limit_lbda0_ = (6563, self.parameters["lbda0"]+2*self.parameters["width"])
        guesses  = self.get_initial_guess(asarray=True, **guess_prop)
        boundaries = self.get_boundaries(asarray=True, **boundaries_prop)
        fiterr = self.get_fiterror(asarray=True, **error_prop)
        fixed = self.get_fixed(asarray=True, **fixed_prop)

        self._function_to_minimize = self.get_priored_chi2 if use_prior else self.get_chi2
        self._minuit = iminuit.Minuit.from_array_func(self._function_to_minimize,
                                           guesses,
                                           error=fiterr,
                                           limit=boundaries,
                                           fix=fixed,
                                           name=self.freeparameters,
                                           errordef=1)
        self.prior_used = use_prior
        # Run the minimisation
        self._minuit.migrad()

        self.fitvalues = self._minuit.values # dict(param:bestvalue, param.err:errorvalue)
        return self.fitvalues

    
    # ------- #
    # SETTER  #
    # ------- #    
    def set_sourcepos(self, sourcex, sourcey):
        """ """
        self.sourcepos = np.asarray([sourcex, sourcey])
        
    def set_parameters(self, parameters):
        """ """
        self.parameters = {k:v for k,v in zip(self.freeparameters,parameters)}
        
    def set_initial_guess(self, **kwargs):
        """ """
        guesses = self.get_initial_guess(**kwargs)
        self.set_parameters([guesses[k] for k in self.freeparameters])
        
    # ------- #
    # GETTER  #
    # ------- #
    def get_effective_position(self, ifux, ifuy):
        """ """
        return np.asarray([ifux+self.parameters["offsetx"], ifuy+self.parameters["offsety"]])
        
    def get_slicegrid(self, size=None):
        """ """
        from pixelproject import grid
        if size is None:
            self.gridsize = int(np.max(np.abs(self.slice.convexhull_vertices)))+1
        else:
            self.gridsize = size
        
        pixels = np.mgrid[0:self.gridsize*2,0:self.gridsize*2]-self.gridsize
        pixels_flat = np.concatenate(pixels.T, axis=0)

        gg = grid.Grid(pixels_flat)
        flagnan = np.asarray(np.sum(np.isnan(self.xy), axis=0), dtype="bool")
        sg = grid.Grid(self.xy.T[~flagnan], self.slice.spaxel_vertices)
        sg.add_data(self.slice.data[~flagnan], "data")
        if self.slice.has_variance():
            sg.add_data(self.slice.variance[~flagnan], "variance")
        return sg.project_to(gg)
    
    # -------- #
    # PLOTTER  #
    # -------- #
    def show_extraction(self, ax=None, ellipses=True, sourcecolor="C1", **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[6,5])
            ax  = fig.add_axes([0.15,0.15, 0.75,0.75])
        else:
            fig = ax.figure

        ax.imshow(self.imgdata, origin="lower", extent=[-self.gridsize,self.gridsize,
                                                        -self.gridsize, self.gridsize], **kwargs)
        
        ax.scatter(*self.slicesources, marker="x", color=sourcecolor)
        if ellipses:
            from matplotlib.patches import Ellipse
            for i,pf in pandas.DataFrame(self._ellipses).iterrows():
                ax.add_patch(Ellipse([pf.x,pf.y], pf.a*2, pf.b*2,
                                    angle=pf.theta*180/np.pi, facecolor="None",
                                    edgecolor=sourcecolor)
                            )
                
        return fig
            
    def show(self):
        """ """
        fig = self.slice.show(vmax="98", zorder=3)
        ax = fig.axes[0]
        ax.scatter(*self.slicesources, color="C1", marker="x", zorder=4, 
                   label="detected")
        ax.scatter(*self.sourcepos, color="k", marker=".", zorder=5, 
                   label="expected")
        ax.scatter(*self.source_effposition, color="k", marker="+", zorder=5, 
                   label="fitted")

        ax.text(0,1.02,"Sources:", va="bottom", ha="left", transform=ax.transAxes)
        ax.legend(loc=[0.15,1.01], ncol=3, frameon=False)
        return fig
    
    # =========== #
    #  Properties #
    # =========== #           
    @property
    def nsources(self):
        """ """
        return len(self.sourcepos.T)
    
    @property
    def source_effposition(self):
        """ """
        return self.get_effective_position(*self.sourcepos)
    
    @property
    def freeparameters(self):
        """ """
        if not hasattr(self, "_freeparameters"):
            self._freeparameters = self.FREEPARAMETERS.copy()
        return self._freeparameters
    
    # - from Slice
    @property
    def normalization(self):
        """ """
        if not hasattr(self,"_normalization"):
            self._normalization = np.nanmean(self.slice.data)
        return self._normalization
    
    @property
    def data(self):
        """ data of the input slice"""
        if not hasattr(self,"_data"):
            self._data = self.slice.data/self.normalization
        return self._data

    @property
    def variance(self):
        """ variance of the slice's data"""
        if not hasattr(self,"_variance"):
            self._variance = self.slice.variance/self.normalization**2
        return self._variance
    
    @property
    def xy(self):
        """ position of the slice's spaxels """
        return self.slice.xy
        
    # - Slice Source
    @property
    def slicesources(self):
        """ """
        if not hasattr(self,"_slicesource"):
            self.load_slicesources()
        return self._slicesources
    
    @property
    def slicesources_skycoord(self):
        """ """
        if not hasattr(self,"_slicesource_sky"):
            self.load_slicesources()
        return self._slicesources_sky
    
    # - Grid
    @property
    def grid(self):
        """ """
        if not hasattr(self, "_grid"):
            self.load_slicegrid()
        return self._grid
    
        
class AstrometryFitter():
    def __init__(self, cube_filenames):
        """ """
        print("*** AstrometryFitter will soon be DEPRECATED ***")
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
    



class WCSIFU():
    """ """

    def __init__(self, date=None):
        """ """
        print("*** WCSIFU will soon be DEPRECATED ***")
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
    print("***   get_astrometry_fitter IS DEPRECATED   ***")
    from ztfquery import sedm as zsedm
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
    p = zsedm.SEDMQuery()
    if download_data:
        _ = [p.download_target_data(name, which="cube") for name in NON_STDTARGET] 
        _ = [p.download_target_data(name, which="astrom") for name in NON_STDTARGET]

    
    target_files = p.get_local_data(NON_STDTARGET,  timerange=timerange)
    return AstrometryFitter(target_files)
