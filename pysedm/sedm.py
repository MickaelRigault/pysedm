#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module containing the part that are directly SEDM oriented. """

import numpy              as np
import warnings

from pyifu.spectroscopy   import Cube, Spectrum
from .utils.tools         import kwargs_update

from .io import PROD_CUBEROOT

__all__ = ["get_sedmcube", "build_sedmcube","kpy_to_e3d"]

# PROD_CUBEROOT e3d
# --- DB Structure
CALIBFILES = ["Hg.fits","Cd.fits","Xe.fits","dome.fits"]

# --- CCD
SEDM_CCD_SIZE = [2048, 2048]
DOME_TRACEBOUNDS = [70,220]
TRACE_DISPERSION = 1.3 # PSF (sigma assuming gaussian) of the traces on the CCD. 
SEDMSPAXELS = np.asarray([[ np.sqrt(3.)/2., 1./2],[0, 1],[-np.sqrt(3.)/2., 1./2],
                          [-np.sqrt(3.)/2.,-1./2],[0,-1],[ np.sqrt(3.)/2.,-1./2]])*2/3.

_EDGES_Y = 20
_EDGES_X = 100
INDEX_CCD_CONTOURS = [[_EDGES_X,_EDGES_Y],[_EDGES_X,1700],
                      [300,2040-_EDGES_Y],[2040-_EDGES_X,2040-_EDGES_Y],
                        [2040-_EDGES_X,_EDGES_Y]]
# --- LBDA
SEDM_LBDA = np.linspace(3700, 9200, 260)

# --- ADR
MLA_ROTATION_RAD= (263) * np.pi / 180.  # degree -> to rad
MLA_ROTMATRIX   = np.asarray([[ np.cos(MLA_ROTATION_RAD),-np.sin(MLA_ROTATION_RAD)], 
                              [ np.sin(MLA_ROTATION_RAD), np.cos(MLA_ROTATION_RAD)]] )
DEFAULT_REFLBDA = 6000 # In Angstrom
IFU_SCALE_UNIT  = 0.43


# --- Palomar Atmosphere
# Palomar Extinction Data from Hayes & Latham 1975
# (Wavelength in Angstroms, Magnitudes per airmass)
PALOMAR_EXTINCTION = np.asarray([ (3200, 1.058),
 (3250, 0.911), (3300, 0.826), (3350, 0.757), (3390, 0.719), (3448, 0.663), (3509, 0.617),
 (3571, 0.575), (3636, 0.537), (3704, 0.500), (3862, 0.428), (4036, 0.364), (4167, 0.325),
 (4255, 0.302), (4464, 0.256), (4566, 0.238), (4785, 0.206), (5000, 0.183), (5263, 0.164),
 (5556, 0.151), (5840, 0.140), (6055, 0.133), (6435, 0.104), (6790, 0.084), (7100, 0.071),
 (7550, 0.061), (7780, 0.055), (8090, 0.051), (8370, 0.048), (8708, 0.044), (9832, 0.036),
 (10255, 0.034), (10610, 0.032), (10795, 0.032), (10870, 0.031)])

def get_palomar_extinction():
    """ The ExtinctionSpectrum object of the Palomar Extinction 
    To correct for atmosphere extinction, see the get_atm_extinction() method

    Return 
    ------
    ExtinctionSpectrum
    """
    from .utils.atmosphere import ExtinctionSpectrum
    spec = ExtinctionSpectrum(None)
    spec.create(lbda=PALOMAR_EXTINCTION.T[0],data=PALOMAR_EXTINCTION.T[1],
                                variance=None, header=None)
    spec._source = "Hayes & Latham 1975"
    return spec


# ------------------ #
#  Builder           #
# ------------------ #
def build_sedmcube(ccd, date, lbda=None, flatfield=None,
                   wavesolution=None, hexagrid=None,
                   flatfielded=True, atmcorrected=True,
                   calibration_ref=None,
                   build_calibrated_cube=False):
    """ Build a cube from the an IFU ccd image. This image 
    should be bias corrected and cosmic ray corrected.

    The standard created cube will be 3Dflat field correct 
    (see flatfielded) and corrected for atmosphere extinction 
    (see atmcorrected). A second cube will be flux calibrated 
    if possible (see build_calibrated_cube).

    Parameters
    ----------
    ccd: [CCD]
        A ccd object from which the cube will be extracted.
        
    date: [string]
        date in usual YYYYMMDD format
    
    lbda: [None/array] -optional-
        wavelength array used to build the cube. 
        If not given the default sedm wavelength range will be used. 
        See pysedm.sedm.SEDM_LBDA.

    // Cube Calibrator //
    
    wavesolution: [WaveSolution] -optional-
        The wavelength solution containing the pixel<->wavelength conversion.
        If None, this will be loaded using `date`.

    hexagrid: [HexaGrid] -optional-
        The Hexagonal Grid tools containing the index<->qr<->xy conversions.
        If None, this will be loaded using `date`.

    flatfield: [Slice] -optional-
        Object containing the relative transmission of the IFU spaxels.
        If None, this will be loaded using `date`.

    // Action Selection //
    
    flatfielded: [bool] -optional-
        Shall the cube be flatfielded?
        - This information will be saved in the header-
    
    atmcorrected: [bool] -optional-
        Shall the cube the corrected for atmosphere extinction?
        - This information will be saved in the header-

    // Additional outcome: Flux calibrated cube //

    build_calibrated_cube: [bool] -optional-
        Shall this method build an additionnal flux calibrated cube?
        
    calibration_ref: [None/string] -optional-
        If you want to build a calibrated cube, you can provide the filename
        of the spectrum containing the inverse-sensitivity (fluxcal*)
        If None, this will load the latest fluxcal object of the night.
        If Nothing found, no flux calibrated cube will be created. 

    Returns
    -------
    Void
    """
    from . import io
    # - INPUT [optional]
    if hexagrid is None:
        hexagrid     = io.load_nightly_hexagonalgrid(date)
    
    if wavesolution is None:
        wavesolution = io.load_nightly_wavesolution(date)
        wavesolution._load_full_solutions_()
    
    if lbda is None:
        lbda = SEDM_LBDA

    if flatfielded and flatfield is None:
        flatfield = io.load_nightly_flat(date)
        
    # - Build the Cube
    cube = ccd.extract_cube(wavesolution, lbda, hexagrid=hexagrid, show_progress=True)

    # - passing the header inforation
    for k,v in ccd.header.items():
        if k not in cube.header:
            cube.header[k] = v

    cube.header['ORIGIN'] = (ccd.filename.split('/')[-1], "CCD filename used to build the cube")
    
    # - Flat Field the cube
    if flatfielded:
        cube.scale_by(flatfield.data)
        cube.header['FLAT3D'] = (True, "Is the Cube FlatFielded")
        cube.header['FLATSRC'] = (flatfield.filename.split('/')[-1], "Object use to FlatField the cube")
    else:
        cube.header['FLAT3D'] = (False, "Is the Cube FlatFielded")
        
    # - Amtphore correction
    if atmcorrected:
        atmspec = get_palomar_extinction()
        extinction = atmspec.get_atm_extinction(cube.lbda, cube.header['AIRMASS'])
        # scale_by devided by
        cube.scale_by(1./extinction)
        cube.header['ATMCORR']  = (True, "Has the Atmosphere extinction been corrected?")
        cube.header['ATMSRC']   = (atmspec._source if hasattr(atmspec,"_source") else "unknown", "Reference of the atmosphere extinction")
        cube.header['ATMSCALE'] = (np.nanmean(extinction), "Mean atm correction over the entire wavelength range")
    else:
        cube.header['ATMCORR']  = (False, "Has the Atmosphere extinction been corrected?")

        
    # - Saving it
    
    if np.any([calibkey_ in ccd.filename for calibkey_ in CALIBFILES]):
        filout = "%s"%(ccd.filename.split("/")[-1].split(".fits")[0])
    else:
        filout = "%s_%s"%(ccd.filename.split("/")[-1].split(".fits")[0], ccd.objname)
        
    fileout =    io.get_datapath(date)+"%s_%s.fits"%(PROD_CUBEROOT,filout) 
    cube.writeto(fileout)

    # - Build Also a flux calibrated cube?
    if build_calibrated_cube:
        build_calibrated_sedmcube(fileout, date=date, calibration_ref=calibration_ref)
        
def build_calibrated_sedmcube(cubefile, date=None, calibration_ref=None, kind=None):
    """ """
    import io
    if calibration_ref is None:
        calfiles = io.get_night_files(date, "spec.fluxcal")
        if len(calfiles)==0:
            warnings.warn("No `fluxcalfiles` for date %s. No calibrated cube created"%date)
            return
        calibration_ref = calfiles[-1]
        if kind is None: kind="defcal"
        print("using %s as a flux calibration reference"%calibration_ref.split("/")[-1] )
        
    else:
        if kind is None: kind="cal"
        
    # - Inverse Sensitivity
    spec = Spectrum(calibration_ref)
    # - Load it
    cube = get_sedmcube(cubefile)
    # - Do it
    cube.scale_by(1./spec.data)
    cube.header['SOURCE']  = (fileout.split('/')[-1] , "the original cube")
    cube.header['FCALSRC'] = (calibration_ref.split('/')[-1], "the calibration source reference")
    cube.header['PYSEDMT'] = ("Flux Calibrated Cube")
    # - Save it
    cube.writeto(cubefile.replace("%s"%PROD_CUBEROOT,"%s_%s"%(PROD_CUBEROOT,kind)))
    
# ------------------ #
#  Main Functions    #
# ------------------ #        
def get_sedmcube(filename, **kwargs):
    """ Load a Cube from the given filename 
    
    Returns
    -------
    Cube
    """
    return SEDMCube(filename, **kwargs)

def get_aperturespectrum(filename):
    """ Load and return an Aperture Spectrum """
    spec = ApertureSpectrum(None,None)
    spec.load(filename)
    return spec

def kpy_to_e3d(filename, lbda, savefile=None):
    """ Converts SEDmachine kpy .npy data into pyifu e3d cubes. 
    (rotation information missing)
    
    Parameters
    ----------
    filename: [string]
        .npy data created using the kpy software

    lbda: [array]
        wavelength array of the cube.

    savefile: [string/None] -optional-
        if you want to save the cube, provide here its fullpath
    
    Returns
    -------
    pyifu Cube 
    """
    from scipy.interpolate import interp1d
    # 
    #  get the data
    data, header = np.load(filename)
    # 
    # empty the cube
    cube = Cube(None)
    cubeflux     = []
    spaxel_map   = {}
    for i, ext in enumerate(data):
        try:
            lbda_nm, flux_ = ext.get_flambda("spec")
            cubeflux.append(interp1d(lbda_nm*10, flux_, kind="cubic")(lbda))
            spaxel_map[i] = np.asarray([ext.X_as,ext.Y_as]*np.sqrt(3))
        except:
            # will crash if given lbda is outside the Extraction wavelength
            continue
        
    # create it
    cube.create(np.asarray(cubeflux).T,lbda=lbda, spaxel_mapping=spaxel_map, variance=None)
    cube.set_spaxel_vertices(SEDMSPAXELS/ np.sqrt(3))
    for k,v in header["header"].items():
        cube.header[k] = v

    # returns it.
    if savefile is not None:
        cube.writeto(savefile)
        
    return cube


# --------------- #
#   PLOTTER       #
# --------------- #
def display_on_hexagrid(value, traceindexes,
                        hexagrid=None, xy=None, 
                        outlier_highlight=None,
                        ax = None, savefile=None, show=True,
                        vmin=None, vmax=None, show_colorbar=True,
                        clabel="",cfontsize="large",
                            **kwargs):
    """ display on the IFU hexagonal grid the given values

    Parameters
    ----------        
    value: [list]
        value defining the color of the hexagones
    
    traceindexes: [list]
        indexes corresponding to the values

    hexagrid: [HexaGrid] -optional if xy given-
        object containing the traceindex<->qr<->xy transformation
    
    xy: [array, array] -optional if hexagrid given-
        x,y, position of the spaxels in the MLA.

    outlier_highlight: [None / positive-float] -optional-
        if a value is `outlier_highlight` sigma away from the core of the
        distribution, it will be highlighted.
        if None, nothing will be done.

    """
    from matplotlib                import patches
    import matplotlib.pyplot           as mpl
    from astrobject.utils.mpladdon import colorbar, insert_ax
    from pyifu.tools import figout

    # - Check input
    if len(value) != len(traceindexes):
        raise ValueError("value and traceindexes do not have the same size (%d vs. %s)"%(len(value),len(traceindexes)))
    else:
        nspaxes = len(value)
        value        = np.asarray(value)

    traceindexes = np.asarray(traceindexes)
    
    # -- Let's go
    if ax is None:
        fig = mpl.figure(figsize=[6,5])
        axim  = fig.add_subplot(111)
    else:
        fig = ax.figure

    
    # - which colors
    if vmin is None:
        vmin = np.percentile(value,0)
    elif type(vmin) == str:
        vmin = np.percentile(value,float(vmin))
    if vmax is None:
        vmax = np.percentile(value,100)
    elif type(vmax) == str:
        vmax = np.percentile(value,float(vmax))
                
    colors = mpl.cm.viridis((value-vmin)/(vmax-vmin))
    # - where
    if xy is None:
        x,y = np.asarray(hexagrid.index_to_xy(hexagrid.ids_to_index(traceindexes)))
    else:
        x,y = xy
    
    # - The Patchs
    ps = [patches.Polygon(SEDMSPAXELS+np.asarray([x[i],y[i]]),
                            facecolor=colors[i], alpha=0.8,**kwargs)
              for i  in range(nspaxes)]
    ip = [axim.add_patch(p_) for p_ in ps]
    axim.autoscale(True, tight=True)

    # - Outlier highlight
    if outlier_highlight is not None:
        outlier_highlight = np.abs(outlier_highlight)
        flagout = np.asarray((np.abs(value) > np.nanmean(value)+np.nanstd(value)*outlier_highlight), dtype="bool")
        if np.any(flagout):
            color = mpl.cm.inferno(0.5,1.)
            for j in np.argwhere(flagout):
                axim.plot(x[j],y[j], marker="x", mfc=color, mew=0, ms=10, zorder=9)
                axim.text(x[j], y[j], "trace %d: %.3f"%(traceindexes[j], value[j]),
                          rotation=45, ha="left", va="bottom", color=color,
                        bbox=dict(facecolor=mpl.cm.binary(0.1,0.5), edgecolor="k", lw=0.5), zorder=9)
            
    if show_colorbar:
        axcbar = axim.insert_ax("right", shrunk=0.88)
        axcbar.colorbar(mpl.cm.viridis,vmin=vmin,vmax=vmax,label=clabel,
                fontsize=cfontsize)
    
    fig.figout(savefile=savefile, show=show)
    return {"ax":axim,"fig":fig}

#################################
#                               #
#    SEDMachine Cube            #
#                               #
#################################
    
class SEDMCube( Cube ):
    """ SEDM Cube """
    DERIVED_PROPERTIES = ["sky"]

    def get_aperture_spec(self, xref, yref, radius, bkgd_annulus=None,
                              refindex=None, adr=True, **kwargs):
        """ 
        bkgd_annulus: [float, float ] -optional-
            coefficient (in radius) defining the background annulus.
            e.g. if the radius is 5 and bkgd_annulus=[1,1.5], the resulting
            annulus will have an inner radius of 5 and an outter radius of 5*1.5= 7.5
        
        """
        from shapely import geometry
        if adr:
            sourcex, sourcey = self.get_source_position(self.lbda, xref=xref, yref=yref, refindex=refindex)
        else:
            sourcex, sourcey = np.ones( len(self.lbda) )*xref,np.ones( len(self.lbda) )*yref

        # - Radius 
        if not hasattr(radius,"__iter__"):
            radius = np.ones(len(self.lbda))*radius
        elif len(radius)!= len(self.lbda):
            raise TypeError("The radius size must be a constant or have the same lenth as self.lbda")
        
        apert = []
        if bkgd_annulus is not None:
            apert_bkgd = []
            
        for i, x, y, r in zip(range(self.nspaxels), sourcex, sourcey, radius):
            sl_ = self.get_slice(index=i, slice_object=True)
            apert.append(sl_.get_aperture(x,y,r, **kwargs))
            if bkgd_annulus is not None:
                apert_bkgd.append(sl_.get_aperture(x,y,r*bkgd_annulus[1],
                                                radius_min=r*bkgd_annulus[0],
                                                **kwargs))
        apert = np.asarray(apert)
        
        # - Setting the background
        
        spec  = ApertureSpectrum(self.lbda, apert.T[0]/apert.T[2], variance=apert.T[1]/apert.T[2]**2 if self.has_variance() else None,
                                    apweight=apert.T[2], header=None)
        if bkgd_annulus is not None:
            apert_bkgd = np.asarray(apert_bkgd)
            bspec = ApertureSpectrum(self.lbda, apert_bkgd.T[0]/apert_bkgd.T[2], variance=apert_bkgd.T[1]/apert_bkgd.T[2]**2 if self.has_variance() else None,
                                    apweight=apert_bkgd.T[2], header=None)
            spec.set_background(bspec)

        for k,v in self.header.items():
            if np.any([entry in k for entry in ["TEL","RA","DEC","DOME","OBJ","OUT","IN_","TEMP","INST",
                                                    "FLAT","ATM","AIR"]]):
                spec.header[k] = v
        if self.filename is not None:
            spec.header["SOURCE"] = self.filename.split("/")[-1]
            
        return spec
        
    def get_source_position(self, lbda, xref=0, yref=0, refindex=None):
        """ The position in the IFU of a spacial element as a function of wavelength.
        Shift caused by the ADR.
        
        Parameters
        ----------
        lbda: [float/array]
            wavelength in angstrom
        
        xref, yref: [float, float]
            x and y position of the spacial element in the IFU at the reference index wavelength
        
        refindex: [int] -optional-
            index of the wavelength slice used as reference. 
        
        Returns
        -------
        [array, array]
        (x and y positions as a function of lbda)
        """
        if self.adr is None or refindex is not None:
            if refindex is None:
                refindex = np.argmin(np.abs(self.lbda-DEFAULT_REFLBDA))
            self.load_adr(lbdaref=self.lbda[refindex])

        x_default, y_default = self.adr.refract(0, 0, lbda, unit=IFU_SCALE_UNIT)
        x, y = np.dot(MLA_ROTMATRIX,np.asarray([x_default,y_default]))
        
        return x+xref, y+yref
    
    
    def load_adr(self, pressure=630, lbdaref=DEFAULT_REFLBDA,
                     **kwargs):
        """ 
        This method will load the ADR based on data from the header.
        You can overwrite these parameter using kwargs:

        Parameters
        ----------
        pressure: [float] -optional-
            Air pressure in mbar
                
        lbdaref: [float] -optional-
            Reference wavelength, the position shift will be given as a function 
            of this wavelength.

        kwargs parameters:
            airmass: [float]
                Airmass of the target

            parangle: [float]
                Parralactic angle in degree

            temperature: [float]
                temperature in Celcius
            
            relathumidity: [float <100]
                Relative Humidity in %
            
        Returns
        ------- 
        Void (loads the self.adr)
        """
        adr_prop = kwargs_update( dict(pressure=pressure,
                                       lbdaref=lbdaref,
                                       temperature=self.header["IN_AIR"], 
                                       relathumidity=self.header["IN_HUM"], 
                                       airmass=self.header['AIRMASS'], 
                                       parangle=self.header['TEL_PA']),
                                **kwargs)
        return super(SEDMCube, self).load_adr(**adr_prop)
    
    def remove_sky(self, nspaxels=50, usemean=False,
                       estimate_from="rawdata", lbda_range=[5000,8000],
                      **kwargs):
        """ Pick the `nspaxels` spaxels and average them out to build a skyspectrum.
        The flux of this skyspectrum is then removed from the cube.

        Parameters
        ----------
        nspaxels: [int]
            the number of spaxels used to estimate the sky.
            These will be the faintest spaxels. 
            (NB: **kwargs are options of the spaxels selection)

        usemean: [bool] -optional-
            If the variance is available, the weighted (1/variance) average will be used
            to combine spectra except if `usemean` is True. In that case, the simple mean 
            will be used.
        
        estimate_from: [string] -optional-
            Attribute that will be used to estimate the `data` of the sky spectrum
 
        lbda_range: [float, float] -optional-
            Which wavelength range is check.

        **kwargs goes to get_faintest_spaxels(): 
                  e.g: lbda_range, avoid_area, avoid_indexes etc.

        Returns
        -------
        Void (affects `data`)
        """
        self._sky = self.get_spectrum(self.get_faintest_spaxels(nspaxels, lbda_range=lbda_range,**kwargs),
                                          usemean=usemean, data=estimate_from)
        self.remove_flux( self._sky.data)


    # - Improved version allowing to add CCD
    def show(self, toshow="data",
                 interactive=False, ccd=None,
                 savefile=None, ax=None, show=True,
                 show_meanspectrum=True, cmap=None,
                 vmin=None, vmax=None, notebook=None,
                 **kwargs):
        """ Display the cube.
        
        Parameters
        ----------
        toshow: [string] -optional-
            Variable you want to display. anything accessible as self.`toshow` that 
            has the same size as the wavelength. 
            If toshow is data or rawdata (or anything containing 'data'), 
            the variance will automatically be added if it exists.
            Do not change this is you have a doubt.
            
        interactive: [bool] -optional- 
           Enable to interact with the plot to navigate through the cube.
           (this might depend on your matplotlib setup.)

        ccd: [bool] -optional-
           Add the CCD image to the interactive plot to enable direct vizualisation of the Traces on the CCD.

        cmap: [matplotlib colormap] -optional-
            Colormap used for the wavelength integrated cube (imshow).

        vmin, vmax: [float /string / None] -optional-
            Lower and upper value for the colormap
            => If the ccd has been given this will affect the ccd image.
            => If not this will affect the projected cube.
            
            3 Formats are available:
            - float: Value in data unit
            - string: percentile. Give a float (between 0 and 100) in string format.
                      This will be converted in float and passed to numpy.percentile
            - None: The default will be used (percentile 0.5 and 99.5 percent respectively).
            (NB: vmin and vmax are independent, i.e. one can be None and the other '98' for instance)

        show_meanspectrum: [bool] -optional-
            If True both a wavelength integrated cube (imshow) and the average spectrum 
            will be displayed. If not, only the wavelength integrated cube (imshow) will.

        ax: [matplotlib.Axes] -optional-
            Provide the axes where the spectrum and/or the wavelength integrated 
            cube  will be drawn. 
            See show_meanspectrum:
               - If True, 2 axes are requested so axspec, aximshow=ax
               - If False, 1 axes is needed, aximshow=ax 
            If None this will create a new axes inside a new figure
            
        savefile: [string/None] -optional-
            Would you like to save the data? If so give the name of this
            file where the plot will be saved.
            You can provide an extention (.pdf or .png) if you don't both the
            .pdf and .png will be created.

        show: [bool] -optional-
            If you do not save the data (see savefile), shall the plot be shown?
        
        notebook: [bool or None] -optional-
            Is this running from a notebook? 
            If True, the plot will be made using fig.show() if not with mpl.show()
            If None, this will try to guess.


        **kwargs goes to matplotlib's imshow 

        Returns
        -------
        Void
        """
        if not interactive or ccd is None:
            return super(SEDMCube, self).show(toshow=toshow, interactive=interactive,
                                           savefile=savefile, ax=ax, show=show,
                                           show_meanspectrum=show_meanspectrum, cmap=cmap,
                                           vmin=vmin, vmax=vmax, notebook=notebook, **kwargs)
        else:
            from .utils.mpl import InteractiveCubeandCCD
            iplot = InteractiveCubeandCCD(self, fig=None, axes=ax, toshow=toshow)
            iplot._nofancy = True
            iplot.set_ccd(ccd)
            iplot.launch(vmin=vmin, vmax=vmax, notebook=notebook)
            return iplot


class ApertureSpectrum( Spectrum ):
    """ Spectrum created with apperture spectroscopy """
    PROPERTIES         = ["apweight", "background"]
    SIDE_PROPERTIES    = []
    DERIVED_PROPERTIES = []

    def __init__(self, lbda, flux, variance=None, apweight=None, header=None):
        """ """
        self.__build__()
        self.set_data(flux, variance=variance, lbda=lbda, logwave=None)
        self._properties['apweight'] = apweight

    # ================ #
    #  Methods         #
    # ================ #
    # ------- #
    # SETTER  #
    # ------- #
    def set_background(self, background):
        """ """
        if type(background) == np.array:
            self._properties['background'] = ApertureSpectrum(self.lbda, background)
        else:
            self._properties['background'] = background
            
    # ------- #
    # GETTER  #
    # ------- #
    
    # ------- #
    # PLOTTER #
    # ------- #
    def show(self, toshow="data", ax=None, savefile=None, show=True,
                 show_background=True,
                 bcolor="0.7", **kwargs):
        """ Display the spectrum.
        
        Parameters
        ----------
        toshow: [string] -optional-
            Variable you want to display. anything accessible as self.`toshow` that 
            has the same size as the wavelength. 
            If toshow is data or rawdata, the variance will automatically be added
            if it exists.
            Do not change this is you have a doubt.

        ax: [matplotlib.Axes] -optional-
            Provide the axes where the spectrum will be drawn.
            If None this will create a new one inside a new figure
            
        savefile: [string/None] -optional-
            Would you like to save the data? If so give the name of this
            file where the plot will be saved.
            You can provide an extention (.pdf or .png) if you don't both the
            .pdf and .png will be created.

        show: [bool] -optional-
            If you do not save the data (see savefile), shall the plot be shown?

        **kwargs goes to specplot (any matplotlib axes.plot entry will work)

        Returns
        -------
        Void
        """
        from pyifu.tools import figout, specplot
        pl = super(ApertureSpectrum, self).show(toshow=toshow, ax=ax, savefile=None, show=False, **kwargs)
        fig = pl["fig"]
        ax  = pl["ax"]
        if show_background and self.has_background():
            alpha = kwargs.pop("alpha",1.)/4.
            super(ApertureSpectrum, self).show(toshow="rawdata", ax=ax,
                                                   savefile=None, show=False, alpha=alpha, **kwargs)
            self.background.show(ax=ax, savefile=None, show=False, alpha=alpha, color=bcolor, **kwargs)
            
        fig.figout(savefile=savefile, show=show)

    # -------- #
    #  I/O     #
    # -------- #
    def _build_hdulist_(self, saveerror=False, savebackground=True):
        """ The fits hdulist that should be saved.

        Parameters
        ----------
        saveerror:  [bool] -optional- 
            Set this to True if you wish to record the error and not the variance
            in you first hdu-table. if False, the table will be called
            VARIANCE and have self.v; if True, the table will be called
            ERROR and have sqrt(self.v)

        savebackground: [bool] -optional-
            Shall the background be saved ?

        Returns
        -------
        Void
        """
        from astropy.io.fits import ImageHDU
        self.header['PYSEDMT']  = ("ApertureSpectrum","Pysedm object Type")
        hdul = super(ApertureSpectrum, self)._build_hdulist_(saveerror=saveerror)

        hduAp  = ImageHDU(self.apweight, name='APWEIGHT')
        hdul.append(hduAp)
        # -- Variance saving
        if self.has_background():
            hduBkgd     = ImageHDU(self.background.data, name='BKGD')
            hdul.append(hduBkgd)
            hduBkgdVar  = ImageHDU(self.background.data, name='BKGDVAR')
            hdul.append(hduBkgdVar)
            hduApBkgd   = ImageHDU(self.background.apweight, name='BKGDAPW')
            hdul.append(hduApBkgd)
            
        return hdul


    def load(self, filename, dataindex=0, varianceindex=1, headerindex=None):
        """ 

        lbda - If an hdu column of the fits file is name:
               "LBDA" or "LAMBDA" or "WAVE" or "WAVELENGTH" or "WAVELENGTHS",
               the column will the used as lbda
        
        """
        super(ApertureSpectrum, self).load(filename, dataindex=dataindex, varianceindex=varianceindex,
                                               headerindex=headerindex)
        
        # Get the LBDA if any
        apweight_ = [f.data for f in self.fits if f.name.upper() in ["APWEIGHT"]]
        self._properties["apweight"] = None if len(apweight_)==0 else apweight_[0]
        
        # Get the LBDA if any
        background_ = [f.data for f in self.fits if f.name.upper() in ["BKGD","BACKGROUND"]]
        bapweight_ = [f.data for f in self.fits if f.name.upper() in ["APWBKGD","BKGDAPW"]]
        bvar_ = [f.data for f in self.fits if f.name.upper() in ["BKGDVAR"]]
        if len(background_)==1:
            
            bck = ApertureSpectrum(self.lbda, background_[0],
                                       variance=None if len(bvar_)==0 else bvar_[0],
                                       apweight=None if len(bapweight_)==0 else bapweight_[0],
                                       header=None)
            self._properties['rawdata'] = self.rawdata + bck.data
            self.set_background(bck)
        
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def data(self):
        """ """
        return self.rawdata - self._backgrounddata
    
    @property
    def apweight(self):
        """ """
        return self._properties['apweight']

    # ----------
    # Background
    @property
    def background(self):
        """ """
        return self._properties['background']
    
    def has_background(self):
        return self._properties['background'] is not None
    
    @property
    def _backgrounddata(self):
        """ """
        return 0 if not self.has_background() else self.background.data



    
########################
#                      #
#    MPL ADDON         #
#                      #
########################
