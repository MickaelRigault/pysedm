#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module containing the part that are directly SEDM oriented. """

import numpy            as np
import matplotlib.pyplot as mpl

from pyifu.spectroscopy import Cube
from pyifu.mplinteractive import InteractiveCube
from .utils.tools       import kwargs_update

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

    

def get_sedmcube(filename, **kwargs):
    """ Load a Cube from the given filename 
    
    Returns
    -------
    Cube
    """
    return SEDMCube(filename, **kwargs)



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


#################################
#                               #
#    SEDMachine Cube            #
#                               #
################################# 
class SEDMCube( Cube ):
    """ SEDM Cube """
    DERIVED_PROPERTIES = ["sky"]
    
    def remove_sky(self, nspaxels=50, usemean=False, estimate_from="rawdata",
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
 
        **kwargs goes to get_faintest_spaxels(): 
                  e.g: lbda_range, avoid_area, avoid_indexes etc.

        Returns
        -------
        Void (affects `data`)
        """
        self.remove_flux( self.get_spectrum(self.get_faintest_spaxels(nspaxels,**kwargs), usemean=usemean,
                                                data=estimate_from).data)


    # - Improved version allowing to add CCD
    def show(self, toshow="data",
                 interactive=False, ccd=None,
                 savefile=None, ax=None, show=True,
                 show_meanspectrum=True, cmap=None,
                 vmin=None, vmax=None, 
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

        **kwargs goes to matplotlib's imshow 

        Returns
        -------
        Void
        """
        if not interactive or ccd is None:
            return super(SEDMCube, self).show(toshow=toshow, interactive=interactive,
                                           savefile=savefile, ax=ax, show=show,
                                           show_meanspectrum=show_meanspectrum, cmap=cmap,
                                           vmin=vmin, vmax=vmax, **kwargs)
        else:
            iplot = InteractiveCubeandCCD(self, fig=None, axes=ax, toshow=toshow)
            iplot._nofancy = True
            iplot.set_ccd(ccd)
            iplot.launch(vmin=vmin, vmax=vmax)
            return iplot


class InteractiveCubeandCCD( InteractiveCube ):
    PROPERTIES = ["ccd", "axccd"]

    def launch(self, *args, **kwargs):
        """ """
        
        self._ccdshow = self.ccdimage.show(ax = self.axccd, aspect="auto", show=False)["imshow"]
        super(InteractiveCubeandCCD, self).launch(*args, **kwargs)

    def set_to_origin(self):
        """ Defines the default parameters """
        super(InteractiveCubeandCCD, self).set_to_origin()
        self._trace_patches = []

    def reset(self):
        """ Set back everything to origin """
        super(InteractiveCubeandCCD, self).reset()
        self.clean_axccd(update_limits=True)
        
    # =========== #
    #  SETTER     #
    # =========== #
    def set_ccd(self, ccd):
        """ """
        self._properties["ccd"] = ccd
        
    def set_figure(self, fig=None, **kwargs):
        """ attach a figure to this method. """
        if fig is None:
            figprop = kwargs_update(dict(figsize=[10,7]), **kwargs)
            self._properties["figure"] = mpl.figure(**figprop)
        elif matplotlib.figure.Figure not in fig.__class__.__mro__:
            raise TypeError("The given fig must be a matplotlib.figure.Figure (or child of)")
        else:
            self._properties["figure"] = mpl.figure(**figprop)

    def set_axes(self, axes=None, **kwargs):
        """ """
        if axes is None and not self.has_figure():
            raise AttributeError("If no axes given, please first set the figure.")
        
        if axes is None:
            figsizes = self.fig.get_size_inches()
            axspec = self.fig.add_axes([0.10,0.6,0.5,0.35])
            axim   = self.fig.add_axes([0.65,0.6,0.75*(figsizes[1]*0.5)/float(figsizes[0]),0.35])

            axccd  = self.fig.add_axes([0.10,0.10,axim.get_position().xmax- 0.1, 0.4])
            
            axspec.set_xlabel(r"Wavelength", fontsize="large")
            axspec.set_ylabel(r"Flux", fontsize="large")
            self.set_axes([axspec,axim,axccd])
            
        elif len(axes) != 3:
            raise TypeError("you must provide 2 axes [axspec and axim] and both have to be matplotlib.axes(._axes).Axes (or child of)")
        else:
            # - actual setting
            self._properties["axspec"], self._properties["axim"], self._properties["axccd"] = axes
            if not self.has_figure():
                self.set_figure(self.axspec.figure)

    # ------------- #
    #  Show Things  #
    # ------------- #
    def show_picked_traces(self):
        """ """
        if not self._hold:
            self.clean_axccd()
            
        self._trace_patches = self.ccdimage.display_traces(self.axccd, self.get_selected_idx(),
                                                            facecolors="None", edgecolors=self._active_color)

    def clean_axccd(self, update_limits=False):
        """ """
        self.axccd.patches = []
        if update_limits:
            self.axccd.set_xlim(0, self.ccdimage.shape[0])
            self.axccd.set_ylim(0, self.ccdimage.shape[1])
        
    # ================= #
    # Change For CCD    #
    # ================= #
    def update_figure_fromaxim(self):
        """ What would happen once the spaxels are picked. """
        if len(self.selected_spaxels)>0:
            self.show_picked_spaxels()
            self.show_picked_spectrum()
            self.show_picked_traces()
            self.fig.canvas.draw()

    # ================= #
    # Properties        #
    # ================= #
    @property
    def axccd(self):
        """ """
        return self._properties["axccd"]
    
    @property
    def ccdimage(self):
        """ """
        return self._properties["ccd"]
