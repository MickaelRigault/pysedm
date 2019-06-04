#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" This modules handles the CCD based images """


"""
// Full usage of ccd functionalities requires: shapely, astrobject, pynverse (all pipable) //

// ccd-x axis is refered to as 'i', y as "j".

show: plot the ccd as imshow.
show_traceindex: plot the ccd using `show()` and overplot the trace coutours.
"""


import warnings
import numpy as np

# Propobject
from propobject            import BaseObject
# Astrobject
from astrobject.photometry import Image
# PyIFU
from pyifu.spectroscopy    import Spectrum


from .utils.tools import kwargs_update

"""
The Idea of the CCD calibration is to first estimate the Spectral Matching.
Then to set this Matching to all the ScienceCCD objects.

- Tracematch holds the spaxel->pixel_area relation.

"""

__all__ = ["get_dome","get_ccd"]

##################################
#                                #
#   Object Generators            #
#                                #
##################################
def get_ccd(lampfile, ccdspec_mask=None,
            tracematch=None, background=None,
            correct_traceflexure=False, savefile_traceflexure=None,
                **kwargs):
    """ Load a SEDmachine ccd image.

    Parameters
    -----------
    lampfile: [string]
        Location of the file containing the object

    tracematch: [Tracematch (pysedm object)] -optional-
        Tracematch object containing the Spectral property on the CCD

    background: [bool] -optional-
        which kind of background do you want (i.e. background=0)
    **kwargs goes to _get_default_background_()

    Returns
    -------
    ScienceCCD (Child of CCD which is a Child of an astrobjec's Image)
    """
    lamp = ScienceCCD(lampfile, background=0)
    if tracematch is not None:
        lamp.set_tracematch(tracematch)
        # Trace Flexure Correction (if any)
        if correct_traceflexure:
            from .flexure import get_ccd_jflexure
            from .sedm    import TRACE_DISPERSION
            # Save the flexure plot
            j_offset = get_ccd_jflexure(lamp, ntraces=200, tracewidth=1, jscan=[-3,3,10],
                                savefile=savefile_traceflexure, get_object=False)

            new_tracematch = lamp.tracematch.get_shifted_tracematch(0, j_offset)
            new_tracematch.set_buffer( TRACE_DISPERSION)
            lamp.set_tracematch(new_tracematch )
            lamp.header["JFLXCORR"] =  (True, "Is TraceMatch corrected for j flexure?")
            lamp.header["CCDJFLX"] =  (j_offset, "amplitude in pixel of the  j flexure Trace correction")
        else:
            lamp.header["JFLXCORR"] =  (False, "Is TraceMatch corrected for j flexure?")
            lamp.header["CCDJFLX"] =  (0, "amplitude in pixel of the  j flexure Trace correction")

    if background is None:
        lamp.set_background(lamp._get_default_background_(**kwargs), force_it=True)

    elif not background == 0:
        lamp.set_background(background, force_it=True)

    return lamp

def get_dome(domefile, tracematch=None,  load_sep=False, **kwargs):
    """ Load a SEDmachine domeflat image.
    (special version of get_ccd. might be moved to get_ccd...)

    Parameters
    -----------
    domefile: [string]
        Location of the file containing the object

    tracematch: [Tracematch (pysedm object)] -optional-
        Tracematch object containing the Spectral property on the CCD

    **kwargs goes to DomeCCD.__init__() if tracematch is None
             else to _get_default_background_
    Returns
    -------
     DomeCCD (Child of CCD which is a Child of an astrobjec's Image)
    """
    if tracematch is not None:
        kwargs["background"] = 0

    dome = DomeCCD(domefile, **kwargs)

    if load_sep:
        dome.datadet = dome.data/np.sqrt(np.abs(dome.data))
        dome.sep_extract(thresh=50., on="datadet")

    if tracematch is not None:
        # = Tracematch that gonna help the background
        dome.set_tracematch(tracematch)
        dome.set_background(dome._get_default_background_(**kwargs), force_it=True)

    return dome


#####################################
#                                   #
#  Raw CCD Images for SED machine   #
#                                   #
#####################################
class BaseCCD( Image ):
    """ """
    def __build__(self,bw=64, bh=64,
                      fw=3, fh=3,**kwargs):
        """ build the structure of the class

        // Doc from SEP
        bw, bh : int, -optional-
            Size of background boxes in pixels. Default is 64 [ndlr (in SEP)].

        fw, fh : int, -optional-
            Filter width and height in boxes. Default is 3 [ndlr (in SEP)].

        """

        super(BaseCCD,self).__build__(**kwargs)
        # -- How to read the image
        self._build_properties["bkgdbox"]={"bh":bh,"bw":bw,"fh":fh,"fw":fw}

    # ==================== #
    #  Internal tools      #
    # ==================== #
    def _get_sep_threshold_(self, thresh):
        """ Trick to automatically get the proper threshold for SEP extract """
        if thresh is None:
            return np.median(self.rawdata)
        return thresh

    # ==================== #
    #  Properties          #
    # ==================== #
    @property
    def data_log(self):
        return np.log10(self.data)

# ============================== #
#                                #
#  Main CCD Object               #
#  Tracing <-> CCD <-> Cube      #
#                                #
# ============================== #
class CCD( BaseCCD ):
    """ Virtual Class For CCD images that have input light """

    PROPERTIES         = ["tracematch"]
    DERIVED_PROPERTIES = ["matched_septrace_index"]

    # ------------------- #
    # Tracematch <-> CCD   #
    # ------------------- #
    def set_tracematch(self, tracematch):
        """ Attach to this instance the Tracematch """
        from .spectralmatching import TraceMatch
        if TraceMatch not in tracematch.__class__.__mro__:
            raise TypeError("The given tracematch must be a TraceMatch object")

        self._properties["tracematch"] = tracematch

    def match_trace_to_sep(self):
        """ matches the SEP ellipse with the current trace vertices.
        You must have ran sep_extract() to be able to use this method.

        (This method need Shapely.)

        You can then use the methods:
        - sepindex_to_traceindex()
        - traceindex_to_sepindex()

        Returns
        -------
        Void
        """
        # -> import and tests
        try:
            from shapely import geometry, vectorized
        except ImportError:
            raise AttributeError("Matching traces to sep requires Shapely. pip install Shapely")

        if not self.has_sepobjects():
            raise AttributeError("sep has not been ran. Do so to be able to match sep output with traces")

        # -> actual code
        x,y,a,b,theta = self.sepobjects.get(["x","y","a","b","theta"]).T
        self._derived_properties["matched_septrace_index"] =\
          {idx: np.argwhere(vectorized.contains( geometry.Polygon(self.tracematch.get_trace_vertices(idx)), x,y)).ravel()
               for idx in self.tracematch.trace_indexes}

    def sepindex_to_traceindex(self, sepindex):
        """ Give the index of an sep entry. This will give the corresponding trace index """
        if not self.has_matchedindex():
            raise AttributeError("spectral match traces has not been matched with sep indexes. Run match_tracematch_and_sep()")
        return [traceindex for traceindex, sepindexes in self.matchedindex.items() if sepindex in sepindexes]

    def traceindex_to_sepindex(self, traceindex):
        """ Give the index of an sep entry. This will give the corresponding trace index """
        if not self.has_matchedindex():
            self.match_trace_to_sep()

        return self.matchedindex[traceindex]

    def set_default_variance(self, force_it=False):
        """ define a default variance using the following formula:

        rawdata + ( median(data) - percentile(data, 16) )**2,

        it supposed to account for poisson noise + potential additional variance.
        This is a really poor's man tools...

        Returns
        -------
        Void
        """
        if self.has_var() and not force_it:
            raise AttributeError("Cannot reset the variance. Set force_it to True to allow overwritting of the variance.")
        delta_sigma = np.percentile(self.data, [16,50])

        self._properties['var'] = self.rawdata+(delta_sigma[1]-delta_sigma[0])**2

    # ----------- #
    #   GETTER    #
    # ----------- #
    def get_trace_cutout(self, traceindex, masked=True, on="data"):
        """ returns a 2D array containg the cutout around the requested trace.
        The trace could be either with or without tracematch mask (i.e. 0 outside
        the trace).

        Parameters
        ----------
        traceindex: [int]
            index of the spaxel trace.

        masked: [bool] -optional-
            do you want the data (see `on`) to be masked out (i.e. ==0) outside
            the trace

        on: [string] -optional-
            On which data source do you want the trace (e.g. data, rawdata, background, variance)

        Returns
        -------
        2d-array
        """
        xmin, xmax = self.tracematch.get_trace_xbounds(traceindex)
        ymin, ymax = self.tracematch.get_trace_ybounds(traceindex)
        if masked:
            return self.get_trace_mask(traceindex)[ymin:ymax,xmin:xmax]*eval("self.%s"%on)[ymin:ymax,xmin:xmax]
        return eval("self.%s"%on)[ymin:ymax,xmin:xmax]

    def get_finetuned_trace(self, traceindex, polydegree=2,
                            width=None, trace_position=False, **kwargs):
        """ The builds a fine tuned trace of the given traceindex.
        The tuning uses detected object from sep.
        => You must have run match_tracematch_and_sep()

        Parameters
        ----------
        traceindex: [int]
            Index of the trace you want to fine tune

        polydegree: [positive-int] -optional-
            Degree of the polynome that will be used to define the trace.
            (See 'width' for details on the width of the trace polygon)
            => If polydegree is higher than the number of sep object detected
               belonging to this trace, polydegree will the reduced to that number
            => If polydegree ends up being lower of equal to 0, None is returned

        width: [float / None] -optional-
            Width of the polygon (in pixels)
            If None, width will be estimated based on the b-values of the
            sep detected objects.

        trace_position: [bool] -optional-
            Get the expected trace central position instead of the vertices of the polygon containing it.


        **kwargs goes to spectralmatching.get_boxing_polygone()

        Returns
        -------
        One of these:
          - None if no fit possible (one of less sep object for this trace )
          - array (vertices)
          - array (x,y if trace_position=True)
        """
        if not self.has_sepobjects():
            warnings.warn("No SEP object loaded. run sep_extract() to enable finetuning. Original vertices returned")
            return self.tracematch.copy()

        _cannot_finetune = False
        try:
            x, y, b = self.sepobjects.get(["x","y","b"], self.traceindex_to_sepindex(traceindex)).T
        except:
            _cannot_finetune = True
            x,y,b = [],[],[]

        if len(x) < polydegree:
            warnings.warn("less sep-points than polynom degree. Degree reduced ")
            polydegree = len(x)-1

        if polydegree <=0:
            _cannot_finetune = True

        if  _cannot_finetune:
            warnings.warn("cannot build finetune tracing for %s. Normal vertices returned"%traceindex)
            return self.tracematch.get_trace_vertices(traceindex)
        else:
            if trace_position:
                return self.tracematch.get_finetuned_trace(traceindex, x, y, polydegree=polydegree, **kwargs)
            else:
                return self.tracematch.get_finetuned_trace_vertices(traceindex, x, y,
                                                            width= np.nanmedian(b)*2. if width is None else width,
                                                            polydegree=polydegree, **kwargs)

    def get_trace_mask(self, traceindex, finetune=False, polydegree=2,
                           subpixelisation=5, **kwargs):
        """ Build a weightmask based on the trace vertices.

        Parameters
        ----------
        traceindex: [int]
            index of the trace for which you want a mask

        finetune: [bool] -optional-
            Should the trace be remeasured based on potential detected sources?

        // The following options apply if finetune is True

        polydegree: [int] -optional-
            Degree of the polynome used to define the traces

        subpixelisation: [int] -optional-
            Our much should the pixel be subdivided to do the polygon-to-image
            interpolation? (the higher the slower)
            Set 1 for no subdivition (fastest)

        **kwargs goes to the method `get_finetuned_trace`
        """
        if not finetune:
            return self.tracematch.get_trace_mask(traceindex)

        from .spectralmatching import polygon_mask, _HAS_SKIMAGE
        if not _HAS_SKIMAGE:
            warnings.warn("get_trace needs skimage to be able to use subpixelisation")
            subpixelisation = 1

        verts = self.get_finetuned_trace(traceindex, polydegree=polydegree, **kwargs)

        mask = np.asarray(polygon_mask( verts*subpixelisation,
                                        self.shape[0]*subpixelisation, self.shape[1]*subpixelisation,
                                        get_fullcolor=False),
                              dtype="float")

        if subpixelisation==1:
            return mask/np.nanmax(mask)

        from .spectralmatching import measure
        return measure.block_reduce(mask, (subpixelisation,subpixelisation) )/float(subpixelisation**2)

    def get_finetuned_tracematch(self, indexes, polydegree=2, width=None, build_masking=False, **kwargs):
        """ """
        from .spectralmatching import TraceMatch
        tmap = TraceMatch()
        tmap.set_trace_vertices({i:self.get_finetuned_trace(i, polydegree=polydegree, width=width, **kwargs)
                                     for i in indexes}, build_masking=build_masking)
        return tmap

    # ---------------- #
    #  Tracematch      #
    # ---------------- #
    def get_spectrum(self, traceindex, on="data", finetune=False):
        """ Get the basic spectrum extracted for the CCD based on the
        TraceMatch object.

        Parameters
        ----------
        traceindex: [int, list of]
            index(es) of the spectrum(a) to return

        on: [str] -optional-
            on which 2d image shall the spectrum be extracted.
            By Default 'data', but you can set e.g. rawdata, background
            or anything accessible as 'self.%s'%on.

        finetune: [bool] -optional-
            Should the trace masking come from finetunning of spectral trace?
            (Remark: The spectral match loaded might already be finetuned ones.)

        Returns
        -------
        flux (or list of) as a function of pixels
        """
        if not self.has_tracematch():
            raise AttributeError("The TraceMatch has not been set. see set_tracematch() ")

        if hasattr(traceindex, "__iter__"):
            return [self.get_spectrum(id_) for id_ in traceindex]

        maskidx  = self.get_trace_mask(traceindex, finetune=finetune)
        return np.sum(eval("self.%s"%on)*maskidx, axis=0)

    def get_xslice(self, i, on="data"):
        """ build a `CCDSlice` based on the ith-column.

        Returns
        -------
        CCDSlice (child of Spectrum)
        """


        slice_ = CCDSlice(None)
        if "data" in on and not self.has_var():
            warnings.warn("Setting the default variance for 'get_xslice' ")
            self.set_default_variance()

        var = self.var.T[i] if 'data' in on else np.ones(np.shape("self.%s"%on))*np.nanstd("self.%s"%on)

        slice_.create(eval("self.%s.T[i]"%on), variance = var,
                    lbda = np.arange(len(self.data.T[i])), logwave=False)

        slice_.set_tracebounds(self.tracematch.get_traces_crossing_x_ybounds(i))

        return slice_

    def fit_background(self, start=2, jump=10, multiprocess=True, set_it=True, smoothing=[0,5], **kwargs):
        """ """
        from .background import get_background, fit_background
        self._background = get_background( fit_background(self, start=start, jump=jump,
                                                            multiprocess=multiprocess, **kwargs),
                                               smoothing=smoothing )
        if set_it:
            self.set_background(self._background.background, force_it=True)


    def fetch_background(self, set_it=True, build_if_needed=True, **kwargs):
        """ """
        from .background import load_background
        from .io import filename_to_background_name
        # ---------------- #
        #  Test it exists  #
        # ---------------- #
        from glob import glob
        if len(glob(filename_to_background_name(self.filename)))==0:
            warnings.warn("No background has been found for %s"%self.filename)
            if not build_if_needed:
                raise IOError("Since build_if_needed=False, No background available.")
            from .background import build_background
            build_background(self, **kwargs)
            warnings.warn("A background has been built")

        self._background = load_background( filename_to_background_name( self.filename ))
        if set_it:
            self.set_background( self._background.background, force_it=True)

    def extract_spectrum(self, traceindex, wavesolution, lbda=None, kind="cubic",
                             get_spectrum=True, pixel_shift=0.):
        """ Build the `traceindex` spectrum based on the given wavelength solution.
        The returned object could be an pyifu's Spectrum or three arrays, lbda, flux, variance.

        The method works as follow for the given traceindex:
        1) Get the flux per pixels [using the get_spectrum() method]
           (Get the variance the same way if any)
        2) Convert the given lbda into pixels
           [using the lbda_to_pixels() method from wavesolution]
        3) Interpolate the flux per pixels into flux per lbda
           (Interpolate the variance the same way)
           [using interp1d from scipy.interpolate]


        Parameters
        ----------
        traceindex: [int]
            The index of the spectrum you want to extract

        wavesolution: [WaveSolution]
            Object containing the method to go from pixels to lbda

        lbda: [array] -optional-
            Shape of the lbda array you want the spectrum to have.

        kind: [str or int] -optional-
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
            where 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of first, second or third order) or as an integer
            specifying the order of the spline interpolator to use.

        get_spectrum: [bool] -optional-
            Which form the returned data should have?

        finetune: [bool] -optional-
            Should the trace masking come from finetunning of spectral trace?
            (Remark: The spectral match loaded might already be a finetuned ones.)

        Returns
        -------
        Spectrum or
        array, array, array/None (lbda, flux, variance)
        """
        f = self.get_spectrum(traceindex, finetune=False)
        v = self.get_spectrum(traceindex, finetune=False, on="var") if self.has_var() else None
        pixs = np.arange(len(f))[::-1]
        minpix, maxpix = self.tracematch.get_trace_xbounds(traceindex)
        mask = pixs[(pixs>minpix)* (pixs<maxpix)][::-1]
        if lbda is not None:
            from scipy.interpolate     import interp1d
            pxl_wanted = wavesolution.lbda_to_pixels(lbda, traceindex) + pixel_shift
            flux = interp1d(pixs[mask], f[mask], kind=kind)(pxl_wanted)
            var  = interp1d(pixs[mask], v[mask], kind=kind)(pxl_wanted) if v is not None else v
        else:
            lbda = wavesolution.pixels_to_lbda(pixs[mask], traceindex)
            flux = f[mask]
            var  = v[mask]

        if get_spectrum:
            spec = Spectrum(None)
            spec.create(flux,variance=var,lbda=lbda)
            return spec

        return lbda, flux, var

    # --------------- #
    #  Extract Cube   #
    # --------------- #
    def extract_cube(self, wavesolution, lbda,
                         hexagrid=None, traceindexes=None, show_progress=False,
                         pixel_shift=0., rotation=None):
        """ Create a cube from the ccd.

        ------------------------------------
        | Central method of the ccd object |
        ------------------------------------

        The method works as follow (see the extract_spectrum() method):
        for each trace (loop object traceindexes)
        1) Get the flux per pixels [using the get_spectrum() method]
           (Get the variance the same way if any)
        2) Convert the given lbda into pixels
           [using the lbda_to_pixels() method from wavesolution]
        3) Interpolate the flux per pixels into flux per lbda
           (Interpolate the variance the same way)
           [using interp1d from scipy.interpolate]
        4) Get the x,y position of the traceindex
           [using the ids_to_index() and index_to_xy() methods from hexagrid]

        If anything false (most likely the interpolation because of wavelength matching)
        the flux (or variance) per lbda will be set to an array of NaNs

        All the spaxels fluxes will be set to a cube (SEDMCube see .sedm)

        Parameters
        ----------
        wavesolution: [WaveSolution]
            The object containing the pixels<->wavelength relation of the night.

        lbda: [float array]
            wavelength array of the cube to be created (in Angstrom)

        hexagrid: [HexagoneProjection] -optional-
            object containing the x,y position of the traces.
            If not given, this will be created based on the instance's TraceMatch.
            (it is advised to give the night hexagrid.)

        traceindexes: [list of int] -optional-
            Which trace should be extracted to build the cube?
            If not given (None) this will used all the traceindexes for which there
            is a wavelength solution (wavesolution)

        show_progress: [bool] -optional-
            Should the progress within the loop over traceindexes should be
            shown (using astropy's ProgressBar)

        Returns
        -------
        SEDMCube (child of pyifu's Cube)
        """
        from .sedm import SEDMSPAXELS, SEDMCube, SEDM_INVERT, SEDM_ROT
        if rotation is None:
            rotation = SEDM_ROT
        # - index check
        if traceindexes is None:
            traceindexes = np.sort(list(wavesolution.wavesolutions.keys()))

        elif np.any(~np.in1d(traceindexes, list(wavesolution.wavesolutions.keys()))):
            raise ValueError("At least some given indexes in `used_indexes` do not have a wavelength solution")

        # - Hexagonal Grid
        if hexagrid is None:
            hexagrid = self.tracematch.extract_hexgrid(traceindexes)

        used_indexes = [i_ for i_ in traceindexes if i_ in hexagrid.ids_index.keys()]
        # - data
        cube      = SEDMCube(None)
        cubeflux_ = {}
        cubevar_  = {} if self.has_var() else None

        # ------------ #
        # MultiProcess #
        # ------------ #
        def _build_ith_flux_(i_):
            try:
                lbda_, flux_, variance_ = self.extract_spectrum(i_, wavesolution, lbda=lbda,
                                                                get_spectrum=False,
                                                                pixel_shift=pixel_shift)
            except:
                warnings.warn("FAILING EXTRACT_SPECTRUM for trace index %d: most likely wavesolution failed for this trace. *NaN Spectrum set*"%i_)
                flux_ = np.ones(len(lbda) )*np.NaN
                variance_ = np.ones(len(lbda) )*np.inf

            cubeflux_[i_] = flux_
            if cubevar_ is not None:
                cubevar_[i_] = variance_
            #except:
            #    warnings.warn("FAILED FOR %s. Most likely, the requested wavelength are not fully covered by the trace"%i_)
            #    cubeflux_[i_] = np.zeros(len(lbda))*np.NaN
            #    if cubevar_ is not None:
            #        cubevar_[i_] = np.zeros(len(lbda))*np.NaN

        # ------------ #
        # - MultiThreading to speed this up
        if show_progress:
            from astropy.utils.console import ProgressBar
            ProgressBar.map(_build_ith_flux_, used_indexes)
        else:
            #from multiprocessing import Pool as ThreadPool
            #pool = ThreadPool(4)
            #pool.map(_build_ith_flux_, used_indexes)
            _ = [_build_ith_flux_(i) for i in used_indexes]

        cubeflux = np.asarray([cubeflux_[i] for i in used_indexes])
        cubevar  = np.asarray([cubevar_[i]   for i in used_indexes]) if cubevar_ is not None else None

        # - Fill the Cube
        #  SEDM DEPENDENT
        hexagrid.set_rot_degree(rotation)
        spaxels_position = np.asarray(hexagrid.index_to_xy( hexagrid.ids_to_index(used_indexes),
                                        invert_rotation=False,
                                        switch_axis=SEDM_INVERT)).T


        spaxel_map = {i:c for i,c in zip(used_indexes, spaxels_position)}

        cube.create(cubeflux.T,lbda=lbda, spaxel_mapping=spaxel_map, variance=cubevar.T)
        cube.set_spaxel_vertices(np.dot(hexagrid.grid_rotmatrix,SEDMSPAXELS.T).T)
        return cube


    # --------------- #
    #  Extract Cube   #
    # --------------- #
    def show(self, toshow="data", ax=None,
                 logscale= False, cmap = None, show_sepobjects=False,
                 vmin = None, vmax = None, savefile=None, show=True, **kwargs):
        """ Highlight the trace on top of the CCD image.
        This method requires that the spectral match has been loaded.

        Parameters
        ----------
        idx: [int]


        vmin, vmax: [float /string / None]
            Upper and lower value for the colormap. 3 Format are available
            - float: Value in data unit
            - string: percentile. Give a float (between 0 and 100) in string format.
                      This will be converted in float and passed to numpy.percentile
            - None: The default will be used (percentile 0.5 and 99.5 percent respectively).
            (NB: vmin and vmax are independent, i.e. one can be None and the other '98' for instance)

        Returns
        -------
        dict ({ax, fig, imshow's output})
        """
        from .utils.mpl import figout, mpl

        if ax is None:
            fig = mpl.figure(figsize=[8,8])
            ax  = fig.add_axes([0.13,0.13,0.8,0.8])
            ax.set_xlabel(r"$\mathrm{x\ [ccd]}$",fontsize = "large")
            ax.set_ylabel(r"$\mathrm{y\ [ccd]}$",fontsize = "large")
        elif "imshow" not in dir(ax):
            raise TypeError("The given 'ax' most likely is not a matplotlib axes. "+\
                             "No imshow available")
        else:
            fig = ax.figure

        # What To Show
        data_ = eval("self.%s"%toshow) if type(toshow) == str else toshow
        if logscale:
            data_ = np.log10(data_)

        if cmap is None:  cmap=mpl.cm.viridis
        vmin = np.percentile(data_, 0.5) if vmin is None else \
          np.percentile(data_, float(vmin)) if type(vmin) == str else\
          vmin
        vmax = np.percentile(data_, 99.5) if vmax is None else \
          np.percentile(data_, float(vmax)) if type(vmax) == str else\
          vmax

        prop = kwargs_update(dict(origin="lower", aspect='auto', vmin=vmin, vmax=vmax), **kwargs)

        # Show It
        sc = ax.imshow(data_, **prop)

        # - sepobject
        if show_sepobjects and self.has_sepobjects():
            self.sepobjects.display(ax)


        fig.figout(savefile=savefile, show=show)

        return {"ax":ax, "fig" : fig, "imshow":sc}

    def show_traceindex(self, traceindex, ax=None,
                          logscale= False, toshow="data", show_finetuned_traces=False,
                          cmap = None, facecolor = "None", edgecolor="k",
                          vmin = None, vmax = None, savefile=None, show=True, **kwargs):
        """ Highlight the trace on top of the CCD image.
        This method requires that the spectral match has been loaded.

        Parameters
        ----------
        traceindex: [int]

        """
        from .utils.mpl import figout

        pl = self.show( toshow=toshow, ax=ax, logscale=logscale, cmap = cmap,
                    vmin = vmin, vmax = vmax, savefile=None, show=False, **kwargs)
        ax, fig = pl["ax"], pl["fig"]
        pl["patch"] = self.display_traces(ax, traceindex, facecolors=facecolor, edgecolors=edgecolor)

        if show_finetuned_traces and self.has_matchedindex():
            from matplotlib import patches
            traces = traceindex if hasattr(traceindex, "__iter__") else [traceindex]
            for idx_ in traces:
                self.sepobjects.display_ellipses(pl["ax"], self.traceindex_to_sepindex(idx_))
                p_ = patches.Polygon(self.get_finetuned_trace(idx_),
                                         facecolor="None", edgecolor="C1")
                pl["ax"].add_patch(p_)


        # Output
        fig.figout(savefile=savefile, show=show)
        return pl

    def show_xslice(self, xpixel, toshow="data", savefile=None, ax=None, show=True,
                        ls="-", color=None, lw=1,
                        show_tracebounds=True,
                        bandalpha=0.5, bandfacecolor="0.7", bandedgecolor="None",bandedgewidth=0,
                        **kwargs):
        """ """
        from .utils.mpl import figout, mpl

        if ax is None:
            fig = mpl.figure(figsize=[9,5])
            ax  = fig.add_subplot(111)
        else:
            fig = ax.figure

        # This plot
        ax.plot( eval("self.%s.T[xpixel]"%toshow), ls=ls, color=color, lw=lw,**kwargs)
        if show_tracebounds:
            [ax.axvspan(*y_, alpha=bandalpha, facecolor=bandfacecolor,
                            edgecolor=bandedgecolor,linewidth=bandedgewidth)
                 for y_ in self.tracematch.get_traces_crossing_x_ybounds(xpixel)]

        fig.figout(savefile=savefile, show=show)


    def show_as_slicer(self, traceindexes, vmin=0 , vmax="90",
                           masked=True, toshow="rawdata"):
        """ """
        import matplotlib.pyplot as mpl
        data = eval("self.%s"%toshow)
        if vmin is None:
            vmin = 0
        if type(vmin) == str:
            vmin = np.percentile(data, vmin)
        if vmax is None:
            vmax = "95"
        if type(vmax) == str:
            vmax = np.percentile(data, vmax)

        #  parameters
        ntraces = len(traceindexes)
        height = 0.85/ntraces

        #  Build the figure
        fig = mpl.figure(figsize=[8,4])

        #  draw the traces
        for i,index_ in enumerate(traceindexes):
            ax  = fig.add_axes([0.1,0.1+height*i, 0.8,height])
            ax.imshow(self.get_trace_cutout(index_, masked=masked, on=toshow), origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_yticks([])
            if i>0:
                ax.set_xticks([])
            else:
                ax.set_xlabel("pixels since trace origin")

        return {"fig":fig}

    def display_traces(self, ax, traceindex, facecolors="None", edgecolors="k",
                             update_limits=True):
        """ """
        pl = self.tracematch.display_traces(ax, traceindex,
                                            facecolors =facecolors, edgecolors=edgecolors)

        # Fancy It
        if update_limits:
            [xmin, xmax], [ymin, ymax] = np.percentile(self.tracematch.trace_vertices[traceindex] if not hasattr(traceindex,"__iter__")  else\
                                                       np.concatenate([self.tracematch.trace_vertices[idx] for idx in traceindex], axis=0),
                                                       [0,100], axis=0).T
            ax.set_xlim(xmin-20, xmax+20)
            ax.set_ylim(ymin-5, ymax+5)

        return pl

    # ================== #
    #  Internal Tools    #
    # ================== #
    def _get_default_background_(self, add_mask=None,
                                     cut_bright_pixels=None,
                                     exclude_edges=False,
                                 scaleup_sepmask=2, apply_sepmask=True,
                                 **kwargs):
        """ This Background has been optimized for SEDm Calibration Lamps """

        if add_mask is None and self.has_tracematch():
            add_mask = np.asarray(~self.tracematch.get_notrace_mask(), dtype="bool")

        if add_mask is not None and cut_bright_pixels is not None:
            data_ = self.rawdata.copy()
            data_[add_mask] = np.NaN
            add_mask = add_mask + (data_>np.percentile(data_[data_==data_],50))

        if exclude_edges:
            falses = np.zeros(self.shape)
            # - x cuts
            xremove, yremove =100, 20
            falses[:,(np.arange(self.shape[1])<xremove) + (np.arange(self.shape[1])>(self.shape[1]-xremove))  ] = 1.
            falses[(np.arange(self.shape[0])<yremove)   + (np.arange(self.shape[0])>(self.shape[0]-yremove)),:] = 1.
            add_mask = add_mask + np.asarray(falses, dtype="bool")

        return self.get_sep_background(doublepass=False, update_background=False,
                                       add_mask=add_mask,
                                       apply_sepmask=apply_sepmask, scaleup_sepmask=scaleup_sepmask,
                                       **kwargs)

    # ================== #
    #   Properties       #
    # ================== #

    # - TraceMatching association
    @property
    def tracematch(self):
        """ """
        return self._properties["tracematch"]

    def has_tracematch(self):
        return self.tracematch is not None

    @property
    def matchedindex(self):
        """ Object containing the relation between the sepindex and the trace index.
        see the methods sepindex_to_traceindex() and traceindex_to_sepindex() """
        if self._derived_properties["matched_septrace_index"] is None:
            self._derived_properties["matched_septrace_index"] = {}
        return self._derived_properties["matched_septrace_index"]

    def has_matchedindex(self):
        """ Is the sep<-> trace index matching done? """
        return self.matchedindex is not None and len(self.matchedindex.keys())>0

    # - Generic properties
    @property
    def objname(self):
        if "Calib" in self.header.get("NAME","no-name"):
            return self.header["NAME"].split()[1]
        return self.header.get("NAME","no-name")

# ============================== #
#                                #
#  Childs Of CCD                 #
#                                #
# ============================== #
class ScienceCCD( CCD ):
    """ Should be used to improve the trace matching. """


class DomeCCD( ScienceCCD ):
    """ Object Build to handle the CCD images of the Dome exposures"""

    # ================== #
    #   Main Tools       #
    # ================== #
    def get_tracematch(self, bound_pixels=None, width="optimal"):
        """ """
        xlim, ylim = self.get_trace_position(bound_pixels=bound_pixels)

        if width is None:
            b = self.sepobjects.get("b")
            width = np.median(b)*2 if width=="median" else \
              b*2 if width!="optimal" else np.clip(b,np.median(b)-2*np.std(b),np.median(b)+2*np.std(b))*2


        return  [np.concatenate(v_) for v_ in zip(np.asarray([xlim,ylim+width]).T,np.asarray([xlim[::-1],ylim[::-1]-width]).T)]


    def get_trace_position(self, bound_pixels=None):
        """ """
        x, y, a, b, theta = self.sepobjects.get(["x","y","a","b","theta"]).T
        # This enables to remove cosmics
        flagout =  a/b<10
        x, y, theta = x[~flagout], y[~flagout], theta[~flagout]

        if bound_pixels is not None:
            print("BOUNDS")
            xlim = np.asarray([x-bound_pixels[0], x+bound_pixels[1]])
            ylim = np.sin(theta)*np.asarray([[-bound_pixels[0], bound_pixels[1]]]).T + y
            return xlim, ylim

        from .sedm import domexy_to_tracesize
        return domexy_to_tracesize(x, y, theta)



    def get_specrectangles(self, length="optimal",height="optimal",theta="optimal",
                               scaleup_red=2.5, scaleup_blue = 7.5, scaleup_heigth=1.5,
                               fixed_pixel_length=None,
                               use_peak_xy=True):
        """ Vertices of the ~rectangles associated to each spectra based on the
        detected SEP ellipses assuming a fixed height given by the median of the
        b-ellipse parameter

        To convert e.g. the i-th vertice into a Shapely polygon:
        ```
        from shapely import geometry
        rect_vert = self.get_specrectangles()
        rect = geometry.polygon.Polygon(rect_vert.T[i].T)
        ```

        Parameters
        ----------

        fixed_pixel_length: [float/None] -optional-
            how many pixel bluer (higher-x) should the traces be in comparison to the
            reddest point (defined by scaleup_red).
            If This is not None, scaleup_blue is ignored
        Returns
        -------
        ndarray (4x2xN) where N is the number of detected ellipses.
        """
        if not self.has_sepobjects():
            self.sep_extract()

        x, y, xpeak, ypeak, a, b, theta = self.sepobjects.get(["x","y","xpeak", "ypeak", "a", "b", "theta"]).T
        x_,y_ = (xpeak, ypeak) if use_peak_xy else (x,y)


        length = np.median(a) if length=="median" else \
          a if length!="optimal" else np.clip(a,np.median(a)-2*np.std(a),np.median(a)+2*np.std(a))

        height = np.median(b)*scaleup_heigth if height=="median" else \
          b*scaleup_heigth if height!="optimal" else np.clip(b,np.median(b)-2*np.std(b),np.median(b)+2*np.std(b))*scaleup_heigth

        angle  = np.median(theta) if theta=="median" else \
          theta if theta!="optimal" else np.clip(theta,np.median(theta)-1.5*np.std(theta),
                                                     np.median(theta)+1.5*np.std(theta))


        left_lim = np.asarray([  x_-length*np.cos(-angle)*scaleup_red,
                                 y_+length*np.sin(-angle)*scaleup_red])
        if fixed_pixel_length is None:
            right_lim = np.asarray([ x_+length*np.cos(-angle)*scaleup_blue,
                                    y_-length*np.sin(-angle)*scaleup_blue])
        else:
            right_lim = np.asarray([ left_lim[0] + np.cos(-angle)*fixed_pixel_length,
                                     left_lim[1] - np.sin(-angle)*fixed_pixel_length])

        return np.asarray([[left_lim[0],left_lim[0],right_lim[0],right_lim[0]],
                               [left_lim[1]-height,left_lim[1]+height,right_lim[1]+height,right_lim[1]-height]]).T


    # ================== #
    #   Internal         #
    # ================== #
    def _get_default_background_(self, mask_prop={},
                                     from_spectmatch=True,
                                apply_sepmask=False, **kwargs):
        """ This Background has been Optimized for SEDm Dome """

        if from_spectmatch and self.has_tracematch():
            return super(DomeCCD, self)._get_default_background_(**kwargs)

        self.set_background(np.percentile(self.rawdata,0.001), force_it=True)
        self.sep_extract(thresh=5, err=np.percentile(self.rawdata,1))

        return self.get_sep_background(doublepass=False, update_background=False,
                                       apply_sepmask=True, **kwargs)


    def _get_sep_extract_threshold_(self):
        """this will be used as a default threshold for sep_extract"""
        warnings.warn("_get_sep_extract_threshold_ called")
        if not hasattr(self,"_sepbackground"):
                _ = self.get_sep_background(update_background=False)
        return self._sepbackground.globalrms*2


# ============================== #
#                                #
#  Main CCD Object               #
#  Tracing <-> CCD <-> Cube      #
#                                #
# ============================== #
class CCDSlice( Spectrum ):
    """ Flux per Pixel Spectrum based on CCD Slicing"""
    PROPERTIES = ["tracebounds"]
    DERIVED_PROPERTIES = ["tracemasks", "contmodel", "flagin"]

    def fit_continuum(self, degree=5, legendre=True, clipping=[2,2],
                          ngauss=0):
        """ """
        from astropy.stats import mad_std
        from modefit import get_polyfit, get_normpolyfit
        y = self.data[self.tracemaskout]

        nmad = mad_std(y[y==y])
        median = np.nanmedian(y)
        self._derived_properties["flagin"] = ((median - clipping[0] * nmad < y) &  (y< median + clipping[1] * nmad)) & (y==y)

        guesses = dict(a0_guess=np.nanmedian(self.data[self.tracemaskout][self.flagin]))
        if ngauss > 0:
            self._derived_properties["contmodel"] = \
              get_normpolyfit(self.lbda[self.tracemaskout][self.flagin], self.data[self.tracemaskout][self.flagin],
                            np.sqrt(self.variance[self.tracemaskout][self.flagin]), degree, ngauss=ngauss, legendre=legendre)
            for ni in range(ngauss):
                guesses['mu%d_guess'%ni]       = np.argmax(self.data)
                guesses['mu%d_boundaries'%ni]  = [0, len(self.data)]
                guesses['sig%d_guess'%ni]      = 150
                guesses['sig%d_boundaries'%ni] = [100,200]
                guesses['ampl%d_guess'%ni]     = np.nanmax(self.data)
                guesses['ampl%d_boundaries'%ni]= [0,None]
        else:
            self._derived_properties["contmodel"] = \
              get_polyfit(self.lbda[self.tracemaskout][self.flagin], self.data[self.tracemaskout][self.flagin],
                           np.sqrt(np.abs(self.variance[self.tracemaskout][self.flagin])), degree, legendre=legendre)

        self.contmodel.use_minuit = True
        self.contmodel.fit(**guesses)


    def show_continuum(self, ax=None, savefile=None, show=True, show_model=True):
        """ """
        from .utils.mpl  import figout, mpl
        from pyifu.tools import specplot

        if ax is None:
            fig = mpl.figure(figsize=[8,4])
            ax  = fig.add_subplot(111)
            ax.set_xlabel(r"$\mathrm{x\ [ccd]}$",fontsize = "large")
            ax.set_ylabel(r"$\mathrm{y\ [ccd]}$",fontsize = "large")
        else:
            fig = ax.figure

        ax.specplot(self.lbda, self.data, var=self.variance, color="0.7")
        dataout = self.data.copy()
        dataout[~self.tracemaskout] = np.NaN
        ax.specplot(self.lbda, dataout, var=None, color="C1")

        if show_model and self.contmodel is not None:
            ax.plot(self.contmodel.xdata, self.contmodel.data,
                        color="C0", zorder=6, ls="None", marker="x",  ms=4)
            ax.plot(self.lbda[self.tracemaskout][self.flagin], self.contmodel.model.get_model(),
                        color="C0", zorder=7)

        fig.figout(savefile=savefile, show=show)

    def get_fit_contrains(self, fixed_pos=True,
                              sigma_guess=1.5,
                              pos_bounds=[1,1], sigma_bounds=[1.2,1.8], **kwargs):
        """ """
        expected_lines = self.get_expected_lines()

        fitprop = {}
        for i in range(self.ntraces):
            fitprop["mu%d_guess"%i]         = expected_lines[i]
            fitprop["mu%d_fixed"%i]         = fixed_pos
            fitprop["mu%d_boundaries"%i]    = [expected_lines[i]-pos_bounds[0],expected_lines[i]+pos_bounds[0]]
            fitprop["sig%d_guess"%i]        = sigma_guess
            fitprop["sig%d_fixed"%i]        = self.data[int(expected_lines[i])]*sigma_guess*2.5 < 200
            fitprop["sig%d_boundaries"%i]   = sigma_bounds
            fitprop["ampl%d_guess"%i]       = np.max([0,self.data[int(expected_lines[i])]*sigma_guess*2.5])
            fitprop["ampl%d_fixed"%i]       = False
            fitprop["ampl%d_boundaries"%i]  = [0,fitprop["ampl%d_guess"%i]*2 if fitprop["ampl%d_guess"%i] > 0 else 1e-4]

        return fitprop

    def get_expected_lines(self):
        """ """
        return np.mean(self.tracebounds, axis=1)

    # =================== #
    #   Properties        #
    # =================== #
    @property
    def tracebounds(self):
        """ boundaries of each traces overlapping with the selected ccd-slice """
        return self._properties["tracebounds"]

    def set_tracebounds(self, tracebounds):
        """ attach to this spectrum boundaries of each traces overlapping with the selected ccd-slice """
        self._properties["tracebounds"] = np.asarray(tracebounds, dtype="float")
        self._derived_properties["tracemasks"] = [(b[0]<self.lbda) * (self.lbda<b[1]) for b in tracebounds]

    @property
    def ntraces(self):
        """ number of traces the Slice is overlapping """
        return len(self.tracebounds)

    @property
    def tracemasks(self):
        """ boolean arrays indicating pixels overlaping spaxel traces """
        return self._derived_properties["tracemasks"]

    @property
    def tracemaskout(self):
        """ Boolean array been True for pixels not within any trace """
        return ~np.asarray(np.sum(self.tracemasks, axis=0), dtype="bool")

    @property
    def flagin(self):
        """ boolean array indicating the data point used for the fit. """
        return self._derived_properties["flagin"]

    # Model
    @property
    def contmodel(self):
        """ Continuum model measured when running fit_continuum """
        return self._derived_properties["contmodel"]



# ============================ #
#                              #
#    Background                #
#                              #
# ============================ #
