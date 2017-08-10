#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" This modules handles the CCD based images """


import warnings
import numpy as np

import matplotlib.pyplot as mpl
from propobject import BaseObject
from astrobject.photometry import Image
from scipy.interpolate     import interp1d
try:
    import shapely # to be removed
    _HAS_SHAPELY = True
except:
    _HAS_SHAPELY = False

from .utils.tools import kwargs_update

EDGES_COLOR = mpl.cm.binary(0.99,0.5, bytes=True)
SPECTID_CMAP = mpl.cm.viridis

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
            tracematch=None, specmatch=None,
                background=None, **kwargs):
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
    if background is None:
        lamp.set_background(lamp._get_default_background_(**kwargs), force_it=True)
    elif not background == 0:
        lamp.set_background(background, force_it=True)
        
    return lamp

def get_dome(domefile, tracematch=None,  **kwargs):
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
    if tracematch is None:
        return DomeCCD(domefile, **kwargs)
    # = Tracematch that gonna help the background
    dome = DomeCCD(domefile, background=0, **kwargs)
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
    def __build__(self,bw=15, bh=15,
                      fw=10, fh=10,**kwargs):
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

    def match_trace_and_sep(self):
        """ SEP extracted object will be matched with trace indexes
        (This might time a bit of time)

        You can then use the methods:
        - sepindex_to_traceindex()
        - traceindex_to_sepindex()
        
        """
        if not self.has_sepobjects():
            raise AttributeError("Sep extract has not been run. This is needed to match the traces with detected lines.")

        
        x,y,a,b,theta = self.sepobjects.get(["x","y","a","b","theta"]).T
        if _HAS_SHAPELY:
            from shapely import geometry, vectorized
            self._derived_properties["matched_septrace_index"] =\
              {idx: np.argwhere(vectorized.contains( geometry.Polygon(self.tracematch.get_trace_vertices(idx)), x,y)).ravel()
                   for idx in self.tracematch.trace_indexes}
        else:
            print("DECREPATED, Install Shapely")
            septrace_index = [self.tracematch.get_trace_source(x_, y_, a=a_*2, b=b_*2, theta=theta_ )
                              for x_,y_,a_,b_,theta_ in zip(x,y,a,b,theta)]
        
            self._derived_properties["matched_septrace_index"] = np.asarray([s[0] if len(s)==1 else np.NaN
                                                                             for s in septrace_index])

    def sepindex_to_traceindex(self, sepindex):
        """ Give the index of an sep entry. This will give the corresponding trace index """
        if not self.has_matchedindex():
            raise AttributeError("spectral match traces has not been matched with sep indexes. Run match_tracematch_and_sep()")
        return [traceindex for traceindex, sepindexes in self.matchedindex.items() if sepindex in sepindexes]
    
    def traceindex_to_sepindex(self, traceindex):
        """ Give the index of an sep entry. This will give the corresponding trace index """
        if not self.has_matchedindex():
            if not _HAS_SHAPELY:
                raise AttributeError("spectral match traces has not been matched with sep indexes. Run match_tracematch_and_sep() or install shapely for onflight tools")
            from shapely import geometry, vectorized
            x,y,a,b,theta = self.sepobjects.get(["x","y","a","b",",theta"]).T
            self.matchedindex[traceindex] = np.argwhere(vectorized.contains( geometry.Polygon(self.tracematch.get_trace_vertices(traceindex)), x,y)).ravel()
        
        return self.matchedindex[traceindex]

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
        _cannot_finetune = False
        try:
            x, y, b = self.sepobjects.get(["x","y","b"], self.traceindex_to_sepindex(traceindex)).T
        except:
            _cannot_finetune = True
            x,y,b = [],[],[]
        
        if len(x) < polydegree:
            warnings.warn("less sep-points than plynom degree. Degree reduced ")
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

    def extract_spectrum(self, traceindex, cubesolution, lbda=None, kind="cubic",
                             get_spectrum=True, finetune=False):
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
            
        cubesolution: [CubeSolution]
            Object containing the method to go fromn pixels to lbda

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
        f = self.get_spectrum(traceindex, finetune=finetune)
        v = self.get_spectrum(traceindex, finetune=finetune, on="var") if self.has_var() else None
        pixs = np.arange(len(f))[::-1]
        minpix, maxpix = self.tracematch.get_trace_xbounds(traceindex)
        mask = pixs[(pixs>minpix)* (pixs<maxpix)][::-1]
        if lbda is not None:
            pxl_wanted = cubesolution.lbda_to_pixels(lbda, traceindex)
            flux = interp1d(pixs[mask], f[mask], kind=kind)(pxl_wanted)
            var  = interp1d(pixs[mask], v[mask], kind=kind)(pxl_wanted) if v is not None else v
        else:
            
            lbda = cubesolution.pixels_to_lbda(pixs[mask], traceindex)
            flux = f[mask]
            var  = v[mask]
        
        if get_spectrum:
            from pyifu.spectroscopy import Spectrum
            spec = Spectrum(None)
            spec.create(flux,variance=var,lbda=lbda)
            return spec
        
        return lbda, flux, var
        
    # --------------- #
    #  Extract Cube   #
    # --------------- #
    def extract_cube(self, wavesolution, lbda,
                         hexagrid=None, traceindexes=None,
                         finetune_trace=False, show_progress=False):
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
        
        finetune_trace: [bool] -optional-
            DECREPEATED
            Should this use the finetuning traces for the masking of the ccd.

        show_progress: [bool] -optional-
            Should the progress within the loop over traceindexes should be 
            shown (using astropy's ProgressBar)

        Returns
        -------
        SEDMCube (child of pyifu's Cube)
        """
        from .sedm import SEDMSPAXELS, SEDMCube
        
        # - index check
        if traceindexes is None:
            traceindexes = np.sort(wavesolution.wavesolutions.keys())
        elif np.any(~np.in1d(traceindexes, wavesolution.wavesolutions.keys())):
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
                                                                get_spectrum=False, finetune=finetune_trace)
                cubeflux_[i_] = flux_
                if cubevar_ is not None:
                    cubevar_[i_] = variance_
            except:
                warnings.warn("FAILED FOR %s. Most likely, the requested wavelength are not fully covered by the trace"%i_)
                cubeflux_[i_] = np.zeros(len(lbda))*np.NaN
                if cubevar_ is not None:
                    cubevar_[i_] = np.zeros(len(lbda))*np.NaN
                    
        # ------------ #
        # - MultiThreading to speed this up
        if show_progress:
            from astropy.utils.console import ProgressBar
            ProgressBar.map(_build_ith_flux_, used_indexes)
        else:
            #from multiprocessing import Pool as ThreadPool
            #pool = ThreadPool(4)
            #pool.map(_build_ith_flux_, used_indexes)
            _ = [_build_ith_flux_[i] for i in used_indexes]
            
        cubeflux = np.asarray([cubeflux_[i] for i in used_indexes])
        cubevar  = np.asarray([cubevar_[i]   for i in used_indexes]) if cubevar_ is not None else None
        
        # - Fill the Cube
        spaxel_map = {i:c
                for i,c in zip(used_indexes,
                    np.asarray(hexagrid.index_to_xy(hexagrid.ids_to_index(used_indexes),invert_rotation=True)).T)
                     }
            
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
        from .utils.mpl import figout

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
        return self.matchedindex is not None

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
    #  Main Tools        #
    # ================== #
    def get_tracematch(self, bound_pixels=None, width="optimal"):
        """ """
        x, y, b, theta = self.sepobjects.get(["x","y","b","theta"]).T
        if bound_pixels is None:
            from .sedm import DOME_TRACEBOUNDS
            bound_pixels = DOME_TRACEBOUNDS

        if width is None:
            width = np.median(b)*2 if width=="median" else \
              b*2 if width!="optimal" else np.clip(b,np.median(b)-2*np.std(b),np.median(b)+2*np.std(b))*2
          
        xlim = np.asarray([x-bound_pixels[0], x+bound_pixels[1]])
        ylim = np.sin(theta)*np.asarray([[-bound_pixels[0], bound_pixels[1]]]).T + y
        
        return  [np.concatenate(v_) for v_ in zip(np.asarray([xlim,ylim+width]).T,np.asarray([xlim[::-1],ylim[::-1]-width]).T)]
        
        
        
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
        print "_get_sep_extract_threshold_ called"
        
        if not hasattr(self,"_sepbackground"):
                _ = self.get_sep_background(update_background=False)
        return self._sepbackground.globalrms*2

    
