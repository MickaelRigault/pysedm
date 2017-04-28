#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" This modules is made to find and handle the position on the spectral traces on the CCDs. """

import warnings
import numpy as np
import matplotlib.pyplot as mpl
from scipy import sparse

from propobject import BaseObject
from .ccd import get_dome, ScienceCCD
from .utils.tools import kwargs_update

try:
    from skimage import measure
    _HAS_SKIMAGE = True
except ImportError:
    warnings.warn("skimage is not available. Tracing wont use subpixelisation. Moire pattern to be expected.")
    _HAS_SKIMAGE = False
    
# ------------------------------- #
#   Attribute SEDM specific       #
# ------------------------------- #
PURE_DOME_ELLIPSE_SCALING = dict(scaleup_red=2.5,  scaleup_blue=9, fixed_pixel_length=275)

# ------------------------------- #
#   PIL Masking tricks            #
# ------------------------------- #
EDGES_COLOR  = mpl.cm.binary(0.99,0.5, bytes=True)
SPECTID_CMAP = mpl.cm.viridis
BACKCOLOR    = (0,0,0,0)
ZOOMING      = 5 if _HAS_SKIMAGE else 1

__all__ = ["load_specmatcher","get_specmatcher"]


def load_tracematcher(tracematchfile):
    """ Build the spectral match object based on the given data.

    Parameters
    ----------
    specmatchfile: [string]
        Path to the .pkl file containing the SpectralMatch data.
        The data must be a dictionary with the following format:
           - {vertices: [LIST_OF_SPECTRAL_VERTICES],
              arclamps: {DICT CONTAINING THE ARCLAMPS INFORMATION IF ANY} -optional-
              }
    Returns
    -------
    SpectralMatch
    """
    smap = TraceMatch()
    smap.load(specmatchfile)
    return smap

def load_specmatcher(specmatchfile):
    """ Build the spectral match object based on the given data.

    Parameters
    ----------
    specmatchfile: [string]
        Path to the .pkl file containing the SpectralMatch data.
        The data must be a dictionary with the following format:
           - {vertices: [LIST_OF_SPECTRAL_VERTICES],
              arclamps: {DICT CONTAINING THE ARCLAMPS INFORMATION IF ANY} -optional-
              }
    Returns
    -------
    SpectralMatch
    """
    smap = SpectralMatch()
    smap.load(specmatchfile)
    tmap = TraceMatch()
    tmap.set_trace_vertices(smap.spectral_vertices, build_masking=False)
    tmap._side_properties['trace_masks'] = smap.idxmask
    return tmap

def get_tracematcher(domefile, build_masking=False, **kwargs):
    """ build the spectral match object on the domefile data. 

    Parameters
    ----------
    domefile: [string]
        Location of the file containing the object
        
    **kwargs goes to the dome method `get_specrectangles` 
    
    Returns
    --------
    SpectralMatch

    """
    # - The Spectral Matcher
    smap = TraceMatch()
    # Dome Data
    dome = get_dome(domefile, background=0)
    dome.sep_extract(thresh=np.nanstd(dome.rawdata))

    # - Initial Guess based on the dome flat.
    prop = kwargs_update( PURE_DOME_ELLIPSE_SCALING, **kwargs)
    smap.set_trace_vertices( dome.get_specrectangles( **prop), build_masking=build_masking)
    
    return smap
    
def polygon_mask(vertices, width=2047, height=2047,
                 facecolor=None, edgecolor=EDGES_COLOR,
                 get_fullcolor=False):
    """ """
    from PIL import Image, ImageDraw
    back = Image.new('RGBA', (width, height), BACKCOLOR)
    mask = Image.new('RGBA', (width, height))
    # PIL *needs* (!!) [(),()] format [[],[]] wont work
    if not hasattr(vertices[0][0],"__iter__"):
        vertices = [vertices]
        
    if facecolor is None:
        npoly = len(vertices)
        facecolor = SPECTID_CMAP(np.linspace(0,1,len(vertices)), bytes=True)
        
    if edgecolor is None or np.shape(edgecolor) != np.shape(facecolor):
        edgecolor = [EDGES_COLOR]*len(vertices)
        
    [ImageDraw.Draw(mask).polygon( [(x_[0]+0.5,x_[1]+0.5) for x_ in vert],
                             fill=tuple(fc),
                             outline=tuple(ec))
     for fc,ec,vert in zip(facecolor, edgecolor, vertices) if len(vertices)>0 ]
        
    back.paste(mask,mask=mask)
    return np.sum(np.array(back), axis=2) if not get_fullcolor else back


def illustrate_traces(ccdimage, spectralmatch,
                     savefile=None, show=True,
                     facecolor=None, edgecolor=None, logscale=True,
                      cmap=None, vmin=None, vmax=None,
                      **kwargs):
    """ """
    from .utils.mpl import figout
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    if cmap is None:cmap=mpl.cm.viridis
    if facecolor is None: facecolor= mpl.cm.bone(0.5,0.3)

    if vmin is None: vmin = np.log10(np.percentile(ccdimage.data, 1 )) if logscale else np.percentile(ccdimage.data, 1)
    if vmax is None: vmax = np.log10(np.percentile(ccdimage.data, 99)) if logscale else np.percentile(ccdimage.data, 99)
        
    # ---------
    fig = mpl.figure(figsize=[10,10])
    
    ax  = fig.add_subplot(111)
    ax.set_xlabel("x [ccd-pixels]", fontsize="large")
    ax.set_ylabel("y [ccd-pixels]", fontsize="large")

    ax.set_title("Spectral Match illustrated on top of %s"%ccdimage.objname,
                     fontsize="large")
    
    
        
    def show_it(ax_, show_poly=True):        
        pl = ccdimage.show(ax=ax_,
                           cmap=cmap,vmin=vmin, vmax=vmax, logscale=logscale,
                           show=False, savefile=None, **kwargs)
        if show_poly:
            spectralmatch.display_polygon(ax_, fc=facecolor, ec=edgecolor)
    # ---------
    
    show_it(ax, False)
    
    # - Top Left 
    axinsTL = zoomed_inset_axes(ax, 6, bbox_to_anchor=[0.3,0.8,0.7,0.1],
                                  bbox_transform=ax.transAxes)  # zoom = 6
    axinsTL.set_xlim(400, 700)
    axinsTL.set_ylim(1900, 2000)
    show_it(axinsTL)
    mark_inset(ax, axinsTL, loc1=2, loc2=1, fc="none", ec="k", lw=1.5)

    
    # - Mid Right 
    axinsMR = zoomed_inset_axes(ax, 6, bbox_to_anchor=[0.23,0.3,0.7,0.1],
                                  bbox_transform=ax.transAxes)  # zoom = 6
    axinsMR.set_xlim(1250, 1550)
    axinsMR.set_ylim(900, 1000)
    show_it(axinsMR)
    mark_inset(ax, axinsMR, loc1=2, loc2=1, fc="none", ec="k", lw=1.5)
    for axins in [axinsMR,axinsTL]:
        [s.set_linewidth(1.5) for s in axins.spines.values()]
        axins.set_xticks([])
        axins.set_yticks([])
    
    fig.figout(savefile=savefile, show=show)


def get_boxing_polygone(x, y, rangex, width, 
                        dy=0.1, polydegree=2, get_vertices=False):
    """ """
    from shapely import geometry
    import modefit
    if not hasattr(dy,"__iter__"): dy = np.ones(len(y))*dy
    pfit = modefit.get_polyfit(x, y, dy, degree=polydegree)
    pfit.fit(a0_guess=np.median(y))
    
    ymodel = pfit.model.get_model(rangex)
    vertices = zip(rangex,ymodel+width) + zip(rangex[::-1],ymodel[::-1]-width)
    if get_vertices:
        return vertices
    
    return  geometry.polygon.Polygon(vertices)


#####################################
#                                   #
#  Spectral Matching Class          #
#                                   #
#####################################
class TraceMatch( BaseObject ):
    """ Object Containing the Relation between trace-ID and trace Location. 
    This object is used to extract trace from the CCD.

    traceindex = ID of the trace (as in _tracecolor and trace_vertices)
    
    """
    PROPERTIES         = ["trace_vertices","subpixelization"]
    SIDE_PROPERTIES    = ["trace_masks"]
    DERIVED_PROPERTIES = ["tracecolor", "facecolor","maskimage","rmap","gmap","bmap"]


    # ===================== #
    #   Main Methods        #
    # ===================== #
    def writeto(self, savefile, savemasks=True):
        """ dump the current object inside the given file. 
        This uses the pkl format.
        
        Parameters
        ----------
        savefile: [string]
            Fullpath of the filename where the data will be saved.
            (shoulf be a FILENAME.pkl)

        savemasks: [bool] -optional-
            shall all the currently loaded idx masking be saved?

        Returns
        -------
        Void
        """
        from .utils.tools import dump_pkl
        data= {"vertices": self.trace_vertices,
               "trace_masks": self.trace_masks if savemasks else None}
            
        dump_pkl(data, savefile)

    def load(self, filename, build_masking=False):
        """ Build the spectral match object based on the given data.

        Parameters
        ----------
        filename: [string]
            Path to the .pkl file containing the SpectralMatch data.
            The data must be a dictionary with the following format:
            - {vertices: {dict containing the LIST_OF_SPECTRAL_VERTICES},
               idxmasks: {dict containing the weighted maps (sparse matrices)}
               }

        build_masking: [bool] -optional-
            Shall the whole mask building tools be set?
            This will be needed if you later on request of trace_masks that is not
            already stored.
            
        Returns
        -------
        Void
        """
        from .utils.tools import load_pkl
        data = load_pkl(filename)
        if "vertices" not in data.keys():
            raise TypeError("The given filename does not have the appropriate format. No 'vertices' entry.")

        self.set_trace_vertices(data["vertices"], build_masking=build_masking)
        
        if "trace_masks" in data.keys():
            self._side_properties['trace_masks'] = data["trace_masks"]
            
            
    # --------- #
    #  SETTER   #
    # --------- #
    def set_trace_vertices(self, vertices, traceindexes=None,
                              build_masking=False, **kwargs):
        """ """
        if not type(vertices) == dict:
            if traceindexes is None:
                traceindexes = np.arange(len(vertices))
            
            self._properties["trace_vertices"] = {i:np.asarray(v) for i,v in zip(traceindexes, vertices)}
        else:
            self._properties["trace_vertices"] = vertices
        
        if build_masking:
            self.build_tracemasking(**kwargs)

    def set_trace_masks(self, masks, traceindexes):
        """ Attach to the current instance masks.
        """
        if hasattr(traceindexes, "__iter__"):
            if len(masks) != len(traceindexes):
                raise ValueError("masks and traceindexes do not have the same size.")
            for i,v in zip(traceindexes, masks):
                self.trace_masks[i] = v
        else:
            self.trace_masks[traceindexes] = masks
        
    # --------- #
    #  GETTER   #
    # --------- #
    def get_trace_vertices(self, traceindex):
        """ traceindex -> vertices 
        
        returns the vertices of the given traceindex

        Returns
        -------
        array (vertices)
        """
        return self.trace_vertices[traceindex]

    def get_finetuned_trace_vertices(self, traceindex, x, y, width, polydegree=2, **kwargs):
        """ The builds a fine tuned trace of the given traceindex.
        The tuning uses x, y position and build a polygon around.
        => You must have run match_tracematch_and_sep()
        
        Parameters
        ----------
        traceindex: [int]
            Index of the trace you want to fine tune

        x, y: [array, array]
            Position around which the polygon will be built.
            
        polydegree: [positive-int] -optional-
            Degree of the polynome that will be used to define the trace.
            (See 'width' for details on the width of the trace polygon)
            => If polydegree is higher than the number of sep object detected
               belonging to this trace, polydegree will the reduced to that number
            => If polydegree ends up being lower of equal to 0, None is returned

        width: [float / None] -optional-
            Width of the polygon (in pixels)

        **kwargs goes to spectralmatching.get_boxing_polygone() 
        """
        xbounds = np.percentile(np.asarray(self.trace_vertices[traceindex]).T[0], [0,100])
        prop = kwargs_update( dict(dy=1), **kwargs )
        return np.asarray(get_boxing_polygone(x, y, rangex=np.linspace(xbounds[0], xbounds[1], polydegree+5),
                                    width= width, polydegree=polydegree, get_vertices=True, **prop))
    
    def get_trace_mask(self, traceindex, update=True, rebuild=False, updateonly=False):
        """ traceindex -> color 
        get a weight mask for the given trace.

        Parameters
        ----------

        Returns
        -------
        Weighting mask (or None if both update and updateonly)
        """
        # - Do you want the one that already exists (if any)?
        if traceindex in self.trace_masks and not rebuild:
            return self.trace_masks[traceindex].toarray()
        
        # - Let's build the mask
        r, g, b, a = self._tracecolor[traceindex]
        mask = ((self._rmap==r)*(self._gmap==g)*(self._bmap==b)).reshape(*self._mapshape)
        final_mask = mask if self.subpixelization == 1 else \
          measure.block_reduce(mask, (self.subpixelization, self.subpixelization) )/float(self.subpixelization**2)

        # - Shall we save it?
        if update:
            self.set_trace_masks(sparse.csr_matrix(final_mask), traceindex)
            if updateonly:
                del final_mask
                return
            
        return final_mask

    def get_notrace_mask(self):
        """ a 2D boolean mask that is True for places in the CCD without trace. """
        if self._maskimage is None:
            self.build_tracemasking(5)
            
        mask = (self._rmap > 0 ).reshape(*self._mapshape)
        final_mask =  mask if self.subpixelization == 1 else \
          measure.block_reduce(mask, (self.subpixelization, self.subpixelization) )/float(self.subpixelization**2)

        return ~np.asarray(final_mask, dtype="bool")
    
    def get_trace_source(self, x, y, a=1, b=1, theta=0):
        """ ccdpixels -> traceindex
        
        The method get the RGBA values within an ellipe centered in `x` and `y` 
        with a major and minor axes length `a` and `b` and angle `theta`.
        The index of the traces that maximize the color overlap is then returned.
        
        Parameters
        ----------
        x, y: [float, float]
            x and y central position of the ellipses

        a, b: [float, float] -optional-
            major and minor axis lengths
        
        theta: [float] -optional-
            rotation angle (in rad) of the ellipse.
        
        Returns
        -------
        traceindex
        """
        try:
            from sep import mask_ellipse
        except:
            raise ImportError("You need sep (Python verion of Sextractor) to run this method => sudo pip install sep")
        
        masking = np.zeros(self._mapshape, dtype="bool")
        mask_ellipse(masking, x*self.subpixelization,
                    y*self.subpixelization, a*self.subpixelization,
                    b*self.subpixelization, theta=theta)
        if not np.any(masking):
            return np.asarray([])
        
        sum_mask = np.sum( np.sum([(self._facecolors.flatten()==v)
                               for v in np.concatenate(self._maskimage.T.T[masking])],
                            axis=0).reshape(self._facecolors.shape), axis=1)
        
        return np.asarray(self.trace_indexes)[sum_mask==sum_mask.max()] if sum_mask.max()>0 else np.asarray([])

    def get_traces_within_polygon(self, polyverts):
        """ Which traces are fully contained within the given polygon (defined by the input vertices) 
        
        Parameters
        ----------
        polyverts: [2D-array]
            Coordinates of the points defining the polygon.

        Returns
        -------
        list of trace indexes
        """
        try:
            import shapely
            _HAS_SHAPELY = True
        except ImportError:
            warnings.warn("You do not have Shapely. get_trace_within_polygon() will use matplotlib but Shapely would be faster")
            _HAS_SHAPELY = False

        if _HAS_SHAPELY:
            from shapely.geometry import Polygon
            globalpoly = Polygon(polyverts)
            return [idx_ for idx_ in self.trace_indexes if globalpoly.contains(Polygon(self.trace_vertices[idx_]))]
        else:
            from matplotlib import patches
            globalpoly = patches.Polygon(polyverts)
            return [idx_ for idx_ in self.trace_indexes if
                    np.all([globalpoly.contains_point(vtr) for vtr in self.trace_vertices[idx_]])]

    # --------- #
    #  PLOTTER  #
    # --------- #
    def display_traces(self, ax, traceindex, facecolors="None", edgecolors="k",
                           autoscale=True,**kwargs):
        """ """
        from matplotlib import patches
        # Several indexes given
        if hasattr(traceindex, "__iter__"):
            if not hasattr(facecolors, "__iter__"):
                facecolors = [facecolors]*len(traceindex)
            if not hasattr(edgecolors, "__iter__"):
                edgecolors = [edgecolors]*len(traceindex)
                
            ps = [patches.Polygon(self.trace_vertices[idx_],
                      facecolor=facecolors[i], edgecolor=edgecolors[i], **kwargs)
                    for i,idx_  in enumerate(traceindex)]
            ip = [ax.add_patch(p_) for p_ in ps]
        # One index given
        else:
            p_ = patches.Polygon(self.trace_vertices[traceindex],
                                facecolor=facecolors, edgecolor=edgecolors, **kwargs)
            ip = [ax.add_patch(p_)]
                                     
        if autoscale:
            ax.autoscale(True, tight=True)
            
        return ip
            
        
    # ------------ #
    #  Methods     #
    # ------------ #
    def build_tracemasking(self, subpixelization=1, width=2047, height=2047):
        """ 
        This will build the internal tools to identify the connections between
        traceindex and ccd-pixels
        
        Returns
        -------
        Void
        """
        self._properties["subpixelization"] = subpixelization

        # - 1 Trace, 1 Color !
        nonunique_RGBA         = np.asarray(zip(np.random.randint(5,90, size=self.ntraces*3),
                                                    np.random.randint(91,175, size=self.ntraces*3),
                                                    np.random.randint(176,254, size=self.ntraces*3),
                                                    [255]*self.ntraces*3))
        
        b = np.ascontiguousarray(nonunique_RGBA).view(np.dtype((np.void,nonunique_RGBA.dtype.itemsize * nonunique_RGBA.shape[1])))
        # This is made for faster identification later on
        self._derived_properties['facecolor']  = nonunique_RGBA[np.unique(b, return_index=True)[1]][:self.ntraces]
        self._derived_properties['tracecolor'] = {i:c for i,c in zip(self.trace_indexes,self._facecolors)}
        verts = [self.trace_vertices[i]*self.subpixelization for i in self.trace_indexes]

        self._derived_properties['maskimage'] = \
          np.asarray(polygon_mask(verts, width*self.subpixelization, height*self.subpixelization,
                           facecolor=self._facecolors, edgecolor=self._facecolors, get_fullcolor=True))

        # - Save detailed information for matching later on
        r, g, b, a = self._maskimage.T        
        self._derived_properties['mapshape'] = (width*self.subpixelization, height*self.subpixelization)
        self._derived_properties['rmap'] = r.ravel(order='F')
        self._derived_properties['gmap'] = g.ravel(order='F')
        self._derived_properties['bmap'] = b.ravel(order='F')

    def extract_hexgrid(self, traceindexes = None, qdistance=None):
        """ Build the array of neightbords.
        This is built on a KDTree (scipy)

        Parameters
        ----------
        usedindex: [list of indexes] -optional-
            Select the indexes you want to use. If None [default] all will be.

        Returns
        -------
        HexagoneProjection (from pysedm.utils.hexagrid)
        """
        from .utils.hexagrid import get_hexprojection
        
        if traceindexes is None:
            traceindexes = self.trace_indexes

        # - position used to define 1 location of 1 spectral_trace
        xydata  = np.asarray([np.nanmean(self.trace_vertices[idx_], axis=0) for idx_ in traceindexes])
        return get_hexprojection(xydata, ids=traceindexes)
    
    # ===================== #
    #   Properties          #
    # ===================== #
    # - Traces I/O
    @property
    def trace_vertices(self):
        """ dictionary containing the Polygon Vertices for the traces. """
        if self._properties["trace_vertices"] is None:
            self._properties["trace_vertices"] = {}
        return self._properties["trace_vertices"]

    @property
    def trace_indexes(self):
        """ Indexes associated with to the traces """
        return self.trace_vertices.keys()
        
    @property
    def ntraces(self):
        """ Number of traces loaded """
        return len(self.trace_indexes)
    
    # ------------ #
    #  Masks       #
    # ------------ #
    @property
    def trace_masks(self):
        """ Weightmap corresponding for the traces in the CCD.
        These are save as dictionary of sparse matrices
        """
        if self._side_properties['trace_masks'] is None:
            self._side_properties['trace_masks'] = {}
        return self._side_properties['trace_masks']
    
    # ------------ #
    #  Matching    #
    # ------------ #
    @property
    def subpixelization(self):
        """ Subpixelization used to draw trace polygon on the image """
        return self._properties["subpixelization"]

    # - Color Tools For Trace identification
    @property
    def _tracecolor(self):
        """ dictionary containing the traceindex <-> RGBA connection """
        return self._derived_properties['tracecolor']
    
    @property
    def _facecolors(self):
        """ trace color in array version """
        return self._derived_properties['facecolor']
        
    @property
    def _maskimage(self):
        """ RGBA image of the trace_vertices. 
        See _tracecolor for the traceindex<->RGBA color connection
        """
        return self._derived_properties['maskimage']
    
    @property
    def _mapshape(self):
        """ width / height of the maskimage """
        return self._derived_properties['mapshape']
        
    @property
    def _rmap(self):
        """ (flatten) R map of RGBA image (see _maskimage) """
        return self._derived_properties['rmap']
    
    @property
    def _gmap(self):
        """ (flatten) G map of RGBA image (see _maskimage)  """
        return self._derived_properties['gmap']
    
    @property
    def _bmap(self):
        """ (flatten) B map of RGBA image (see _maskimage)  """
        return self._derived_properties['bmap']
    

class SpectralMatch( BaseObject ):
    """ """
    PROPERTIES         = ["spectral_vertices","vertices_indexes"]
    SIDE_PROPERTIES    = ["arclamp", "zooming"]
    DERIVED_PROPERTIES = ["maskimage","specpoly","spectral_fc",
                           "rmap", "gmap", "bmap", "amap",
                           "rmap_flat", "gmap_flat", "bmap_flat", "amap_flat", # To speed up
                           "idxmasks"]
    
    # ================== #
    #  Main Tools        #
    # ================== #
    # ----------- #
    #  I/O        #
    # ----------- #
    def writeto(self, savefile, savearcs=True, savemasks=True):
        """ dump the current object inside the given file. 
        This uses the pkl format.
        
        Parameters
        ----------
        savefile: [string]
            Fullpath of the filename where the data will be saved.
            (shoulf be a FILENAME.pkl)

        savearcs: [bool] -optional-
            shall the arclamps attribute be saved?
            It is recommended.

        savemasks: [bool] -optional-
            shall all the currently loaded idx masking be saved?

        Returns
        -------
        Void
        """
        from .utils.tools import dump_pkl
        data= {"vertices": self.spectral_vertices,
               "arclamps": self.arclamps if savearcs else None,
               "idxmasks": self.idxmask if savemasks else None}
            
        dump_pkl(data, savefile)


    def load(self, filename):
        """ Build the spectral match object based on the given data.

        Parameters
        ----------
        filename: [string]
            Path to the .pkl file containing the SpectralMatch data.
            The data must be a dictionary with the following format:
            - {vertices: [LIST_OF_SPECTRAL_VERTICES],
               arclamps: {DICT CONTAINING THE ARCLAMPS INFORMATION IF ANY} -optional-
               }
        Returns
        -------
        Void
        """
        from .utils.tools import load_pkl
        data = load_pkl(filename)
        if "vertices" not in data.keys():
            raise TypeError("The given filename does not have the appropriate format. No 'vertices' entry.")

        self.set_specrectangle(data["vertices"])
        
        if "arclamps" in data.keys() and data["arclamps"] is not None:
            self._side_properties["arclamp"] = data["arclamps"]
        
        if "idxmasks" in data.keys():
            self._derived_properties["idxmasks"] = data["idxmasks"]
        
    # ----------- #
    #  Structure  #
    # ----------- #
    def reset(self):
        """ """
        self._properties['spectral_vertices'] = None
        for k in self.DERIVED_PROPERTIES:
            self._derived_properties[k] = None

    # ----------- #
    #  Builder    #
    # ----------- #
    def set_specrectangle(self, spectral_vertices, indexes=None,
                              width=2047, height=2047):
        """ """
        self.reset()
        if indexes is None:
            indexes = np.arange(len(spectral_vertices))
            
        self._properties['spectral_vertices'] = np.asarray(spectral_vertices)
        self._properties['vertices_indexes']  = np.asarray(indexes)
        
        self.build_maskimage(width=width, height=height)

    def build_maskimage(self, width=2047, height=2047):
        """ """
        self._derived_properties['maskimage'] = \
          polygon_mask([np.asarray(poly.exterior.xy).T*self._zooming for poly in self.spectral_polygon],
                           width*self._zooming, height*self._zooming,
                           facecolor=self.spectral_facecolor,
                           edgecolor=self.spectral_facecolor,
                           get_fullcolor=True)
        
        self._derived_properties['rmap'],self._derived_properties['gmap'],self._derived_properties['bmap'],self._derived_properties['amap'] = np.asarray(self.maskimage).T
        
    def extract_hexgrid(self, usedindex = None, qdistance=None):
        """ Build the array of neightbords.
        This is built on a KDTree (scipy)

        Parameters
        ----------
        usedindex: [list of indexes] -optional-
            Select the indexes you want to use. If None [default] all will be.

        """
        from shapely.geometry import MultiPoint
        from .utils.hexagrid import get_hexprojection
        
        # - Index game
        all_index = np.arange(self.nspectra)
        if usedindex is None:
            usedindex = all_index
        # Careful until the end of this method
        # The indexes will be index of 'usedindex'
        # they will have the sidx_ name

        # - position used to define 1 location of 1 spectral_trace
        # spectra_ref = [np.asarray(self.spectral_polygon[int(i)].centroid.xy).T[0] for i in usedindex]
        spectra_ref          = MultiPoint([self.spectral_polygon[int(i)].centroid for i in usedindex]) # shapely format
        xydata  = np.asarray([np.concatenate(c.xy) for c in spectra_ref]) # array format for kdtree
        
        return get_hexprojection(xydata, ids=usedindex)

    # -------------- #
    #  Fetch Index   #
    # -------------- #
    def pixel_to_index(self, x, y, around=1):
        """ The method get the RGBA values of the `x` and `y` pixels and all the one `around` 
        (forming a square centered on `x` `y`) and returns the median index value.
        Returns
        -------
        int or None (if the x,y area is not attached to any index)
        """
        x,y,around = x*self._zooming, y*self._zooming, around*self._zooming
        allindex = [i for i in np.concatenate([[self.color_to_index(self._rmap[x_,y_],
                                                self._gmap[x_,y_],
                                                self._bmap[x_,y_]) 
                      for x_ in np.arange(x-around,x+around)]
                     for y_ in np.arange(y-around,y+around)])
                     if i is not None]
            
        return np.median(allindex) if len(allindex)>0 else np.NaN
    
    def color_to_index(self, r,g,b,a=None):
        """ get the color associated with the given color list.
        If None is found, this returns None, it returns the index otherwise.

        Returns
        -------
        int or None
        """
        index = np.argwhere((self.spectral_facecolor.T[0]==r)*(self.spectral_facecolor.T[1]==g)*(self.spectral_facecolor.T[2]==b))
        return self.trace_indexes[index[0][0]] if len(index)>0 else None
    
    # ----------- #
    #  GETTER     #
    # ----------- #
    def get_idx_color(self, index):
        """ """
        return self._index_to_(index, "facecolor")

    def get_spectral_mask(self, format="array"):
        """ General masking of global area.
        Where are the traces in the CCD

        Parameters
        ----------
        format: ["array","sparse"] -optional-
            how would you like your mask to be returned? 
            - array: 2D weight array
            - sparse: Sparse matrix of the array (scipy.sparse.csr_matrix) 
                      if so simply do '.toarray()' to convert it back to array.

        Returns
        -------
        2d-array bool
        """
        mask = (self._rmap.T > 0 )
        mask_= mask if self._zooming == 1 else \
          measure.block_reduce(mask, (self._zooming, self._zooming) )/float(self._zooming**2)

        if format in ["array"]:
            return mask_
        elif format in ["sparse","sparsematrix"]:
            return sparse.csr_matrix(mask_)
        
        raise ValueError("Unknown format %s (use 'array' or 'sparse')"%format)
    
    def get_idx_mask(self, index, format="array", rebuild=False):
        """ Weight mask of the corresponding index trace on the CCD.
        The weighted will depend on the 'zooming' used. If 1, then the weight mask
        will only be 0 and 1, if 'zooming'>1 then sub pixelization will be used to 
        identify the overlap between polygon. In that latter case, weight will be values
        between 0 and 1 (including 0 and 1). 
        (See self._zooming for the current zooming)

        Parameters
        ----------
        index: [int]
            Id of the trace.

        format: ["array","sparse"] -optional-
            how would you like your mask to be returned? 
            - array: 2D weight array
            - sparse: Sparse matrix of the array (scipy.sparse.csr_matrix) 
                      if so simply do '.toarray()' to convert it back to array.
        rebuild: [bool] -optional-
            If the mask for the requested index is saved in self.idxmask, shall this use the 
            saved value (rebuild=False) or shall this rebuild it (rebuild=True)
        Returns
        -------
        2D array or sparse matrix (see format option)
        """
        # - Composite mask from multiple indexes
        if hasattr(index,"__iter__"):
            return np.asarray(np.sum([ self.get_idx_mask(i_, include=include)
                                           for i_ in index], axis=0), dtype="bool")
        # Load existing mask
        # ------------------
        if not rebuild and index in self.idxmask.keys():
            if format in ["array"]:
                return self.idxmask[index].toarray()
            elif format in ["sparse","sparsematrix"]:
                return self.idxmask[index]
            raise ValueError("Unknown format %s (use 'array' or 'sparse')"%format)
        
        # Build the mask
        # --------------
        fr, fg, fb, fa = self._index_to_(index, "facecolor")
        mask = (((self._rmap_flat==fr)*(self._bmap_flat==fb)*(self._gmap_flat==fg))).reshape(self._rmap.shape)

        # Resample it.
        mask_ = mask if self._zooming == 1 else \
          measure.block_reduce(mask, (self._zooming, self._zooming) )/float(self._zooming**2)
        
        if format in ["array"]:
            return mask_
        elif format in ["sparse","sparsematrix"]:
            return sparse.csr_matrix(mask_)
        raise ValueError("Unknown format %s (use 'array' or 'sparse')"%format)

    def load_idx_masks(self, indexes=None, show_progress=False):
        """ Build the 'idx mask' for the given indexes and load them to the `idxmask` attribute
        
        Parameters
        ----------
        indexes: [None / list of  indexes] -optional-
            list of indexes to be loaded. If None, all the indexes will be loaded
            
        show_progress: [bool] -optional-
            If you have astropy installed you can show progression bar.
            If you do n have astropy, this will be forced to False.


        Returns
        -------
        Void
        """
        # Internal Method
        def load_idx_mask(idx_):
            self.idxmask[idx_] = self.get_idx_mask(idx_, format="sparse", rebuild=True)

        # Do we use ProgressBar ?
        if show_progress:
            try:
                from astropy.utils.console import ProgressBar
            except ImportError:
                warning.warn(ImportError("astropy not installed. No ProgressBar available"))
                show_progress = False

        # Which index are we going to save?
        if indexes is None: indexes = np.arange(self.nspectra)

        # -> Let's do it then!
        if show_progress:
            ProgressBar.map(load_idx_mask, indexes)
        else:
            [load_idx_mask(idx) for idx in indexes]
            
    # -------------- #
    #  Select idx    #
    # -------------- #
    def get_idx_within_bounds(self, xbounds, ybounds):
        """ Returns the list of indexes for which the indexth's polygon 
        is fully contained within xbounds, ybounds (in CCD index) """
        from shapely.geometry import Polygon
        
        rectangle = Polygon([[xbounds[0],ybounds[0]],[xbounds[0],ybounds[1]],
                             [xbounds[1],ybounds[1]],[xbounds[1],ybounds[0]]])
        
        return self.get_idx_within_polygon(rectangle)
    
    def get_idx_within_polygon(self, polygon_):
        """ Returns the list of indexes for which the indexth's polygon 
        is fully contained within the given polygon """
        return np.arange(self.nspectra)[ np.asarray([polygon_.contains(p_)
                                         for p_ in self.spectral_polygon], dtype="bool") ]
                      
    # ----------- #
    #  Plotting   #
    # ----------- #
    def display_polygon(self, ax, idx=None, **kwargs):
        """ display on the given axis the polygon used to define the spectral match """
        from astrobject.utils.shape import draw_polygon
        if idx is None:
            return ax.draw_polygon(self.spectral_polygon, **kwargs)
        if not hasattr(idx, "__iter__"):
            idx = [idx]
        return [ax.draw_polygon(self.spectral_polygon[i], **kwargs)  for i in idx]

    
    def show_traces(self, ax=None, savefile=None, show=True, cmap=None,
                 add_colorbar=True, index=None,
                 **kwargs):
        """ """
        from astrobject.utils.mpladdon import figout
        if not self.has_maskimage():
            raise AttributeError("maskimage not set. If you set the `specrectangles` run build_maskimage()")
        
        if ax is None:
            fig = mpl.figure(figsize=[10,7])
            ax  = fig.add_subplot(1,1,1)
        else:
            fig = ax.figure

        # -- Imshow
        prop = kwargs_update(dict(origin="lower", interpolation="nearest"), **kwargs)
        if index is None:
            im   =  ax.imshow(self.maskimage, **prop)
        else:
            boolmask = self.get_idx_mask(index)
            tmp = np.sum(np.asarray(self.maskimage, dtype="float"), axis=2)
            tmp[~boolmask] *= np.NaN
            im   =  ax.imshow(tmp, **prop)
            
        # -- Output
        fig.figout(savefile=savefile, show=show)

    def show_index_trace(self, index, ax=None, savefile=None,
                         show=True, legend=True, draw_polygon=True):
        """ """
        from .utils.mpl import get_lamp_color, figout
        from astrobject.utils.shape import draw_polygon
        if ax is None:
            fig = mpl.figure(figsize=[8,6])
            ax  = fig.add_subplot(1,1,1)
            ax.set_xlabel("x [ccd pixels]", fontsize="large")
            ax.set_ylabel("y [ccd pixels]", fontsize="large")
        else:
            fig = ax.figure

        for i, name in enumerate(self.arclamps.keys()):
            x,y = self.get_arcline_positions(name, index)
            ax.plot(x,y, marker="o", ms=10,
                        mfc=get_lamp_color(name, 0.5), mew=1,
                        mec=get_lamp_color(name, 0.9),
                        ls="None", label=name)

        if draw_polygon:
            if len(self.arclamps.keys())>0:
                ax.draw_polygon(self.get_arcbased_polygon(index), ec="0.5")
                
            ax.draw_polygon(self.spectral_polygon[index])
            
            
        if legend:
            ax.legend(loc="upper left", frameon=False, fontsize="medium",
                        markerscale=0.6)
            
        fig.figout(savefile=savefile, show=show)


    
    # --------------- #
    #  Arc Lines      #
    # --------------- #
    def add_arclamp(self, arc, match=False):
        """ """
        if type(arc) == str:
            arc = ScienceCCD(arc, background=0)
        elif ScienceCCD not in arc.__class__.__mro__:
            raise TypeError("The given arc must be a string or an ScienceCCD object")

        if not arc.has_sepobjects():
            arc.sep_extract(thresh=np.nanstd(arc.rawdata))
            
        x,y,a,b,t = arc.sepobjects.get(["x","y","a","b","theta"]).T
        self.arclamps[arc.objname] = {"arcsep":{"x":x,"y":y,"a":a,"b":b,"t":t},
                                      "index":None}
        
        # - shall the matching be done? It takes several seconds.
        if match:
            self.match_arc(arc.objname)
            
    def match_arc(self, arcname):
        """ Match the detected (by SEP) arc emission line with 
        existing spectral features. 
        """
        self._test_arc_(arcname, test_matching=False)
        x= self.arclamps[arcname]["arcsep"]["x"]
        y= self.arclamps[arcname]["arcsep"]["y"]
        b= self.arclamps[arcname]["arcsep"]["b"]
        self.arclamps[arcname]["index"] = \
          np.asarray([self.pixel_to_index(x_,y_, around=b_*2)
                        for x_,y_,b_ in zip(x,y,b)])
    # ---------- #
    # ARC SETTER #
    # ---------- #
    def set_arcbased_specmatch(self, width=2, polydegree=2):
        """ """
        if len(self.arclamps.keys())==0:
            raise AttributeError("No lamp loaded.")

        if not hasattr(width,"__iter__"):  width = np.ones(self.nspectra)*width
            
        vertices = []
        for i in range(self.nspectra):
            try:
                newvert = self.get_arcbased_polygon(i, width=width[i],
                                            polydegree=polydegree, get_vertices=True)
            except:
                newvert = self.spectral_vertices[i]
                warnings.warn("No Vertices changed for %d"%i)
                
            vertices.append(newvert)
            
        self.set_specrectangle(vertices)
        
            
    # ---------- #
    # ARC GETTER #
    # ---------- #
    def get_arcline_positions(self, arcname, index):
        """ x and y ccd-coordinates of the detected arclines associated
        with the given `index`

        Returns
        -------
        x,y
        """
        arcindex = self.index_to_arcindex(arcname, index)
        if len(arcindex) == 0 :
            return None,None
        return self.arclamps[arcname]["arcsep"]["x"][arcindex],\
          self.arclamps[arcname]["arcsep"]["y"][arcindex]

    def get_arcbased_polygon(self, index, width=2., polydegree=2,
                                 xbounds=None, get_vertices=False,
                                 include_dome=True):
        """ """
        arc_pos_x, arc_pos_y = [], []
        for arc in self.arclamps.keys():
            x_, y_ = self.get_arcline_positions(arc,index)
            arc_pos_x.append(x_)
            arc_pos_y.append(y_)

        if xbounds is None:
            xbounds = np.percentile(np.asarray(self.spectral_vertices[index]).T[0], [0,100])
            
        if xbounds[0] is None: np.percentile(np.asarray(self.spectral_vertices[index]).T[0], 0)
        if xbounds[1] is None: np.percentile(np.asarray(self.spectral_vertices[index]).T[0], 100)
        
        return get_boxing_polygone(np.concatenate(arc_pos_x), np.concatenate(arc_pos_y), 
                            rangex= np.linspace(xbounds[0],xbounds[1],polydegree+5),
                            width=width, dy=1, polydegree=polydegree,
                            get_vertices=get_vertices)
    
    # - Index Matching
    def arcindex_to_index(self, arcname, arcindex):
        """ """
        self._test_arc_(arcname, test_matching=True)
        return self.arclamps[arcname]["index"][arcindex]

    def index_to_arcindex(self, arcname, index):
        """ """
        self._test_arc_(arcname, test_matching=True)
        index = np.argwhere(self.arclamps[arcname]["index"]==index)
        return np.concatenate(index).tolist() if len(index)>0 else []

    def _test_arc_(self, arcname, test_matching=False):
        """ """
        if arcname not in self.arclamps:
            raise AttributeError("No arclamp loaded named %s"%arcname)
        if test_matching and self.arclamps[arcname]["index"] is None:
            raise AttributeError("The matching for %d has not been made. see match_arc()"%arcname)
        
    @property
    def arclamps(self):
        """ Arc lamp associated to the spectral matcher """
        if self._side_properties["arclamp"] is None:
            self._side_properties["arclamp"] = {}
        return self._side_properties["arclamp"]
    
    # ================== #
    #  Properties        #
    # ================== #
    def _index_to_(self, index, what):
        """ """
        return eval("self.spectral_%s[index]"%what)

    # ================== #
    #  Properties        #
    # ================== #
    @property
    def nspectra(self):
        """ number of spectra loaded """
        return len(self.spectral_vertices) if self.spectral_vertices is not None else 0
    
    # -----------------
    # - Spectral Info
    @property
    def spectral_vertices(self):
        """ Vertices for the polygon defining the location of the spectra """
        return self._properties["spectral_vertices"]
    
    @property
    def spectral_polygon(self):
        """ Shapely Multi Polygon associated to the given vertices """
        from shapely import geometry
        if self._derived_properties["specpoly"] is None:
            self._derived_properties["specpoly"] = geometry.MultiPolygon([ geometry.polygon.Polygon(rect_)
                                                                            for rect_ in self.spectral_vertices])
        return self._derived_properties["specpoly"]

    @property
    def spectral_facecolor(self):
        """ Random Color Associated to the spectra """
        if self._derived_properties['spectral_fc'] is None:
            self._load_random_color_()  
        return self._derived_properties['spectral_fc']
    

    def _load_random_color_(self):
        """ """
        nonunique_RGBA         = np.random.randint(100,250,size=[self.nspectra*2,4])
        nonunique_RGBA_ec      = np.random.randint(50,99,size=[self.nspectra*2,4])
        nonunique_RGBA.T[3]    = 255 # explicit here to avoid PIL / mpl color variation
        nonunique_RGBA_ec.T[3] = 255 # explicit here to avoid PIL / mpl color variation
            
        b = np.ascontiguousarray(nonunique_RGBA).view(np.dtype((np.void,nonunique_RGBA.dtype.itemsize * nonunique_RGBA.shape[1])))
        b_ec = np.ascontiguousarray(nonunique_RGBA_ec).view(np.dtype((np.void,nonunique_RGBA_ec.dtype.itemsize * nonunique_RGBA_ec.shape[1])))
        
        self._derived_properties['spectral_fc'] = \
          nonunique_RGBA[np.unique(b, return_index=True)[1]][:self.nspectra]
          
    @property
    def maskimage(self):
        """ Masking image core of the class. """
        return self._derived_properties['maskimage']
    
    def has_maskimage(self):
        return self.has_maskimage is not None

    @property
    def idxmask(self):
        """ dictionary containing the sparse matrix of the (loaded) indexes """
        if self._derived_properties["idxmasks"] is None:
            self._derived_properties["idxmasks"] = {}
        return self._derived_properties["idxmasks"]
        
    # ----------
    # Internal
    @property
    def _zooming(self):
        """ subpixelization of the PIL tools """
        if self._side_properties["zooming"] is None:
            self._side_properties["zooming"] = ZOOMING
        return self._side_properties["zooming"]

    # Image in R
    @property
    def _rmap(self):
        return self._derived_properties["rmap"]
    
    @property
    def _rmap_flat(self):
        """ Raveled version of _rmap"""
        if self._derived_properties["rmap_flat"] is None:
           self._derived_properties["rmap_flat"] = self._rmap.ravel(order='F')
        return self._derived_properties["rmap_flat"]

    # Image in G
    @property
    def _gmap(self):
        return self._derived_properties["gmap"]
    @property
    def _gmap_flat(self):
        """ Raveled version of _gmap"""
        if self._derived_properties["gmap_flat"] is None:
           self._derived_properties["gmap_flat"] = self._gmap.ravel(order='F')
        return self._derived_properties["gmap_flat"]
    
    # Image in B
    @property
    def _bmap(self):
        return self._derived_properties["bmap"]
    @property
    def _bmap_flat(self):
        """ Raveled version of _bmap"""
        if self._derived_properties["bmap_flat"] is None:
           self._derived_properties["bmap_flat"] = self._bmap.ravel(order='F')
        return self._derived_properties["bmap_flat"]

    # Image in A
    @property
    def _amap(self):
        return self._derived_properties["amap"]
    @property
    def _amap_flat(self):
        """ Raveled version of _amap"""
        if self._derived_properties["amap_flat"] is None:
           self._derived_properties["amap_flat"] = self._amap.ravel(order='F')
        return self._derived_properties["amap_flat"]
