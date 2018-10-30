#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" This modules is made to find and handle the position on the spectral traces on the CCDs. """

import warnings
import numpy as np
import matplotlib.pyplot as mpl
from scipy import sparse
from astropy.utils.console import ProgressBar

from propobject import BaseObject
from .ccd import get_dome, ScienceCCD
from .utils.tools import kwargs_update, is_arraylike

from .sedm import SEDM_CCD_SIZE
try:
    from shapely import vectorized, geometry
    _HAS_SHAPELY = True
except:
    warnings.warn("You do not have Shapely. trace masking will be slower.")
    _HAS_SHAPELY = False
    
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
_BASEPIX     = np.asarray([[0,0],[0,1],[1,1],[1,0]])

__all__ = ["load_tracematcher","get_tracematcher"]


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
    smap.load(tracematchfile)
    return smap

def get_tracematcher(domefile, build_tracemask=False, width=None, **kwargs):
    """ build the spectral match object on the domefile data. 

    Parameters
    ----------
    domefile: [string]
        Location of the file containing the object
        
    **kwargs goes to the dome method `get_specrectangles` 
    
    Returns
    --------
    TraceMatch

    """
    from .sedm import TRACE_DISPERSION
    # - The Spectral Matcher
    smap = TraceMatch()
    # Dome Data
    dome = get_dome(domefile, background=0, load_sep=True)

    # - Initial Guess based on the dome flat.
    if width is None:
        width = TRACE_DISPERSION

    xlim, ylim = dome.get_trace_position()
    smap.set_trace_line(np.asarray([xlim, ylim]).T, width=width, build_tracemask=build_tracemask)
    return smap
    
def polygon_mask(vertices, width=2048, height=2048,
                 facecolor=None, edgecolor=EDGES_COLOR,
                 get_fullcolor=False):
    """ """
    from PIL import Image, ImageDraw
    back = Image.new('RGBA', (width, height), BACKCOLOR)
    mask = Image.new('RGBA', (width, height))
    # PIL *needs* (!!) [(),()] format [[],[]] wont work
    if not is_arraylike(vertices[0][0]):
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
            spectralmatch.display_traces(ax_, spectralmatch.trace_indexes, fc=facecolor, ec=edgecolor)
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
    if not is_arraylike(dy):
        dy = np.ones(len(y))*dy
    pfit = modefit.get_polyfit(x, y, dy, degree=polydegree)
    pfit.fit(a0_guess=np.median(y))
    
    ymodel = pfit.model.get_model(rangex)
    if width == 0:
        if not get_vertices:
            raise ValueError("A polygon cannot have a width of 0")
        return zip(rangex,ymodel+width)
    
    vertices = zip(rangex,ymodel+width) + zip(rangex[::-1],ymodel[::-1]-width)
    if get_vertices:
        return vertices
    
    return  geometry.Polygon(vertices)

# ------------------------- # 
#   MultiProcessing Tracing #
# ------------------------- #
def verts_to_mask(verts):
    """ Based on the given vertices (and using the CCD size from semd.SEDM_CCD_SIZE)
    this create a weighted mask:
    - pixels outise of the vertices have 0
    - pixels fully included within the vertices have 1
    - pixels on the edge only have a fraction of 1 
    

    = Based on Shapely = 
    
    Returns
    -------
    [NxM] array (size of semd.SEDM_CCD_SIZE)
    """
    from shapely.vectorized import contains
    verts = verts+np.asarray([0.5,0.5])
    
    xlim, ylim = np.asarray(np.round(np.percentile(verts, [0,100], axis=0)), dtype="int").T + np.asarray([-1,1])
    polytrace  = geometry.Polygon(verts)
        
    sqgrid     = np.asarray([[_BASEPIX + np.asarray([x_,y_])
                                  for y_ in np.arange(*ylim)] for x_ in np.arange(*xlim)]
                                ).reshape((ylim[1]-ylim[0]) * (xlim[1]-xlim[0]),*np.shape(_BASEPIX))
    
    maskins = vectorized.contains(polytrace, *sqgrid.T).T
    
    maskfull = np.zeros(SEDM_CCD_SIZE)
    maskfull[xlim[0]:xlim[1],ylim[0]:ylim[1]] = \
      (np.asarray([  0 if not np.any(maskin_) else 1 if np.all(maskin_) else polytrace.intersection(geometry.Polygon(sq_)).area
            for maskin_,sq_ in zip(maskins,sqgrid)]).reshape(xlim[1]-xlim[0],ylim[1]-ylim[0]))
    return maskfull.T
    

def load_trace_masks(tmatch, trace_indexes=None, multiprocess=True,
                         notebook=True, ncore=None):
    """ """
    if trace_indexes is None:
        trace_indexes = tmatch.trace_indexes
    
    if multiprocess:
        import multiprocessing
        bar = ProgressBar( len(trace_indexes), ipython_widget=notebook)
        if ncore is None:
            ncore = np.max([multiprocessing.cpu_count() - 1, 1])
                
        p = multiprocessing.Pool(ncore)
        for j, mask in enumerate( p.imap(verts_to_mask, [tmatch.trace_vertices[i_] for i_ in trace_indexes])):
            tmatch.set_trace_masks(sparse.csr_matrix(mask), trace_indexes[j])
            bar.update(j)
        bar.update( len(trace_indexes) )
        
    else:
        raise NotImplementedError("Use multiprocess = True (load_trace_masks)")

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
    PROPERTIES         = ["trace_linestring","trace_vertices","subpixelization", "trace_indexes", "width"]
    SIDE_PROPERTIES    = ["trace_masks","ij_offset", ]
    DERIVED_PROPERTIES = ["tracecolor", "facecolor", "maskimage",
                          "rmap", "gmap", "bmap",
                          "trace_polygons"]

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
        data= {"xys": self._xys,
               "trace_indexes": self.trace_indexes,
               "width": self.width,
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
        if "xys" in data.keys():
            self.set_trace_line(data["xys"], width=data["width"],
                                    trace_indexes=data["trace_indexes"], build_tracemask=False)
        elif "vertices" in data.keys():
            raise TypeError("OLDER FORMAT NOT AVAILABLE FOR THE MOMENT. WILL BE SOON. (If urgent contact Mickael) ")
            #self.set_trace_vertices(data["vertices"], build_masking=build_masking)
        else:
            raise TypeError("UNRECOGNIZED TRACEMATCH FORMAT")
        
        if "trace_masks" in data.keys():
            self._side_properties['trace_masks'] = data["trace_masks"]
            

    def add_trace_offset(self, i_offset, j_offset):
        """ """
        print("DEPRECATED use, get_shifted_tracematch ")
        new_verts = {i:v + np.asarray([i_offset, j_offset]) for i,v in self.trace_vertices.items()}
        self._side_properties['ij_offset'] = np.asarray([i_offset, j_offset])
        self.set_trace_vertices(new_verts)
        self._side_properties['trace_masks'] = None
        
    # --------- #
    #  SETTER   #
    # --------- #

    # => Baseline Definition
    
    def set_trace_line(self, xys, width=None, trace_indexes=None, build_tracemask=False):
        """ """
        if trace_indexes is None:
            trace_indexes = np.arange( len(xys) )
        self._xys = xys
        self.set_trace_indexes(trace_indexes)
        lines = [geometry.LineString( xy ) for xy in xys]
        self._properties["trace_linestring"] = {i:l for i,l in zip(self.trace_indexes, lines)}
            
        self.set_buffer(width, build_tracemask=build_tracemask)

    def set_trace_indexes(self, trace_indexes):
        """ ID of the traces (spaxels) """
        self._properties["trace_indexes"] = trace_indexes

    def set_buffer(self, width, build_tracemask=False):
        """ """
        self._properties["width"] = width
        if self._properties["trace_linestring"] is not None:
            self._update_trace_polygon_(build_tracemask)

    def _update_trace_polygon_(self, build_tracemask=False):
        """ """
        self._derived_properties["trace_polygons"] = {i:p.buffer(self.width) for i, p in self.trace_linestring.items()}
        self._properties["trace_vertices"] = {i:np.asarray(p.exterior.coords.xy).T for i, p in self.trace_polygons.items()}
        self._side_properties['trace_masks'] = None
        if build_tracemask:
            self.build_tracemask(**kwargs)

    # => Deriv the rest
    def set_trace_masks(self, masks, trace_indexes):
        """ Attach to the current instance masks. """
        if is_arraylike(trace_indexes):
            if len(masks) != len(trace_indexes):
                raise ValueError("masks and trace_indexes do not have the same size.")
            for i,v in zip(trace_indexes, masks):
                self.trace_masks[i] = v
        else:
            self.trace_masks[trace_indexes] = masks


    # --------- #
    #  Shift    #
    # --------- #
    def get_yshifted_traceweight_mask(self, yshift, subpixelization=5):
        """ """
        return self.get_shifted_tracematch(0, yshift, build_tracemask=False).get_traceweight_mask(subpixelization)
    
    
    def get_shifted_tracematch(self, xshift, yshift, build_tracemask=False):
        """ """
        tmap = TraceMatch()
        tmap.set_trace_line(self._xys + np.asarray([[xshift,yshift],[xshift,yshift]]),
                                trace_indexes= self.trace_indexes,
                                width=self.width, build_tracemask=build_tracemask)
        return tmap
    
    # --------- #
    #  GETTER   #
    # --------- #
    def get_sub_tracematch(self, traces, build_tracemask=False):
        """ """
        tmap_ = TraceMatch()
        flagin = np.in1d(self.trace_indexes, traces)
        tmap_.set_trace_line(self._xys[flagin],
                                trace_indexes = self.trace_indexes[flagin],
                                width=self.width, build_tracemask=build_tracemask)
        return tmap_

    # Trace crossing 
    def get_traces_crossing_x(self, xpixel, ymin=-1, ymax=1e5):
        """ traceindexes of the traces crossing the 'xpixel' vertical line
        
        Returns
        -------
        list of indexes
        """
        return self.get_traces_crossing_line([xpixel,ymin],[xpixel,ymax])

    def get_traces_crossing_x_ybounds(self, xpixel, ymin=-1, ymax=1e5):
        """ traceindexes of the traces crossing the 'xpixel' vertical line
        
        Returns
        -------
        list of indexes
        """
        line = geometry.LineString([[xpixel,ymin],[xpixel,ymax]])
        mpoly = geometry.MultiPolygon([self.trace_polygons[i_]
                            for i_ in self.get_traces_crossing_x(xpixel, ymin=ymin, ymax=ymax) ])
        
        return np.asarray([m.intersection(line).xy[1]for m in mpoly])
            
        
    def get_traces_crossing_y(self, ypixel, xmin=-1, xmax=1e5):
        """ traceindexes of the traces crossing the 'ypixel' horizonthal line
        
        Returns
        -------
        list of indexes
        """
        return self.get_traces_crossing_line([xmin,ypixel],[xmax,ypixel])
        
    def get_traces_crossing_line(self, pointa, pointb):
        """ traceindexes of traces crossing the vertival line formed by the [a,b] vector 
        Parameters
        ----------
        pointa, pointb: [xcoord, ycoord]
            coordinates of the 2 points defining the line

        Returns
        -------
        list of indexes
        """
        line = geometry.LineString([pointa,pointb])
        return [idx for idx in self.trace_indexes if self.trace_polygons[idx].crosses(line)]

        
    # Boundaries
    def get_trace_xbounds(self, traceindex):
        """ get the extremal x-ccd coordinates covered by the trace """
        return np.asarray(np.round(np.percentile(np.asarray(self.trace_vertices[traceindex]).T[0], [0,100])), dtype="int")
    
    def get_trace_ybounds(self, traceindex):
        """ get the extremal y-ccd coordinates covered by the trace """
        return np.asarray(np.round(np.percentile(np.asarray(self.trace_vertices[traceindex]).T[1], [0,100])), dtype="int")
    
    def get_trace_vertices(self, traceindex):
        """ traceindex -> vertices 
        
        returns the vertices of the given traceindex

        Returns
        -------
        array (vertices)
        """
        return self.trace_vertices[traceindex]

    def get_finetuned_trace_vertices(self, traceindex, x, y, width,
                                         polydegree=2, **kwargs):
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
        xbounds = self.get_trace_xbounds(traceindex)
        prop = kwargs_update( dict(dy=1), **kwargs )
        return np.asarray(get_boxing_polygone(x, y, rangex=np.linspace(xbounds[0], xbounds[1], polydegree+5),
                                    width= width, polydegree=polydegree, get_vertices=True, **prop))
    
    def get_finetuned_trace(self, traceindex, x, y, polydegree=2, **kwargs):
        """ Builds the best guess fine tuning of the trace given the x, y position

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

        **kwargs goes to spectralmatching.get_boxing_polygone() [cannot be width, fixed to 0]
        Returns
        -------
        x, y 
        """
        xbounds = self.get_trace_xbounds(traceindex)
        prop = kwargs_update( dict(dy=1), **kwargs )
        return np.asarray(get_boxing_polygone(x, y, rangex=np.arange(*xbounds),
                                    width= 0, polydegree=polydegree, get_vertices=True, **prop))

    #  Masking  #  
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
        if _HAS_SHAPELY:
            mask = self._get_shapely_trace_mask_(traceindex)
        else:
            mask = self._get_color_trace_mask_(traceindex)
            
        # - Shall we save it?
        if update:
            self.set_trace_masks(sparse.csr_matrix(mask), traceindex)
            if updateonly:
                del mask
                return
            
        return mask

    def _get_color_trace_mask_(self, traceindex):
        """ Use the tracebuild colors trick
        = Time depends on the subpixelization, 5 takes about 1s =
        """
        r, g, b, a = self._tracecolor[traceindex]
        mask = ((self._rmap==r)*(self._gmap==g)*(self._bmap==b)).reshape(*self._mapshape)
        final_mask = mask if self.subpixelization == 1 else \
              measure.block_reduce(mask, (self.subpixelization, self.subpixelization) )/float(self.subpixelization**2)
              
    def _get_shapely_trace_mask_(self, traceindex):
        """ Based on Shapely, measure the intersection area between traces and pixels.
        = Takes about 1s =
        """
        return verts_to_mask(self.trace_vertices[traceindex])
    
    def _load_trace_mask_(self, traceindexe ):
        """ """
        _ = self.get_trace_mask(traceindexe, updateonly=True)

    def get_traceweight_mask(self, subpixelization=5):
        """ """
        if self._maskimage is None or subpixelization != self.subpixelization:
            self.build_tracemask(subpixelization)
            
        mask = (self._rmap > 0 ).reshape(*self._mapshape)
        return  mask if self.subpixelization == 1 else \
          measure.block_reduce(mask, (self.subpixelization, self.subpixelization) )/float(self.subpixelization**2)
          
    def get_notrace_mask(self, subpixelization=5, asbool=True):
        """ a 2D boolean mask that is True for places in the CCD without trace. """
        return ~np.asarray(self.get_traceweight_mask(subpixelization), dtype="bool")

    # Trace Location #
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
        if _HAS_SHAPELY:
            globalpoly = geometry.Polygon(polyverts)
            return [idx_ for idx_ in self.trace_indexes if globalpoly.contains(geometry.Polygon(self.trace_vertices[idx_]))]
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
        if is_arraylike(traceindex):
            if not is_arraylike(facecolors):
                facecolors = [facecolors]*len(traceindex)
            if not is_arraylike(edgecolors):
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
    def build_tracemask(self, subpixelization=5, width=2048, height=2048):
        """ 
        This will build the internal tools to identify the connections between
        traceindex and ccd-pixels
        
        Returns
        -------
        Void
        """
        self._properties["subpixelization"] = subpixelization
        # - 1 Trace, 1 Color !
        # List because of python3
        nonunique_RGBA         = np.asarray( list(zip(np.random.randint(5,90, size=self.ntraces*3),
                                                    np.random.randint(91,175, size=self.ntraces*3),
                                                    np.random.randint(176,254, size=self.ntraces*3),
                                                    [255]*self.ntraces*3)) )
        
        b = np.ascontiguousarray(nonunique_RGBA).view(np.dtype((np.void,nonunique_RGBA.dtype.itemsize * nonunique_RGBA.shape[1])))
        # This is made for faster identification later on
        self._derived_properties['facecolor']  = nonunique_RGBA[np.unique(b, return_index=True)[1]][:self.ntraces]
        self._derived_properties['tracecolor'] = {i:c for i,c in zip(self.trace_indexes,self._facecolors)}
        verts = [(self.trace_vertices[i]+np.asarray([0.5,0.5]))*self.subpixelization for i in self.trace_indexes]

        self._derived_properties['maskimage'] = \
          np.asarray(polygon_mask(verts, width*self.subpixelization, height*self.subpixelization,
                           facecolor=self._facecolors, edgecolor=self._facecolors, get_fullcolor=True))

        # - Save detailed information for matching later on
        r, g, b, a = self._maskimage.T        
        self._derived_properties['mapshape'] = (width*self.subpixelization, height*self.subpixelization)
        self._derived_properties['rmap'] = r.ravel(order='F')
        self._derived_properties['gmap'] = g.ravel(order='F')
        self._derived_properties['bmap'] = b.ravel(order='F')

    def extract_hexgrid(self, trace_indexes = None, qdistance=None, theta=None):
        """ Build the array of neightbords.
        This is built on a KDTree (scipy)

        Parameters
        ----------
        usedindex: [list of indexes] -optional-
            Select the indexes you want to use. If None [default] all will be.

        theta: [angle in radian]
            rotation of the IFU with respect to north
            
        
        Returns
        -------
        HexagoneProjection (from pysedm.utils.hexagrid)
        """
        from .utils.hexagrid import get_hexprojection
        
        if trace_indexes is None:
            trace_indexes = self.trace_indexes

        # - position used to define 1 location of 1 spectral_trace
        xydata  = np.asarray([np.nanmean(self.trace_vertices[idx_], axis=0) for idx_ in trace_indexes])
        
        return get_hexprojection(xydata, ids=trace_indexes, theta=theta)

    
    # ===================== #
    #   Properties          #
    # ===================== #
    # - Traces I/O
    @property
    def trace_linestring(self):
        """ Central line of the trace | Shapely.MultiLineString """
        return self._properties["trace_linestring"]
    
    @property
    def ntraces(self):
        """ """
        return None if self.trace_linestring is None else len( self.trace_linestring )
    
    @property
    def width(self):
        """ size of the buffer aroung the trace lines"""
        return self._properties["width"]
    
    @property
    def trace_indexes(self):
        """ ID of the traces (spaxels) """
        if self._properties["trace_indexes"] is None:
            if self.trace_linestring is not None:
                self._properties["trace_indexes"] = np.arange( self.ntraces )
                
        return self._properties["trace_indexes"]
    
    @property
    def trace_vertices(self):
        """ dictionary containing the Polygon Vertices for the traces. """
        if self._properties["trace_vertices"] is None:
            self._properties["trace_vertices"] = {}
        return self._properties["trace_vertices"]

        
    @property
    def trace_polygons(self):
        """ Shapely polygon of the traces based on their vertices"""
        if not _HAS_SHAPELY:
            raise ImportError("You do not have shapely. this porpoerty needs it. pip install Shapely")
        return self._derived_properties["trace_polygons"]

    @property
    def ij_offset(self):
        """ By how much the traces are offseted in comparison to the night_tracematch """
        if self._side_properties["ij_offset"] is None:
            return [0,0]
        return self._side_properties["ij_offset"]
    
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
    
