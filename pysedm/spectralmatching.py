#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" This modules is made to find and handle the position on the spectral traces on the CCDs. """

import warnings
import numpy as np
import matplotlib.pyplot as mpl


from propobject import BaseObject
from .ccd import get_dome, get_lamp, LampCCD
from .utils.tools import kwargs_update

# ------------------------------- #
#   Attribute SEDM specific       #
# ------------------------------- #
PURE_DOME_ELLIPSE_SCALING = dict(scaleup_red=2.5,  scaleup_blue=9)

# ------------------------------- #
#   PIL Masking tricks            #
# ------------------------------- #
EDGES_COLOR  = mpl.cm.binary(0.99,0.5, bytes=True)
SPECTID_CMAP = mpl.cm.viridis
BACKCOLOR    = (0,0,0,0)


__all__ = ["load_specmatcher","get_specmatcher"]


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
    return smap


def get_specmatcher(domefile, arcfiles=None, **kwargs):
    """ build the spectral match object on the domefile data. 

    Parameters
    ----------
    domefile: [string]
        Location of the file containing the object
        
    hgfile: [string/None]
        Location of a file containing the Mercury arclamp cube.

    **kwargs goes to the dome method `get_specrectangles` 
    
    Returns
    --------
    SpectralMatch
    """
    # - The Spectral Matcher
    smap = SpectralMatch()
    # Dome Data
    dome = get_dome(domefile, background=0)
    dome.sep_extract(thresh=np.nanstd(dome.rawdata))
    
    # only dome data
    if arcfiles is None: 
        warnings.warn("Only dome data used to derived the full spectral matching. This migh not be accurate enough.")
        prop = kwargs_update( PURE_DOME_ELLIPSE_SCALING, **kwargs)
        smap.set_specrectangle(dome.get_specrectangles(**prop))
        return smap

    # You provided ArcFiles? Let's uses them then!
    for arcfile in arcfiles:
        arcccd = LampCCD(arcfile, background=0)
        if arcccd.objname == "Xe":
            arcccd.sep_extract(thresh=np.nanstd(arcccd.rawdata)*2)
        else:
            arcccd.sep_extract(thresh=np.nanstd(arcccd.rawdata))
        smap.add_arclamp(arcccd, match=True)
        
    # Improve the Rectangle patching based on the arcs
    smap.set_arcbased_specmatch()
    return smap
        
    
    
def polygon_mask(polygons, width=2047, height=2047,
                     facecolor=None, edgecolor=EDGES_COLOR,
                     get_fullcolor=False):
    """ """
    from PIL import Image, ImageDraw
    back = Image.new('RGBA', (width, height), BACKCOLOR)
    mask = Image.new('RGBA', (width, height))
    # PIL *needs* (!!) [(),()] format [[],[]] wont work
    if not hasattr(polygons,"__iter__"):
        polygons = [polygons]
        
    if facecolor is None:
        npoly = len(polygons)
        facecolor = SPECTID_CMAP(np.linspace(0,1,len(polygons)), bytes=True)
        
    if edgecolor is None or np.shape(edgecolor) != np.shape(facecolor):
        edgecolor = [EDGES_COLOR]*len(polygons)
        
    [ImageDraw.Draw(mask).polygon( [(x_[0]+0.5,x_[1]+0.5) for x_ in np.asarray(polygon_.exterior.xy).T],
                             fill=tuple(fc),
                             outline=tuple(ec))
     for fc,ec,polygon_ in zip(facecolor, edgecolor, polygons)]
        
    back.paste(mask,mask=mask)
    return np.sum(np.array(back), axis=2) if not get_fullcolor else back



def illustrate_traces(ccdimage, spectralmatch,
                     savefile=None, show=True,
                     facecolor=None, 
                      cmap=None, vmin=None, vmax=None,
                      **kwargs):
    """ """
    from .utils.mpl import figout
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    if cmap is None:cmap=mpl.cm.viridis
    if facecolor is None: facecolor= mpl.cm.bone(0.5,0.3)
    # ---------
    fig = mpl.figure(figsize=[10,10])
    
    ax  = fig.add_subplot(111)
    ax.set_xlabel("x [ccd-pixels]", fontsize="large")
    ax.set_ylabel("y [ccd-pixels]", fontsize="large")

    ax.set_title("Spectral Match illustrated on top of %s"%ccdimage.objname,
                     fontsize="large")
    
    def show_it(ax_, show_poly=True):
        pl = ccdimage.show(ax=ax_,
                           cmap=cmap,vmin=vmin, vmax=vmax, 
                           show=False, savefile=None)
        if show_poly:
            spectralmatch.display_polygon(ax_, fc=facecolor,
                                          **kwargs)
    # ---------
    
    show_it(ax, False)
    
    # - Top Left 
    axinsTL = zoomed_inset_axes(ax, 6, bbox_to_anchor=[0.3,0.8,0.7,0.1],
                                  bbox_transform=ax.transAxes)  # zoom = 6
    axinsTL.set_xlim(200, 500)
    axinsTL.set_ylim(1900, 2000)
    show_it(axinsTL)
    mark_inset(ax, axinsTL, loc1=3, loc2=1, fc="none", ec="k", lw=1.5)

    
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
class SpectralMatch( BaseObject ):
    """ """
    PROPERTIES         = ["spectral_vertices"]
    SIDE_PROPERTIES    = ["arclamp"]
    DERIVED_PROPERTIES = ["maskimage","specpoly","spectral_fc","spectral_ec",
                           "rmap", "gmap", "bmap", "amap"]
    
    # ================== #
    #  Main Tools        #
    # ================== #
    # ----------- #
    #  I/O        #
    # ----------- #
    def writeto(self, savefile, savearcs=True):
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

        Returns
        -------
        Void
        """
        from .utils.tools import dump_pkl
        data= {"vertices": self.spectral_vertices,
               "arclamps": self.arclamps if savearcs else None
                }
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
    def set_specrectangle(self, spectral_vertices,
                              width=2047, height=2047):
        """ """
        self.reset()
        self._properties['spectral_vertices'] = np.asarray(spectral_vertices)
        self.build_maskimage(width=width, height=height)

    def build_maskimage(self, width=2047, height=2047):
        """ """
        self._derived_properties['maskimage'] = \
          polygon_mask(self.spectral_polygon, width, height,
                           facecolor=self.spectral_facecolor,
                           edgecolor=self.spectral_edgecolor,
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
        allindex = [i for i in np.concatenate([[self.color_to_index(self._rmap[x_,y_],
                                                self._gmap[x_,y_],
                                                self._bmap[x_,y_]) 
                      for x_ in np.arange(x-around,x+around)]
                     for y_ in np.arange(y-around,y+around)])
                     if i is not None]
            
        return np.median(allindex) if len(allindex)>0 else np.NaN
    
    def color_to_index(self, r,g,b,a=None):
        """ get the color associated with the given color list.
        This will first fetch in the facecolor and if nothing is found this will use the edgecolor.
        If None is found, this returns None, it returns the index otherwise.

        Returns
        -------
        int or None
        """
        index = np.argwhere((self.spectral_facecolor.T[0] == r) * (self.spectral_facecolor.T[1] == g) * (self.spectral_facecolor.T[2] == b))
        if len(index) == 0:
            index = np.argwhere((self.spectral_edgecolor.T[0] == r) * (self.spectral_edgecolor.T[1] == g) * (self.spectral_edgecolor.T[2] == b))
        
        return index[0][0] if len(index)>0 else None
    
    # ----------- #
    #  GETTER     #
    # ----------- #
    def get_idx_color(self, index):
        """ """
        return self._index_to_(index, "facecolor")

    def get_spectral_mask(self, include="all"):
        """ General masking of global area.

        Parameters
        ----------
        include: [string: all/face/edge] -optional-
            Which part of the masking do you want to get:
            - both: True (1) for spectral regions including face and edges
            - face: True (1) for the core of the spectral masking
            - edge: True (1) for the edge of the spectral regions
        Returns
        -------
        2d-array bool
        """
        # - Code help. See _load_random_color_()
        # face r goes [100->250]
        # edge r goes [50->99]
        rmap, gmap, bmap, amap = np.asarray(self.maskimage).T
        mask = (rmap.T > 0 ) if include in ["all",'both','faceandedge'] else\
          (rmap.T >= 100 ) if include in ['core',"face"] else \
          (rmap.T < 100 ) * (rmap.T > 0 ) if include in ['edge'] else\
          None
        if mask is None:
            raise ValueError("include not understood: 'all'/'both' or 'face'/'core' or 'edge'")
        
        return mask
          
    def get_idx_mask(self, index, include="all"):
        """ boolean 2D mask for the given idx. """
        # To Be Done. Could be fasten
        
        # - Composite mask from multiple indexes
        if hasattr(index,"__iter__"):
            return np.asarray(np.sum([ self.get_idx_mask(i_, include=include)
                                           for i_ in index], axis=0), dtype="bool")
        # - Single index
        # value we hare looking for
        fr,fg,fb,fa = self._index_to_(index, "facecolor")
        er,eg,eb,ea = self._index_to_(index, "edgecolor")
        # masking edge and face
        maskface = (self._rmap.T == fr) * (self._bmap.T == fb) * (self._gmap.T == fg)
        maskedge = (self._rmap.T == er) * (self._bmap.T == eb) * (self._gmap.T == eg)
        
        # let's build the actual mask
        mask = maskface + maskedge if include in ["all",'both','faceandedge'] else\
          maskface if include in ['core',"face"] else\
          maskedge if include in ['edge'] else None
          
        if mask is None:
            raise ValueError("include not understood: 'all'/'both' or 'face'/'core' or 'edge'")
        
        return mask

    # -------------
    # - Select idx
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
    def display_polygon(self, ax, idx=None,**kwargs):
        """ display on the given axis the polygon used to define the spectral match """
        from astrobject.utils.shape import draw_polygon
        if idx is None:
            return ax.draw_polygon(self.spectral_polygon, **kwargs)
        return [ax.draw_polygon(self.spectral_polygon[i])  for i in idx]

    
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
            arc = LampCCD(arc, background=0)
        elif LampCCD not in arc.__class__.__mro__:
            raise TypeError("The given arc must be a string or an LampCCD object")

        
        if not arc.has_sepobjects():
            arc.sep_extract(thresh=np.nanstd(arc.rawdata))
        x,y,a,b,t = arc.sepobjects.get(["x","y","a","b","t"]).T
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
                                 xbounds=None, get_vertices=False):
        """ """
        xXe,yXe = self.get_arcline_positions("Xe",index)
        xCd,yCd = self.get_arcline_positions("Cd",index)
        xHg,yHg = self.get_arcline_positions("Hg",index)
        
        if xbounds is None:
            xbounds = np.percentile(self.spectral_vertices[index].T[0], [0,100])
            
        if xbounds[0] is None: np.percentile(self.spectral_vertices[index].T[0], 0)
        if xbounds[1] is None: np.percentile(self.spectral_vertices[index].T[0], 100)
        
        return get_boxing_polygone(np.concatenate([xXe,xCd,xHg]), 
                            np.concatenate([yXe,yCd,yHg]), 
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
            raise AttributeError("No arclamp loaded named %d"%arcname)
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
    
    @property
    def spectral_edgecolor(self):
        """ Spectral color associated to the Core of the polygon """
        if self._derived_properties['spectral_ec'] is None:
            self._load_random_color_()
        return self._derived_properties['spectral_ec']

    def _load_random_color_(self):
        """ """
        nonunique_RGBA    = np.random.randint(100,250,size=[self.nspectra*2,4])
        nonunique_RGBA_ec = np.random.randint(50,99,size=[self.nspectra*2,4])
        nonunique_RGBA.T[3]    = 255 # explicit here to avoid PIL / mpl color variation
        nonunique_RGBA_ec.T[3] = 255 # explicit here to avoid PIL / mpl color variation
            
        b = np.ascontiguousarray(nonunique_RGBA).view(np.dtype((np.void,nonunique_RGBA.dtype.itemsize * nonunique_RGBA.shape[1])))
        b_ec = np.ascontiguousarray(nonunique_RGBA_ec).view(np.dtype((np.void,nonunique_RGBA_ec.dtype.itemsize * nonunique_RGBA_ec.shape[1])))
        
        self._derived_properties['spectral_fc'] = \
          nonunique_RGBA[np.unique(b, return_index=True)[1]][:self.nspectra]
        self._derived_properties['spectral_ec'] = \
          nonunique_RGBA_ec[np.unique(b_ec, return_index=True)[1]][:self.nspectra]
          
    @property
    def maskimage(self):
        """ Masking image core of the class. """
        return self._derived_properties['maskimage']
    
    def has_maskimage(self):
        return self.has_maskimage is not None


    # ----------
    # Internal
    @property
    def _rmap(self):
        return self._derived_properties["rmap"]
    @property
    def _gmap(self):
        return self._derived_properties["gmap"]
    @property
    def _bmap(self):
        return self._derived_properties["bmap"]
    @property
    def _amap(self):
        return self._derived_properties["amap"]
