#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" This modules contains the Wavelength solution tools. """

import warnings
import numpy as np
import matplotlib.pyplot as mpl


from propobject import BaseObject
from .ccd import get_dome, get_lamp
from .utils.tools import kwargs_update

PURE_DOME_ELLIPSE_SCALING = dict(scaleup_red=2.5,  scaleup_blue=8)
HGSHIFT_ELLIPSE_SCALING   = dict(scaleup_red=-2.5, scaleup_blue=8)

# - Mapping
EDGES_COLOR = mpl.cm.binary(0.99,0.5, bytes=True)
SPECTID_CMAP = mpl.cm.viridis
BACKCOLOR    = (0,0,0,0)


__all__ = ["get_specmatcher"]


def get_specmatcher(domefile, hgfile=None, **kwargs):
    """ build the spectral match object on the domefile data. 

    Parameters
    ----------
    domefile: [string]
        Location of the file containing the object
        
    Returns
    --------
    SpectralMatch
    """
    # - The Spectral Matcher
    smap = SpectralMatch()
    # Dome Data
    dome = get_dome(domefile, background=0, **kwargs)
    dome.sep_extract(thresh=np.nanstd(dome.rawdata))
    
    # only dome data
    if hgfile is None: 
        warnings.warn("Only dome data used to derived the full spectral matching. This migh not be accurate enough.")
        prop = kwargs_update( PURE_DOME_ELLIPSE_SCALING, **kwargs)
        smap.set_specrectangle(dome.get_specrectangles(**prop))
        return smap
    
    # Hg Data
    hg = get_lamp(hgfile, background=0)
    hg.sep_extract(thresh=np.nanstd(hg.rawdata))
    
    # - Build new rectangles
    smap.set_specrectangle(dome.get_specrectangles(**HGSHIFT_ELLIPSE_SCALING))
    x,y,a = hg.sepobjects.get(["x","y","a"]).T
    # quick slow ~10s
    smapindex = np.asarray([smap.pixel_to_index(x_,y_, around=a_*2) for x_,y_,a_ in zip(x,y,a)])

    # Enable to do hgsepindex <-> [smapindex=domesepindex]
    

    
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

#####################################
#                                   #
#  Spectral Matching Class          #
#                                   #
#####################################
class SpectralMatch( BaseObject ):
    """ """
    PROPERTIES = ["spectral_vertices"]
    DERIVED_PROPERTIES = ["maskimage","specpoly","spectral_fc","spectral_ec",
                           "rmap", "gmap", "bmap", "amap"]
    
    # ================== #
    #  Main Tools        #
    # ================== #
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
    def set_specrectangle(self, spectral_vertices, width=2047, height=2047):
        """ """
        self._properties['spectral_vertices'] = np.asarray(spectral_vertices)
        self.build_maskimage()

    def build_maskimage(self, width=2047, height=2047):
        """ """
        self._derived_properties['maskimage'] = \
          polygon_mask(self.spectral_polygon, width, height,
                           facecolor=self.spectral_facecolor,
                           edgecolor=self.spectral_edgecolor,
                           get_fullcolor=True)
        self.reset()
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
    
    def show(self, ax=None, savefile=None, show=True, cmap=None,
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
