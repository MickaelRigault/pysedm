#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" This modules handles the CCD based images """


import warnings
import numpy as np
import shapely # to be removed
import matplotlib.pyplot as mpl
from propobject import BaseObject
from astrobject.photometry import Image
from astrobject.utils.tools import kwargs_update

EDGES_COLOR = mpl.cm.binary(0.99,0.5, bytes=True)
SPECTID_CMAP = mpl.cm.viridis

"""
The Idea of the CCD calibration is to first estimate the Spectral Matching. 
Then to set this Matching to all the ScienceCCD objects. 

- SpectralMatch holds the spaxel->pixel_area relation.

"""
##################################
#                                #
#   Object Generators            #
#                                #
##################################

def get_dome(domefile, specmatch=None,  **kwargs):
    """ Load a SEDmachine domeflat image. 

    Parameters
    -----------
    domefile: [string]
        Location of the file containing the object
    
    specmatch: [SpectralMatch (pysedm object)] -optional-
        SpectralMatch object containing the Spectral property on the CCD

    **kwargs goes to DomeCCD.__init__() if specmatch is None 
             else to _get_default_background_
    Returns
    -------
    DomeCCD (Child of CCD which is a Child of an astrobjec's Image)
    """
    if specmatch is None:
        return DomeCCD(domefile, **kwargs)
    # = SpectralMatch that gonna help the background
    dome = DomeCCD(domefile, background= 0, **kwargs)
    dome.set_specmatch(specmatch)
    dome.set_background(dome._get_default_background_(**kwargs), force_it=True)
    return dome

def get_lamp(lampfile, ccdspec_mask=None,
                 specmatch=None, **kwargs):
    """ Load a SEDmachine lamp image. 

    Parameters
    -----------
    lampfile: [string]
        Location of the file containing the object
    
    specmatch: [SpectralMatch (pysedm object)] -optional-
        SpectralMatch object containing the Spectral property on the CCD

    **kwargs goes to _get_default_background_()

    Returns
    -------
    LampCCD (Child of CCD which is a Child of an astrobjec's Image)
    """
    lamp = LampCCD(lampfile, background=0)
    if specmatch is not None:
        lamp.set_specmatch(specmatch)
    lamp.set_background(lamp._get_default_background_(**kwargs), force_it=True)
    return lamp

def get_specmatcher(domefile, matchprop={},**kwargs):
    """ build the spectral match object on the domefile data. 

    Parameters
    ----------
    domefile: [string]
        Location of the file containing the object
        
    Returns
    --------
    SpectralMatch
    """
    dome = get_dome(domefile, **kwargs)
    return dome.derive_spectralmatch(**matchprop)


##################################
#                                #
#   Internal Tools               #
#                                #
##################################

def polygon_mask(polygons, width=2047, height=2047,
                     facecolor=None, edgecolor=EDGES_COLOR,
                     get_fullcolor=False):
    """ """
    from PIL import Image, ImageDraw
    back = Image.new('RGBA', (width, height), (0,0,0,0))
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
                          ]

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
    #  Geometry   #
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

    def index_to(self, index, what):
        """ """
        return eval("self.spectral_%s[index]"%what)

    def build_hexneighbors(self, usedindex = None, qdistance=None):
        """ Build the array of neightbords.
        This is built on a KDTree (scipy)

        Parameters
        ----------
        usedindex: [list of indexes] -optional-
            Select the indexes you want to use. If None [default] all will be.

        

        
        """
        from shapely.geometry import MultiPoint
        from .utils import hexagrid
        
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
        
        self._derived_properties["hexneighbors"] = hexagrid.HexagoneProject(xydata)
        
    # ----------- #
    #  GETTER     #
    # ----------- #
    def get_idx_color(self, index):
        """ """
        return self.index_to(index, "facecolor")

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
        rmap, gmap, bmap, amap = np.asarray(self.maskimage).T
        # value we hare looking for
        fr,fg,fb,fa = self.index_to(index, "facecolor")
        er,eg,eb,ea = self.index_to(index, "edgecolor")
        # masking edge and face
        maskface = (rmap.T == fr) * (bmap.T == fb) * (gmap.T == fg)
        maskedge = (rmap.T == er) * (bmap.T == eb) * (gmap.T == eg)
        
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
    @property
    def nspectra(self):
        """ number of spectra loaded """
        return len(self.spectral_vertices) if self.spectral_vertices is not None else 0

    # -----------------
    # - Hexagonal Grid
    @property
    def hexneighbors(self):
        """ """
        return self._derived_properties["hexneighbors"]

    def has_hexneighbors(self):
        """ Tests if the hexneighbors dictionary has been built. """
        return self.hexneighbors is not None

    @property
    def hexgrid(self):
        """ Coordinate (Q,R) of the given index """
        if self._derived_properties["hexgrid"] is None:
            self._derived_properties["hexgrid"]  = {i:None for i in range(self.nspectra)}
        return self._derived_properties["hexgrid"]
    
    # -----------------
    # - Spectral Info
    @property
    def spectral_vertices(self):
        """ Vertices for the polygon defining the location of the spectra """
        return self._properties["spectral_vertices"]
    
    @property
    def spectral_polygon(self):
        """ Shapely Multi Polygon associated to the given vertices """
        if self._derived_properties["specpoly"] is None:
            self._derived_properties["specpoly"] = shapely.geometry.MultiPolygon([ shapely.geometry.polygon.Polygon(rect_)
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

#####################################
#                                   #
#  Position Matching Class          #
#                                   #
#####################################
""" Going from Spectral Index to Spatial Position """
class PositionMatching( BaseObject ):
    """ """
    PROPERTIES = []
    def set_polygons(self, polygons, index):
        """ """
    
#####################################
#                                   #
#  Raw CCD Images for SED machine   #
#                                   #
#####################################
class CCD( Image ):
    """ """
    def __build__(self,bw=300, bh=300,
                      fw=3, fh=3,**kwargs):
        """ build the structure of the class

        // Doc from SEP
        bw, bh : int, -optional-
            Size of background boxes in pixels. Default is 64 [ndlr (in SEP)].

        fw, fh : int, -optional-
            Filter width and height in boxes. Default is 3 [ndlr (in SEP)].

        """
        
        super(CCD,self).__build__(**kwargs)
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


    
class ScienceCCD( CCD ):
    """ Virtual Class For CCD images that have input light """
    PROPERTIES = ["spectralmatch"]
    
    def set_specmatch(self, specmatch):
        """ """
        if SpectralMatch not in specmatch.__class__.__mro__:
            raise TypeError("The given specmatch must be a SpectralMatch object")
        
        self._properties["spectralmatch"] = specmatch


    def extract_spectrum(self, specidx, on="data"):
        """ Get the basic spectrum extracted for the CCD based on the 
        SpectralMatch object. 
        
        Parameters
        ----------
        sepcidx: [int, list of]
            index(es) of the spectrum(a) to return

        on: [str] -optional-
            on which 2d image shall the spectrum be extracted.
            By Default 'data', but you can set e.g. rawdata, background 
            or anything accessible as 'self.%s'%on. 
            
        Returns
        -------
        flux (or list of) as a function of pixels
        """
        if not self.has_specmatch():
            raise AttributeError("The SpectralMatch has not been set. see set_specmatch() ")
            
        if hasattr(specidx, "__iter__"):
            return [self.get_spectrum(id_) for id_ in specidx]

        maskidx  = self.specmatch.get_idx_mask(specidx)
        return np.sum(eval("self.%s"%on)*maskidx, axis=0)

    # ================== #
    #  Internal Tools     #
    # ================== #
    def _get_default_background_(self, add_mask=None,
                                 scaleup_sepmask=2, apply_sepmask=True,
                                 **kwargs):
        """ This Background has been Optimized for SEDm Calibration Lamps """
        if add_mask is None and self.has_specmatch():
            add_mask = self.specmatch.get_spectral_mask()
            
        return self.get_sep_background(doublepass=False, update_background=False,
                                       add_mask=add_mask,
                                       apply_sepmask=apply_sepmask, scaleup_sepmask=scaleup_sepmask,
                                       **kwargs)
    
    # ================== #
    #   Properties       #
    # ================== #
    @property
    def specmatch(self):
        """ """
        return self._properties["spectralmatch"]
    
    def has_specmatch(self):
        return self.specmatch is not None    

    @property
    def objname(self):
        if "Calib" in self.header["NAME"]:
            return self.header["NAME"].split()[1]
        return self.header["NAME"]
    
# ============================== #
#                                #
#  Childs Of CCD                 #
#                                #
# ============================== #

class DomeCCD( ScienceCCD ):
    """ Object Build to handle the CCD images of the Dome exposures"""

    # ================== #
    #  Main Tools        #
    # ================== #
    def get_specrectangles(self, length="optimal",height="optimal",theta="optimal",
                               scaleup_red=2.5, scaleup_blue = 7.5, scaleup_heigth=1.5,
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
        right_lim = np.asarray([ x_+length*np.cos(-angle)*scaleup_blue,
                                 y_-length*np.sin(-angle)*scaleup_blue])        

        return np.asarray([[left_lim[0],left_lim[0],right_lim[0],right_lim[0]],
                               [left_lim[1]-height,left_lim[1]+height,right_lim[1]+height,right_lim[1]-height]]).T

    
    def derive_spectralmatch(self, **kwargs):
        """ Returns the mask of the spectrally covered regions (True if covered) 
        This masking is based on the sep detected regions. 
        See the method get_specrectangles() 
        Returns
        ------
        2d boolen array (width x height)
        """
        #rectangles = self.get_specrectangles(**kwargs)
        smap = SpectralMatch()
        smap.set_specrectangle(self.get_specrectangles(**kwargs))
        return smap

    # ================== #
    #   Internal         #
    # ================== #
    def _get_default_background_(self, mask_prop={},
                                     from_spectmatch=True,
                                apply_sepmask=False, **kwargs):
        """ This Background has been Optimized for SEDm Dome """

        if from_spectmatch and self.has_specmatch():
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

    
class LampCCD( ScienceCCD ):
    """ """
