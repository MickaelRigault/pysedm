#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" This modules handles the CCD based images """


import warnings
import numpy as np
import shapely # to be removed
import matplotlib.pyplot as mpl
from propobject import BaseObject
from astrobject.photometry import Image

from .utils.tools import kwargs_update

EDGES_COLOR = mpl.cm.binary(0.99,0.5, bytes=True)
SPECTID_CMAP = mpl.cm.viridis

"""
The Idea of the CCD calibration is to first estimate the Spectral Matching. 
Then to set this Matching to all the ScienceCCD objects. 

- SpectralMatch holds the spaxel->pixel_area relation.

"""



SEDMSPAXELS = np.asarray([[np.sqrt(3.)/2.,1./2],[0,1],[-np.sqrt(3.)/2.,1./2],
                              [-np.sqrt(3.)/2.,-1./2],[0,-1],[np.sqrt(3.)/2.,-1./2],
                              ])*2/3.


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


    def get_spectrum(self, specidx, on="data"):
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

    def extract_spectrum(self, specidx, cubesolution, get_spectrum=True):
        """ Buiuld the `specidx` spectrum based on the given wavelength solution.
        The returned object could be an pyifu's Spectrum or three arrays, lbda, flux, variance.

        Parameters
        ----------
        specidx: [int]
            The index of the spectrum you want to extract
            
        cubesolution: [CubeSolution]
            Object containing the method to go fromn pixels to lbda

        get_spectrum: [bool] -optional-
            Which form the returned data should have?

        Returns
        -------
        Spectrum or
        array, array, array/None (lbda, flux, varirance)
        """
        
        f = self.get_spectrum(specidx)
        pixs = np.arange(len(f))[::-1]
        # Black magic, but it works.
        # - Faster with masking
        hasflux = np.argwhere(f>0.0)
        mask = pixs[(pixs>np.min(hasflux))* (pixs<np.max(hasflux))][::-1]
        
        lbda = cubesolution.pixels_to_lbda(pixs[mask], specidx)
        variance = self.get_spectrum(specidx, on="var")[mask] if self.has_var() else None
        
        if get_spectrum:
            from pyifu.spectroscopy import Spectrum
            spec = Spectrum(None)
            spec.create(f[mask],variance=variance,lbda=lbda)
            return spec
        
        return lbda, f[mask], variance
        
    # --------------- #
    #  Extract Cube   #
    # --------------- #
    def extract_cube(self, wavesolution, lbda,
                    hexagrid=None, used_indexes=None):
        """ """
        from pyifu.spectroscopy import Cube
        from scipy.interpolate import interp1d
        
        # - index check
        if used_indexes is None:
            used_indexes = np.sort(wavesolution.wavesolutions.keys())
        elif np.any(~np.in1d(used_indexes, wavesolution.wavesolutions.keys())):
            raise ValueError("At least some given indexes in `used_indexes` do not have a wavelength solution")
        
        # - Hexagonal Grid
        if hexagrid is None:
            hexagrid = self.smap.build_hexgrid(used_indexes)
        used_indexes = [i_ for i_ in used_indexes if i_ in hexagrid.ids_index.keys()]
        # - data
        cube     = Cube(None)
        cubeflux = []
        cubevar  = [] if self.has_var() else None
        for i in used_indexes:
            lbda_, flux_, variance_ = self.extract_spectrum(i, wavesolution, get_spectrum=False)
            cubeflux.append(interp1d(lbda_, flux_, kind="cubic")(lbda))
            if cubevar is not None:
                cubevar.append(interp1d(lbda_, variance_, kind="cubic")(lbda))
            
        spaxel_map = {i:c for i,c in enumerate(np.asarray(hexagrid.index_to_xy(hexagrid.ids_to_index(used_indexes),
                                                        invert_rotation=True)).T)}
            
        cube.create(np.asarray(cubeflux).T,lbda=lbda, spaxel_mapping=spaxel_map, variance=np.asarray(cubevar).T)
        cube.set_spaxel_vertices(np.dot(hexagrid.grid_rotmatrix,SEDMSPAXELS.T).T)
        return cube



        
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
