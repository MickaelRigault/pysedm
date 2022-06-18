#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" This library contains the high level functionnality to go from x,y,lbda (cube coords) <-> i,j (ccd-coords) """

import numpy as np
from propobject import BaseObject

from shapely.geometry import Point, LineString

from .utils.tools import is_arraylike


# = Pysedm Information = #
from .sedm import SEDMSPAXELS   as SPAXEL_SHAPE
from .sedm import SEDM_CCD_SIZE as CCD_SHAPE
INVERTED_LBDA_X = True


##########################
#                        #
#   The Mapping Class    #
#                        #
##########################
class Mapper( BaseObject ):
    """ """
    PROPERTIES = ["tracematch","hexagrid", "wavesolution"]
    DERIVED_PROPERTIES = ["spaxel_mapping","spaxelslice"]
    # ================= #
    #  Initialization   #
    # ================= #

    def __init__(self, tracematch=None, hexagrid=None, wavesolution=None):
        """ """
        if tracematch is not None:
            self.set_tracematch(tracematch)
        if hexagrid is not None:
            self.set_hexagrid(hexagrid)
        if wavesolution is not None:
            self.set_wavesolution(wavesolution)

    # - derived
    def derive_spaxel_mapping(self, traceindexes):
        """ """
        import pyifu
        xy = np.asarray(self.hexagrid.index_to_xy(self.hexagrid.ids_to_index(traceindexes))).T
        self._derived_properties["spaxel_mapping"] = {i:xy_ for i,xy_ in zip(traceindexes,xy)}
        
        self._derived_properties["spaxel_slice"] = \
          pyifu.get_slice( self.traceindexes, xy, spaxel_vertices= np.dot( self.hexagrid.grid_rotmatrix, SPAXEL_SHAPE.T ).T )
          
        self._derived_properties["spaxel_polygon"] = {i:p_ for i,p_ in zip(traceindexes,self.spaxel_slice.get_spaxel_polygon(remove_nan=True)) }
    # ================= #
    #  Methods          #
    # ================= #

    # ------------- #
    #   GETTER      #
    # ------------- #
    def get_lbda_ij(self, lbda, traceindexes=None):
        """ returns all i,j coordinates for the given wavelength

        Parameters
        ----------
        lbda: [float] 
            wavelength in Angstrom

        traceindexes: [list/array] -optional-
            list of traces you want to use. 
            If None all the traces in self.traceindexes will be used.

        Returns
        -------
        dictionary 
        [{traceindex:[i,j] ... }
        """
        if traceindexes is None:
            traceindexes = self.traceindexes

        return {traceindex:self.lbda_to_ij(lbda, traceindex) for traceindex in self.traceindexes}
    
    def get_ij(self, x, y, lbda):
        """ 
        Reminder: i,j are the CCD coordinates
                  q,r are the MLA hexagonal coordiates
                  x,y are the MLA coordinate (in spaxels)
                  traceindex is the unique ID of a spaxel on the ccd
        """
        traceindex = self.xy_to_traceindex(x,y)
        return self.lbda_to_ij(lbda, traceindex)
        
    def get_xylbda(self, i, j):
        """ 
        
        Reminder: i,j are the CCD coordinates
                  q,r are the MLA hexagonal coordiates
                  x,y are the MLA coordinate (in spaxels)
                  traceindex is the unique ID of a spaxel on the ccd
        """
        traceindex = self.ij_to_traceindex(i,j)
        lbda       = self.ij_to_lbda(i, traceindex)
        x,y        = self.traceindex_to_xy(traceindex)
        return np.asarray([x,y,lbda])

    def get_expected_j(self, i,j):
        """ fetch the traceindex associated to the i,j coordinates
        and then returns where the j is supposed to be if the trace where perfectly 
        aligned. 
        
        Reminder: i,j are the CCD coordinates
                  q,r are the MLA hexagonal coordiates
                  x,y are the MLA coordinate (in spaxels)
                  traceindex is the unique ID of a spaxel on the ccd
            
        """
        traceindex = self.ij_to_traceindex(i,j)
        if is_arraylike(traceindex):
            return [self.traceindexi_to_j(t_,i) for t_ in traceindex]
        
        return self.traceindexi_to_j(traceindex,i)[1]
            
    # ------------- #
    #  CONVERSION   #
    # ------------- #
    # .................... #
    #  lbda <-> i,j        #
    # .................... #
    def ij_to_lbda(self, i, traceindex):
        """ converts the i-th ccd pixel to a wavelength for a given traceindex """
        i_eff = (CCD_SHAPE[1]-1)-i if INVERTED_LBDA_X else i # -1 because starts at 0
        return self.wavesolution.pixels_to_lbda(i_eff, traceindex)
    
    def lbda_to_ij(self, lbda, traceindex):
        """ provide the central i,j coordinate for given traceindex at the given wavelength.
        Remark: j is the center of the trace for the column corresponding to the wavelength 
        """
        i =  self.wavesolution.lbda_to_pixels(lbda, traceindex)
        return self.traceindexi_to_j(traceindex, i, inverted=INVERTED_LBDA_X)
    
    def traceindexi_to_j(self, traceindex, i, maxlines=1e4, inverted=False):
        """ get thethe center of the trace `traceindex` for the column `i`"""
        if is_arraylike(i):
            return np.asarray([self.traceindexi_to_j(traceindex, i_) for _ in i])
        
        if i is None: return np.asarray([None,None])

        i_eff = (CCD_SHAPE[1]-1)-i if inverted else i # -1 because starts at 0
        try:
            return i_eff, np.mean(self.tracematch.trace_polygons[traceindex].intersection(LineString([[i_eff,0],[i_eff, maxlines]])), axis=0)[1]
        except:
            return i_eff, np.NaN

    # .................... #
    #  i,j <-> traceindex  #
    # .................... #
    def ij_to_traceindex(self, i, j):
        """ provide the traceindex associated to the pixel i, j.
        If the pixel is not in any spaxel, a None is set

        Reminder: i,j are the CCD coordinates
                  q,r are the MLA hexagonal coordiates
                  x,y are the MLA coordinate (in spaxels)
                  traceindex is the unique ID of a spaxel on the ccd

        Returns
        -------
        list
        """
        if is_arraylike(i):
            return [self.ij_to_traceindex(i_,j_) for i_,j_ in zip(i,j)]
        
        pij = Point(i,j)
        traceindex = [i_ for i_ in self.traceindexes if self.tracematch.trace_polygons[i_].contains(pij)]
        if len(traceindex) > 1:
            raise ValueError("the ccd-pixel coordinates %d,%d belongs to more than 1 spaxel-trace"%(i,j))

        return traceindex[0] if len(traceindex)==1 else None

    def traceindex_to_ij(self, traceindex):
        """ """
        return self.tracematch.get_trace_vertices(traceindex)
    
    # .................... #
    #  x,y <-> traceindex  #
    # .................... #
    def xy_to_traceindex(self, x, y):
        """ each spaxel has a unique x,y, location that 
        can be converted into a unique index (1d-array) entry.
        This tools enables to know what is the index corresponding to the given
        2D (x,y) position.
        

        Reminder: i,j are the CCD coordinates
                  q,r are the MLA hexagonal coordiates
                  x,y are the MLA coordinate (in spaxels)
                  traceindex is the unique ID of a spaxel on the ccd
                  
        Parameters
        ----------
        xy: [2d array]
            x and y position(s) in the following structure:
            [x,y] or [[x0,y0],[x1,y1]]

        Returns
        -------
        list of indexes 
        """
        # - Following set_spaxels_mapping, v are list
        if is_arraylike(x):
            return [self.xy_to_traceindex(x_,y_) for x_,y_ in zip(x,y)]
        
        pxy = Point(x,y)
        traceindex = [i_ for i_ in self.traceindexes if self._spaxel_polygon[i_].contains(pxy)]
        if len(traceindex) > 1:
            raise ValueError("the spaxel coordinates (x=%d,y=%d) belongs to more than 1 spaxel-trace"%(x,y))

        return traceindex[0] if len(traceindex)==1 else None
        
    def traceindex_to_xy(self, traceindex):
        """ Each spaxel has a unique index (1d-array) entry.
        This tools enables to know what is the 2D (x,y) position 
        of this index

        Reminder: i,j are the CCD coordinates
                  q,r are the MLA hexagonal coordiates
                  x,y are the MLA coordinate (in spaxels)
                  traceindex is the unique ID of a spaxel on the ccd

        Returns
        -------
        [int,int] (x,y)
        """
        if is_arraylike(traceindex):
            return np.asarray([self.traceindex_to_xy(index_) for index_ in traceindex])
        
        if traceindex in self.spaxel_mapping.keys():
            return self.spaxel_mapping[traceindex]
        return None, None
        
        
    # ------------- #
    #   SETTER      #
    # ------------- #
    # - set
    def set_hexagrid(self, hgrid):
        """ """
        self._properties["hexagrid"] = hgrid
        
    def set_tracematch(self, tmap):
        """ """
        self._properties["tracematch"] = tmap
        
    def set_wavesolution(self, wsol):
        """ """
        self._properties["wavesolution"] = wsol
        
    # ------------- #
    #   PLOTTER     #
    # ------------- #
    
    # ================= #
    #  Properties       #
    # ================= #
    # Sources
    @property
    def hexagrid(self):
        """ object containing the spaxel index <-> x,y coordinates conversion """
        return self._properties["hexagrid"]
    
    @property
    def tracematch(self):
        """ object containing the spaxel index <-> i,j coordinates conversion """
        return self._properties["tracematch"]
    

    @property
    def wavesolution(self):
        """ object containing the lbda<-> pixel coordinates """
        return self._properties["wavesolution"]

    # Derived
    @property
    def traceindexes(self):
        """ 'ID' (int) of the spaxels """
        return np.asarray(list(self.spaxel_mapping.keys()))

    @property
    def spaxel_slice(self):
        """ pyifu Slice using traceindex as data """
        if self._derived_properties["spaxel_slice"] is None:
            raise AttributeError("spaxel_slice not defined. See `derive_spaxel_mapping()` method")
        return self._derived_properties["spaxel_slice"]
    
    @property
    def spaxel_mapping(self):
        """ Dictionary containing the central coordinate of traceindex spaxels 
        {traceindex_i:[x_i, y_], ...} for the ith spaxel. """
        if self._derived_properties["spaxel_mapping"] is None:
            raise AttributeError("spaxel_mapping not defined. See `derive_spaxel_mapping()` method")
        return self._derived_properties["spaxel_mapping"]
    
    @property
    def _spaxel_polygon(self):
        """ list of Shapely Polygon derived by the `derive_spaxel_mapping()` method """
        if self._derived_properties["spaxel_polygon"] is None:
            raise AttributeError("spaxel_mapping not defined. See `derive_spaxel_mapping()` method")
        return self._derived_properties["spaxel_polygon"]

