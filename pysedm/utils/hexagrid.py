#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy         as np
from scipy.spatial   import KDTree, distance
from propobject      import BaseObject




class HexagoneProject( BaseObject ):
    """ This class enable to build th Q,R hexagone grid
    based on given reference positions 
    """
    PROPERTIES         = ["xy","qdistance"]
    DERIVED_PROPERTIES = ["neighbors","hexgrid","tree","multipoint"]

    def __init__(self, xy, qdistance=None, load=True):
        """ """
        self.set_xy(xy)
        self.set_qdistance(qdistance)
        if load:
            self.load_hexneighbor()

            
    # -------------- #
    #   SETTER       #
    # -------------- # 
    def set_xy(self, xy):
        """ """
        self._properties["xy"] = np.asarray(xy)
        self._derived_properties["tree"] = KDTree(self.xy)
        self._derived_properties["hexgrid"] = None # reset
        
    def set_qdistance(self, qdistance=None):
        """ Distance at which the KDtree will be queried

        What should be the distance used to search the neighbors.
        If None, this optimal distance will be automatically built
        based on the density around the central point.
        If you are not sure, lease this to None
        """
        if qdistance is None:
            # We want 7 (the point  + 6 neiggtbors). So at what distance the 8th is?
            dist_8, index_8 = self.tree.query(self.xy[self.get_central_index()],8) 
            qdistance = np.mean(dist_8[-2:])
        
        self._properties["qdistance"] = qdistance

    def load_hexneighbor(self):
        """ """
        self._derived_properties["neighbors"] = self.tree.query_ball_tree(self.tree, r=self.qdistance)

    # ---------- #
    #   GETTER   #
    # ---------- #
    def get_central_index(self):
        """ the closest point to the centroid """
        if self.centroid is None:
            raise ValueError("This cannot estimate the centroid (no Shapely?).")
        
        return np.argmin([distance.euclidean(xydata_,self.centroid) for xydata_ in self.xy])
    
    def get_idx_neighbors(self, index):
        """ """
        # No test here for fast recursive tricks
        arr = np.asarray(self.neighbors[index])
        return arr[arr!=index]

    def get_shared_neighbors(self, index1, index2):
        """ Get the index that are neighbors of both indexes """
        return np.intersect1d(self.get_idx_neighbors(index1),
                              self.get_idx_neighbors(index2),
                              assume_unique=True)

    # ---------- #
    #  Builder   #
    # ---------- #
    def build_qr_grid(self, origin, )
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def xy(self):
        """ Data of the three """
        return self._properties["xy"]
    
    @property
    def npoints(self):
        """ """
        return len(self.xy) if self.xy is not None else None
    
    @property
    def qdistance(self):
        """ Distance used the query the neighbors """
        if self._properties["qdistance"] is None:
            self.set_qdistance(None)
        return self._properties["qdistance"]

    # ----------
    # Derived
    @property
    def tree(self):
        """ The KDtree associated with the xy data"""
        return self._derived_properties["tree"]

    @property
    def neighbors(self):
        """ The neightbors """
        return self._derived_properties["neighbors"]


    @property
    def hexgrid(self):
        """ Dictionary containing the relation between index and (Q,R)"""
        if self._derived_properties["hexgrid"] is None:
            self._derived_properties["hexgrid"] = {i:None for i in self.npoints}
        return self._derived_properties["hexgrid"]
    # ----------
    # Centroid 
    @property
    def centroid(self):
        """ """
        if self._multipoint is not None:
            return np.asarray(self._multipoint.centroid.xy).T[0]
        return None

    @property
    def _multipoint(self):
        """ """
        try:
            from shapely.geometry import MultiPoint, Point
        except ImportError:
            warnings.warn("No Shapely, no MultiPoint")
            return None
        
        if self._derived_properties["multipoint"] is None:
            self._derived_properties["multipoint"] = MultiPoint([Point(x,y) for x,y in self.xy])
            
        return self._derived_properties["multipoint"]
