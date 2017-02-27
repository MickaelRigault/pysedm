#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy         as np
from scipy.spatial   import KDTree, distance
from propobject      import BaseObject




class HexagoneProject( BaseObject ):
    """ This class enable to build th Q,R hexagone grid
    based on given reference positions 
    """
    PROPERTIES         = ["xy","qdistance"]
    SIDE_PROPERTIES    = ["ref_idx"]
    DERIVED_PROPERTIES = ["neighbors","hexgrid","hexafilled",
                              "tree","multipoint","rotmatrix"]

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

    def get_forth_index(self, n0, n1, ref):
        """ If n1 and n2 are neightbors, they should have two shared neighbors.
        One is ref the other one is returned.
        
        Returns
        -------
        index or None (if no forth index found)
        """
        index = [i for i in self.get_shared_neighbors(n0,n1) if i not in [ref]]
        if len(index)==0:
            return None
        if len(index)>2:
            warnings.warn("Several 'forth' indexes for n0 %d, n1 %d (ref %d)"%(n0,n1, ref))
        return index[0]
        
    # -------------- #
    #  Coordinates   #
    # -------------- #
    def index_to_qr(self, index):
        """ get the (Q,R) hexagonal coordinates of the given index"""
        return self.hexgrid[index]

    def index_to_xy(self, index, invert_rotation=True):
        """ """
        q,r = np.asarray([self.index_to_qr(i) for i in index]).T\
          if hasattr(index,"__iter__") else self.index_to_qr(index)
        return self.qr_to_xy(q,r, invert_rotation=invert_rotation)
    
    def qr_to_xy(self, q,r, invert_rotation=True):
        """ Convert (q,r) hexagonal grid coordinates into (x,y) system 
        Returns
        -------
        x,y
        """
        x,y = np.asarray([ (2*q + r)/np.sqrt(3.),  r])
        if not invert_rotation:
            return x,y 
        return np.dot(self.grid_rotmatrix,np.asarray([x,y]))
    
    # -------------- #
    # GRID BUILDING  #
    # -------------- #
    def set_grid_reference(self, ref_00, ref_01, ref_10, theta=None):
        """  The three indexes defining (0,0),(0,1),(1,0).

        They must be neighbors.

        theta is the hexagonal grid angle (in radian)
        """
        if ref_10 not in self.get_shared_neighbors(ref_00, ref_01):
            raise ValueError("The given reference indexes are not neighbors.")
        
        self.hexgrid[ref_00] = [0,0]
        self.hexgrid[ref_01] = [0,1]
        self.hexgrid[ref_10] = [1,0]
        
        self._side_properties["ref_idx"] = [ref_00,ref_01,ref_10]
        self._derived_properties["hexafilled"] = None
        
        # - derived the baseline rotation matrix
        x,y = np.asarray(self.index_to_xy([ref_00,ref_01,ref_10], invert_rotation=False))
        if theta is None:
            v1 = np.asarray([x[1],y[1]])
            v2 = np.asarray([x[2],y[2]])
            theta = np.arccos(np.clip(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)), -1.0, 1.0))
            
        self._derived_properties["grid_theta"] = theta
            
        
    def populate(self, idx, neighbor0, neighbor1, ref):
        """ Based on the given neighbors and the reference index,
        the Q,R coordinates are set to the given idx.
        
        In the hexagonal grid, two neighbords N0 and N1 share
        two neighbors: M0 and M1. If you want to populate `idx=M0` then
        the `ref=M1`.

        The algorithm is as follows:
        M0_Q = (N0_Q - M1_Q) + N1_Q
        M0_R = (N0_R - M1_R) + N1_R.

        The Q,R coordinates are saved in self.hexgrid.
        
        If M1 (the reference) has no Q,R coordinates set yet, nothing happen. 0 returned
        If M0 (the index) already has Q,R coordinates, nothing happen. -1 returned
        If M0 (the index) has been populated, congrats, and 1 returned

        Returns
        -------
        1 : if index populated
        0 : if no Reference coords
        -1: if no need to population index
        (-2: if at least 1 neighbor is None)
        """
        
        # - No Reference
        if self.hexgrid[ref] is None:
            return 0
        if self.hexgrid[idx] is not None:
            return -1
        if self.hexgrid[neighbor0] is None or self.hexgrid[neighbor1] is None:
            return -2
        
        q_ref, r_ref = self.hexgrid[ref]
        q_0, r_0     = self.hexgrid[neighbor0]
        q_1, r_1     = self.hexgrid[neighbor1]
        self.hexgrid[idx] = [(q_0 - q_ref) + q_1, (r_0 - r_ref) + r_1]
        return 1


    def fetch_hexagon_references(self, hexcentral):
        """ """
        neightbors = self.get_idx_neighbors(hexcentral)
        for n1 in neightbors:
            if self.hexgrid[n1] is not None:
                n2 = [n2_ for n2_ in self.get_shared_neighbors(hexcentral, n1)
                          if self.hexgrid[n2_] is not None]
                if len(n2)>0:
                    return n1,n2[0]
        return None
        
    def populate_hexagon(self, central_hexagon):
        """ """
        refs = self.fetch_hexagon_references(central_hexagon)
        if refs is None:
            return
        n1_, ref1_ = refs
        index_to_populate = self.get_forth_index(central_hexagon, n1_, ref1_)
        
        # run to all. Passed it not necessary
        neighbors = self.get_idx_neighbors(central_hexagon)
        for i in neighbors:
            if index_to_populate is None:
                break
            out = self.populate(index_to_populate, central_hexagon, n1_, ref1_)
            n1_, ref1_        = index_to_populate, n1_
            index_to_populate = self.get_forth_index(central_hexagon, n1_, ref1_)
            
        self._hexafilled[central_hexagon] = True
        return neighbors

        
    # ---------- #
    #  Builder   #
    # ---------- #
    def build_qr_grid(self, ref_idx):
        """ 
        Parameters
        ----------
        ref_idx: [3 ints]
            The three indexes defining (0,0),(0,1),(1,0).
            They must be neighbors. (see `set_grid_reference`)
        """
        if len(ref_idx) != 3:
            raise TypeError("The given ref_idx must be a list of 3 indexes")
        # - Defining the Axis
        self.set_grid_reference(*ref_idx)


        # - How will that be built?
        n0, n1, ref1 = self.ref_idx
        goahead      = True
        running_i    = 0
        next_to_run  = [n0]
        
        while(goahead and running_i<self.npoints):
            potential_next = self.populate_hexagon(n0)
            if potential_next is not None:
                next_          = [n for n in potential_next if not self._hexafilled[n]]
            else:
                next_      = []
                
            next_to_run    = [n_ for n_ in next_to_run if n_ not in [n0]] + next_
            if len(next_to_run)>1:
                n0 = np.random.choice(next_to_run)
                running_i +=1
            else:
                goahead = False
            
                
                
        
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

    # - Building the Q,R grid
    @property
    def hexgrid(self):
        """ List containing the relation between index and (Q,R) """
        if self._derived_properties["hexgrid"] is None:
            self._derived_properties["hexgrid"] = np.asarray([None for i in range(self.npoints)])
        return self._derived_properties["hexgrid"]

    @property
    def _hexafilled(self):
        """ Dictionary containing the relation between index and (Q,R)"""
        if self._derived_properties["hexafilled"] is None:
            self._derived_properties["hexafilled"] = {i:False for i in range(self.npoints)}
        return self._derived_properties["hexafilled"]

    
    @property
    def ref_idx(self):
        """ This indexes used to define (0,0), (0,1), (1,0). This is filled when running `set_grid_reference` """
        return self._side_properties["ref_idx"]

    @property
    def grid_theta(self):
        """ """
        return self._derived_properties["grid_theta"]
    @property
    def grid_rotmatrix(self):
        """ Rotation matrix associated to reference id of the grid """
        return np.matrix([[np.cos(self.grid_theta),np.sin(self.grid_theta)],
                            [-np.sin(self.grid_theta), np.cos(self.grid_theta)]])
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
