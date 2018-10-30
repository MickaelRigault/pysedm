#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy         as np
from scipy.spatial   import KDTree, distance
from propobject      import BaseObject

from .tools          import is_arraylike

__all__ = ["get_hexprojection"]


def load_hexprojection(hexfile):
    """ """
    hex_ = HexagoneProjection(None, empty=True)
    hex_.load(hexfile)
    return hex_

def get_hexprojection(xy, ids=None, qdistance=None,
                      reference_ids=None, build=True, theta=None, **kwargs):
    """ """
    hgrid = HexagoneProjection(xy, ids=ids, qdistance=qdistance, **kwargs)
    
    reference_idx = hgrid.get_default_grid_reference() \
      if reference_ids is None else hgrid.ids_to_index(reference_ids)

    hgrid.set_grid_reference(*reference_idx, theta=theta)
    if build:
        hgrid.build_qr_grid(hgrid.ref_idx)
        
    return hgrid

class HexagoneProjection( BaseObject ):
    """ This class enable to build th Q,R hexagone grid
    based on given reference positions 
    """
    PROPERTIES         = ["xy","qdistance","hexgrid", "neighbors"]
    SIDE_PROPERTIES    = ["ref_idx","index_ids", "rot_degree"]
    DERIVED_PROPERTIES = ["hexafilled","grid_theta",
                              "tree","multipoint","rotmatrix",
                              "ids_index"]

    def __init__(self, xy, qdistance=None, load=True, ids=None,
                     empty=False):
        """ """
        if empty:
            return
        
        self.set_xy(xy, ids=ids)
        self.set_qdistance(qdistance)
        if load:
            self.fetch_neighbors()

    # -------------- #
    #   I/O          #
    # -------------- #
    def writeto(self, savefile):
        """ """
        from .tools import dump_pkl
        data = {
            "neighbors": self.neighbors,
            "ids":       self.index_ids,
            "hexgrid":   self.hexgrid,
            "ref_idx":   self.ref_idx,
            "qdistance": self.qdistance
            }
            
        dump_pkl(data, savefile)

    def load(self, hexfile):
        """ load the hexagone grid from the pkl file containing the data """
        from .tools import load_pkl
        data = load_pkl(hexfile)
        
        if "neighbors" not in data.keys():
            raise TypeError("The given filename does not have the appropriate format. No 'neighbors' entry.")

            
        # You can rerun everything with this
        self.set_neighbors(data["neighbors"])
        
        #  Valuable information
        if "ref_idx" in data.keys():
            self.set_grid_reference(*data["ref_idx"])

        #  No need to run anything with that again
        if "hexgrid" in data.keys(): #
            self.set_hexgrid(data["hexgrid"])
            
            
        if "ids" in data.keys():
            self.set_ids(data["ids"])

        if "qdistance" in data.keys():
            self.set_qdistance(data["qdistance"])


    def show(self, ax=None, switch_axis=False, colored_by=None, **kwargs):
        """ """
        from ..sedm import SEDMSPAXELS
        import matplotlib.pyplot as mpl
        from matplotlib      import patches
        if ax is None:
            fig = mpl.figure(figsize=[5,5])
            ax  = fig.add_subplot(111)
        else:
            fig = ax.figure

        indexes = list(self.ids_index.keys())
        if colored_by is None:
            colors = mpl.cm.viridis(np.random.uniform(size=len(indexes)))
        else:
            vmin, vmax = np.percentile(colored_by, [0,100])
            colors = mpl.cm.viridis( (colored_by-vmin)/(vmax-vmin) )
            
        ps = [patches.Polygon(SEDMSPAXELS + self.index_to_xy(self.ids_to_index(id_), switch_axis=switch_axis),
                           facecolor=colors[i], alpha=0.8) for i,id_  in enumerate(indexes)]

        ip = [ax.add_patch(p_) for p_ in ps]
        ax.autoscale(True, tight=True)
        
        fig.show()
    # -------------- #
    #   SETTER       #
    # -------------- # 
    def set_xy(self, xy, ids=None):
        """ """
        self._properties["xy"] = np.asarray(xy)
        self._derived_properties["tree"] = KDTree(self.xy)
        self._properties["hexgrid"] = None # reset
        self.set_ids(ids)
        
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

    def set_ids(self, ids):
        """ The id corresponding the given coordinates (xy). """
        self._side_properties["index_ids"] = ids
        
    def fetch_neighbors(self):
        """ """
        self.set_neighbors( self.tree.query_ball_tree(self.tree, r=self.qdistance) )

    def set_neighbors(self, neighbors):
        """ """
        self._properties["neighbors"] = neighbors
        
    def set_hexgrid(self, hexgrid):
        """ directly provide the (Q,R) coordinates for the indexes. """
        self._properties["hexgrid"] = hexgrid
        
    # ---------- #
    #   GETTER   #
    # ---------- #
    def get_central_index(self):
        """ the closest point to the centroid """
        if self.centroid is None:
            raise ValueError("This cannot estimate the centroid (no Shapely?).")
        
        return np.argmin([distance.euclidean(xydata_,self.centroid) for xydata_ in self.xy])

    def get_default_grid_reference(self):
        """ Look For the central value and 3 neightbors to define the 
        (0,0), (0,1) and (1,0) coordinates. 
        Returns
        -------
        3 indexes (with Q,R coords: (0,0), (0,1) and (1,0))
        """
        r00 = self.get_central_index()
        r01 = self.get_idx_neighbors(r00)[0]
        r10 = self.get_shared_neighbors(r00,r01)[0]
        return r00, r01, r10
    
    def get_idx_neighbors(self, index):
        """ Gives the name of all the `index` neightbors.
        There should be 6 (since it is an haxagonal grid) but less is expected
        if the index is at the edge of the detector. 
        More would be sign of problem

        Returns
        -------
        list of indexes
        """
        # No test here for fast recursive tricks
        arr = np.asarray(self.neighbors[index])
        return arr[arr!=index]

    def get_shared_neighbors(self, index1, index2):
        """ Get the indexes that are neighbors of both indexes """
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
    def index_to_ids(self, index):
        """ given the index of the given ids """
        return self.index_ids[index]

    def ids_to_index(self, ids):
        """ given the id of the given index """        
        if is_arraylike(ids):
            return np.asarray([self.ids_to_index(ids_) for ids_ in ids])
        
        return self.ids_index[ids]
    
    def index_to_qr(self, index):
        """ get the (Q,R) hexagonal coordinates of the given index"""
        return self.hexgrid[index]

    def index_to_xy(self, index, invert_rotation=True, switch_axis=False):
        """ """
        qr = np.asarray(self.index_to_qr(index)).T

        if not is_arraylike(index):
            q,r = qr if qr is not None else [np.NaN, np.NaN]
        else:
            q,r = np.asarray([qr_ if qr_ is not None else [np.NaN, np.NaN]
                                  for qr_ in qr]).T
            
        return self.qr_to_xy(q,r, invert_rotation=invert_rotation,
                            switch_axis=switch_axis)
    
    def qr_to_xy(self, q, r, invert_rotation=True,  switch_axis=False):
        """ Convert (q,r) hexagonal grid coordinates into (x,y) system 
        Returns
        -------
        x,y
        """
        # INVERTING X, Y on purpose to match the natural ifu_x ifu_y coordinates
        x,y = np.asarray([ (2*q + r)/np.sqrt(3.),  r])
        if invert_rotation:
           x,y = np.dot(self.grid_rotmatrix, np.asarray([x,y]))

        if switch_axis:
            x,y = y,x
            
        if self.rot_degree is not None and self.rot_degree != 0:
            _rot = self.rot_degree  * np.pi / 180
            rotmat = np.asarray([[ np.cos(_rot), -np.sin(_rot)],[ np.sin(_rot), np.cos(_rot)]])
            x,y = np.dot(rotmat, np.asarray([x,y]))
            
        return x,y
    
    # -------------- #
    # Grid Building  #
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
            
    def set_rot_degree(self, rot_degree):
        """ Rotation with respect to the north, in degree"""
        self._side_properties["rot_degree"] = rot_degree
        
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
            next_          = [n for n in potential_next if not self._hexafilled[n]] if potential_next is not None\
              else []                
            next_to_run    = [n_ for n_ in next_to_run if n_ not in [n0]] + next_
            if len(next_to_run)>1:
                n0 = np.random.choice(next_to_run)
                running_i +=1
            else:
                goahead = False


    # ----------- #
    #   PLOTTER   #
    # ----------- #

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
        if self._properties["xy"] is not None:
            return len(self.xy)
        return len(self._properties["neighbors"]) if self._properties["neighbors"] is not None else None
        
    
    @property
    def qdistance(self):
        """ Distance used the query the neighbors """
        if self._properties["qdistance"] is None:
            self.set_qdistance(None)
        return self._properties["qdistance"]

    @property
    def index_ids(self):
        """ IDS corresponding the to indexes. If nothing set, this simply is this index. """
        if self._side_properties["index_ids"] is None:
            self._side_properties["index_ids"] = np.arange(self.npoints)
            
        return self._side_properties["index_ids"]

    @property
    def ids_index(self):
        """ IDs corresponding the to indexes. If nothing set, this simply is this index. """
        if self._derived_properties["ids_index"] is None:
            self._derived_properties["ids_index"] = {self.index_ids[i]:i for i in np.arange(self.npoints)}
            
        return self._derived_properties["ids_index"]


    # ----------
    # Side
    @property
    def rot_degree(self):
        """ """
        return self._side_properties["rot_degree"]
    # ----------
    # Derived
    @property
    def tree(self):
        """ The KDtree associated with the xy data"""
        return self._derived_properties["tree"]

    @property
    def neighbors(self):
        """ The neightbors """
        return self._properties["neighbors"]

    # - Building the Q,R grid
    @property
    def hexgrid(self):
        """ List containing the relation between index and (Q,R) """
        if self._properties["hexgrid"] is None:
            self._properties["hexgrid"] = np.asarray([None for i in range(self.npoints)])
        return self._properties["hexgrid"]

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
        """ Angle [rad] that the (0,1), (0,0) (1,0) coordinates do. """
        if self._derived_properties["grid_theta"] is None:
            self._derived_properties["grid_theta"] = 0
        return self._derived_properties["grid_theta"] 
    
    @property
    def grid_rotmatrix(self):
        """ Rotation matrix associated to the grid. (based on self.grid_theta) """
        return np.asarray([[np.cos(self.grid_theta),np.sin(self.grid_theta)],
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
