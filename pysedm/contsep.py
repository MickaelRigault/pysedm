#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is to get target or host spaxels in SEDM cube data.
NOTE: This module is working with the cube observed after Nov 10 2019.
      Before that date, an error related to 'old format' is raised.
"""

import numpy as np
import matplotlib.pyplot as mpl
from shapely import geometry

from . import io
from . import astrometry
from . import sedm

####################
#                  #
#   CLASS          #
#                  #
####################

def get_spaxels_from_constsep(date, targetid):
    """
    Parameters
    ----------
    date: str
        YYYYMMDD
    targetid: str
        unique part of the filename, i.e. ZTF name or timestamp like 06_10_45

    Return
    ----------
    SEDM_CONTOUR class
    """
    return SEDM_CONTOUR.from_sedmid(date, targetid)


class SEDM_CONTOUR():

    #
    # Initializing
    #

    def __init__(self, astromfile, cube, isomag_range=[20, 26, 13], fake_mag=18.0):
        """
        "astromfile" will be provided with its full path.
        "isomag_range": lists to make an isomag contour, [start, end, step number]
        "fake_mag": magnitude to make a fake target in reference image.

        1. Set astrometry and download PS BytesIO data, so it takes time.
        2. Then, make a fake target in the reference PS image with 'fake_mag',
        3. Then, set isomag contour in reference PS data ("refcount") and IFU data ("ifucount").

        return self.target_consep_mag, self.target_polygon_loc

        """
        self.isomag = np.linspace(isomag_range[0], isomag_range[1], isomag_range[2])
        self.cube = cube

        self.set_target_astrometry(astromfile)
        self.set_ps_reference()
        self.set_fake_target_in_ref(fake_mag)
        self.set_ref_iso_contours()

        self.offset = np.asarray( [0,0] )

    @classmethod
    def from_sedmid(cls, date, targetid):
        """
        Parameters
        ----------
        date: str
            YYYYMMDD
        targetid: str
            unique part of the filename, i.e. ZTF name or timestamp like 06_10_45

        Returns
        -------
        Instance of the class (like a __init__)
        """
        filename = io.get_night_files(date, "cube", targetid)[0]
        cube = sedm.get_sedmcube(filename)

        return cls(filename, cube)

    #
    #  SETTER
    #

    def set_target_astrometry(self, astromfile):
        """ """
        self.astro = astrometry.Astrometry(astromfile)

    def set_ps_reference(self):
        """ download PS BytesIO data (it takes time.). """
        self.iref = astrometry.IFUReference.from_astrometry(self.astro)

    def set_ref_iso_contours(self):
        """ get iso mag contours in reference PS data ("refcounts"). """

        self.refcounts_ = self.iref.get_iso_contours(where="ref", isomag=self.isomag )

    def set_fake_target_in_ref(self, fake_mag):
        """ get a fake target with magnitude of "fake_mag" in the reference PS imaga to make a contour for IFU data. """
        return self.iref.photoref.build_pointsource_image(fake_mag)

    def set_ifu_iso_contours(self):
        """ get iso mag contours in IFU data ("ifucounts"). """

        self._ifucounts = self.iref.get_iso_contours(where="ifu", isomag=self.isomag )

    def set_target_contour_location(self):
        """
        find a location of a target contour in each isomag contour in the cube.

        Returns a list of [ (isomag, ifucount key of isomag, isomag index, 1 th array in isomag contour)]
        """

        #ifucounts = self.get_ifu_iso_contours()

        __target_polygon_loc = []

        for mag_index in range( 0, len(self.ifucounts) ):

            for i in range(0, len(self.ifucounts[ list(self.ifucounts.keys())[mag_index] ])):
                poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[mag_index] ][i])
                target_polygon_ = poly.contains( geometry.Point( self.iref.get_target_coordinate("ifu")) )

                if target_polygon_:
                    __target_polygon_loc.append( (self.isomag[mag_index], list(self.ifucounts.keys())[mag_index], mag_index, i) )

        self._target_polygon_loc = np.array(__target_polygon_loc)

    #
    #  GETTER
    #

    def get_refimage_coords_in_ifu(self):
        """ get refimage coordinates in ifu, considering offset! """

        refimage_coords_in_ifu = self.iref.get_refimage_sources_coordinates("ifu", inifu=True)

        return refimage_coords_in_ifu

    def get_target_faintest_contour_w_area_method(self):
        """
        get a faintest contour for the target by comparing contour area
        ,between the i th and (i-1) th contours, starting from the faintest contour.

        Basic concept is that
        1) if areas for both contours are big (here >50), that means both contours include (most likely) with other sources, like a galxy.
        2) or if the ith contour area is as small as 10, that contour is for the target.
        3) or if the contour area difference (i th - (i-1) th) is larger than 40, i th contour is for the target.
        """

        mag_step_diff = len(self.ifucounts) - len(self.target_polygon_loc)

        for i in range(1, len(self.target_polygon_loc)+mag_step_diff):
            poly_ref = geometry.Polygon( self.ifucounts[ self.target_polygon_loc[len(self.target_polygon_loc) - i ][1] ][int(self.target_polygon_loc[len(self.target_polygon_loc)- i ][3])] )
            #print(self.isomag[ len(self.target_polygon_loc) - i + mag_step_diff ], "mag contour area:", poly_ref.area)

            i_comp = i + 1
            poly_comp = geometry.Polygon( self.ifucounts[ self.target_polygon_loc[len(self.target_polygon_loc) - i_comp ][1] ][int(self.target_polygon_loc[len(self.target_polygon_loc) - i_comp ][3])] )

            if (poly_ref.area>50) & (poly_comp.area>50):
                continue
                #print("too big\n")
            elif poly_ref.area<10:
                #print("Faint mag contour for target is in", self.isomag[ len(self.target_polygon_loc) - i + mag_step_diff ], " mag contour.")
                return self.isomag[ len(self.target_polygon_loc) - i + mag_step_diff ]
            else:
                poly_criteria = np.abs(poly_ref.area - poly_comp.area)
                if poly_criteria > 40:
                    #print("Faintest mag contour only for the target is in", self.isomag[ len(self.target_polygon_loc) - i_comp + mag_step_diff ], "mag contour.")
                    return self.isomag[ len(self.target_polygon_loc) - i_comp + mag_step_diff ]

    def get_target_faintest_contour_w_counting_method(self):
        """
        get a faintest contour for the target by counting how many sources are in i th contour.
        We want to have only '1 target source' in i th contour.
        This is possible
        1) because we know the location of the target contour from the function "get_target_contour_location()"
        2) BUT in reference PS image, there is no target.
        """

        mag_step_diff = len(self.ifucounts) - len(self.target_polygon_loc)

        refimag_coords_in_ifu = self.get_refimage_coords_in_ifu()

        refimag_x = refimag_coords_in_ifu[0]
        refimag_y = refimag_coords_in_ifu[1]

        # starts from the faintest in target_polygon_loc
        for i in range(1, len(self.target_polygon_loc)+mag_step_diff):
            #print(i, self.isomag[ len(self.ifucounts) - i ], "mag contour:")
            poly_ = geometry.Polygon( self.ifucounts[ self.target_polygon_loc[len(self.target_polygon_loc)-i][1] ][int(self.target_polygon_loc[len(self.target_polygon_loc)-i][3])] )

            chk_poly_contain_src = []
            for src_num in range(0, len(refimag_x)):
                chk_poly_contain_src.append(poly_.contains( geometry.Point([refimag_x[src_num], refimag_y[src_num]]) ))

            if any(chk_poly_contain_src) is False:
                #print("Faintest mag contour only for the target is in", self.isomag[ len(self.ifucounts) - i ], " mag contour.")
                return self.isomag[ len(self.ifucounts) - i ]

    def get_target_faintest_contour(self, method="both"):
        """
        NOTE: When there is only 1 marked target in the cube,
              counting method always returns the faintest defined isomag, e.g., here 26.0 mag.
              This case, let's try only with 'area' method: "get_target_faintest_contour(method='area')."
        """

        if method is "both":
            area_method = self.get_target_faintest_contour_w_area_method()
            counting_method = self.get_target_faintest_contour_w_counting_method()

            if area_method == counting_method:
                target_consep_mag_ = area_method
                #print("From area method = ", area_method, "mag contour.")
                #print("From counting method = ", counting_method, "mag contour.")

                return target_consep_mag_
            else:
                raise ValueError("Check the number of sources in the cube. If 1, use method='area'.")

        elif method is "area":
            area_method = self.get_target_faintest_contour_w_area_method()
            counting_method = "Not used."

            #print("From area method = ", area_method, "mag contour.")
            #print("From counting method = ", counting_method)

            target_consep_mag_ = area_method

            return target_consep_mag

        elif method is "counting":
            counting_method = self.get_target_faintest_contour_w_counting_method()
            area_method = "Not used."

            #print("From area method = ", area_method)
            #print("From counting method = ", counting_method, "mag contour.")

            target_consep_mag_ = counting_method

            return target_consep_mag_

    def get_target_contsep_information(self):
        """
        get target faintes contour information, like faintes contour magnitude and its location in the target polygon.
        """
        refimage_coords_in_ifu = self.get_refimage_coords_in_ifu()

        ## When there is only 1 marked target in the cube, use 'area' method.
        if refimage_coords_in_ifu.size == 0:
            #ifucounts = self.get_ifu_iso_contours()
            target_contsep_mag = self.get_target_faintest_contour(method="area", target_polygon_loc=target_polygon_loc)
        else:
            target_contsep_mag  = self.get_target_faintest_contour()

        target_contsep_mag_index_ = int( self.target_polygon_loc[self.target_polygon_loc[:,0] == target_contsep_mag][0][2] )
        target_contsep_array_index_  = int( self.target_polygon_loc[self.target_polygon_loc[:,0] == target_contsep_mag][0][3] )

        return target_contsep_mag_index_, target_contsep_array_index_

    #
    # SHOW
    #

    def show_ref_ifu(self, offset=[0.0, 0.0], savefile=None):
        """
        !!! offset will be given by hand after a visual check !!!
        """

        if any(self.offset != offset) is True:
            self.iref.set_ifu_offset( offset[0], offset[1] )
            self.ifucounts
            self.offset = np.asarray( offset )
        self.iref.show()

        if savefile is not None:
            fig.savefig( savefile )

    def show_catdata(self, savefile=None):
        """ """

        fig = mpl.figure()

        ax = fig.add_subplot(111)

        ax.imshow(np.log10(self.iref.photoref.fakeimage), origin="lower", cmap=mpl.cm.binary)

        ii=0
        for i,couts_ in self.refcounts_.items():
            for n, contour in enumerate(couts_):
                ax.plot(contour[:, 0], contour[:, 1], linewidth=2, color="C%d"%ii)
            ii+=1

        ra,dec,mag = self.iref.photoref._pstarget.catdata[["raMean","decMean","rPSFMag"]].values.T
        x,y = self.iref.photoref.refimage.coords_to_pixel(ra,dec).T
        flagout = (mag<0)
        ax.scatter(x[~flagout],y[~flagout], marker=".", c=mag[~flagout])

        if savefile is not None:
            fig.savefig( savefile )

    def show_ifudata(self, wcontour=True, wtargetspaxel=False, wotherspaxel=False, savefile=None):
        """ """

        fig = mpl.figure()

        ax = fig.add_subplot(111)
        _ = self.iref.show_slice(ax=ax, vmin="5", vmax="99")

        if wcontour:
            ii=0
            for couts_ in self.ifucounts.keys():
                for n, contour in enumerate(self.ifucounts[couts_]):
                    if n==0:
                        ax.plot(contour[:, 0], contour[:, 1], linewidth=2, color="C%d"%ii, zorder=8, label=self.isomag[ii])
                    else:
                        ax.plot(contour[:, 0], contour[:, 1], linewidth=2, color="C%d"%ii, zorder=8)
                ii+=1
            ax.legend()

        if wtargetspaxel:
            target_ids_from_contsep = self.get_target_spaxels()
            spaxel_patches = self.cube._display_im_(axim=ax, vmin="5", vmax="99")

            for i in target_ids_from_contsep:
                spaxel_patches[i].set_edgecolor("red")
                spaxel_patches[i].set_linewidth(1)
                spaxel_patches[i].set_zorder(10)

        if wotherspaxel:
            others_ids_from_contsep = self.get_others_spaxels()
            spaxel_patches = self.cube._display_im_(axim=ax, vmin="5", vmax="99")

            for i in others_ids_from_contep:
                spaxel_patches[i].set_edgecolor("k")
                spaxel_patches[i].set_linewidth(1)
                spaxel_patches[i].set_zorder(9)

        if savefile is not None:
            fig.savefig( savefile )

    ######################
    # Method for Spaxels #
    ######################

    def get_target_spaxels(self):
        """
        get target spaxels from the faintest consep.

        return spaxel ids in numpy array.
        """

        target_contsep_mag_index, target_contsep_array_index = self.get_target_contsep_information()

        target_contsep_poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[target_contsep_mag_index] ][target_contsep_array_index] )

        target_contsep_poly_index = self.cube.get_spaxels_within_polygon(target_contsep_poly) #spaxel index
        target_contsep_spaxel_ids = np.arange(self.cube.nspaxels)[np.isin(self.cube.indexes,target_contsep_poly_index)] #spaxel index to id for drawing.

        return target_contsep_spaxel_ids

    def get_others_spaxels(self):
        """
        get other sources' spaxels (including host and so on).
        !!! It should be updated to select only host spaxels !!!
        """

        target_contsep_mag_index, target_contsep_array_index = self.get_target_contsep_information()

        others_contsep_mag_index = target_contsep_mag_index
        others_contsep_array_index = [i for i in range(0,len( self.ifucounts[ list(self.ifucounts.keys())[others_contsep_mag_index] ] )) if i != target_contsep_array_index]

        __others_contsep_spaxel_ids = []

        for i in range(0, len(others_contsep_array_index)):
            others_contsep_polys = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[others_contsep_mag_index] ][others_contsep_array_index[i]] )

            others_contsep_poly_index = self.cube.get_spaxels_within_polygon(others_contsep_polys) #spaxel index
            __others_contsep_spaxel_ids.append( np.arange(self.cube.nspaxels)[np.isin(self.cube.indexes,others_contsep_poly_index)] ) #spaxel index to id for drawing.

        _others_contsep_spaxel_ids = __others_contsep_spaxel_ids[0]

        for i in range(0, len(__others_contsep_spaxel_ids)):
            if len(__others_contsep_spaxel_ids[i] > 0):
                _others_contsep_spaxel_ids = np.append(_others_contsep_spaxel_ids, __others_contsep_spaxel_ids[i])

        others_contsep_spaxel_ids = np.unique(_others_contsep_spaxel_ids)

        return others_contsep_spaxel_ids

    ###################
    # Properties      #
    ###################

    @property
    def ifucounts(self):
        """
        IFU iso contours.
        If an offset has changed, re-set ifucounts considering the offset.
         """
        if (not hasattr(self, "_ifucounts")) | any( (self.offset != self.iref.ifu_offset) ) :
            self.set_ifu_iso_contours()

        return self._ifucounts

    @property
    def target_polygon_loc(self):
        """
        """
        if not hasattr(self, "_target_polygon_loc"):
            self.set_target_contour_location()

        return self._target_polygon_loc
