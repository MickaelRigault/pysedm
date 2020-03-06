#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" This module is to get target or host spaxels in SEDM cube data. """

import pyifu
import numpy as np
import matplotlib.pyplot as mpl
from shapely import geometry

import pysedm
from pysedm import astrometry

from sedmhaz import sedmz

####################
#                  #
#   CLASS          #
#                  #
####################

def get_spaxels_from_constsep(date, targetid):
    """ get target spaxels
    !!! TO DO THAT, cube from pysedm should be loaded.
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

        self.set_target_astrometry(astromfile)
        self.set_ps_reference()
        self.set_fake_target_in_ref(fake_mag)
        self.isomag = np.linspace(isomag_range[0], isomag_range[1], isomag_range[2])
        self.cube = cube
        self.set_target_consep_information()

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
        cube = sedmz.get_cleaned_sedmcube(date, targetid)
        filename = pysedm.io.get_night_files(date, "cube", targetid)[0]
        return cls(filename, cube)

    #
    #  Methods
    #

    def set_target_astrometry(self, astromfile):
        """ """
        self.astromfile = astromfile
        self.astro = astrometry.Astrometry(self.astromfile)

    def set_ps_reference(self):
        """ download PS BytesIO data (it takes time.). """
        self.iref = astrometry.IFUReference.from_astrometry(self.astro)
        self.refimage_coords_in_ifu = self.iref.get_refimage_sources_coordinates("ifu", inifu=True)

        #return self.iref

    def set_fake_target_in_ref(self, fake_mag):
        """ get a fake target with magnitude of "fake_mag" in the reference PS imaga to make a contour for IFU data. """
        return self.iref.photoref.build_pointsource_image(fake_mag)

    def set_target_consep_information(self):
        """
        """
        ## When there is only 1 marked target in the cube, use 'area' method.
        if self.refimage_coords_in_ifu.size == 0:
            target_consep_mag, target_polygon_loc = self.get_target_faintest_contour(method="area")
        else:
            target_consep_mag, target_polygon_loc = self.get_target_faintest_contour()

        self.target_consep_mag_index = int( target_polygon_loc[target_polygon_loc[:,0] == target_consep_mag][0][2] )
        self.target_consep_array_index  = int( target_polygon_loc[target_polygon_loc[:,0] == target_consep_mag][0][3] )

        #return self.target_consep_mag_index, self.target_consep_array_index

    def get_iso_contours(self):
        """ get iso mag contours in reference PS data ("refcounts") and IFU data ("ifucounts"). """

        self.refcounts = self.iref.get_iso_contours(where="ref", isomag=self.isomag )

        self.ifucounts = self.iref.get_iso_contours(where="ifu", isomag=self.isomag )

        return self.refcounts, self.ifucounts

    def get_target_contour_location(self):
        """
        find a location of a contour, which contains a target, in each isomag contour.

        Returns a list of [ (isomag, ifucount key of isomag, isomag index, 1 th array in isomag contour)]
        """

        self.get_iso_contours()

        self.target_polygon_loc = []

        for mag_index in range( 0, len(self.ifucounts) ):

            for i in range(0, len(self.ifucounts[ list(self.ifucounts.keys())[mag_index] ])):
                poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[mag_index] ][i])
                self.target_polygon = poly.contains( geometry.Point( self.iref.get_target_coordinate("ifu")) )


                if self.target_polygon:
                    self.target_polygon_loc.append( (self.isomag[mag_index], list(self.ifucounts.keys())[mag_index], mag_index, i) )

        return self.target_polygon_loc

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
            poly_ref = geometry.Polygon( self.ifucounts[ self.target_polygon_loc[len(self.target_polygon_loc) - i ][1] ][self.target_polygon_loc[len(self.target_polygon_loc)- i ][3]] )
            #print(self.isomag[ len(self.target_polygon_loc) - i + mag_step_diff ], "mag contour area:", poly_ref.area)

            i_comp = i + 1
            poly_comp = geometry.Polygon( self.ifucounts[ self.target_polygon_loc[len(self.target_polygon_loc) - i_comp ][1] ][self.target_polygon_loc[len(self.target_polygon_loc) - i_comp ][3]] )

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

        #refimage_coords_in_ifu_ = self.iref.get_refimage_sources_coordinates("ifu", inifu=True)

        refimag_x = self.refimage_coords_in_ifu[0]
        refimag_y = self.refimage_coords_in_ifu[1]

        # starts from the faintest in target_polygon_loc
        for i in range(1, len(self.target_polygon_loc)+mag_step_diff):
            #print(i, self.isomag[ len(self.ifucounts) - i ], "mag contour:")
            poly_ = geometry.Polygon( self.ifucounts[ self.target_polygon_loc[len(self.target_polygon_loc)-i][1] ][self.target_polygon_loc[len(self.target_polygon_loc)-i][3]] )

            chk_poly_contain_src = []
            for src_num in range(0, len(self.refimage_coords_in_ifu[0])):
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

        self.get_target_contour_location()

        if method is "both":
            area_method = self.get_target_faintest_contour_w_area_method()
            counting_method = self.get_target_faintest_contour_w_counting_method()

            if area_method == counting_method:
                self.target_consep_mag = area_method
                #print("From area method = ", area_method, "mag contour.")
                #print("From counting method = ", counting_method, "mag contour.")

                return self.target_consep_mag, np.array(self.target_polygon_loc)
            else:
                raise ValueError("Check the number of sources in the cube. If 1, use method='area'.")

        elif method is "area":
            area_method = self.get_target_faintest_contour_w_area_method()
            counting_method = "Not used."

            #print("From area method = ", area_method, "mag contour.")
            #print("From counting method = ", counting_method)

            self.target_consep_mag = area_method

            return self.target_consep_mag, np.array(self.target_polygon_loc)

        elif method is "counting":
            counting_method = self.get_target_faintest_contour_w_counting_method()
            area_method = "Not used."

            #print("From area method = ", area_method)
            #print("From counting method = ", counting_method, "mag contour.")



    #
    # Show method
    #

    def show_ref_ifu(self, offset=(0.0, 0.0), savefile=None):
        """
        !!! offset will be given by hand after a visual check !!!
        """
        self.x_offset = offset[0]
        self.y_offset = offset[1]

        #self.iref.set_ifu_offset( offset[0], offset[1] )
        self.iref.set_ifu_offset( self.x_offset, self.y_offset )
        self.iref.show()

        if savefile is not None:
            fig.savefig( savefile )

    def show_catdata(self, savefile=None):
        """ """
        self.get_iso_contours()

        fig = mpl.figure()

        ax = fig.add_subplot(111)

        ax.imshow(np.log10(self.iref.photoref.fakeimage), origin="lower", cmap=mpl.cm.binary)

        ii=0
        for i,couts_ in self.refcounts.items():
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
        self.get_iso_contours()

        fig = mpl.figure()

        ax = fig.add_subplot(111)
        self.iref.set_ifu_offset( self.x_offset, self.y_offset )
        _ = self.iref.show_slice(ax=ax, vmax="96")

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
            target_ids_from_consep = self.get_target_spaxels()
            spaxel_patches = self.cube._display_im_(axim=ax, vmin="68", vmax="97")

            for i in target_ids_from_consep:
                spaxel_patches[i].set_edgecolor("red")
                spaxel_patches[i].set_linewidth(1)
                spaxel_patches[i].set_zorder(10)

        if wotherspaxel:
            others_ids_from_consep = self.get_others_spaxels()
            spaxel_patches = self.cube._display_im_(axim=ax, vmin="68", vmax="97")

            for i in others_ids_from_consep:
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

        target_consep_poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[self.target_consep_mag_index] ][self.target_consep_array_index])

        target_consep_poly_index = self.cube.get_spaxels_within_polygon(target_consep_poly) #spaxel index
        target_consep_spaxel_ids = np.arange(self.cube.nspaxels)[np.isin(self.cube.indexes,target_consep_poly_index)] #spaxel index to id for drawing.

        return target_consep_spaxel_ids

    def get_others_spaxels(self):
        """
        get other sources' spaxels (including host and so on).
        """

        others_consep_mag_index = self.target_consep_mag_index
        others_consep_array_index = [i for i in range(0,len( self.ifucounts[ list(self.ifucounts.keys())[others_consep_mag_index] ] )) if i != self.target_consep_array_index]

        __others_consep_spaxel_ids = []

        for i in range(0, len(others_consep_array_index)):
            others_consep_polys = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[others_consep_mag_index] ][others_consep_array_index[i]])
            others_consep_poly_index = self.cube.get_spaxels_within_polygon(others_consep_polys) #spaxel index
            __others_consep_spaxel_ids.append( np.arange(self.cube.nspaxels)[np.isin(self.cube.indexes,others_consep_poly_index)] ) #spaxel index to id for drawing.

        _others_consep_spaxel_ids = __others_consep_spaxel_ids[0]

        for i in range(0, len(__others_consep_spaxel_ids)):
            if len(__others_consep_spaxel_ids[i] > 0):
                _others_consep_spaxel_ids = np.append(_others_consep_spaxel_ids, __others_consep_spaxel_ids[i])

        others_consep_spaxel_ids = np.unique(_others_consep_spaxel_ids)

        return others_consep_spaxel_ids
