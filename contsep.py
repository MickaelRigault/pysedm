#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is to get target or host spaxels in SEDM cube data.
-v.20200702: 'get_others_spaxels': fix 'spaxel_id' option, when 'forced_addcontsep_mag' works automatically.
-v.20200622: consider when no other spaxels in contsep mag, e.g. 20200620_ZTF20abffaxl.
-v.20200619: consider 'target_others_dist' for multiple ref sources.
-v.20200618: consider 'np.nanmin(target_others_dist) > 2' when 'refimg_coords_in_ifu.size' == 0.
-v.20200612: From 20200611_ZTF18aaiykoz, which show mostly overlappbed coordinates b/w the target and ref sources.
             1. add criteria such that 'np.nanmin(target_others_dist) > 2'. '2' is from minimum size of polygon diameter.
             > for that case, "self.target_contsep_mag = -99.0".
-v.20200609: From 20200609_ZTF20abahhml, which has 'nan' in contour arrays,
             1. add "not np.isnan(t_).any()" and change "geometry.Polygon(t_).area>2 (before 1)"
                for '_ifucounts_cleaned'.
-v.20200608: From 20200604_ZTF20abaszgh, which has lots of segemented small polygons,
             1. add '_ifucounts_cleaned'.
             2. For counting method, add 'if poly_.area < 100:'.
-v.20200427-28: 1.show_ifudata: add showing 'offset', 'contsep_mag' and 'fake_mag'.
             2. add 'fake_mag' (affects target) and 'forced_mag' (affects host) option.
-v.20200420: 'offset' option.
-v.20200407.
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

def get_spaxels_from_constsep(date, targetid, offset=None, fake_mag=None, forced_mag=None, isomag=None):
    """
    Parameters
    ----------
    date: str
        YYYYMMDD
    targetid: str
        unique part of the filename, i.e. ZTF name or timestamp like 06_10_45
    offset: list of 2 floats, default=[0.0, 0.0],
        x and y offset between the cube and reference images.
    fake_mag: float, default=16.0,
        fake target mag for the reference image.
    forced_mag: float, default=0.0, (1.0 is recommended to use host spectra)
        add forced_mag to contsep_mag in steps of 0.5 to get more host spaxels.
        ! if contpse_mag = 26 mag (the faintest), it raises error.

    Return
    ----------
    SEDM_CONTOUR class
    """

    if offset is None:
        offset = [0.0, 0.0]
    if fake_mag is None:
        fake_mag = 16.0
    if forced_mag is None:
        forced_mag = 0.0
    if isomag is None:
        isomag = [20, 26, 13]

    return SEDM_CONTOUR.from_sedmid(date, targetid, offset, fake_mag, forced_mag, isomag)


class SEDM_CONTOUR():

    #
    # Initializing
    #

    def __init__(self, astromfile, cube, offset, fake_mag, forced_mag, isomag_range):
        """
        "astromfile" will be provided with its full path.
        "isomag_range": lists to make an isomag contour, [start, end, step number]

        1. Set astrometry and download PS BytesIO data, so it takes time.
        2. Then, make a fake target in the reference PS image with 'fake_mag',
        3. Then, set isomag contour in reference PS data ("refcount") and IFU data ("ifucount").

        return self.target_consep_mag, self.target_polygon_loc

        """
        self.isomag = np.linspace(isomag_range[0], isomag_range[1], isomag_range[2])
        self.cube = cube

        self.set_target_astrometry(astromfile)
        self.set_ps_reference()
        self.set_offset(offset)

        self.set_fake_mag(fake_mag)
        self.set_fake_target_in_ref(self.fake_mag)
        self.set_ref_iso_contours()

        #self.set_forced_addcontsep_mag(forced_mag)
        self.forced_addcontsep_mag = forced_mag # to tweak host (others) contour.

    @classmethod
    def from_sedmid(cls, date, targetid, offset, fake_mag, forced_mag, isomag):
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

        return cls(filename, cube, offset, fake_mag, forced_mag, isomag)

    #
    #  SETTER
    #

    def set_target_astrometry(self, astromfile):
        """ """
        self.astro = astrometry.Astrometry(astromfile)

    def set_target_coords_in_ifu(self):
        """ """
        self._target_coords_in_ifu = self.iref.get_target_coordinate(where="ifu")

    def set_ps_reference(self):
        """ download PS BytesIO data (it takes time.). """
        self.iref = astrometry.IFUReference.from_astrometry(self.astro)

    def set_refimg_coords_in_ifu(self):
        """ """
        self._refimg_coords_in_ifu = self.iref.get_refimage_sources_coordinates("ifu", inifu=True)

    def set_target_others_dist(self):
        self._target_others_dist = []

        if self.refimg_coords_in_ifu.size > 0:
            for i in range(0, len(self.refimg_coords_in_ifu[0])):
                __target_others_dist = np.sqrt((self.target_coords_in_ifu[0] - self.refimg_coords_in_ifu[: ,i][0])**2 +
                                               (self.target_coords_in_ifu[1] - self.refimg_coords_in_ifu[: ,i][1])**2 )
                self._target_others_dist.append(__target_others_dist)
        else:
            # if there is no others in the cube, return half of cube size.
            self._target_others_dist = 20

    def set_offset(self, offset):
        """ set astrometry offset between reference image and cube image. """
        self.iref.set_ifu_offset( offset[0], offset[1] )
        self.offset = np.asarray( offset )

    def set_fake_mag(self, fake_mag):
        """ """
        self.fake_mag = fake_mag

    def set_fake_target_in_ref(self, fake_mag):
        """ get a fake target with magnitude of "fake_mag" in the reference PS imaga to make a contour for IFU data. """
        self.iref.photoref.build_pointsource_image(fake_mag)

    def set_ref_iso_contours(self):
        """ get iso mag contours in reference PS data ("refcounts"). """

        self.refcounts_ = self.iref.get_iso_contours(where="ref", isomag=self.isomag )

    def set_ifu_iso_contours(self):
        """ get iso mag contours in IFU data ("ifucounts"). """
        _ifucounts = self.iref.get_iso_contours(where="ifu", isomag=self.isomag )

        self._ifucounts_cleaned = {}
        for k, v in _ifucounts.items():
            self._ifucounts_cleaned[k] = [t_ for t_ in v if (len(t_)>4) and (not np.isnan(t_).any()) and (geometry.Polygon(t_).area>2) ]

    def set_target_contour_location(self):
        """
        find a location of a target contour in each isomag contour in the cube.

        Returns a list of [ (isomag, ifucount key of isomag, isomag index, 1 th array in isomag contour)]
        """

        __target_polygon_loc = []

        for mag_index in range( 0, len(self.ifucounts) ):

            for i in range(0, len(self.ifucounts[ list(self.ifucounts.keys())[mag_index] ])):
                poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[mag_index] ][i])
                target_polygon_ = poly.contains( geometry.Point( self.iref.get_target_coordinate("ifu")) )

                if target_polygon_:
                    __target_polygon_loc.append( (self.isomag[mag_index], list(self.ifucounts.keys())[mag_index], mag_index, i) )

        self._target_polygon_loc = np.array(__target_polygon_loc)

    #
    # RESET
    #

    def _reset_forced_addcontsep_mag(self, forced_mag):
        """ for host spaxels """
        self.forced_addcontsep_mag = forced_mag

    def _reset_fake_mag(self, fake_mag):
        """ """
        self.fake_mag = fake_mag
        self.set_fake_target_in_ref(self.fake_mag)
        self.set_ref_iso_contours()
        self.set_ifu_iso_contours()
        self.set_target_contour_location()

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
            poly_ref = geometry.Polygon( self.ifucounts[self.target_polygon_loc[len(self.target_polygon_loc) - i ][1] ][int(self.target_polygon_loc[len(self.target_polygon_loc)- i ][3])] )

            i_comp = i + 1
            poly_comp = geometry.Polygon( self.ifucounts[ self.target_polygon_loc[len(self.target_polygon_loc) - i_comp ][1] ][int(self.target_polygon_loc[len(self.target_polygon_loc) - i_comp ][3])] )

            if (poly_ref.area>50) & (poly_comp.area>50):
                continue

            elif poly_ref.area<10:
                return self.target_polygon_loc[ len(self.target_polygon_loc) - i + mag_step_diff ][0]

            else:
                poly_criteria = np.abs(poly_ref.area - poly_comp.area)
                if poly_criteria > 40:
                    return self.target_polygon_loc[ len(self.target_polygon_loc) - i_comp + mag_step_diff ][0]

    def get_target_faintest_contour_w_counting_method(self):
        """
        get a faintest contour for the target by counting how many sources are in i th contour.
        We want to have only '1 target source' in i th contour.
        This is possible
        1) because we know the location of the target contour from the function "get_target_contour_location()"
        2) BUT in reference PS image, there is no target.
        """

        mag_step_diff = len(self.ifucounts) - len(self.target_polygon_loc)
        target_faintest_isomag_key = self.target_polygon_loc[len(self.target_polygon_loc)-1][1]

        #refimg_coords_in_ifu = self.get_refimage_coords_in_ifu()

        refimg_x = self.refimg_coords_in_ifu[0]
        refimg_y = self.refimg_coords_in_ifu[1]

        # starts from the faintest in target_polygon_loc
        for i in range(1, len(self.target_polygon_loc)+mag_step_diff):
            poly_ = geometry.Polygon( self.ifucounts[ self.target_polygon_loc[len(self.target_polygon_loc)-i][1] ][int(self.target_polygon_loc[len(self.target_polygon_loc)-i][3])] )

            if poly_.area < 100:
                chk_poly_contain_src = []
                for src_num in range(0, len(refimg_x)):
                    chk_poly_contain_src.append(poly_.contains( geometry.Point([refimg_x[src_num], refimg_y[src_num]]) ))

                if any(chk_poly_contain_src) is False:
                    return self.target_polygon_loc[len(self.target_polygon_loc)-i][0]

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
                return target_consep_mag_

            else:
                #raise ValueError("Check the number of sources in the cube. If 1, use method='area'.")
                target_consep_mag_ = counting_method
                return target_consep_mag_

        elif method is "area":
            area_method = self.get_target_faintest_contour_w_area_method()
            counting_method = "Not used."

            target_contsep_mag_ = area_method

            return target_contsep_mag_

        elif method is "counting":
            counting_method = self.get_target_faintest_contour_w_counting_method()
            area_method = "Not used."

            target_consep_mag_ = counting_method

            return target_consep_mag_


    def get_target_contsep_information(self, forced_addcontsep=False):
        """
        get target faintes contour information, like faintes contour magnitude and its location in the target polygon.
        """

        if np.nanmin(self.target_others_dist) > 2.0:

            ## When there is only 1 marked target in the cube, use 'area' method.
            if self.refimg_coords_in_ifu.size == 0:
                self.target_contsep_mag = self.get_target_faintest_contour(method="area")
            else:
                self.target_contsep_mag  = self.get_target_faintest_contour()

            if forced_addcontsep:
                self.target_contsep_mag = self.target_contsep_mag + self.forced_addcontsep_mag

            target_contsep_mag_index_ = int( self.target_polygon_loc[self.target_polygon_loc[:,0] == self.target_contsep_mag][0][2] )
            target_contsep_array_index_  = int( self.target_polygon_loc[self.target_polygon_loc[:,0] == self.target_contsep_mag][0][3] )
            target_contsep_key_  = int( self.target_polygon_loc[self.target_polygon_loc[:,0] == self.target_contsep_mag][0][1] )

            return target_contsep_mag_index_, target_contsep_array_index_, target_contsep_key_

        else:
            self.target_contsep_mag = -99.0

            target_contsep_mag_index_ = np.nan
            target_contsep_array_index_  = np.nan
            target_contsep_key_ = np.nan

            return target_contsep_mag_index_, target_contsep_array_index_, target_contsep_key_

    #
    # Host
    #

    def set_host_coords_in_ifu(self):
        """ host identification and get its coordnates in pixel from PS wcs """

        if self.refimg_coords_in_ifu.size == 0:
            self._host_coords_in_ifu = self.iref.get_refimage_sources_coordinates(inifu=False, where='ifu')
            #self._host_coords_in_ifu = (-99.0, -99.0)
        else:

            host_coords_in_pspixel = self.iref.photoref.get_host_coords(inpixels=True)
            host_candi_coords_in_rcpixel = self.iref.get_refimage_sources_coordinates(inifu=True, where='ref')

            ps_rc_diff = []

            for i in range(0, len(host_candi_coords_in_rcpixel[0])):
                x_diff = host_coords_in_pspixel[0] - host_candi_coords_in_rcpixel[0][i]
                y_diff = host_coords_in_pspixel[1] - host_candi_coords_in_rcpixel[1][i]
                ps_rc_diff.append(np.sqrt(x_diff**2 + y_diff**2))

            host_index = np.argmin(ps_rc_diff)

            self._host_coords_in_ifu = (self.refimg_coords_in_ifu[0][host_index], self.refimg_coords_in_ifu[1][host_index])

    def set_host_contour_location(self):
        """
        find a location of a host contour in each isomag contour in the cube.

        Returns a list of [ (isomag, ifucount key of isomag, isomag index, 1 th array in isomag contour)]
        """

        __host_polygon_loc = []

        for mag_index in range( 0, len(self.ifucounts) ):

            for i in range(0, len(self.ifucounts[ list(self.ifucounts.keys())[mag_index] ])):
                poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[mag_index] ][i])
                host_polygon_ = poly.contains( geometry.Point( self.host_coords_in_ifu ) )

                if host_polygon_:
                    __host_polygon_loc.append( (self.isomag[mag_index], list(self.ifucounts.keys())[mag_index], mag_index, i) )

        self._host_polygon_loc = np.array(__host_polygon_loc)


    def get_host_radec_in_degree(self):
        """
        ???
        """

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
            self.target_coords_in_ifu
            self.refimg_coords_in_ifu
            self.host_coords_in_ifu
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

    def show_ifudata(self, wcontour=True, wtargetspaxel=False, wotherspaxel=False, whostspaxel=False, savefile=None):
        """ """

        self.get_target_contsep_information()

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
            target_ids_from_contsep = self.get_target_spaxels(spaxels_id=True)
            spaxel_patches = self.cube._display_im_(axim=ax, vmin="5", vmax="99")

            for i in target_ids_from_contsep:
                spaxel_patches[i].set_edgecolor("red")
                spaxel_patches[i].set_linewidth(1)
                spaxel_patches[i].set_zorder(10)

        if wotherspaxel:
            others_ids_from_contsep = self.get_others_spaxels(spaxels_id=True)
            spaxel_patches = self.cube._display_im_(axim=ax, vmin="5", vmax="99")

            for i in others_ids_from_contsep:
                spaxel_patches[i].set_edgecolor("k")
                spaxel_patches[i].set_linewidth(1)
                spaxel_patches[i].set_zorder(9)

        if whostspaxel: # to be revised!!!
            # host identification
            if len(self.host_coords_in_ifu) > 0:
                ax.scatter(self.host_coords_in_ifu[0], self.host_coords_in_ifu[1],
                           color="pink", s=100, zorder=10)

            host_ids_from_contsep = self.get_host_spaxels(spaxels_id=True)
            spaxel_patches = self.cube._display_im_(axim=ax, vmin="5", vmax="99")

            for i in host_ids_from_contsep:
                spaxel_patches[i].set_edgecolor("pink")
                spaxel_patches[i].set_linewidth(1)
                spaxel_patches[i].set_zorder(9)

        t1 = "offset=(%.1f, %.1f), contsep_mag for target=%.1f mag (other=%.1f mag)" % (self.offset[0], self.offset[1], self.target_contsep_mag-self.forced_addcontsep_mag, self.target_contsep_mag)
        t2 = "fake_mag=%.1f mag, forced_mag=%.1f mag" % (self.fake_mag, self.forced_addcontsep_mag)

        ax.set_xlim(-20,19)
        ax.set_ylim(-24,22)

        ax.text(-20, 24.5, t1, fontsize=10)
        ax.text(-20, 22.5, t2, fontsize=10)

        if savefile is not None:
            fig.savefig( savefile )

    ######################
    # Method for Spaxels #
    ######################

    def get_target_spaxels(self, spaxels_id=True):
        """
        get target spaxels from the faintest consep.

        Parameters
        ----------
        spaxels_id: bool -optional-
            Deafult is return spaxel id (not index).
            If True, return spaxels idenx (for using pysedm).

        Returns
        -------
        Return spaxel ids or indexes in numpy array.
        """

        target_contsep_mag_index, target_contsep_array_index, target_contsep_key = self.get_target_contsep_information(forced_addcontsep=False)

        if np.nanmin(self.target_others_dist) > 2.0:
            target_contsep_poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[target_contsep_mag_index] ][target_contsep_array_index] )

        else:
            target_contsep_poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[0] ][0] )

        target_contsep_spaxel_index = self.cube.get_spaxels_within_polygon(target_contsep_poly) #spaxel index

        if spaxels_id:
            target_contsep_spaxel_ids = np.arange(self.cube.nspaxels)[np.isin(self.cube.indexes,target_contsep_spaxel_index)] #spaxel index to id for drawing.
            return target_contsep_spaxel_ids
        else:
            return np.array(target_contsep_spaxel_index)


    def get_others_spaxels(self, spaxels_id=True):
        """
        get other sources' spaxels (including host and so on).

        Parameters
        ----------
        spaxels_id: bool -optional-
            Deafult is return spaxel id (not index).
            If True, return spaxels idenx (for using pysedm).
        forced_addcontsep_mag: float in 0.5 step -optional-
            To change 'target_contsep_mag' to get more wider other sources spaxels.
            e.g., when SN is exploded very close to the bright center, the number of target spaxels are mostly less than 10.


        Returns
        -------
        Return spaxel ids or indexes in numpy array.
        """

        spaxels_id_for_forcedmag = spaxels_id

        if np.nanmin(self.target_others_dist) > 2.0:
            _target_contsep_spaxel_index = self.get_target_spaxels(spaxels_id=False)

            #if len(_target_contsep_spaxel_index) < 11: # when SN is exploded close to the bright center.

            if (self.forced_addcontsep_mag > 0.0) and (self.get_target_faintest_contour() < 26.0): # TO BE CAHGNED
                 target_contsep_mag_index, target_contsep_array_index, target_contsep_key = self.get_target_contsep_information(forced_addcontsep=True)
                 others_contsep_mag_index = target_contsep_mag_index
                 others_contsep_array_index = [i for i in range(0,len( self.ifucounts[ list(self.ifucounts.keys())[others_contsep_mag_index] ] )) ]

                 __others_contsep_spaxel_ids = []
                 _others_contsep_spaxel_index = []

                 for i in range(0, len(others_contsep_array_index)):
                     others_contsep_polys = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[others_contsep_mag_index] ][others_contsep_array_index[i]] )

                     __others_contsep_spaxel_index = self.cube.get_spaxels_within_polygon(others_contsep_polys) #spaxel index
                     if len(__others_contsep_spaxel_index) > 0:
                         _others_contsep_spaxel_index += __others_contsep_spaxel_index

                 others_contsep_spaxel_index = np.unique(_others_contsep_spaxel_index) #spaxel index

                # remove target spaxel
                 for spax_ in _target_contsep_spaxel_index:
                     others_contsep_spaxel_index = others_contsep_spaxel_index[others_contsep_spaxel_index != spax_]

            else:
                target_contsep_mag_index, target_contsep_array_index, target_contsep_key = self.get_target_contsep_information(forced_addcontsep=False)
                others_contsep_mag_index = target_contsep_mag_index
                others_contsep_array_index = [i for i in range(0,len( self.ifucounts[ list(self.ifucounts.keys())[others_contsep_mag_index] ] )) if i != target_contsep_array_index]

                __others_contsep_spaxel_ids = []
                _others_contsep_spaxel_index = []

                for i in range(0, len(others_contsep_array_index)):
                    others_contsep_polys = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[others_contsep_mag_index] ][others_contsep_array_index[i]] )

                    __others_contsep_spaxel_index = self.cube.get_spaxels_within_polygon(others_contsep_polys) #spaxel index
                    if len(__others_contsep_spaxel_index) > 0:
                        _others_contsep_spaxel_index += __others_contsep_spaxel_index

                others_contsep_spaxel_index = np.unique(_others_contsep_spaxel_index) #spaxel index

            if (others_contsep_spaxel_index.size < 10) and (self.forced_addcontsep_mag < 1.6):
                self.forced_addcontsep_mag += 0.5
                return self.get_others_spaxels(spaxels_id=spaxels_id_for_forcedmag)

            if spaxels_id:
                others_contsep_spaxel_ids = np.arange(self.cube.nspaxels)[np.isin(self.cube.indexes, others_contsep_spaxel_index)] #spaxel index to id for drawing.
                return others_contsep_spaxel_ids
            else:
                return others_contsep_spaxel_index

        else:
            self.target_contsep_mag = -99.0

            others_contsep_spaxel_index = np.empty(0)
            return others_contsep_spaxel_index


    def get_host_spaxels(self, spaxels_id=True, update=False):
        """
        get host spaxels from the faintest consep (or + forced_addcontsep_mag).

        Parameters
        ----------
        spaxels_id: bool -optional-
            Deafult is return spaxel id (not index).
            If True, return spaxels idenx (for using pysedm).

        Returns
        -------
        Return spaxel ids or indexes in numpy array.
        """

        _spaxels_id = spaxels_id

        if len(self.host_coords_in_ifu) > 0:
            #if (self.forced_addcontsep_mag > 0.0) and (self.get_target_faintest_contour() < 26.0):
        #if np.nanmin(self.target_others_dist) > 2.0:

            if update is False:
                self._set_host_contsep_information(again=False)

            only_host_check = self._host_polygon_check(self.host_contsep_poly)

            if only_host_check is not 1:
                # means there are host+other src.
                self._set_host_contsep_information(again=True)
                self.get_host_spaxels(spaxels_id=_spaxels_id, update=True)

            host_contsep_spaxel_index = self.cube.get_spaxels_within_polygon(self.host_contsep_poly) #spaxel index

            if spaxels_id:
                host_contsep_spaxel_ids = np.arange(self.cube.nspaxels)[np.isin(self.cube.indexes, host_contsep_spaxel_index)] #spaxel index to id for drawing.
                return host_contsep_spaxel_ids
            else:
                return np.array(host_contsep_spaxel_index)

        #else:
        #    host_contsep_poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[0] ][0] )

    def _set_host_contsep_information(self, again=False):
        """
        """
        if again is False:
            target_contsep_mag_index, _, _ = self.get_target_contsep_information(forced_addcontsep=False)
            self.host_contsep_mag_index = target_contsep_mag_index
            self.host_contsep_array_index  = int( self.host_polygon_loc[self.host_polygon_loc[:,0] == self.target_contsep_mag][0][3] )
            self.host_contsep_mag = self.target_contsep_mag
        else:
            self.host_contsep_mag_index -= 1
            self.host_contsep_array_index = int( self.host_polygon_loc[self.host_polygon_loc[:,2] == self.host_contsep_mag_index][0][3] )
            self.host_contsep_mag = self.host_polygon_loc[self.host_polygon_loc[:,2] == self.host_contsep_mag_index][0][0]

        self.host_contsep_poly = geometry.Polygon(self.ifucounts[ list(self.ifucounts.keys())[self.host_contsep_mag_index] ][self.host_contsep_array_index] )

    def _host_polygon_check(self, host_contsep_poly):
        """
        Check there is only host in the host polygon.
        """

        refimg_x = self.refimg_coords_in_ifu[0]
        refimg_y = self.refimg_coords_in_ifu[1]

        chk_host_poly_contain_others = []

        for src_num in range(0, len(refimg_x)):
            chk_host_poly_contain_others.append(host_contsep_poly.contains( geometry.Point([refimg_x[src_num], refimg_y[src_num]]) ))

        return chk_host_poly_contain_others.count(True)




    ###################
    # Properties      #
    ###################

    @property
    def ifucounts(self):
        """
        IFU iso contours.
        If an offset has changed, re-set ifucounts considering the offset.
         """
        if (not hasattr(self, "_ifucounts_cleaned")) | any( (self.offset != self.iref.ifu_offset) ) :
            self.set_ifu_iso_contours()

        return self._ifucounts_cleaned

    @property
    def target_polygon_loc(self):
        """
        """
        if not hasattr(self, "_target_polygon_loc"):
            self.set_target_contour_location()

        return self._target_polygon_loc

    @property
    def target_coords_in_ifu(self):
        """
        """
        if ( not hasattr(self, "_target_coords_in_ifu") ) | any( (self.offset != self.iref.ifu_offset) ):
            self.set_target_coords_in_ifu()

        return self._target_coords_in_ifu

    @property
    def refimg_coords_in_ifu(self):
        """
        """
        if ( not hasattr(self, "_refimg_coords_in_ifu") ) | any( (self.offset != self.iref.ifu_offset) ):
            self.set_refimg_coords_in_ifu()

        return self._refimg_coords_in_ifu

    @property
    def target_others_dist(self):
        """
        """
        if not hasattr(self, "_target_others_dist"):
            self.set_target_others_dist()

        return self._target_others_dist

    @property
    def host_coords_in_ifu(self):
        """
        """
        if ( not hasattr(self, "_host_coords_in_ifu") ) | any( (self.offset != self.iref.ifu_offset) ):
            self.set_host_coords_in_ifu()

        return self._host_coords_in_ifu

    @property
    def host_polygon_loc(self):
        """
        """
        if not hasattr(self, "_host_polygon_loc"):
            self.set_host_contour_location()

        return self._host_polygon_loc
