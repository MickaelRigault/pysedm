#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is to remove cosmic ray spaxels in the SEDM datacube.
-v20201218: add "show_cr_spaxels".
-v20201110.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
from shapely.geometry import Point

from . import sedm
from . import io

def get_cr_spaxels_from_byecr(date, targetid):
    """
    Parameters
    ----------
    date: str
        YYYYMMDD
    targetid: str
        unique part of the filename, i.e. ZTF name or timestamp like 06_10_45

    Return
    ----------
    SEDC_BYECR class
    """

    return SEDM_BYECR.from_sedmid(date, targetid)

####################
#                  #
#   CLASS          #
#                  #
####################

class SEDM_BYECR():

    def __init__(self, date, cube):
        """
        set cube, hexagrid, and normalized cube.
        Then, derived_df also defines with spatial neighbors indexes.
        """

        self.cube = cube

        self.set_hexagrid(date)
        self.set_normalized_cube()

        self.derived_df = self.get_spatial_neighbors_indexes().reset_index()


    @classmethod
    def from_sedmid(cls, date, targetid):
        """
        """

        filename = io.get_night_files(date, "cube", targetid)[0]
        cube = sedm.get_sedmcube(filename)

        return cls(date, cube)


    #
    # SETTER
    #

    def set_hexagrid(self, date):
        """ NEED date_HexaGrid.pkl file. """
        self.hexagrid = io.load_nightly_hexagonalgrid(date)

    def set_normalized_cube(self):
        """
        Normalize every spaxel in a slice with same spaxels in -10 +10 slices.

        Returns
        -------
        normalized cube data and error (not variance).
        """

        #normalized_cube_data = np.empty( [self.cube.data.shape[0], self.cube.data.shape[1]])
        #normalized_cube_err = np.empty( [self.cube.data.shape[0], self.cube.data.shape[1]])

        cube_data_df = pd.DataFrame(self.cube.data, index=['lbda'+str(i) for i in range(0, self.cube.data.shape[0])]).T
        cube_var_df = pd.DataFrame(self.cube.variance, index=['lbda'+str(i) for i in range(0, self.cube.data.shape[0])]).T
        _lbda_num = len(self.cube.lbda)

        norm_cube_data_df = cube_data_df.copy()
        norm_cube_err_df = cube_var_df.copy()

        for _lbda_index in range(0, _lbda_num):

            if _lbda_index < 10:
                _median = np.median(cube_data_df.iloc[:, 0:_lbda_index+11], axis=1)
                norm_cube_data_df["lbda"+str(_lbda_index)] = cube_data_df.iloc[:, _lbda_index] / _median
                norm_cube_err_df["lbda"+str(_lbda_index)] = np.sqrt(cube_var_df.iloc[:, _lbda_index]) / _median
            elif 209 < _lbda_index:
                _median = np.median(cube_data_df.iloc[:, _lbda_index-10:_lbda_num], axis=1)
                norm_cube_data_df["lbda"+str(_lbda_index)] = cube_data_df.iloc[:, _lbda_index] / _median
                norm_cube_err_df["lbda"+str(_lbda_index)] = np.sqrt(cube_var_df.iloc[:, _lbda_index]) / _median
            else:
                _median = np.median(cube_data_df.iloc[:, _lbda_index-10:_lbda_index+11], axis=1)
                norm_cube_data_df["lbda"+str(_lbda_index)] = cube_data_df.iloc[:, _lbda_index] / _median
                norm_cube_err_df["lbda"+str(_lbda_index)] = np.sqrt(cube_var_df.iloc[:, _lbda_index]) / _median

        self.norm_cube_data = norm_cube_data_df.T.values
        self.norm_cube_err = norm_cube_err_df.T.values

        from pyifu import spectroscopy

        self.norm_cube = spectroscopy.get_cube(data=self.norm_cube_data, variance=self.norm_cube_err**2,
                                               header=self.cube.header, lbda=self.cube.lbda,
                                               spaxel_mapping=self.cube.spaxel_mapping,
                                               spaxel_vertices=self.cube.spaxel_vertices)


    #
    # GETTER
    #

    def get_spatial_neighbors_indexes(self):
        """ Only consider a spaxel witch has 6 neighbors to remove a spaxel at the edge of SEDM datacube."""

        _p_list = []

        for _test_index in range(0, self.cube.nspaxels):

            _neighbors_indexes = self.hexagrid.get_idx_neighbors(_test_index)

            if len(_neighbors_indexes) == 6:
                _neighbors_indexes_dict = {f"nei{i+1}": v for i, v in enumerate(_neighbors_indexes)}
                #_p_list.append(pd.DataFrame({"test_spx":_test_index, **_neighbors_indexes_dict}, index=[_test_index] ))
                _p_list.append(pd.DataFrame({**_neighbors_indexes_dict}, index=[_test_index] ))

        return pd.concat(_p_list, sort=True)

    def get_spatial_neighbors_info(self, lbda_index=0):
        """ """
        #print("Run get_spatial_neighbors_info")

        self.derived_df["flux1_norm"] = self.norm_cube_data[lbda_index][self.derived_df["nei1"]]
        self.derived_df["flux1_norm_err"] = self.norm_cube_err[lbda_index][self.derived_df["nei1"]]
        self.derived_df["flux2_norm"] = self.norm_cube_data[lbda_index][self.derived_df["nei2"]]
        self.derived_df["flux2_norm_err"] = self.norm_cube_err[lbda_index][self.derived_df["nei2"]]
        self.derived_df["flux3_norm"] = self.norm_cube_data[lbda_index][self.derived_df["nei3"]]
        self.derived_df["flux3_norm_err"] = self.norm_cube_err[lbda_index][self.derived_df["nei3"]]
        self.derived_df["flux4_norm"] = self.norm_cube_data[lbda_index][self.derived_df["nei4"]]
        self.derived_df["flux4_norm_err"] = self.norm_cube_err[lbda_index][self.derived_df["nei4"]]
        self.derived_df["flux5_norm"] = self.norm_cube_data[lbda_index][self.derived_df["nei5"]]
        self.derived_df["flux5_norm_err"] = self.norm_cube_err[lbda_index][self.derived_df["nei5"]]
        self.derived_df["flux6_norm"] = self.norm_cube_data[lbda_index][self.derived_df["nei6"]]
        self.derived_df["flux6_norm_err"] = self.norm_cube_err[lbda_index][self.derived_df["nei6"]]

        #self.derived_df["flux_norm_median"] = np.median(derived_df[["flux1_norm", "flux2_norm", "flux3_norm", "flux4_norm", "flux5_norm", "flux6_norm"]], axis=1)
        #self.derived_df["flux_norm_mad"] = stats.median_absolute_deviation(derived_df[["flux1_norm", "flux2_norm", "flux3_norm", "flux4_norm", "flux5_norm", "flux6_norm"]], axis=1)

        self.derived_df["nei_norm_mean"] = np.mean(self.derived_df[["flux1_norm", "flux2_norm", "flux3_norm", "flux4_norm", "flux5_norm", "flux6_norm"]], axis=1)
        self.derived_df["nei_norm_std"] = np.std(self.derived_df[["flux1_norm", "flux2_norm", "flux3_norm", "flux4_norm", "flux5_norm", "flux6_norm"]], axis=1)
        #self.derived_df["nei_norm_std_sqrtn"] = derived_df["nei_norm_std"] / np.sqrt(6) # number of neighbors = 6
        self.derived_df["nei_norm_mean_err"] = (1/6)*np.sqrt(self.derived_df["flux1_norm_err"]**2 +
                                                             self.derived_df["flux2_norm_err"]**2 +
                                                             self.derived_df["flux3_norm_err"]**2 +
                                                             self.derived_df["flux4_norm_err"]**2 +
                                                             self.derived_df["flux5_norm_err"]**2 +
                                                             self.derived_df["flux6_norm_err"]**2)


        return self.derived_df

    def get_spectral_neighbors_info(self, lbda_index=0):
        """ """

        if lbda_index is 0:
            self.derived_df["spec_nei_flux1_norm"] = self.norm_cube_data[lbda_index+1][self.derived_df["index"]]
            self.derived_df["spec_nei_flux1_norm_err"] = self.norm_cube_err[lbda_index+1][self.derived_df["index"]]
            self.derived_df["spec_nei_flux2_norm"] = self.norm_cube_data[lbda_index+2][self.derived_df["index"]]
            self.derived_df["spec_nei_flux2_norm_err"] = self.norm_cube_err[lbda_index+2][self.derived_df["index"]]

        elif lbda_index is (len(self.cube.lbda)-1):
            self.derived_df["spec_nei_flux1_norm"] = self.norm_cube_data[lbda_index-1][self.derived_df["index"]]
            self.derived_df["spec_nei_flux1_norm_err"] = self.norm_cube_err[lbda_index-1][self.derived_df["index"]]
            self.derived_df["spec_nei_flux2_norm"] = self.norm_cube_data[lbda_index-2][self.derived_df["index"]]
            self.derived_df["spec_nei_flux2_norm_err"] = self.norm_cube_err[lbda_index-2][self.derived_df["index"]]

        else:
            self.derived_df["spec_nei_flux1_norm"] = self.norm_cube_data[lbda_index-1][self.derived_df["index"]]
            self.derived_df["spec_nei_flux1_norm_err"] = self.norm_cube_err[lbda_index-1][self.derived_df["index"]]
            self.derived_df["spec_nei_flux2_norm"] = self.norm_cube_data[lbda_index+1][self.derived_df["index"]]
            self.derived_df["spec_nei_flux2_norm_err"] = self.norm_cube_err[lbda_index+1][self.derived_df["index"]]

        self.derived_df["spec_nei_mean"] = np.mean(self.derived_df[["spec_nei_flux1_norm", "spec_nei_flux2_norm"]], axis=1)
        self.derived_df["spec_nei_std"] = np.std(self.derived_df[["spec_nei_flux1_norm", "spec_nei_flux2_norm"]], axis=1)
        self.derived_df["spec_nei_mean_err"] = (1/2)*np.sqrt(self.derived_df["spec_nei_flux1_norm_err"]**2 +
                                                             self.derived_df["spec_nei_flux2_norm_err"]**2)

        return self.derived_df

    def get_test_spaxel_info(self, lbda_index=0, wspectral=True):
        """ """

        self.derived_df["test_spaxel_flux_norm"] = self.norm_cube_data[lbda_index][self.derived_df["index"]]#.byteswap().newbyteorder()
        self.derived_df["test_spaxel_flux_norm_err"] = self.norm_cube_err[lbda_index][self.derived_df["index"]]#.byteswap().newbyteorder()

        self.derived_df["test_spaxel_norm_diff"] = self.derived_df["test_spaxel_flux_norm"] - self.derived_df["nei_norm_mean"]
        self.derived_df["test_spaxel_norm_diff_err"] = np.sqrt(self.derived_df["test_spaxel_flux_norm_err"]**2 + self.derived_df["nei_norm_mean_err"]**2)
        self.derived_df["test_spaxel_norm_diff_sigma"] = self.derived_df["test_spaxel_norm_diff"] / self.derived_df["test_spaxel_norm_diff_err"]

        if wspectral:
            self.derived_df["test_spaxel_spec_norm_diff"] = self.derived_df["test_spaxel_flux_norm"] - self.derived_df["spec_nei_mean"]
            self.derived_df["test_spaxel_spec_norm_diff_err"] = np.sqrt(self.derived_df["test_spaxel_flux_norm_err"]**2 + self.derived_df["spec_nei_mean_err"]**2)
            self.derived_df["test_spaxel_spec_norm_diff_sigma"] = self.derived_df["test_spaxel_spec_norm_diff"] / self.derived_df["test_spaxel_spec_norm_diff_err"]

        return self.derived_df

    #def get_cr_spaxel_info(self, lbda_index=0, wspectral=False, cut_criteria=5):
    def get_cr_spaxel_info(self, lbda_index=None, wspectral=False, cut_criteria=5):
        """
        Return cr spaxel information, e.g., wavelength, normalized flux, neighbors mean etc.

        Parameters
        ----------
        lbda_index: float.
            specify lbda_index you want to investigate.
            'None' means run through whole lbda_index. Default.
        wspectral: bool.
            if you want to use a spectral filtering together with spatial one, put 'wspectral=True'.
            But it is not yet validated.
        cut_criteria: float.
            cut criteria values we want to use.
            Default is '5'.

        Return
        ----------
        cosmic ray information DataFrame.
        """

        self.cut_criteria = cut_criteria

        cr_info_df = pd.DataFrame(columns=["cr_spaxel_index", "cr_spaxel_id", "cr_lbda", "cr_lbda_index", "cr_diff_norm_sigma",
                                           "test_spaxel_flux_norm", "test_spaxel_flux_norm_err",
                                           "nei_norm_mean", "nei_norm_mean_err"])

        if lbda_index is not None: #for specific lbda_index to test

            _lbda_index = lbda_index

            self.get_spatial_neighbors_info(lbda_index=_lbda_index)
            self.get_test_spaxel_info(lbda_index=_lbda_index, wspectral=False)

            _cr_info_df = pd.DataFrame(columns=["cr_spaxel_index", "cr_lbda", "cr_lbda_index", "cr_diff_norm_sigma",
                                                "test_spaxel_flux_norm", "test_spaxel_flux_norm_err",
                                                "nei_norm_mean", "nei_norm_mean_err"])

            _cr_temp_df = self.derived_df[ abs(self.derived_df["test_spaxel_norm_diff_sigma"]) > self.cut_criteria ]

            if len(_cr_temp_df) > 0:

                _cr_info_df["cr_spaxel_index"] = _cr_temp_df["index"]
                _cr_info_df["cr_spaxel_id"] = self.cube.indexes[_cr_info_df["cr_spaxel_index"]]
                _cr_info_df["cr_lbda"] = self.cube.lbda[_lbda_index]
                _cr_info_df["cr_lbda_index"] = _lbda_index
                _cr_info_df["cr_diff_norm_sigma"] = _cr_temp_df["test_spaxel_norm_diff_sigma"]
                _cr_info_df["test_spaxel_flux_norm"] = _cr_temp_df["test_spaxel_flux_norm"]
                _cr_info_df["test_spaxel_flux_norm_err"] = _cr_temp_df["test_spaxel_flux_norm_err"]
                _cr_info_df["nei_norm_mean"] = _cr_temp_df["nei_norm_mean"]
                _cr_info_df["nei_norm_mean_err"] = _cr_temp_df["nei_norm_mean_err"]

                cr_info_df = cr_info_df.append( _cr_info_df, ignore_index=True)

        else: #run whole lbda_index
            for _lbda_index in range(0, len(self.cube.lbda)):

                if wspectral:
                    self.get_spatial_neighbors_info(lbda_index=_lbda_index)
                    self.get_spectral_neighbors_info(lbda_index=_lbda_index)
                    self.get_test_spaxel_info(lbda_index=_lbda_index, wspectral=True)

                    _cr_info_df = pd.DataFrame(columns=["cr_spaxel_index", "cr_spaxel_id", "cr_lbda", "cr_lbda_index", "cr_diff_norm_sigma",
                                                        "test_spaxel_flux_norm", "test_spaxel_flux_norm_err",
                                                        "nei_norm_mean", "nei_norm_mean_err"])

                    _cr_temp_df = self.derived_df[ abs(self.derived_df["test_spaxel_norm_diff_sigma"]) > self.cut_criteria ]

                    if len(_cr_temp_df) > 0:

                        _cr_info_df["cr_spaxel_index"] = _cr_temp_df["index"]
                        _cr_info_df["cr_spaxel_id"] = self.cube.indexes[_cr_info_df["cr_spaxel_index"]]
                        _cr_info_df["cr_lbda"] = self.cube.lbda[_lbda_index]
                        _cr_info_df["cr_lbda_index"] = _lbda_index
                        _cr_info_df["cr_diff_norm_sigma"] = _cr_temp_df["test_spaxel_norm_diff_sigma"]
                        _cr_info_df["test_spaxel_flux_norm"] = _cr_temp_df["test_spaxel_flux_norm"]
                        _cr_info_df["test_spaxel_flux_norm_err"] = _cr_temp_df["test_spaxel_flux_norm_err"]
                        _cr_info_df["nei_norm_mean"] = _cr_temp_df["nei_norm_mean"]
                        _cr_info_df["nei_norm_mean_err"] = _cr_temp_df["nei_norm_mean_err"]

                        cr_info_df = cr_info_df.append( _cr_info_df, ignore_index=True)

                else: # wspectral=False, default
                    self.get_spatial_neighbors_info(lbda_index=_lbda_index)
                    self.get_test_spaxel_info(lbda_index=_lbda_index, wspectral=False)

                    _cr_info_df = pd.DataFrame(columns=["cr_spaxel_index", "cr_spaxel_id", "cr_lbda", "cr_lbda_index", "cr_diff_norm_sigma",
                                                        "test_spaxel_flux_norm", "test_spaxel_flux_norm_err",
                                                        "nei_norm_mean", "nei_norm_mean_err"])

                    _cr_temp_df = self.derived_df[ abs(self.derived_df["test_spaxel_norm_diff_sigma"]) > self.cut_criteria ]

                    if len(_cr_temp_df) > 0:

                        _cr_info_df["cr_spaxel_index"] = _cr_temp_df["index"]
                        _cr_info_df["cr_spaxel_id"] = self.cube.indexes[_cr_info_df["cr_spaxel_index"]]
                        _cr_info_df["cr_lbda"] = self.cube.lbda[_lbda_index]
                        _cr_info_df["cr_lbda_index"] = _lbda_index
                        _cr_info_df["cr_diff_norm_sigma"] = _cr_temp_df["test_spaxel_norm_diff_sigma"]
                        _cr_info_df["test_spaxel_flux_norm"] = _cr_temp_df["test_spaxel_flux_norm"]
                        _cr_info_df["test_spaxel_flux_norm_err"] = _cr_temp_df["test_spaxel_flux_norm_err"]
                        _cr_info_df["nei_norm_mean"] = _cr_temp_df["nei_norm_mean"]
                        _cr_info_df["nei_norm_mean_err"] = _cr_temp_df["nei_norm_mean_err"]

                        cr_info_df = cr_info_df.append( _cr_info_df, ignore_index=True)



        dtype_dict = {"cr_spaxel_index": int, "cr_lbda": float, "cr_lbda_index": int, "cr_diff_norm_sigma": float,
                      "test_spaxel_flux_norm": float, "test_spaxel_flux_norm_err": float,
                      "nei_norm_mean": float, "nei_norm_mean_err": float}
        cr_info_df = cr_info_df.astype(dtype_dict)

        return cr_info_df.sort_values(["cr_spaxel_index", "cr_lbda"])

    #
    # SHOW
    #

    def show_cr_spaxels(self, lbda_index=None, wspectral=False, cut_criteria=5, savefile=None):
        """ """

        cr_df = self.get_cr_spaxel_info(lbda_index=lbda_index, wspectral=wspectral, cut_criteria=cut_criteria)

        fig = mpl.figure(figsize=[5,5])

        ax = fig.add_subplot(111)
        spaxel_patches = self.cube._display_im_(ax, vmin="5", vmax="99")

        for i_cr in np.unique(cr_df["cr_spaxel_index"]):
            spaxel_patches[i_cr].set_edgecolor("blue")
            spaxel_patches[i_cr].set_linewidth(2)
            spaxel_patches[i_cr].set_zorder(13)

        _text = "Number of detected cosmic rays = %i\nNumber of affected spaxels: %i" %( len(cr_df), len(np.unique(cr_df["cr_spaxel_index"])) )
        ax.text(-20, 23.0, _text, fontsize=10)

        if savefile is not None:
            fig.savefig( savefile )
