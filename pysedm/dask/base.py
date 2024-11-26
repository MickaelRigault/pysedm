""" Basic Dask Tools  """

import warnings
import pandas
import numpy as np
import dask

from .. import io, get_sedmcube
from .. import fluxcalibration
from ..io import REDUXPATH
import glob
import os
from astropy.io.fits import getval


def fetch_hypergalfluxcal(date, range_night=2):
    """ """
    from datetime import datetime, timedelta
    files = []
    d_ = datetime.strptime(date, '%Y%m%d')
    dates = [(d_ + timedelta(n)).strftime('%Y%m%d').replace('-', '')
             for n in range(-range_night, range_night+1)]
    for date_ in dates:
        for file in glob.glob(os.path.join(REDUXPATH, date_)+"/fluxcal_hypergal*"):
            files.append(file)
    return files


def fetch_pysedmfluxcal(date, range_night=0):
    """ """
    from datetime import datetime, timedelta
    files = []
    d_ = datetime.strptime(date, '%Y%m%d')
    dates = [(d_ + timedelta(n)).strftime('%Y%m%d').replace('-', '')
             for n in range(-range_night, range_night+1)]
    
    for date_ in dates:
        for file in glob.glob(os.path.join(REDUXPATH, date_)+"/fluxcal*"):
            files.append(file)

    if len(files) == 0:
        warnings.warn(f"No fluxcal found for night {date}")
    return files


def get_fluxcal_file(cube, hgfirst=False, update=False):
    """ """
    date = cube.header["OBSDATE"].replace("-", "")
    # - Make sure you have all the fluxcalibration files locally
    if hgfirst:  # Use hypergal fluxcalibration file if available
        filefluxcal = fetch_hypergalfluxcal(date)
        for f_ in filefluxcal:
            try:
                spec = fluxcalibration.load_fluxcal_spectrum(f_)
                if np.min(spec.data) < 0:
                    filefluxcal.remove(f_)
            except TypeError:
                filefluxcal.remove(f_)
        if len(filefluxcal) == 1:
            return filefluxcal[0]
        elif len(filefluxcal) > 1:
            fluxcal_mjd_obs = [getval(f, "MJD_OBS") for f in filefluxcal]
            mjd = cube.header.get("MJD_OBS")
            target_mjd_obs = mjd
            return filefluxcal[np.argmin(np.abs(target_mjd_obs - np.asarray(fluxcal_mjd_obs)))]
        
    from ztfquery import sedm
    squery = sedm.SEDMQuery()
    # downloads what is missing.
    try:
        import json
        _ = squery.get_night_fluxcal(date, nprocess=1, show_progress=False)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(f"Corrupted file from whatdata at date {date}")
    # - grab the nearest
    try:
        fluxcal = io.fetch_nearest_fluxcal(mjd=cube.header.get("MJD_OBS"))
        
    except:
        print("enter inside the except.")
        filefluxcal = fetch_pysedmfluxcal(date)
        for f_ in filefluxcal:
            try:
                spec = fluxcalibration.load_fluxcal_spectrum(f_)
                if np.min(spec.data) < 0:
                    filefluxcal.remove(f_)
            except TypeError:
                filefluxcal.remove(f_)
            except FileNotFoundError:
                filefluxcal.remove(f_)
                warnings.warn(f"{f_} is a Corrupted file from whatdata at date {date}")

        if len(filefluxcal) == 0:
            raise ValueError("no filefluxcal found.")
        
        if len(filefluxcal) == 1:
            return filefluxcal[0]
        
        elif len(filefluxcal) > 1:
            fluxcal_mjd_obs = [getval(f, "MJD_OBS") for f in filefluxcal]
            mjd = cube.header.get("MJD_OBS")
            target_mjd_obs = mjd
            fluxcal = filefluxcal[np.argmin(
                np.abs(target_mjd_obs - np.asarray(fluxcal_mjd_obs)))]
            
    return fluxcal


def calibrate_cube(cube, fluxcalfile, airmass=None, backup_airmass=1.1, store_data=False):
    """ """
    if airmass is None:
        airmass = cube.header.get("AIRMASS", backup_airmass)

    fluxcal = fluxcalibration.load_fluxcal_spectrum(fluxcalfile)
    cube.scale_by(fluxcal.get_inversed_sensitivity(cube.header.get("AIRMASS", backup_airmass)),
                  onraw=False)
    cube.set_filename(cube.filename.replace("e3d", "cale3d"))
    header = {**dict(cube.header), **
              dict({'FLUXCAL': f'{os.path.basename(fluxcalfile)}'})}
    cube.set_header(header)

    if store_data:
        cube.writeto(cube.filename)

    return cube


def _no_delayed_(func):
    return func


def get_delayed_func(as_dask):
    """ """
    return dask.delayed if as_dask else _no_delayed_


class ClientHolder(object):

    def __init__(self, client=None):
        """ """
        if client is not None:
            self.set_client(client)

    def set_client(self, client):
        """ """
        self._client = client

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def client(self):
        """ Dask client used for computation """
        if not hasattr(self, "_client"):
            return None
        return self._client

#
#
#
#
#


class _SEDMFileHolder_(ClientHolder):

    def __init__(self, client=None, files=None):
        """ """
        _ = super().__init__(client=client)
        if files is not None:
            self.set_files(files)
    # =============== #
    # Initialisation  #
    # =============== #

    @classmethod
    def from_files(cls, files, client):
        """ shortcut to __init__ for simplicity """
        return cls(client=client, files=files)

    @classmethod
    def from_name(cls, name, client, **kwargs):
        """ """
        files = cls.get_files(client=client, targetname=name, **kwargs)
        return cls.from_files(files=files, client=client)

    @classmethod
    def from_date(cls, date, client, **kwargs):
        """ """
        files = cls.get_files(client=client, dates=date, **kwargs)
        return cls.from_files(files=files, client=client)

    @classmethod
    def from_daterange(cls, daterange, client, **kwargs):
        """ """
        from ztfquery import sedm
        start, end = daterange
        dates = sedm.build_datelist(start=start, end=end)
        return cls.from_date(dates, client, **kwargs)

    @classmethod
    def from_month(cls, year, month, client, **kwargs):
        """ """
        from calendar import monthrange
        monthdates = [f'{year:04d}{month:02d}{d:02d}' for d in range(
            1, monthrange(year, month)[1] + 1)]
        return cls.from_date(monthdates, client=client, **kwargs)

    @classmethod
    def from_year(cls, year, client, **kwargs):
        """ """
        return cls.from_daterange([f"{year}-01-01", f"{year}-12-31"], client, **kwargs)

    # - Static Get Files
    @staticmethod
    def get_files(client, dates=None, targetname=None,  **kwargs):
        """ """
        raise NotImplementedError("get_file mush be implemented")

    # =============== #
    # Methods         #
    # =============== #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_files(self, files, unique=True):
        """ """
        if len(files) == 0:
            raise ValueError("no files given")

        # cubefiles
        self._files = files if not unique else list(np.unique(files))
        # dataziles
        datafile = pandas.DataFrame(self.files, columns=["filepath"])
        dataall = datafile["filepath"].str.split("/", expand=True)
        datafile["basename"] = dataall[dataall.columns[-1]].values

        info = pandas.DataFrame.from_records(
            datafile["basename"].apply(io.parse_filename))
        datafile = pandas.merge(
            datafile, info, left_index=True, right_index=True)
        datafile["is_std"] = datafile["name"].str.contains("STD")
        self._datafiles = datafile

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def files(self):
        """ """
        if not hasattr(self, "_files"):
            return None
        return self._files

    def has_files(self):
        """ """
        return self.files is not None and len(self.files) > 0

    @property
    def datafiles(self):
        """ stored version of get_datafiles """
        if not hasattr(self, "_datafiles"):
            return None
        return self._datafiles


class DaskCCD(_SEDMFileHolder_):
    """ """
    # =============== #
    # Initialisation  #
    # =============== #
    @staticmethod
    def get_files(client, dates=None, targetname=None,  **kwargs):
        """ """
        if targetname is None and dates is None:
            raise ValueError("either dates or targetname must be given")

        from ztfquery import sedm
        squery = sedm.SEDMQuery()
        files = []
        if dates is not None:
            files += list(squery.get_night_crr(dates, client=client, **kwargs))

        if targetname is not None:
            files += list(squery.get_target_crr(targetname, client=client))

        return files


class DaskCube(_SEDMFileHolder_):

    # =============== #
    # Initialisation  #
    # =============== #

    @staticmethod
    def get_files(client, dates=None, targetname=None, incl_astrom=True, **kwargs):
        """ """
        if targetname is None and dates is None:
            raise ValueError("either dates or targetname must be given")

        from ztfquery import sedm
        squery = sedm.SEDMQuery()
        cubes = []
        astrom = []
        if dates is not None:
            cubes += list(squery.get_night_cubes(dates,
                          client=client, **kwargs))
            if incl_astrom:
                astrom += list(squery.get_night_astrom(dates, client=client))
        if targetname is not None:
            cubes += list(squery.get_target_cubes(targetname, client=client))
            if incl_astrom:
                astrom += list(squery.get_target_astrom(targetname,
                               client=client))

        return cubes

    # -------- #
    #  SETTER  #
    # -------- #

    # -------- #
    #  GETTER  #
    # -------- #
    def get_std_basename(self, excluse_std=None, avoid_noncalspec=True, avoid_bad=True):
        """ get the cubefile_dataframe entry associated to the standard stars """
        dstd = self.datafiles[self.datafiles["is_std"]]

        if excluse_std is not None:
            re_exclude = "|".join(list(np.atleast_1d(excluse_std)))
            dstd = dstd[~dstd["name"].str.contains(re_exclude)]

        if avoid_noncalspec:
            noncalspec_std = io.get_noncalspec_standards()
            dstd = dstd[~dstd["name"].str.contains("|".join(noncalspec_std))]

        if avoid_bad:
            bad_std = io.get_bad_standard_exposures()
            dstd = dstd[~dstd["basename"].str.contains("|".join(bad_std))]

        return dstd["basename"]

    def get_datafiles(self, excluse_std=None,
                      avoid_noncalspec=True, avoid_bad=True,
                      add_stdcalib=True, index_per_calib=True):
        """ """
        datafiles = self.datafiles.copy()
        if add_stdcalib:
            std_datafiles = self.get_std_basename(excluse_std=excluse_std,
                                                  avoid_noncalspec=avoid_noncalspec,
                                                  avoid_bad=avoid_bad)
            df_std = datafiles.loc[std_datafiles.index]
            id_ = np.argmin(
                np.abs(datafiles["mjd"].values-df_std["mjd"].values[:, None]), axis=0)
            datafiles["std_calib"] = df_std["basename"].iloc[id_].values

            if index_per_calib:
                return datafiles.set_index(["std_calib", datafiles.index])

        return datafiles

    # -------- #
    #  DASK    #
    # -------- #
    @staticmethod
    def get_calibrated_cube(cubefile_, fluxcalfile=None, hgfirst=False,
                            apply_byecr=True, get_filename=False, as_dask=True, **kwargs):
        """ """
        # dask.delayed or doing nothing
        delayed = get_delayed_func(as_dask)

        # 1. Get cube
        cube = delayed(get_sedmcube)(cubefile_, apply_byecr=apply_byecr)

        # 2. Get flux calibration file (if any)
        if fluxcalfile is None:
            fluxcalfile = delayed(get_fluxcal_file)(
                cube, hgfirst)  # could be None

        # 3. Flux calibrating the cube
        calibrated_cube = delayed(calibrate_cube)(cube, fluxcalfile, **kwargs)

        if get_filename:
            return calibrated_cube.filename

        return calibrated_cube
