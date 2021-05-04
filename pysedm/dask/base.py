""" Basic Dask Tools  """

import pandas
import numpy as np
from astropy import time
from .. import io

def parse_filename(filename):
    """ """
    e3d, crr,b, ifudate, hh,mm,ss, *targetname=filename.split("_")
    targetname = "-".join(targetname).replace(" ","")
    date = ifudate.replace("ifu","")
    mjd = time.Time(f"{date[:4]}-{date[4:6]}-{date[6:]}"+" "+f"{hh}:{mm}:{ss}", format="iso").mjd
    return {"date":date, 
           "mjd":mjd, 
           "name":targetname.split(".")[0]}

class ClientHolder( object ):

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
        if not hasattr(self,"_client"):
            return None
        return self._client


class DaskCube( ClientHolder ):
    
    
    # =============== #
    # Initialisation  #
    # =============== #
    def __init__(self, client=None, cubefiles=None):
        """ """
        _ = super().__init__(client=client)
        if cubefiles is not None:
            self.set_cubefiles(cubefiles)

    @classmethod
    def from_cubefiles(cls, cubefiles, client):
        """ shortcut to __init__ for simplicity """
        return cls(client=client, cubefiles=cubefiles)

    @classmethod
    def from_name(cls, name, client, **kwargs):
        """ """
        cubefiles = cls.get_cubes(client=client, targetname=name, **kwargs)[0]
        return cls.from_cubefiles(cubefiles=cubefiles, client=client)

    @classmethod
    def from_date(cls, date, client, **kwargs):
        """ """
        cubefiles = cls.get_cubes(client=client, dates=date, **kwargs)[0]
        return cls.from_cubefiles(cubefiles=cubefiles, client=client)

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
        monthdates = [f'{year:04d}{month:02d}{d:02d}' for d in range(1, monthrange(year, month)[1] + 1)]
        return cls.from_date(monthdates, client=client, **kwargs)

    @classmethod
    def from_year(cls, year, client, **kwargs):
        """ """
        return cls.from_daterange([f"{year}-01-01",f"{year}-12-31"], client, **kwargs)

    @staticmethod
    def get_cubes(client, dates=None, targetname=None, incl_astrom=True, **kwargs):
        """ """
        if targetname is None and dates is None:
            raise ValueError("either dates or targetname must be given")

        from ztfquery import sedm
        squery = sedm.SEDMQuery()
        cubes = []
        astrom = []
        if dates is not None:
            cubes += list(squery.get_night_cubes(dates, client=client, **kwargs))
            if incl_astrom:
                astrom += list(squery.get_night_astrom(dates, client=client))
        if targetname is not None:
            cubes += list(squery.get_target_cubes(targetname, client=client))
            if incl_astrom:
                astrom += list(squery.get_target_astrom(targetname, client=client))

        return cubes, astrom
        
    # -------- #
    #  SETTER  #
    # -------- #
    def set_cubefiles(self, cubefiles):
        """ """
        # cubefiles
        self._cubefiles = cubefiles

        # dataziles
        datafile = pandas.DataFrame(self.cubefiles, columns=["filepath"])
        dataall = datafile["filepath"].str.split("/", expand=True)
        datafile["basename"] = dataall[dataall.columns[-1]].values

        info = pandas.DataFrame.from_records(datafile["basename"].apply(parse_filename))
        datafile = pandas.merge(datafile, info, left_index=True, right_index=True)
        datafile["is_std"] = datafile["name"].str.contains("STD")
        self._datafiles = datafile
        
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
            id_ = np.argmin(np.abs(datafiles["mjd"].values-df_std["mjd"].values[:,None]), axis=0)
            datafiles["std_calib"] = df_std["basename"].iloc[id_].values

            if index_per_calib:
                return datafiles.set_index(["std_calib", datafiles.index])
            
        return datafiles

    
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def cubefiles(self):
        """ """
        if not hasattr(self,"_cubefiles"):
            return None
        return self._cubefiles

    def has_cubefiles(self):
        """ """
        return self.cubefiles is not None and len(self.cubefiles)>0
    
    @property
    def datafiles(self):
        """ stored version of get_datafiles """
        if not hasattr(self,"_datafiles"):
            return None
        return self._datafiles
