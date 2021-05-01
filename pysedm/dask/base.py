""" Basic Dask Tools  """

import pandas
import numpy as np
from astropy import time

def parse_filename(filename):
    """ """
    e3d,crr,b, ifudate, *times, targetname=filename.split("_")
    date = ifudate.replace("ifu","")
    mjd = time.Time(f"{date[:4]}-{date[4:6]}-{date[6:]}"+" "+":".join(times), format="iso").mjd
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
    def from_name(cls, name, client):
        """ """
        cubefiles = cls.get_cubes(client=client, targetname=name, **kwargs)[0]
        return cls.from_cubefiles(cubefiles=cubefiles, client=client)

    @classmethod
    def from_date(cls, date, client, **kwargs):
        """ """
        cubefiles = cls.get_cubes(client=client, dates=date, **kwargs)[0]
        return cls.from_cubefiles(cubefiles=cubefiles, client=client)

    @classmethod
    def from_month(cls, year, month, client):
        """ """
        from calendar import monthrange
        monthdates = [f'{year:04d}{month:02d}{d:02d}' for d in range(1, monthrange(year, month)[1] + 1)]
        return cls.from_date(monthdates, client=client)
    
    @staticmethod
    def get_cubes(client, dates=None, targetname=None, incl_astrom=True):
        """ """
        if targetname is None and dates is None:
            raise ValueError("either dates or targetname must be given")

        from ztfquery import sedm
        squery = sedm.SEDMQuery()
        cubes = []
        astrom = []
        if dates is not None:
            cubes += list(squery.get_night_cubes(dates, client=client))
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
        self._cubefiles = cubefiles

    def get_cubefile_dataframe(self, index_per_calib=True):
        """ """
        datafile = pandas.DataFrame(self.cubefiles, columns=["filepath"])
        dataall = datafile["filepath"].str.split("/", expand=True)
        datafile["basename"] = dataall[dataall.columns[-1]].values

        info = pandas.DataFrame.from_records(datafile["basename"].apply(parse_filename))
        datafile = pandas.merge(datafile, info,
                                    left_index=True, right_index=True)
        datafile["is_std"] = datafile["name"].str.contains("STD")

        df_std = datafile[datafile["is_std"]]
        id_ = np.argmin(np.abs(datafile["mjd"].values-df_std["mjd"].values[:,None]), axis=0)
        datafile["std_calib"] = df_std["basename"].iloc[id_].values

        if index_per_calib:
            return datafile.set_index(["std_calib", datafile.index])
        return datafile
    
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

    
