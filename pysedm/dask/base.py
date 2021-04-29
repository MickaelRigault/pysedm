""" Basic Dask Tools  """


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
    def from_name(cls, name, client):
        """ """
        from ztfquery import sedm
        squery = sedm.SEDMQuery()
        cubefiles = squery.get_target_cubes(name, client=client)
        return cls.from_cubefiles(cubefiles=cubefiles, client=client)

    @classmethod
    def from_date(cls, date, client):
        """ """
        from ztfquery import sedm
        squery = sedm.SEDMQuery()
        cubefiles = squery.get_night_cubes(date, client=client)
        return cls.from_cubefiles(cubefiles=cubefiles, client=client)

    @classmethod
    def from_month(cls, year, month, client):
        """ """
        from calendar import monthrange
        monthdates = [f'{year:04d}{month:02d}{d:02d}' for d in range(1, monthrange(year, month)[1] + 1)]
        return cls.from_date(monthdates, client=client)
    
    @classmethod
    def from_cubefiles(cls, cubefiles, client):
        """ shortcut to __init__ for simplicity """
        return cls(client=client, cubefiles=cubefiles)
    # -------- #
    #  SETTER  #
    # -------- #
    def set_cubefiles(self, cubefiles):
        """ """
        self._cubefiles = cubefiles
        
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

