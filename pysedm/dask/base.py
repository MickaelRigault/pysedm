""" Basic Dask Tools  """


class ClientHolder( object ):
    """ """
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


