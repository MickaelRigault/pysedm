#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to I/O the data """

import os

REDUXPATH   = os.getenv('SEDMREDUXPATH',default="~/redux/")


def get_datapath(YYYYMMDD):
    """ Return the full path of the current date """
    return REDUXPATH+"%s/"%YYYYMMDD




