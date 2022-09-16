#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 

# --- CCD
SEDM_CCD_SIZE = [2048, 2048]
DOME_TRACEBOUNDS = [70,230]
TRACE_DISPERSION = 1.2*2 # PSF (sigma assuming gaussian) of the traces on the CCD.

SEDM_INVERT = False #  Shall the x and y axis extracted in the hexagrid be inverted ?
SEDM_ROT    = 103 # SEDM alignment to have north up
SEDM_MLA_RADIUS = 25

# --- LBDA
SEDM_LBDA = np.linspace(3700, 9300, 220)
LBDA_PIXEL_CUT = 3

DEFAULT_REFLBDA = 6000 # In Angstrom
IFU_SCALE_UNIT  = 0.55

# ----- WCS
SEDM_ASTROM_PARAM  = [ 7.28968990e-01,  6.89009309e-02, -6.57804812e-03, -7.94252856e-01,
                           1.02682050e+03 , 1.01659890e+03]

SEDM_ASTROM_PARAM_since_20180928 = [ 6.63023938e-01,  6.57283519e-02, -1.97868377e-02, -7.71650238e-01,
                                         1.01768812e+03,  1.01237730e+03]

SEDM_ASTROM_PARAM_since_20190201 = [ 6.20197410e-01,  1.02551606e-01,  3.84158750e-02, -8.63030378e-01,
                                         1.03498483e+03,  1.01326973e+03]

SEDM_ASTROM_PARAM_since_20190417 = [ 6.85554379e-01, -2.01025422e-02,  2.46681500e-02, -6.71790508e-01,
                                         1.02143124e+03,  1.02278900e+03]

# --- Palomar Atmosphere
# Palomar Extinction Data from Hayes & Latham 1975
# (Wavelength in Angstroms, Magnitudes per airmass)
SITE_EXTINCTION = np.asarray([ (3200, 1.058),
 (3250, 0.911), (3300, 0.826), (3350, 0.757), (3390, 0.719), (3448, 0.663), (3509, 0.617),
 (3571, 0.575), (3636, 0.537), (3704, 0.500), (3862, 0.428), (4036, 0.364), (4167, 0.325),
 (4255, 0.302), (4464, 0.256), (4566, 0.238), (4785, 0.206), (5000, 0.183), (5263, 0.164),
 (5556, 0.151), (5840, 0.140), (6055, 0.133), (6435, 0.104), (6790, 0.084), (7100, 0.071),
 (7550, 0.061), (7780, 0.055), (8090, 0.051), (8370, 0.048), (8708, 0.044), (9832, 0.036),
 (10255, 0.034), (10610, 0.032), (10795, 0.032), (10870, 0.031)])


SITE_COORDS = {"latitude":   33.3563, # North
                  "longitude":-116.8648, # West
                  "altitude":1712,       #meter
                      }

SEDM_XRED_EXTENTION = [5.972e-03,56]# 54.4443
SEDM_XBLUE_EXTENTION = [7.232e-03,-238] #236.4647

# -------------- #
#  SEDM          #
# -------------- #
# From wavesolution.py but is SEDM specific so moving here
REFWAVELENGTH = 7000

# For Information:
#    Sodium skyline fitted by SNIFS is at 5892.346 Angstrom
SODIUM_SKYLINE_LBDA = 5900#5896
TELLURIC_REF_LBDA   = 7630 #  7624.5

_REFORIGIN = 69

LINES= {"Hg": # IN VACUUM
            { 
               np.mean([5771.210, 5792.276])   : {"ampl":13. ,"mu":201-_REFORIGIN,
                           "doublet":False,
                        "info":"merge of 5771.210, 5792.276 blended"},
               5462.268 : {"ampl":62.,"mu":187-_REFORIGIN},
               4359.560   : {"ampl":50. ,"mu":127-_REFORIGIN},
               4047.708 : {"ampl":10. ,"mu":105-_REFORIGIN}, 
              
               },
        "Cd":  # IN VACUUM
              {4679.325 : {"ampl":14. ,"mu":143-_REFORIGIN}, 
               4801.254 : {"ampl":40. ,"mu":153-_REFORIGIN},
               5087.239 : {"ampl":60. ,"mu":169-_REFORIGIN},
               6440.249 : {"ampl":19. ,"mu":227-_REFORIGIN}, 
               # - Cd and Xe seems to have the same lines
               # Almost same wavelength but diffent enough to save the code
               8280.01  : {"ampl": 1. ,"mu":275-_REFORIGIN,
                            "backup":"Xe", # used only if Xe is not there
                            "doublet":True,"info":"merge of 8230, 8341"},
               #8818.51  : {"ampl": 0.8,"mu":295-_REFORIGIN},
               #9000.01  : {"ampl": 0.5,"mu":302-_REFORIGIN,
               #             "doublet":True,"info":"merge of 8945, 9050"},
                },
        # Important for Xe. Keep the Bright line, the First one. or change get_line_shift()
#        "Xe": {8280.    : {"ampl": 1. ,"mu":280-_REFORIGIN,
#                            "doublet":True,"info":"merge of 8230, 8341"},
#               8818.5   : {"ampl": 0.8,"mu":295-_REFORIGIN},
#               9000     : {"ampl": 0.5,"mu":302-_REFORIGIN,
#                            "doublet":True,"info":"merge of 8945, 9050"},
#               }

        "Xe": {np.average([8233.90, 8282.39], weights=[2,1])  : {"ampl": 16. ,"mu":280-_REFORIGIN,
                            "doublet":False},# yes but really close
                            
               np.average([8349.11, 8411.00], weights=[2,3])   : {"ampl": 3. ,"mu":283-_REFORIGIN,
                            "doublet":False}, # yes but really close
                            
                            
               8821.83    : {"ampl": 11.,"mu":294-_REFORIGIN},
                   
               np.average([8954.71,9047.93]) : {"ampl": 11.,"mu":298-_REFORIGIN,
                             "doublet":True , "info": "merge of lines 9854.71,9047.93"},
               9165.16    : {"ampl": 4.5,"mu":303-_REFORIGIN},
               
               # small lines but isolated
               7644.12    : {"ampl": 1.,"mu":264-_REFORIGIN},
               
               #9425.    : {"ampl": 2.,"mu":309-_REFORIGIN,
               #                "doublet":True , "info": "merge of lines 9400 and 9450"},
               
               #9802.8    : {"ampl": 0.4,"mu":317-_REFORIGIN},
               #9938.0    : {"ampl": 0.4,"mu":320-_REFORIGIN},
               
               }
        }