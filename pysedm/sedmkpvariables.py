#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# --- CCD
SEDM_CCD_SIZE = [2048, 2048]
DOME_TRACEBOUNDS = [70,230]
TRACE_DISPERSION = 1.5#1.2*2 # PSF (sigma assuming gaussian) of the traces on the CCD.

SEDM_INVERT = False #  Shall the x and y axis extracted in the hexagrid be inverted ?
SEDM_ROT    = 103 # SEDM alignment to have north up
SEDM_MLA_RADIUS = 26#25

# --- LBDA
SEDM_LBDA = np.linspace(4000, 7500, 137)
LBDA_PIXEL_CUT = 3

DEFAULT_REFLBDA = 6000 # In Angstrom
IFU_SCALE_UNIT  = 0.75

# ----- WCS
SEDM_ASTROM_PARAM  = [ 7.28968990e-01,  6.89009309e-02, -6.57804812e-03, -7.94252856e-01,
                           1.02682050e+03 , 1.01659890e+03]

SEDM_ASTROM_PARAM_since_20180928 = [ 6.63023938e-01,  6.57283519e-02, -1.97868377e-02, -7.71650238e-01,
                                         1.01768812e+03,  1.01237730e+03]

SEDM_ASTROM_PARAM_since_20190201 = [ 6.20197410e-01,  1.02551606e-01,  3.84158750e-02, -8.63030378e-01,
                                         1.03498483e+03,  1.01326973e+03]

SEDM_ASTROM_PARAM_since_20190417 = [ 6.85554379e-01, -2.01025422e-02,  2.46681500e-02, -6.71790508e-01,
                                         1.02143124e+03,  1.02278900e+03]

## Kitt Peak extinction
SITE_EXTINCTION = np.asarray([(3200. , 1.017),
 (3250. , 0.881),(3300. , 0.787),(3350. , 0.731),(3400. , 0.683),(3450. , 0.639),(3500. , 0.600),
 (3571. , 0.556),(3636. , 0.518),(3704. , 0.484),(3760. , 0.457),(3816. , 0.433),(3862. , 0.414),
 (3945. , 0.383),(4000. , 0.365),(4036. , 0.353),(4065. , 0.344),(4167. , 0.315),(4190. , 0.310),
 (4255. , 0.293),(4290. , 0.284),(4370. , 0.268),(4464. , 0.250),(4498. , 0.243),(4566. , 0.231),
 (4604. , 0.225),(4678. , 0.215),(4735. , 0.206),(4785. , 0.202),(4825. , 0.197),(4914. , 0.186),
 (5000. , 0.180),(5067. , 0.174),(5145. , 0.167),(5206. , 0.163),(5263. , 0.161),(5341. , 0.158),
 (5440. , 0.153),(5504. , 0.150),(5556. , 0.148),(5630. , 0.147),(5710. , 0.146),(5775. , 0.144),
 (5840. , 0.137),(5910. , 0.133),(5980. , 0.133),(6056. , 0.132),(6140. , 0.125),(6220. , 0.119),
 (6290. , 0.114),(6365. , 0.109),(6436. , 0.104),(6473. , 0.101),(6530. , 0.097),(6600. , 0.093),
 (6670. , 0.090),(6740. , 0.085),(6790. , 0.083),(6850. , 0.081),(6975. , 0.075),(7055. , 0.073),
 (7100. , 0.072),(7150. , 0.070),(7220. , 0.068),(7270. , 0.066),(7365. , 0.065),(7460. , 0.063),
 (7550. , 0.061),(7580. , 0.058),(7725. , 0.056),(7780. , 0.055),(7820. , 0.055),(7910. , 0.054),
 (8000. , 0.052),(8090. , 0.051),(8210. , 0.050),(8260. , 0.049),(8370. , 0.048),(8708. , 0.030),
 (9832. , 0.053),(10256., 0.023)])

SITE_COORDS = {"latitude":  31.9633, #North
                   "longitude": -111.6, #West
                   "altitude": 2120,  #meter
                      }


SEDM_XRED_EXTENTION = [5.972e-03, 30]# 54.4443
SEDM_XBLUE_EXTENTION = [7.232e-03,-220] #236.4647

# -------------- #
#  SEDMv2        #
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
               4359.560   : {"ampl":50. ,"mu":112-_REFORIGIN},
               4047.708 : {"ampl":10. ,"mu":82-_REFORIGIN}, 
              
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
                            
                            
               8821.83    : {"ampl": 11.,"mu":289-_REFORIGIN},
                   
               np.average([8954.71,9047.93]) : {"ampl": 11.,"mu":293-_REFORIGIN,
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