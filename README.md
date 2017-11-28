# pysedm
Data Reduction Pipeline for the SEDmachine (no ready yet)

*This module is under development and is not yet ready to be used*

# Installation
Not ready yet

# Dependencies
- numpy, scipy, matpotlib
- shapely (pip install shapely)
  - _for the polygon matching in CCD to Spectrum. Might be moved to matplotlib_  
- propobject (pip install propobject)
  - _for the structure of the classes_
- astrobject (pip install astrobject)
  - _for the basic Image objects as well as low level tools_
- pyifu (pip install pyifu)
  - _for the basics Spectrum and Cube objects_
- modefit and iminuit (pip install modefit ; pip install iminuit)
  - _modefit is used to fit the background and emission lines. It needs iminuit to do so_
- pynverse (pip install pynverse)
  - _for a faster lbda<->pixels conversion in the wavelength solution (it's slower without pynverse)_


# Modules

## CCD

## wavelength solution
