# pysedm
Data Reduction Pipeline for the SEDmachine (no ready yet)

*This module is under development and is not yet ready to be used*

# Installation

```bash
git pull https://github.com/MickaelRigault/pysedm.git
cd pysedm
python setup.py install
```


# Dependencies

works in python 2.7+ and 3.x, but the 2.7+ might not be supported in the future. 

### Basics

- numpy, scipy, matpotlib, astropy
- propobject (pip install propobject) _for the structure of the classes_
- pyifu (pip install pyifu or see https://github.com/MickaelRigault/pyifu) _cube and spectra object library_

### If you want to reproduce the spectral extraction:

- psfcube (pip install psfcube or see https://github.com/MickaelRigault/psfcube) 
  _psfcube depends on Minuit (fitter) and modefit (structure), which are automatically installed if needed_

### If you need to reproduce the cube creation:

- shapely (pip install shapely) _for the polygon matching in CCD to Spectrum_  
- astrobject (pip install astrobject) _for the basic Image objects as well as low level tools_
- pynverse (pip install pynverse) _for a faster lbda<->pixels conversion in the wavelength solutio_

# Running a manual spectral extraction

...doc ongoing...

# Modules
### Astrometry `pysedm.astrometry.py`

This module contains tools to determine the target position inside the MLA given metaguider rainbow camera wcs solution.
(see `rainbowcam.py`)

Main functionalities:
- `get_object_ifu_pos(cube_)` expected target position in the given cube "cube_"
- `get_ccd_coords(cube_)` target position in metaguider given its wcs solution.

### Background (ccd) `pysedm.background.py`

This module contains tools to build the ccd background image. 

_low level module_

### CCD `pysedm.ccd.py`

_Full usage of ccd functionalities requires: shapely, astrobject, pynverse (all "pipable")_

_ccd-x axis is refered to as 'i', y as "j"_

Core module allowing to extract a 3D cube from a ccd. 

Main functionalities:
- `get_ccd(sedm_crr_filename)` returns a `ScienceCCD` object.

Main Object:
- `ScienceCCD`: loads a _sedm_crr_filename_ and contains method to interact with it.
   
   // CCD (2D) to 1D spectrum
   - `get_trace_mask(traceindex)`: 2D weight-mask of a given trace. each ccd-pixel has weight (from 0 to 1) corresponding to its overlap with the trace location definition (1 fully in, 0 fully out, 0.x edge cases)
   - `get_spectrum(traceindex)`: load given traceindex trace_mask (see `get_trace_mask()`) and converts the masked-ccd into a 1D-flux (in cdd pixel unit). 
   
   // Extract wavelength-calibrated Cube and individual spectrum.
   - `extrat_spectrum(traceindex, wavesolution)`: gets traceindex 1D-flux (see `get_spectrum()`) and returns: wavelength_array_in_angstrom, flux and variance (in cdd pixel unit). _this method is the core of `extract_cube()`_
   - `extrat_cube(wavesolution, lbda, hexagrid)`: get the traceindex 1D-flux (see `get_spectrum()`) and returns: wavelength_array_in_angstrom, flux and variance (in cdd pixel unit). _this method is the core of `extract_cube()`_
   
   // Visualization
   - `show()`: plot the ccd as imshow.
   - `show_traceindex()`: plot the ccd using `show()` and overplot the trace coutours.

### wavelength solution


# Definitions
- traceindex: ID of a trace on a ccd. Defined by `TraceMatcher`

