# pysedm
Data Reduction Pipeline for the SEDmachine (no ready yet)

_This module is under development and is not yet ready to be used_

_works in python 2.7+ and 3.x, but the 2.7+ might not be supported in the future._

# Installation

```bash
pip install pysedm
```

or 
```bash
git pull https://github.com/MickaelRigault/pysedm.git
cd pysedm
python setup.py install
```


Basic installation detailed here enables you to:
- *load*, *visualize* and *use* pysedm product objects (spectra, cubes, but also calibration object)
- *extract* spectra from cubes. 

# Running a manual spectral extraction

Here is are some examples to use `pysedm`. 

### Open a Cube.

Say you have a cubefile name `e3d_crr_date_id_ztfname.fits`

```python
import pysedm
# Load the cube
cube = pysedm.get_sedmcube('e3d_crr_date_id_ztfname.fits')
# See the cube and enable to clic on the cube to visualize spaxels:
cube.show(interactive=True)
```
![](examples/display_cube_example.gif)


# Modules

See details [here](pysedm)

# Dependencies

The following dependencies are automatically installed (if needed only):

- _numpy_, _scipy_, _matpotlib_, _astropy_ (basic anaconda)

- _propobject_ (pip install propobject) _for the structure of the classes_

- _pyifu_ (pip install pyifu or see https://github.com/MickaelRigault/pyifu) _cube and spectra object library_

- _psfcube_ (pip install psfcube or see https://github.com/MickaelRigault/psfcube) 
  _psfcube depends on Minuit (fitter) and modefit (structure), which are automatically installed if needed_


See details [here](pysedm) for additional dependencies you will need for full pipeline functionalities (like re-creating the wavelength solution etc.)
