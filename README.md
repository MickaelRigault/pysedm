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

## Open and display a Cube.

Say you have a cubefile name `e3d_crr_date_fileid_ztfname.fits`

```python
import pysedm
# Load the cube
cube = pysedm.get_sedmcube('e3d_crr_date_fileid_ztfname.fits')
# See the cube and enable to clic on the cube to visualize spaxels:
cube.show(interactive=True)
```
![](examples/display_cube_example.gif)

_**what is going on?** if you click on spaxels you see the spaxel spectra on the left. If you click once on "**control**" you will not replace the spectrum on the left but **see new spaxel's spectra with a new color** (colors match bewtween spaxel contours and spectra. Click once again on "control" to turns this off. 
If you **drag your mouse on the IFU**, the display spectrum will be the average of the rectangle defined by the draging. Click on **"shift" to draw any polygon**. Click again on "shift" to turns it off. Click on **"option"and your dragging will define an aperture radius**. Click on **"escape" to clean everything switch back to original mode**_ 

You can directly display the cube without opening ipython by doing:
```bash
display_cube.py e3d_crr_date_id_ztfname.fits
```

## Manual cube extraction.

You want to manually extract a spectrum from a cube `e3d_crr_date_fileid_ztfname.fits` (`fileid` is HH_MM_SS)

From your shell do (with date been YYYYMMDD):
```bash
extract_star.py DATE --auto FILE_ID --display --tag manual
```
![](examples/extract_star_example.gif)

_**what is going on?** (1) double clicked to locate the expected centroid of the target (creating the  black cross) and (2) click and "shift" and draw a countour avoiding the host. The contours should be a ~5 spaxels large if possible. Finally (3) close the window to launch the PSF extraction.


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
