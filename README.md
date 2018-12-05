# PhenomPy

Python version of the algorithm by Khan et al (1508.07253, 1508.07250) for non-precessing, phenomenological waveform modeling. The main file, gr.py, reproduces a GR waveform for given source frame masses, dimensionless spins, luminosity distances, and Neutron Star flag (True or False) through the IMRPhenomD class, and derivative classes that modify the fisher analysis, but not the theory of gravity. All these quantities are in seconds, so use the constants defined in the utilities.py file to convert (see example code). The module, modified_gr.py, contains extensions to GR waveforms, as defined by the parameterized post-Einsteinian framework (0909.3328). The Modified_IMRPhenomD_Full_Freq.py includes a phase modification throughout the entire frequency range (generally suitable for propagation effects), while Modified_IMRPhenomD_Ins_Freq.py only includes the modification in the inspiral portion of the waveform (generally suitable for generation effects)

## Getting Started

You will need the following packages not commonly included in Python distributions:
multiprocessing
autograd
astropy

All are maintained on common package management systems like conda and pip.
```
pip install {autograd,multiprocessing,astropy}
```


### Installing

To be able to access this code through your path, either download this project into a folder already included in the $PYTHONPATH variable, or add the downloaded folder to your path manually, through the .bashrc file (most common solution).



## Classes

In gr.py:

IMRPhenomD - 
mass1 - Mass of the larger object
mass2 - Mass of the smaller object 
spin1 - Dimensionless spin of object 1 
spin2 - Dimensionless spin of object 2 
collision_time - time of coalescence (set to 0 generally) 
collision_phase - phase at the time of coalescence (set to 0 generally) 
Luminosity_Distance - Luminosity distance to the source  
cosmo_model - Cosmological model from astropy (default is Planck15), see http://docs.astropy.org/en/stable/cosmology/ 
NSflag - Either true or false to set the limits of integration
N_Detectors - The number of detectors (set to 1 by default)

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system



## Contributing

Please email me at scottperkins2@montana.edu if you are interested in contributing.


## Authors

* **Scott Perkins** - *Initial work and Maintenance* - [ScottPerkins](https://github.com/scottperkins)
* **Nico Yunes** - *Advising and guidance*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks to Kent Yagi, whose code was used to calibrate and verify this project
