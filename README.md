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

## Files

gr.py contains a majority of the code. It houses the GR PhenomD algorithm and Fisher analysis, as well as the extensions to GR that don't consitute modifications to the theory of gravity.

modified_gr.py contains the classes that model extended theories of gravity, including a general PPE class for modelling parameterized post-Einsteinian modifications for the full and inspiral only frequency ranges, with and without SPA correction terms to the phase for both the ppE parameter and the GR phase. It also houses classes for including the transition frequency from inspiral to Merger-Ringdown as an extra parameter.

utilities.py contains general functions utilized in all the other files, including chirp mass and symmetric mass ratio calculations, etc.

noise_utilities.py includes all the spectral noise densities for various detectors.

analysis_utilities.py includes functions that are useful for post- model creation analysis, including some mapping functions from ppE parameters to physical parameters. 


## Classes

In gr.py:

IMRPhenomD - Class that computes waveforms, waveform derivatives, and Fisher matrices in GR

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


IMRPhenomD_Full_Freq_SPA - Same as IMRPhenomD, but with a phase term that is the next order correction to the SPA approximation. See https://arxiv.org/abs/gr-qc/9901076

IMRPhenomD_Inspiral_Freq_SPA - Same as above, but it only includes the correction to the inspiral portion of the phase.


In modified_gr.py

Modified_IMRPhenomD_Full_Freq - Class that incorporates a ppE phase parameter into the GR waveform of IMRPhenomD - same arguments with the following additions:

bppe - power of the modification:  delta phi = (pi Chirp Mass f)^(-bppe/3) 

Modified_IMRPhenomD_Inspiral_Freq - same as above, but the modification is only present in the inspiral portion. Good for modelling generation effects where the correction might not be accurate for higher frequencies. 

Modified_IMRPhenomD_Full_Freq_SPA - ppE waveform with SPA correction for both the GR phase and the ppE parameter 

Modified_IMRPhenomD_Inspiral_Freq_SPA - same as above, but only for the inspiral portion

Modified_IMRPhenomD_Transition_Freq - GR Waveform that treats the transition frequency as a model parameter for Fisher Calculations - extra argument 

f_int_mr - frequency in Hz at which the model should transition from the PN waveform to the merger-ringdown waveform

Modified_IMRPhenomD_All_Transition_Freq - same as above, but also includes the transition frequency from inspiral to intermediate waveforms as well



## Contributing

Please email me at scottperkins2@montana.edu if you are interested in contributing.


## Authors

* **Scott Perkins** - *Initial work and Maintenance* - [ScottPerkins](https://github.com/scottperkins)
* **Nico Yunes** - *Advising and guidance*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks to Kent Yagi, whose code was used to calibrate and verify this project
