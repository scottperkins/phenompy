# IMRPhenomD
# IMRPhenomD

Python version of the algorithm by Khan et al (1508.07253, 1508.07250) for non-precessing, phenomenological waveform modeling. The main file, IMRPhenomD.py, reproduces a GR waveform for given source frame masses, dimensionless spins, luminosity distances, and Neutron Star flag (True or False). All these quantities are in seconds, so use the constants defined in the initial lines to convert (see example code). The other class files contain extensions to GR waveforms, as defined by the parameterized post-Einsteinian framework (0909.3328). The IMRPhenomD_full_mod.py includes a phase modification throughout the entire frequency range (generally suitable for propagation effects), while IMRPhenomD_ins_mod.py only includes the modification in the inspiral portion of the waveform (generally suitable for generation effects)

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

To be able to access this code through your path, either download this project into a folder already included in the $PYTHONPATH variable, or add the downloaded folder to your path manually, through the .bashrc file.



## Running the tests

Explain how to run the automated tests for this system

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

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Scott Perkins** - *Initial work and Maintenance* - [PurpleBooth](https://github.com/scottperkins)
* **Nico Yunes** - *Advising and guidance*
See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks to Kent Yagi, whose code was used to calibrate and verify this project
