####################################################################
"""Sample code that creates a model within GR, then plots the amplitude,
phase, waveform, and one of the parameter derivatives"""
####################################################################

from phenompy.gr import IMRPhenomD as imr
import matplotlib.pyplot as plt
import numpy as np
from phenompy.utilities import c, mpc, s_solm #importing constants - s_solm is the mass of the 
                                               #sun in seconds, mpc is a megaparsec in seconds,
                                                # and c is c in meters/sec    
import astropy.cosmology as cosmology


#Defining Parameters for the model 
dl = 420*mpc
mass1 =36*s_solm
mass2 =29*s_solm
spin1 = 0.32
spin2 = 0.44
detect = 'aLIGO'#This will be the detector for calculating Fisher Matricies
NSflag = False

#Create the model with the physical parameters
model1 = imr(mass1 = mass1,mass2 = mass2, spin1 = spin1,spin2 = spin2,collision_time=0,collision_phase= 0,Luminosity_Distance = dl,  N_detectors = 1,NSflag=NSflag, cosmo_model = cosmology.Planck15)

#Calculate the fisher and inverse of the fisher for the detector detect (aLIGO)
fisher, inverse_fisher = model1.calculate_fisher_matrix_vector(detector='aLIGO')

print(np.sqrt(np.diagonal(inverse_fisher)))

