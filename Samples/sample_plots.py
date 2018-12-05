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

show_plots = True

#Defining Parameters for the model 
dl = 420*mpc
mass1 =36*s_solm
mass2 =29*s_solm
spin1 = 0.32
spin2 = 0.44
detect = 'aLIGO'#This will be the detector for calculating Fisher Matricies
NSflag = False

model1 = imr(mass1 = mass1,mass2 = mass2, spin1 = spin1,spin2 = spin2,collision_time=0,collision_phase= 0,Luminosity_Distance = dl,  N_detectors = 1,NSflag=NSflag, cosmo_model = cosmology.Planck15)



#For speed in caclulating Fishers, some parts of the derivatives are precomputed and stored, so this command must first be run to initialize those values:
model1.calculate_derivatives()


# Plot Example Output
frequencies = np.linspace(1,5000,1e6)
#Calculates the waveform - 
Amp,phase,h = model1.calculate_waveform_vector(frequencies)

#Calculate the derivative wrt the symmetric mass ratio, which is parameter 5, :
eta_deriv = model1.calculate_waveform_derivative_vector(frequencies,5)
fig, axes = plt.subplots(2,2)
axes[0,0].plot(frequencies,Amp)
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].set_title('Amplitude')
axes[0,0].set_ylabel("Amplitude")
axes[0,0].set_xlabel("Frequency (Hz)")

axes[0,1].plot(frequencies,phase)
axes[0,1].set_title('Phase')
axes[0,1].set_ylabel("Phase")
axes[0,1].set_xlabel("Frequency (Hz)")

axes[1,0].plot(frequencies,h,linewidth=0.5)
axes[1,0].set_title('Full Waveform')
axes[1,0].set_ylabel("Waveform (s)")
axes[1,0].set_xlabel("Frequency (Hz)")
#axes[1,0].set_xlim(0,50)

axes[1,1].plot(frequencies,eta_deriv)
axes[1,1].set_title(r'$\partial{h}/\partial{log(\eta)}$')
axes[1,1].set_ylabel("Waveform (s)")
axes[1,1].set_xlabel("Frequency (Hz)")
axes[1,1].set_xscale('log')
axes[1,1].set_yscale('log')

plt.suptitle("Example Plots for Sample Model",fontsize = 16)
if show_plots:
    plt.show()
plt.close()
