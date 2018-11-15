####################################################################
"""Sample code that creates a model within GR, then plots the amplitude,
phase, waveform, and one of the parameter derivatives"""
####################################################################


import IMRPhenomD as imr
import matplotlib.pyplot as plt
import numpy as np

show_plots = True

mpc = imr.mpc
c = imr.c
s_solm = imr.s_solm

dl = 420*mpc
mass1 =36*s_solm
mass2 =29*s_solm
spin1 = 0.32
spin2 = 0.44
detect = 'aLIGO'
NSflag = False

model1 = imr.IMRPhenomD(mass1,mass2,spin1,spin2,0,0,dl,N_detectors = 1,NSflag=NSflag)

model1.calculate_derivatives()#This is only necessary to plot the derivative - precompiles various derivatives for speed

# Plot Example Output
frequencies = np.linspace(1,5000,1e6)
frequencies = np.linspace(1e-4,.001,1e5)
Amp,phase,h = model1.calculate_waveform_vector(frequencies)

eta_deriv = model1.log_factors[5]*model1.calculate_waveform_derivative_vector(model1.split_freqs_amp(frequencies),model1.split_freqs_phase(frequencies),5)
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
