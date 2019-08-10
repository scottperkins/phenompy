from phenompy.gr import IMRPhenomD as imr
from utilities import c,s_solm,mpc
import numpy as np
from time import time
from phenompy.analysis_utilities import log_likelihood

#start = time()
mass1 = 50.6*s_solm
mass2 = 34.3*s_solm
spin1 = 0.8
chi_eff = .36
spin2 = (chi_eff * (mass1 + mass2) - mass1*spin1)/mass2
NS = False
N_detectors = 3
detector = 'Hanford_O2'
dl = 500
#model = imr(mass1=mass1,mass2=mass2,collision_phase=0,collision_time=0,
#                spin1=spin1,spin2=spin2,Luminosity_Distance=dl*mpc,NSflag=NS,N_detectors=N_detectors)
#print(model.chirpm,model.symmratio)
#freqs = np.asarray([.001])
#amp,phase,h = model.calculate_waveform_vector(freqs)
#model.calculate_derivatives()
#h = model.calculate_waveform_derivative(freqs,4)
#print(time()-start)
#print(10*(complex(np.cos(np.pi/4),np.sin(np.pi/4))))
loops = 13
length = 2000*32
data = np.random.rand(length)*1e-21
frequencies = np.linspace(11,5000,length)
model = imr(mass1=mass1,mass2=mass2,collision_phase=0,collision_time=0,
                spin1=spin1,spin2=spin2,Luminosity_Distance=dl*mpc,NSflag=NS,N_detectors=N_detectors)
freqs = np.asarray([.001])
amp,phase,h = model.calculate_waveform_vector(freqs)
print(amp)
print("SNR: {}".format(model.calculate_snr(detector=detector)))
amp,phase,h = model.calculate_waveform_vector(frequencies)
data =   np.multiply(amp,np.add(np.cos(phase),1j*np.sin(phase)))
start = time()
for i in range(loops):
    A0 = model.A0
    t_c = 0
    phi_c = 0
    chirpm =model.chirpm
    symmratio = model.symmratio
    chi_s = model.chi_s
    chi_a = model.chi_a
    beta = 0.0
    bppe = -1
    NSflag= False
    N_detectors = 3
    detector = 'Hanford_O2'
    
    logl = log_likelihood(Data=data,frequencies = frequencies,A0 = A0,t_c = t_c, phi_c = phi_c, chirpm=chirpm,symmratio=symmratio,chi_s=chi_s,chi_a=chi_a,beta=beta, bppe=bppe,NSflag=NSflag, detector=detector,N_detectors=N_detectors)
    print(logl)

print((time()-start)/loops)
