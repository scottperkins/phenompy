import autograd.numpy as np
import astropy.cosmology as cosmology
from scipy.optimize import fsolve
import IMRPhenomD as imr

"""Useful functions for analysis of waveforms. NOT used for waveform production, like the utilities.py file.
These functions usually require creating or passing an already created model."""



"""Calculate luminositiy distance for a desired SNR and model"""
###########################################################################################
# def LumDist_SNR(mass1, mass2,spin1,spin2,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors=1,detector='aLIGO',SNR_target=10):
#     temp_model = imr.IMRPhenomD(mass1=mass1, mass2=mass2,spin1=spin1,spin2=spin2, collision_time=0, \
#                     collision_phase=0,Luminosity_Distance=50*imr.mpc,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors=1)
#     SNR_temp = temp_model.calculate_snr(detector=detector)
#     D_L_target = np.sqrt((SNR_temp / SNR_target) *(50*imr.mpc)**2)
#     return D_L_target/imr.mpc

def LumDist_SNR(mass1, mass2,spin1,spin2,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors=1,detector='aLIGO',SNR_target=10,lower_freq=None,upper_freq=None):
    D_L_target = fsolve(
            lambda l: SNR_target - LumDist_SNR_assist(mass1=mass1, mass2=mass2,spin1=spin1,spin2=spin2,DL=l,cosmo_model = cosmo_model,NSflag = NSflag,N_detectors=N_detectors,detector=detector,lower_freq=lower_freq,upper_freq=upper_freq),
            100*imr.mpc)[0]
    return D_L_target/imr.mpc
def LumDist_SNR_assist(mass1, mass2,spin1,spin2,DL,cosmo_model,NSflag ,N_detectors,detector,lower_freq,upper_freq):
    temp_model = imr.IMRPhenomD(mass1=mass1, mass2=mass2,spin1=spin1,spin2=spin2, collision_time=0, \
                    collision_phase=0,Luminosity_Distance=DL,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors=1)
    SNR_temp = temp_model.calculate_snr(detector=detector,lower_freq=lower_freq,upper_freq=upper_freq)
    return SNR_temp
###########################################################################################
