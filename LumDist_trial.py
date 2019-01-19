import analysis_utilities
from gr import IMRPhenomD_detector_frame as imrdf
from time import time 
from utilities import s_solm, mpc
N_detectors = 3

mass1 = 100*s_solm
mass2 = 80*s_solm
spin1 = 0.9
spin2 = 0.7
DL_start = 100
model = imrdf(mass1=mass1, mass2=mass2,spin1=spin1,spin2=spin2, collision_time=0, \
                    collision_phase=0,Luminosity_Distance=DL_start*mpc,NSflag = False,N_detectors=N_detectors)
print(model.calculate_snr('Hanford_O2'))
time1 = time()

DL_start = analysis_utilities.LumDist_SNR_lite(model.chirpm,model.symmratio,SNR_target=10.8,N_detectors=N_detectors,detector='Hanford_O2')/mpc
DL =analysis_utilities.LumDist_SNR(mass1,mass2,spin1,spin2,SNR_target=10.8,initial_guess=DL_start, N_detectors=N_detectors,detector='Hanford_O2')
time2 = time()
print(time2-time1)

model = imrdf(mass1=mass1, mass2=mass2,spin1=spin1,spin2=spin2, collision_time=0, \
                    collision_phase=0,Luminosity_Distance=DL*mpc,NSflag = False,N_detectors=N_detectors)
print(model.calculate_snr('Hanford_O2'))
