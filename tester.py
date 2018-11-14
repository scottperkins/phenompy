import IMRPhenomD as imr
import Modified_IMRPhenomD as modimr
import numpy as np
from analysis_utilities import LumDist_SNR

mass1 = np.asarray([2,10,1e6],dtype=float)*imr.s_solm
mass2 = np.asarray([1,9,1e5],dtype=float)*imr.s_solm
spin1 = np.asarray([.01,.8,.4],dtype=float)
spin2 = np.asarray([-.01,.3,.7],dtype=float)
dl = np.asarray([100,800,4000],dtype=float)*imr.mpc
ns = [True,False,False]
detects = ['aLIGO','aLIGOFitted','LISA']
DL = LumDist_SNR(mass1[1], mass2[1],spin1[1],spin2[1],SNR_target=100,detector='ET-D',lower_freq=5,upper_freq=400)
model = imr.IMRPhenomD(mass1=mass1[1],mass2=mass2[1],spin1=spin1[1],spin2=spin2[1],
        collision_phase=0,collision_time=0,Luminosity_Distance=DL*imr.mpc,NSflag=ns[1])
print(model.calculate_snr(detector='ET-D',lower_freq=5,upper_freq=400))

model = modimr.Modified_IMRPhenomD_Transition_Freq(mass1=mass1[1],mass2=mass2[1],spin1=spin1[1],spin2=spin2[1],
        collision_phase=0,collision_time=0,Luminosity_Distance=DL*imr.mpc,NSflag=ns[1])

fish4, invfish4, cholo = model.calculate_fisher_matrix_vector('aLIGO')

model = modimr.IMRPhenomD_Full_Freq_SPA(mass1=mass1[1],mass2=mass2[1],spin1=spin1[1],spin2=spin2[1],
        collision_phase=0,collision_time=0,Luminosity_Distance=DL*imr.mpc,NSflag=ns[1])

fish1, invfish1, cholo = model.calculate_fisher_matrix_vector('aLIGO')
print(fish1)

model = imr.IMRPhenomD(mass1=mass1[1],mass2=mass2[1],spin1=spin1[1],spin2=spin2[1],
        collision_phase=0,collision_time=0,Luminosity_Distance=DL*imr.mpc,NSflag=ns[1])

fish2, invfish2, cholo = model.calculate_fisher_matrix_vector('aLIGO')
print(fish2)
print(invfish1-invfish2)


model = modimr.IMRPhenomD_Inspiral_Freq_SPA(mass1=mass1[1],mass2=mass2[1],spin1=spin1[1],spin2=spin2[1],
        collision_phase=0,collision_time=0,Luminosity_Distance=DL*imr.mpc,NSflag=ns[1])

fish3, invfish3, cholo = model.calculate_fisher_matrix_vector('aLIGO')
print(fish3)
print(invfish1-invfish3)
print(invfish2-invfish3)

print('\n \n \n \n')
print(np.sqrt(np.diagonal(invfish4)))
print(np.sqrt(np.diagonal(invfish2)))
