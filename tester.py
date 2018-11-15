import IMRPhenomD as imr
import Modified_IMRPhenomD as modimr
import numpy as np
from analysis_utilities import LumDist_SNR

mass1 = np.asarray([2,50,1e6],dtype=float)*imr.s_solm
mass2 = np.asarray([1,20,1e5],dtype=float)*imr.s_solm
spin1 = np.asarray([.01,.8,.4],dtype=float)
spin2 = np.asarray([-.01,.3,.7],dtype=float)
dl = np.asarray([100,500,4000],dtype=float)*imr.mpc
ns = [True,False,False]
# detects = ['aLIGO','aLIGOFitted','LISA']

detects = ['aLIGO','aLIGO','LISA']
count = 2

DL = LumDist_SNR(mass1[count], mass2[count],spin1[count],spin2[count],SNR_target=10,detector=detects[count])
model = imr.IMRPhenomD(mass1=mass1[count],mass2=mass2[count],spin1=spin1[count],spin2=spin2[count],
        collision_phase=0,collision_time=0,Luminosity_Distance=DL*imr.mpc,NSflag=ns[count])
print(model.calculate_snr(detector=detects[count]))



model = modimr.Modified_IMRPhenomD_Transition_Freq(mass1=mass1[count],mass2=mass2[count],spin1=spin1[count],spin2=spin2[count],
        collision_phase=0,collision_time=0,Luminosity_Distance=dl[count],NSflag=ns[count])
fish4, invfish4, cholo = model.calculate_fisher_matrix_vector(detects[count])
print('\n \n \n')
print(model.f_trans1)
print(model.upper_freq)
print(np.sqrt(np.diagonal(invfish4)))
print('\n \n \n')

model = modimr.IMRPhenomD_Full_Freq_SPA(mass1=mass1[count],mass2=mass2[count],spin1=spin1[count],spin2=spin2[count],
        collision_phase=0,collision_time=0,Luminosity_Distance=dl[count],NSflag=ns[count])
fish1, invfish1, cholo = model.calculate_fisher_matrix_vector(detects[count])
print(fish1)

model = imr.IMRPhenomD(mass1=mass1[count],mass2=mass2[count],spin1=spin1[count],spin2=spin2[count],
        collision_phase=0,collision_time=0,Luminosity_Distance=dl[count],NSflag=ns[count])
fish2, invfish2, cholo = model.calculate_fisher_matrix_vector(detects[count])
print(fish2)
print(invfish1-invfish2)


model = modimr.IMRPhenomD_Inspiral_Freq_SPA(mass1=mass1[count],mass2=mass2[1],spin1=spin1[count],spin2=spin2[count],
        collision_phase=0,collision_time=0,Luminosity_Distance=dl[count],NSflag=ns[count])
fish3, invfish3, cholo = model.calculate_fisher_matrix_vector(detects[count])



model = modimr.Modified_IMRPhenomD_Full_Freq_SPA(mass1=mass1[count],mass2=mass2[count],spin1=spin1[count],spin2=spin2[count],
        collision_phase=0,collision_time=0,Luminosity_Distance=dl[count],NSflag=ns[count])
fish5, invfish5, cholo = model.calculate_fisher_matrix_vector(detects[count])

model = modimr.Modified_IMRPhenomD_Full_Freq(mass1=mass1[count],mass2=mass2[count],spin1=spin1[count],spin2=spin2[count],
        collision_phase=0,collision_time=0,Luminosity_Distance=dl[count],NSflag=ns[count])
fish6, invfish6, cholo = model.calculate_fisher_matrix_vector(detects[count])

# print(np.sqrt(np.diagonal(invfish5)))
# print(np.sqrt(np.diagonal(invfish6)))
for x in np.arange(8):
    # print(np.sqrt(np.diagonal(invfish5))[x])
    # print(np.sqrt(np.diagonal(invfish6))[x])
    print((np.sqrt(np.diagonal(invfish5))[x]-np.sqrt(np.diagonal(invfish6))[x])/(np.sqrt(np.diagonal(invfish5))[x]))

for x in np.arange(7):
    # print(np.sqrt(np.diagonal(invfish5))[x])
    # print(np.sqrt(np.diagonal(invfish6))[x])
    print((np.sqrt(np.diagonal(invfish1))[x]-np.sqrt(np.diagonal(invfish2))[x])/(np.sqrt(np.diagonal(invfish1))[x]))
