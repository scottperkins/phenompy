import Modified_IMRPhenomD as modimr
import IMRPhenomD as imr
import numpy as np

mass1 = np.asarray([2,10,1e6],dtype=float)*imr.s_solm
mass2 = np.asarray([1,9,1e5],dtype=float)*imr.s_solm
spin1 = np.asarray([.01,.8,.4],dtype=float)
spin2 = np.asarray([-.01,.3,.7],dtype=float)
dl = np.asarray([100,800,4000],dtype=float)*imr.mpc
ns = [True,False,False]
detects = ['aLIGO','aLIGOFitted','LISA']

model = modimr.Modified_IMRPhenomD_Transition_Freq(mass1=mass1[1],mass2=mass2[1],spin1=spin1[1],spin2=spin2[1],
        collision_phase=0,collision_time=0,Luminosity_Distance=dl[1],NSflag=ns[1])
fish, invfish, cholo = model.calculate_fisher_matrix_vector('aLIGO')
print(fish)
modelGR = imr.IMRPhenomD(mass1=mass1[1],mass2=mass2[1],spin1=spin1[1],spin2=spin2[1],
        collision_phase=0,collision_time=0,Luminosity_Distance=dl[1],NSflag=ns[1])

fishGR, invfishGR, cholo = modelGR.calculate_fisher_matrix_vector('aLIGO')
for i in np.arange(len(fish)):
    for j in np.arange(i):
        if i < len(fishGR) and j < len(fishGR):
            print(fishGR[i][j]-fish[i][j])
        else:
            print("EXTRA")
            print(fish[i][j])
print(np.sqrt(np.diagonal(invfish)))
print(np.sqrt(np.diagonal(invfishGR)))
