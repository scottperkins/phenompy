import IMRPhenomD_full_mod as imrfull
import IMRPhenomD_ins_mod as imrins
import IMRPhenomD as imr
import numpy as np
import csv

mass1 = np.asarray([2,10,1e6],dtype=float)*imr.s_solm
mass2 = np.asarray([1,9,1e5],dtype=float)*imr.s_solm
spin1 = np.asarray([.01,.8,.4],dtype=float)
spin2 = np.asarray([-.01,.3,.7],dtype=float)
dl = np.asarray([100,800,4000],dtype=float)*imr.mpc
ns = [True,False,False]
detects = ['aLIGO','aLIGOFitted','LISA']
fishs, invfishs,total = [],[],[]
for x in np.arange(len(mass1)):
    model = imrfull.Modified_IMRPhenomD(mass1=mass1[x],mass2=mass2[x],spin1=spin1[x],spin2=spin2[x],
            collision_phase=0,collision_time=0,Luminosity_Distance=dl[x],NSflag=ns[x])
    fish,invfish,cholo = model.calculate_fisher_matrix_vector(detector=detects[x])
    fishs.append(fish)
    invfishs.append(invfish)
    for x in fish:
        for y in x:
            total.append(y)
    for x in invfish:
        for y in x:
            total.append(y)

compfishs = []
compinvfishs = []
with open('verification_data_full_mod.csv','r') as f:
    read = csv.reader(f,delimiter=',')
    for row in read:
        for item in row:
            compfishs.append(float(item))

total= np.asarray(total)
compfishs = np.asarray(compfishs)

print(np.nonzero(total-compfishs))
##############################################################################################
fishs, invfishs,total = [],[],[]
for x in np.arange(len(mass1)):
    model = imrins.insModified_IMRPhenomD(mass1=mass1[x],mass2=mass2[x],spin1=spin1[x],spin2=spin2[x],
            collision_phase=0,collision_time=0,Luminosity_Distance=dl[x],NSflag=ns[x])
    fish,invfish,cholo = model.calculate_fisher_matrix_vector(detector=detects[x])
    fishs.append(fish)
    invfishs.append(invfish)
    for x in fish:
        for y in x:
            total.append(y)
    for x in invfish:
        for y in x:
            total.append(y)

compfishs = []
compinvfishs = []
with open('verification_data_ins_mod.csv','r') as f:
    read = csv.reader(f,delimiter=',')
    for row in read:
        for item in row:
            compfishs.append(float(item))

total= np.asarray(total)
compfishs = np.asarray(compfishs)

print(np.nonzero(total-compfishs))
##############################################################################################
fishs, invfishs,total = [],[],[]
for x in np.arange(len(mass1)):
    model = imr.IMRPhenomD(mass1=mass1[x],mass2=mass2[x],spin1=spin1[x],spin2=spin2[x],
            collision_phase=0,collision_time=0,Luminosity_Distance=dl[x],NSflag=ns[x])
    fish,invfish,cholo = model.calculate_fisher_matrix_vector(detector=detects[x])
    fishs.append(fish)
    invfishs.append(invfish)
    for x in fish:
        for y in x:
            total.append(y)
    for x in invfish:
        for y in x:
            total.append(y)

compfishs = []
compinvfishs = []
with open('verification_data.csv','r') as f:
    read = csv.reader(f,delimiter=',')
    for row in read:
        for item in row:
            compfishs.append(float(item))

total= np.asarray(total)
compfishs = np.asarray(compfishs)

print(np.nonzero(total-compfishs))
