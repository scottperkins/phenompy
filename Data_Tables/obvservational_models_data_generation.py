import numpy as np
import IMRPhenomD as pd
import csv
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
"""This code is to create models mimicing the observations of LIGO and tabulating the resulting
lambda_g / Vainshtein radius relationship """

"""Parameter References:
GW150914: PRL 116, 241102 (2016)
GW151226:: 10.1103/PhysRevLett.116.241103 (No Spins quoted? High probability that one BH a > .2)
GW170104: PRL.118.221101 / https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.118.221101/GW170104-Supplement.pdf
GW170814: 10.1103/PhysRevLett.119.141101 (No spins?)
GW170817: 1805.11579 (Picked low spin prior -- Also different values quoted for high spin
                    -- also just averaged masses from upper/lower bounds, but are the bounds asymmetric?)
GW170608:IOP 851:L35 (Poorly constrained spins - set to 0 for now)
"""
points = 5000
data_filetree = os.path.dirname(os.path.realpath(__file__))
names = ['GW150914','GW151226','GW170104','GW170814','GW170817','GW170608']
mass1 = np.asarray([36,14.2,31.2,30.5,1.48,12],dtype=float)*pd.s_solm
mass2 = np.asarray([29,7.5,19.4,25.3,1.26,7],dtype=float)*pd.s_solm
spin1 = np.asarray([.32,.2,.45,.01,.02,.0],dtype=float)
spin2 = np.asarray([.44,.01,.47,.01,.04,.0],dtype=float)
LumD = np.asarray([410,440,880,540,44.7,340],dtype=float)*pd.mpc
NSFlag = [False,False,False,False,True,False]
detector ='aLIGO'


###############################################################################################
#Script to populate Data Tables for plotting degeneracy plots for overlaying
###############################################################################################
# for i in np.arange(len(names)):
#     model = pd.Modified_IMRPhenomD(mass1=mass1[i],mass2=mass2[i],spin1=spin1[i],spin2=spin2[i],\
#         collision_time=0,collision_phase=0,Luminosity_Distance=LumD[i],NSflag=NSFlag[i],N_detectors=2)
#     fish,invfish,cholo = model.calculate_fisher_matrix_vector('aLIGO')
#     DB = np.sqrt(np.diagonal(invfish)[-1])
#     Rv = np.linspace(10e-5*model.DL/2,.9999999*model.DL/2,points)
#     lambdag = model.degeneracy_function_lambda(DB,Rv)
#     with open(data_filetree+'/'+names[i]+'_fisher_deg.csv','w') as f:
#         writer = csv.writer(f,delimiter=',')
#         output = [[Rv[t],lambdag[t]] for t in np.arange(len(Rv))]
#         for i in output:
#             writer.writerow(list(i))
###############################################################################################

###############################################################################################
#Code to plot each observation in a grid
###############################################################################################
# colnum = 2
# rownum = 3
# fig, axes = plt.subplots(rownum,colnum,sharex='col',sharey='row',figsize=(10,4))
# points = 5000
# maxx, maxy, minx, miny = [],[],[],[]
# k,j = -1, -1
# for i in np.arange(len(names)):
#     if k == 2:
#         k = 0
#     else:
#         k+=1
#     if j == 1:
#         j = 0
#     else:
#         j+=1
#     model = pd.Modified_IMRPhenomD(mass1=mass1[i],mass2=mass2[i],spin1=spin1[i],spin2=spin2[i],\
#         collision_time=0,collision_phase=0,Luminosity_Distance=LumD[i],NSflag=NSFlag[i],N_detectors=2)
#     fish,invfish,cholo = model.calculate_fisher_matrix_vector(detector)
#     x,y= model.create_degeneracy_data(points = points)
#     x = np.divide(x,pd.mpc)
#     y = np.multiply(y,pd.c)
#     maxx.append(x[-1])
#     maxy.append(y[0])
#     minx.append([0])
#     miny.append([-1])
#     lower = np.zeros(points)
#     axes[k][j].fill_between(x,y,lower,alpha = 0.5)
# max_x, min_x = max(maxx),min(minx)[0]
# max_y, min_y = max(maxy), min(miny)[0]
# counter = 0
# for i in np.arange(rownum):
#     for j in np.arange(colnum):
#         axes[i][j].set_xlim([min_x,max_x])
#         axes[i][j].set_ylim([min_y,max_y])
#         axes[i][j].annotate(names[counter],[350,.75e16])
#         counter+=1
#
# axes[1][0].set_ylabel(r'Bound on $\lambda_g$ (meters)')
# axes[2][0].set_xlabel(r'Bound on Vainshtein Radius (Mpc)')
# fig.subplots_adjust(hspace=0)
# fig.suptitle('LIGO Observations - Graviton Mass Bound Incorporating Screening')
# script_dir = os.path.dirname(os.path.realpath(__file__))
# os.chdir(script_dir)
# plt.savefig('../../../Figures/Scott/ligo_observations_deg.png')
# plt.show()
# plt.close()

###############################################################################################

###############################################################################################
#LISA Predictions
###############################################################################################
# points = 5000
# mass1 = np.asarray([6e6,6e6,5e6,5e5,4e4,4e4],dtype=float)*pd.s_solm
# mass2 = np.asarray([5e6,4e5,4e5,4e4,3e4,3e4],dtype=float)*pd.s_solm
# spin1 = np.asarray([.32,.2,.45,.01,.02,.0],dtype=float)
# spin2 = np.asarray([.44,.01,.47,.01,.04,.0],dtype=float)
# LumD = np.asarray([4100,4400,8800,5400,8000,1000],dtype=float)*pd.mpc
# NSFlag = [False,False,False,False,False,False]
# detector ='LISA'
#
# colnum = 2
# rownum = 3
# fig, axes = plt.subplots(rownum,colnum,sharex='col',sharey='row',figsize=(10,4))
# points = 5000
# maxx, maxy, minx, miny = [],[],[],[]
# k,j = -1, -1
# for i in np.arange(len(mass1)):
#     if k == 2:
#         k = 0
#     else:
#         k+=1
#     if j == 1:
#         j = 0
#     else:
#         j+=1
#     model = pd.Modified_IMRPhenomD(mass1=mass1[i],mass2=mass2[i],spin1=spin1[i],spin2=spin2[i],\
#         collision_time=0,collision_phase=0,Luminosity_Distance=LumD[i],NSflag=NSFlag[i],N_detectors=1)
#     fish,invfish,cholo = model.calculate_fisher_matrix_vector(detector)
#     x,y= model.create_degeneracy_data(points = points)
#     x = np.divide(x,pd.mpc)
#     y = np.multiply(y,pd.c)
#     maxx.append(x[-1])
#     maxy.append(y[0])
#     minx.append([0])
#     miny.append([-1])
#     lower = np.zeros(points)
#     axes[k][j].fill_between(x,y,lower,alpha = 0.5)
# max_x, min_x = max(maxx),min(minx)[0]
# max_y, min_y = max(maxy), min(miny)[0]
# counter = 0
# for i in np.arange(rownum):
#     for j in np.arange(colnum):
#         axes[i][j].set_xlim([min_x,max_x])
#         axes[i][j].set_ylim([min_y,max_y])
#         axes[i][j].annotate('M1 : {} \n M2: {} \n DL: {}'.format(mass1[counter]/pd.s_solm,
#         mass2[counter]/pd.s_solm,LumD[counter]/pd.mpc),[3000,.1e20])
#         counter+=1
#
# axes[1][0].set_ylabel(r'Bound on $\lambda_g$ (meters)')
# axes[2][0].set_xlabel(r'Bound on Vainshtein Radius (Mpc)')
# fig.subplots_adjust(hspace=0)
# fig.suptitle('LISA Predictions - Graviton Mass Bound Incorporating Screening')
# script_dir = os.path.dirname(os.path.realpath(__file__))
# os.chdir(script_dir)
# plt.savefig('../../../Figures/Scott/lisa_deg.png')
# plt.show()
# plt.close()
###############################################################################################

###############################################################################################
#dRGT specific - solving for the graviton mass including screening - LIGO observations
###############################################################################################
points = 5000
mass1 = np.asarray([36,14.2,31.2,30.5,1.48,12],dtype=float)*pd.s_solm
mass2 = np.asarray([29,7.5,19.4,25.3,1.26,7],dtype=float)*pd.s_solm
spin1 = np.asarray([.32,.2,.45,.01,.02,.0],dtype=float)
spin2 = np.asarray([.44,.01,.47,.01,.04,.0],dtype=float)
LumD = np.asarray([410,440,880,540,44.7,340],dtype=float)*pd.mpc
NSFlag = [False,False,False,False,True,False]
detector ='aLIGO'

lambdag = []
def Dfunc(beta,lambdag):
    return lambdag**2 - pd.Dfunc()
for i in np.arange(len(mass1)):
    model = pd.Modified_IMRPhenomD(mass1=mass1[i],mass2=mass2[i],spin1=spin1[i],spin2=spin2[i],\
            collision_time=0,collision_phase=0,Luminosity_Distance=LumD[i],NSflag=NSFlag[i],N_detectors=2)
    fish,invfish,cholo = model.calculate_fisher_matrix_vector(detector)
    beta = np.sqrt(np.diagonal(invfish))[-1]
    gal_mass = 5.8e11*pd.s_solm
    # print((gal_mass*(1e16/(pd.hplanck*pd.c))**2)**(1/3)/pd.mpc)
    lambdag.append(fsolve(lambda l: l-model.degeneracy_function_lambda(beta,(gal_mass*(l/pd.hplanck)**2)**(1/3)),1e14/pd.c)[0])
    # x = np.linspace(.8e15/pd.c,.8e16/pd.c,1000)
    # y = list(map(lambda l: l-model.degeneracy_function_lambda(beta,(gal_mass*(l/pd.hplanck)**2)**(1/3) ),x))
    # plt.plot(np.asarray(x)*pd.c,y)
    # plt.show()
    # plt.close()
for x in lambdag:
    print("Compton Wavelength: {}x10^16".format(x*pd.c/1e16))


###############################################################################################

###############################################################################################
#dRGT specific - solving for the graviton mass including screening - LISA
###############################################################################################
points = 5000
mass1 = np.asarray([6e6,6e6,5e6,5e5,4e4,4e4],dtype=float)*pd.s_solm
mass2 = np.asarray([5e6,4e5,4e5,4e4,3e4,3e4],dtype=float)*pd.s_solm
spin1 = np.asarray([.32,.2,.45,.01,.02,.0],dtype=float)
spin2 = np.asarray([.44,.01,.47,.01,.04,.0],dtype=float)
LumD = np.asarray([4100,4400,8800,5400,8000,1000],dtype=float)*pd.mpc
NSFlag = [False,False,False,False,False,False]
detector ='LISA'


lambdag = []
def Dfunc(beta,lambdag):
    return lambdag**2 - pd.Dfunc()
for i in np.arange(len(mass1)):
    model = pd.Modified_IMRPhenomD(mass1=mass1[i],mass2=mass2[i],spin1=spin1[i],spin2=spin2[i],\
            collision_time=0,collision_phase=0,Luminosity_Distance=LumD[i],NSflag=NSFlag[i],N_detectors=1)
    fish,invfish,cholo = model.calculate_fisher_matrix_vector(detector)
    beta = np.sqrt(np.diagonal(invfish))[-1]
    gal_mass = 5.8e11*pd.s_solm
    # print((gal_mass*(1e16/(pd.hplanck*pd.c))**2)**(1/3)/pd.mpc)
    lambdag.append(fsolve(lambda l: l-model.degeneracy_function_lambda(beta,(gal_mass*(l/pd.hplanck)**2)**(1/3)),1e15/pd.c)[0])
    # x = np.linspace(.8e15/pd.c,.8e16/pd.c,1000)
    # y = list(map(lambda l: l-model.degeneracy_function_lambda(beta,(gal_mass*(l/pd.hplanck)**2)**(1/3) ),x))
    # plt.plot(np.asarray(x)*pd.c,y)
    # plt.show()
    # plt.close()
for x in lambdag:
    print("Compton Wavelength: {}x10^16".format(x*pd.c/1e16))


###############################################################################################
