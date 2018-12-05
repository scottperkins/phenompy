"""This file contains all the noise curve data, unpacked from ./Data_Tables,
 as well as analytic curves for various detectors"""
import csv
import os
import autograd.numpy as np

"""Path variables"""
IMRPD_dir = os.path.dirname(os.path.realpath(__file__))
IMRPD_tables_dir = IMRPD_dir + '/Data_Tables'


"""Read in csv file with noise curve data for Fisher Analysis - aLIGO is column 1 and freq is column 0"""
noise = []
for i in range(11):
    noise.append([])
with open(IMRPD_tables_dir+'/curves.csv', 'r',encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        for i in np.arange(len(row)):
            noise[i].append(float(row[i]))
noise_lisa = [[],[],[]]
with open(IMRPD_tables_dir+'/NewLISATable.dat','r',encoding='utf-8') as file:
    reader = csv.reader(file,delimiter="\t")
    for line in reader:
        noise_lisa[0].append(float(line[0]))
        noise_lisa[1].append(float(line[1]))
        noise_lisa[2].append(float(line[2]))
noise_hanford_O2 = [[],[]]
with open(IMRPD_tables_dir+'/Hanford_O2_Strain.csv', 'r',encoding='utf-8') as file:
    reader = csv.reader(file,delimiter =',')
    for line in reader:
        noise_hanford_O2[0].append(float(line[0]))
        noise_hanford_O2[1].append(float(line[1])) 

"""Analytic noise curve from Will '97, for testing - aLIGOAnalytic - arXiv:gr-qc/9709011v1
returns S**(1/2)"""
def sym_noise_curve(f):
    S = 3e-48
    fknee = 70.
    return np.sqrt(S * ((fknee/f)**4 + 2 + 2*(f/fknee)**2)/5)


"""Returns a fitted S**(1/2) curve of Hanford O(1) noise data - see 1603.08955 appendix C"""
def noise_hanford_O1( f):
    a_vec = np.array([47.8466,-92.1896,35.9273,-7.61447,0.916742,-0.0588089,0.00156345],dtype=float)
    S0 = .8464
    x = np.log(f)
    return (np.sqrt(S0) * np.exp(a_vec[0] + a_vec[1]*x + a_vec[2]*x**2 +
            a_vec[3] * x**3 + a_vec[4]*x**4 + a_vec[5]*x**5 + a_vec[6]*x**6))
    

"""Returns noise**(1/2) of DECIGO for frequency f in Hz
https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.109901"""
def noise_decigo(f):
    S0 = 7.05e-48
    S1 = 4.8e-51
    S2 = 5.33e-52
    fp = 7.36
    return np.sqrt(S0*(1+ (f/fp)**2)  + S1 * (f)**(-4) * (1 / (1+ (f/fp)**2) ) + S2 * f**(-4))


###########################################################################################
# """Plot Noise Curves to check data etc"""
#import matplotlib.pyplot as plt
#names = [ 'aLIGO', 'A+', 'A++', 'Vrt', 'Voyager', 'CE1', 'CE2 wide', 'CE2 narrow', 'ET-B', 'ET-D']
#fig,ax = plt.subplots()
#for i in np.arange(len(noise[1:])):
#    ax.plot(noise[0],noise[i+1],label = names[i])
#
#ax.plot(np.linspace(10.,2500,1e5),noise_hanford_O1(np.linspace(10.,2000,1e5)),label='fitted Hanford Noise',linestyle='--')
#
#ax.plot(np.linspace(1e-3,100,1e5),noise_decigo(np.linspace(1e-3,100,1e5)),label='DECIGO',linestyle='-.')
#
#ax.plot(noise_lisa[0],noise_lisa[1],label='LISA')
#ax.plot(noise_hanford_O2[0],noise_hanford_O2[1],label='O2 hanford')
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_title("Noise Curves")
#ax.set_ylabel(r'$S^{1/2}$')
#ax.set_xlabel('Frequency (Hz)')
#ax.legend()
#plt.show()
#plt.close()
###########################################################################################
