import autograd.numpy as np
import numpy
from scipy import integrate
import autograd.scipy.linalg as spla
import math
import csv
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from autograd import grad
from time import time
from functools import partial
from autograd.extend import primitive, defvjp
from autograd import elementwise_grad as egrad
import astropy.cosmology as cosmology
from astropy.coordinates import Distance
from astropy import units as u
import astropy.constants as consts
from scipy.interpolate import interp1d

from noise_utilities import *
import utilities
print("YOU JUST IMPORTED THE NEW IMRPHENOMD !!!!")
plt.style.use('seaborn-whitegrid')

"""Euler's Number (Not in SciPy or NumPy Constants)"""
gamma_E = 0.5772156649015328606065120900824024310421
"""Physical Constants - importable with IMRPhenomD.constant"""
c = 299792458#consts.c #Speed of light m/s
G = consts.G.to('m3/(s2 kg)').value*consts.M_sun.value #Gravitational constant in m**3/(s**2 SolMass)=6.674*10**(-11)*(1.98855*10**30)
s_solm = G / consts.c.value**3#G/c**3#seconds per solar mass =492549095*10**(-14)
mpc = 1/consts.c.to('Mpc/s').value#consts.kpc.to('m')*1000/c#Mpc in sec =3085677581*10**(13)/c
H0 = cosmology.Planck15.H0#6780*10**(-2)/(3 * 10**5)#67.80/(3.086*10**19) #Hubble constant in [1/Mpc]
hplanck = consts.h.to('eV s').value #Planck Constant in eV s

"""Path variables"""
IMRPD_dir = os.path.dirname(os.path.realpath(__file__))
IMRPD_tables_dir = IMRPD_dir + '/Data_Tables'

###########################################################################################
#Read in data
###########################################################################################
"""Read in csv file with IMRPhenomD Phenomological Parameters
Array is Lambda[i][j] - i element of {rho_n,v2,gamma_n,sigma_n,beta_n,alpha_n}
and j element of lambda{00,10,01,11,21,02,12,22,03,13,23}"""
Lambda = np.zeros((19,11))
with open(IMRPD_tables_dir+'/IMRPhenomDParameters_APS.csv','r') as f:
    reader = csv.reader(f)
    i = -1
    for row in reader:
        if i == -1:
            i += 1
            continue
        rowconvert = []
        for x in row[1:]:
            rowconvert.append(float(eval(x)))
        Lambda[i] = rowconvert
        i += 1
#
# """Read in csv file with noise curve data for Fisher Analysis - aLIGO is column 1 and freq is column 0"""
# noise = []
# for i in range(11):
#     noise.append([])
# with open(IMRPD_tables_dir+'/curves.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         for i in np.arange(len(row)):
#             noise[i].append(float(row[i]))
# noise_lisa = [[],[],[]]
# with open(IMRPD_tables_dir+'/NewLISATable.dat','r') as file:
#     reader = csv.reader(file,delimiter="\t")
#     for line in reader:
#         noise_lisa[0].append(float(line[0]))
#         noise_lisa[1].append(float(line[1]))
#         noise_lisa[2].append(float(line[2]))


"""For Data imported below, see the script ./Data_Tables/tabulate_data.py for production of values and accuracy testing"""
"""Read in tabulated data for the luminosity distance to Z conversion - tabulated and interpolated for speed"""
LumDZ = [[],[]]
with open(IMRPD_tables_dir+'/tabulated_LumD_Z.csv', 'r') as f:
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        LumDZ[0].append(float(row[0]))
        LumDZ[1].append(float(row[1]))
Zfunc = interp1d(LumDZ[0],LumDZ[1])

"""Read in the tabulated data for the integral in the cosmological distance defined in Will '97
 - Tabulated and interpolated for speed"""
ZD = [[],[]]
with open(IMRPD_tables_dir+'/tabulated_Z_D.csv','r') as f:
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        ZD[0].append(float(row[0]))
        ZD[1].append(float(row[1]))
Dfunc = interp1d(ZD[0],ZD[1])
###########################################################################################



"""Class for Inspiral-Merger-Ringdown Phenomenological model D
See paper by Khan et al - arxiv:1508.07253 and 1508.01250v2 for algorithm and parameter values
Only deviation from algorithm from Khan et. al. is the adjustment of the overall amplitude scale A_0

Required Packages beyond the standard libraries: multiprocessing, autograd

Example code below this class will excute only if this file is run directly, not if it's imported - shows some basic method call examples

The table of Numerical Fit parameters (Lambda Parameters) and the noise curve data must be in the same folder as this program
-------mass 1 must be larger-------"""
class IMRPhenomD():
    """parameters: mass 1, mass 2, spin PARAMETERS 1 and 2,tc - collision time, phase at tc,
     the luminosity distance, Cosmology to use (must be a supported cosmology in the astropy.cosmology package), and NSflag (True or False)
      and N_detectors is the number of detectors that observed the event
      - all should be in units of [s] or [1/s] - use constants defined above for conversion
      - all parameters are in the SOURCE frame"""
    def __init__(self, mass1, mass2,spin1,spin2, collision_time, \
                    collision_phase,Luminosity_Distance,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors=1):
        """Populate model variables"""
        self.N_detectors = N_detectors
        self.NSflag = NSflag
        self.cosmo_model = cosmo_model
        self.DL = Luminosity_Distance
        self.tc = float(collision_time)
        self.phic = float(collision_phase)
        self.symmratio = (mass1 * mass2) / (mass1 + mass2 )**2
        self.chirpme =  (mass1 * mass2)**(3/5)/(mass1 + mass2)**(1/5)
        self.delta = self.calculate_delta(self.symmratio)
        self.Z =Distance(Luminosity_Distance/mpc,unit=u.Mpc).compute_z(cosmology = self.cosmo_model)
        self.chirpm = self.chirpme*(1+self.Z)
        self.M = utilities.calculate_totalmass(self.chirpm,self.symmratio)
        self.m1 = self.calculate_mass1(self.chirpm,self.symmratio)
        self.m2 = self.calculate_mass2(self.chirpm,self.symmratio)
        self.A0 =(np.pi/30)**(1/2)*self.chirpm**2/self.DL * (np.pi*self.chirpm)**(-7/6)
        self.totalMass_restframe = mass1+mass2
        """Spin Variables"""
        self.chi1 = spin1
        self.chi2 = spin2
        self.chi_s = (spin1 + spin2)/2
        self.chi_a = (spin1 - spin2)/2

        """Post Newtonian Phase"""
        self.pn_phase = np.zeros(8)
        for i in [0,1,2,3,4,7]:
            self.pn_phase[i] = self.calculate_pn_phase(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1,i)

        """Numerical Fit Parameters"""
        self.parameters =[]
        for i in np.arange(len(Lambda)):
            self.parameters.append(self.calculate_parameter(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))

        """Post Newtonian Amplitude"""
        self.pn_amp = np.zeros(7)
        for i in np.arange(7):
            self.pn_amp[i]=self.calculate_pn_amp(self.symmratio,self.delta,self.chi_a,self.chi_s,i)

        """Post Merger Parameters - Ring Down frequency and Damping frequency"""
        self.fRD = self.calculate_postmerger_fRD(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fdamp = self.calculate_postmerger_fdamp(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fpeak = self.calculate_fpeak(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6])

        """Calculating the parameters for the intermediate amplitude region"""
        self.param_deltas = np.zeros(5)
        for i in np.arange(5):
            self.param_deltas[i] = self.calculate_delta_parameter(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i)

        """Phase continuity parameters"""
        """Must be done in order - beta1,beta0,alpha1, then alpha0"""
        self.beta1 = self.phase_cont_beta1(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s)
        self.beta0 = self.phase_cont_beta0(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1)
        self.alpha1 = self.phase_cont_alpha1(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1)
        self.alpha0 = self.phase_cont_alpha0(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1)
        self.var_arr = [self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a]

    """Calculates the total mass given symmratio and chirp mass"""
    # def calculate_totalmass(self,chirpm,symmratio):return chirpm*symmratio**(-3/5)

    """Calculates the redshift given a luminosity distance"""
    def calculate_Z(self,DL):
        """Interpolates the luminosity distance vs Z relationship for the standard cosmology
        argument: Luminosity Distance in sec"""
        OmegaM = .3
        OmegaL = .7
        zTable = np.arange(0,10.01,.01)
        DLTable = [ (1+z1)/H0 * integrate.quad(lambda z2: ((OmegaM)*(1+z2)**3 +OmegaL)**(-1/2) ,0 , z1)[0] for z1 in zTable]
        return CubicSpline(DLTable,zTable)(DL/mpc)

    """Calculates the individual masses given symmratio and chirp mass"""
    ###########################################################################################################
    def calculate_mass1(self,chirpm,symmratio):
        return 1/2*(chirpm / symmratio**(3/5) \
        + np.sqrt(1-4*symmratio)*chirpm / symmratio**(3/5))

    def calculate_mass2(self,chirpm,symmratio):
        return 1/2*(chirpm / symmratio**(3/5) \
        - np.sqrt(1-4*symmratio)*chirpm / symmratio**(3/5))
    ###########################################################################################################

    """calculate fpeak"""
    def calculate_fpeak(self,M,fRD,fdamp,gamma2,gamma3):return abs(fRD + fdamp*gamma3*(np.sqrt(1-gamma2**2)-1)/gamma2)

    """Calculate delta parameter"""
    def calculate_delta(self,symmratio):return np.sqrt(1-4*symmratio)

    ###########################################################################################################
    """Calculates the parameters beta0, alpha0, beta1, and alpha1 based on the condition that the phase is continuous
    across the boundary"""
    def phase_cont_beta1(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s):
        M = self.assign_totalmass(chirpm,symmratio)
        f1 = 0.018/M
        pn_phase =[]
        for x in np.arange(len(self.pn_phase)):
            pn_phase.append(self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f1,x))
            # pn_phase.append(self.calculate_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f1,x))
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
        sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
        sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
        ins_grad = egrad(self.phi_ins,0)
        return (1/M)*ins_grad(f1,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)*symmratio\
         - (beta2*(1/(M*f1)) + beta3*(M*f1)**(-4))

    def phase_cont_alpha1(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1):
        M = self.assign_totalmass(chirpm,symmratio)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        f2 = fRD*0.5
        return ((1/M)*egrad(self.phi_int,0)(f2,M,symmratio,beta0,beta1,beta2,beta3)*symmratio -
            symmratio/M * egrad(self.phi_mr,0)(f2,chirpm,symmratio,0,0,alpha2,alpha3,alpha4,alpha5,fRD,fdamp))
            #(alpha2*(1/(M*f2)**2) + alpha3*(M*f2)**(-1/4) + alpha4*(1/(fdamp+(f2-alpha5*fRD)**2/(fdamp)))/M))

    def phase_cont_beta0(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1):
        M = self.assign_totalmass(chirpm,symmratio)
        f1 = 0.018/M
        pn_phase =[]
        for x in np.arange(len(self.pn_phase)):
            pn_phase.append(self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f1,x))
            # pn_phase.append(self.calculate_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f1,x))
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
        sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
        sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
        return (self.phi_ins(f1,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)*symmratio -
            ( beta1*f1*M + beta2*np.log(M*f1) - beta3/3 *(M*f1)**(-3)))

    def phase_cont_alpha0(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1):
        M = self.assign_totalmass(chirpm,symmratio)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        f2 = fRD*0.5
        return (self.phi_int(f2,M,symmratio,beta0,beta1,beta2,beta3) *symmratio -
            symmratio*self.phi_mr(f2,chirpm,symmratio,0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp))
         #(alpha1*f2*M -alpha2*(1/(f2*M)) + (4/3)*alpha3*(f2*M)**(3/4) + alpha4*np.arctan((f2-alpha5*fRD)/(fdamp))))
    ###########################################################################################################

    """Calculates the delta parameters involved in the intermediate amplitude function - see utilities file for functions"""
    def calculate_delta_parameter(self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i):
        rho0 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,0)
        rho1 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,1)
        rho2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,2)
        gamma1=self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,4)
        gamma2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5)
        gamma3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6)
        pn_amp =[]
        for x in np.arange(len(self.pn_amp)):
            pn_amp.append(self.assign_pn_amp(symmratio,massdelta,chi_a,chi_s,x))
        M = self.assign_totalmass(chirpm,symmratio)
        f1 = 0.014/M
        f2 = (f1+f3)/2

        ###########################################################################
        #Testing
        ###########################################################################
        #A2 = np.sqrt(2*symmratio/(3*np.pi**(1/3)))*(M*f2)**(-7/6)
        ###########################################################################
        v1 = self.amp_ins(f1,M,rho0,rho1,rho2,pn_amp)
        v2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,3)
        v3 = self.amp_mr(f3,gamma1,gamma2,gamma3,fRD,fdamp,M)
        dd1 = egrad(self.amp_ins,0)(f1,M,rho0,rho1,rho2,pn_amp)
        dd3 = egrad(self.amp_mr,0)(f3,gamma1,gamma2,gamma3,fRD,fdamp,M)

        if i == 0: return utilities.calculate_delta_parameter_0(f1,f2,f3,v1,v2,v3,dd1,dd3,M)

        elif i ==1: return utilities.calculate_delta_parameter_1(f1,f2,f3,v1,v2,v3,dd1,dd3,M)

        elif i ==2: return utilities.calculate_delta_parameter_2(f1,f2,f3,v1,v2,v3,dd1,dd3,M)

        elif i ==3: return utilities.calculate_delta_parameter_3(f1,f2,f3,v1,v2,v3,dd1,dd3,M)

        else: return utilities.calculate_delta_parameter_4(f1,f2,f3,v1,v2,v3,dd1,dd3,M)


    ###########################################################################################################
    """Caclulate the post merger paramters fRD and fdamp
    from the kerr parameter a = J/M**2 - the relationship between a and fRD,f_damp
    is numerical - interpolated data from http://www.phy.olemiss.edu/~berti/ringdown/ - 0905.2975
    a has a numerical fit from the khan phenomD paper and is a function of the
    symmetric mass ratio and the total initial spin
    the final parameters are then determined from omega = omegaM/(M - energry radiated)"""
    def calculate_postmerger_fRD(self,m1,m2,M,symmratio,chi_s,chi_a):
        chi1 = chi_s+chi_a
        chi2 = chi_s - chi_a
        S = (chi1*m1**2 + chi2*m2**2)/M**2 #convert chi to spin s in z direction
        S_red = S/(1-2*symmratio)

        a = S + 2*np.sqrt(3)*symmratio - 4.399*symmratio**2 + 9.397*symmratio**3 - \
        13.181*symmratio**4 +(-0.085*S +.102*S**2 -1.355*S**3 - 0.868*S**4)*symmratio + \
        (-5.837*S -2.097*S**2 +4.109*S**3 +2.064*S**4)*symmratio**2

        E_rad_ns = 0.0559745*symmratio +0.580951*symmratio**2 - \
        0.960673*symmratio**3 + 3.35241*symmratio**4

        E_rad = E_rad_ns*(1+S_red*(-0.00303023 - 2.00661*symmratio +7.70506*symmratio**2)) / \
        (1+ S_red*(-0.67144 - 1.47569*symmratio +7.30468*symmratio**2))

        #Calculate the post merger frequencies from numerical data
        MWRD = (1.5251-1.1568*(1-a)**0.1292)
        fRD = (1/(2*np.pi))*(MWRD)/(M*(1 - E_rad))
        return fRD

    def calculate_postmerger_fdamp(self,m1,m2,M,symmratio,chi_s,chi_a):
        chi1 = chi_s+chi_a
        chi2 = chi_s - chi_a
        S = (chi1*m1**2 + chi2*m2**2)/M**2 #convert chi to spin s in z direction
        S_red = S/(1-2*symmratio)

        a = S + 2*np.sqrt(3)*symmratio - 4.399*symmratio**2 + 9.397*symmratio**3 - \
        13.181*symmratio**4 +(-0.085*S +.102*S**2 -1.355*S**3 - 0.868*S**4)*symmratio + \
        (-5.837*S -2.097*S**2 +4.109*S**3 +2.064*S**4)*symmratio**2

        E_rad_ns = 0.0559745*symmratio +0.580951*symmratio**2 - \
        0.960673*symmratio**3 + 3.35241*symmratio**4

        E_rad = E_rad_ns*(1+S_red*(-0.00303023 - 2.00661*symmratio +7.70506*symmratio**2)) / \
        (1+ S_red*(-0.67144 - 1.47569*symmratio +7.30468*symmratio**2))

        #Calculate the post merger frequencies from numerical data
        MWdamp = ((1.5251-1.1568*(1-a)**0.1292)/(2*(0.700 + 1.4187*(1-a)**(-.4990))))
        fdamp = (1/(2*np.pi))*(MWdamp)/(M*(1 - E_rad))
        return fdamp
    ###########################################################################################################

    """Caluculates the parameters from the Lambda matrix defined above.
    Indices are as follows: parameters[i] for i element of {0,19}:
    order of parameters: rho{1,2,3},v2,gamma{1,2,3},sigma{1,2,3,4},beta{1,2,3},alpha{1,2,3,4,5}"""
    def calculate_parameter(self,chirpm,symmratio,chi_a,chi_s,i):
        m1=self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        M = self.assign_totalmass(chirpm,symmratio)
        chi1 = chi_a+chi_s
        chi2 = chi_s-chi_a
        chi_eff = (m1*(chi1)+ m2*(chi2))/M
        chi_pn = chi_eff - (38*symmratio/113)*(2*chi_s)
        self.chi_pn = chi_pn
        param_list = Lambda[i]
        spin_coeff = chi_pn - 1

        parameter = param_list[0] + param_list[1]*symmratio + \
            (spin_coeff)*(param_list[2] + param_list[3]*symmratio + param_list[4]*symmratio**2) + \
            (spin_coeff)**2*(param_list[5] + param_list[6]*symmratio+param_list[7]*symmratio**2) + \
            (spin_coeff)**3*(param_list[8] + param_list[9]*symmratio+param_list[10]*symmratio**2)

        return parameter

    """Post Newtonian approximation for the inspiral amplitude for the ith (\el {0,1,2,3,4,5,6}) parameter"""
    def calculate_pn_amp(self,symmratio,massdelta,chi_a,chi_s,i):
        if i == 0: return 1.
        elif i ==1: return 0.
        elif i == 2: return -323/224 + 451*symmratio/168
        elif i ==3: return 27*massdelta*chi_a/8 + (27/8 - 11*symmratio/6)*chi_s
        elif i == 4: return -27312085/8128512 - 1975055*symmratio/338688 + \
        105271*symmratio**2/24192 + (-81/32 + 8*symmratio)*chi_a**2 - \
        81*massdelta*chi_a*chi_s/16 + (-81/32 + 17*symmratio/8)*chi_s**2

        elif i == 5: return -85*np.pi/64 + 85*np.pi*symmratio/16 + massdelta*(285197/16128 - \
        1579*symmratio/4032)*chi_a + (285197/16128 - 15317*symmratio/672 - \
        2227*symmratio**2/1008)*chi_s

        else: return -177520268561/8583708672 + (545384828789/5007163392 - 205*np.pi**2/48)*symmratio - \
        3248849057*symmratio**2/178827264 + 34473079*symmratio**3/6386688 + \
        (1614569/64512 - 1873643*symmratio/16128 + 2167*symmratio**2/42)*chi_a**2 + \
        (31*np.pi/12 - 7*np.pi*symmratio/3)*chi_s + (1614569/64512 - 61391*symmratio/1344 + \
        57451*symmratio**2/4032)*chi_s**2 + \
        massdelta*chi_a*(31*np.pi/12 + (1614569/32256 - 165961*symmratio/2688)*chi_s)


    """calculates the 8 coefficients for PN expansion for the phase as a function of freq. f for the ith (\el {0,1,2,3,4,5,6,7,8}) parameter"""
    def calculate_pn_phase(self, chirpm,symmratio,delta,chi_a,chi_s,f,i):
        """5 and 6 depend on the given freq."""
        M = self.assign_totalmass(chirpm,symmratio)
        if i == 0:return 1.
        elif i == 1: return 0.
        elif i == 2: return 3715/756 + 55*symmratio/9
        elif i == 3: return -16*np.pi + 113*delta*chi_a/3 + \
        (113/3 - 76*symmratio/3)*chi_s

        elif i ==4: return 15293365/508032 + 27145*symmratio/504 + 3085*symmratio**2/72 + \
        (-405/8 + 200*symmratio)*chi_a**2 - \
        (405/4)*delta*chi_a*chi_s +\
        (-405/8 + 5*symmratio/2)*chi_s**2

        elif i == 7: return 77096675*np.pi/254016 + 378515*np.pi*symmratio/1512 - \
        74045*np.pi*symmratio**2/756 + delta*(-25150083775/3048192 + \
        26804935*symmratio/6048 - 1985*symmratio**2/48)*chi_a + \
        (-25150083775/3048192 + 10566655595*symmratio/762048 - \
        1042165*symmratio**2/3024 + 5345*symmratio**3/36)*chi_s

        elif i ==5: return (1 + np.log(np.pi*M*f))* (38645*np.pi/756 - 65*np.pi*symmratio/9 + \
        delta*(-732985/2268 - 140*symmratio/9)*chi_a + \
        (-732985/2268 + 24260*symmratio/81 + 340*symmratio**2/9)*chi_s)

        else: return  11583231236531/4694215680 - 6848*gamma_E/21 -\
         640*np.pi**2/3 + (-15737765635/3048192 + 2255*np.pi**2/12)*symmratio + \
         76055*symmratio**2/1728 - 127825*symmratio**3/1296 \
          + 2270*delta*chi_a*np.pi/3 + \
         (2270*np.pi/3 - 520*np.pi*symmratio)*chi_s - 6848*np.log(64*np.pi*M*f)/63

    """Fundamental Waveform Functions"""
    ###########################################################################################################
    """Calculates the phase of the inspiral range of the GW as a function of freq. f"""
    def phi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase):
        """calculate pn phase - Updates the PN coeff. for a given freq."""
        M = self.assign_totalmass(chirpm,symmratio)
        temp5 = self.calculate_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f,5)
        temp6 = self.calculate_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f,6)
        """autograd doesn't handle array assignment very well - need to re-instantiate array"""
        phasepn = [pn_phase[0],pn_phase[1],pn_phase[2],pn_phase[3],pn_phase[4],temp5,temp6,pn_phase[7]]
        pnsum = 0
        for i in np.arange(len(self.pn_phase)):
            pnsum += phasepn[i]* (np.pi * M*f )**(i/3)
        phiTF2 = 2*np.pi * f * tc - phic -np.pi/4+ \
        3/(128*symmratio)*(np.pi *M* f )**(-5/3)*pnsum

        """Calculates the full freq. with pn terms and and NR terms
        - sigma0  and sigma1 are an overall phase factor and derivative factor and are arbitrary (absorbed into phic and tc)"""
        sigma0 = 0
        sigma1 =0
        return phiTF2 + (1/symmratio)*(sigma0 + sigma1*M*f + \
        (3/4)*sigma2*(M*f)**(4/3) + (3/5)*sigma3*(M*f)**(5/3) + \
        (1/2)*sigma4*(M*f)**(2))
    """Amplitude of inspiral"""
    def amp_ins(self,f,M,rho0,rho1,rho2,pn_amp):
        """Calculate PN Amplitude"""
        amp_pn = 0
        for i in np.arange(len(pn_amp)):
            amp_pn = amp_pn + pn_amp[i]*(np.pi*M*f)**(i/3)

        """NR corrections:
        exponential is (7+i) instead of (6+i) because loop var. {0,1,2}"""
        parameters = [rho0,rho1,rho2]
        amp_nr = 0
        for i in np.arange(3):
            amp_nr = amp_nr + parameters[i] * (M*f)**((7+i)/3)

        """full amplitude, including NR parameters:"""
        return (amp_pn + amp_nr)

    """Frequency of intermediate stage"""
    def phi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3):
        return (1/symmratio)*(beta0+ beta1*(M*f) + beta2*np.log(M*f) - beta3/3 *(M*f)**(-3))
    """Amplitude of intermediate stage"""
    def amp_int(self,f,deltas,M):
        return (deltas[0]+ deltas[1]*M*f +deltas[2]*(M*f)**2 + \
        deltas[3]*(M*f)**3 + deltas[4]*(M*f)**4)#*self.A0(f)

    """Phase for Merger-Ringdown"""
    def phi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp):
        M = self.assign_totalmass(chirpm,symmratio)
        return (1/symmratio)*(alpha0 +alpha1*(M*f) -alpha2*(1/(M*f)) + (4/3)*alpha3*(M*f)**(3/4) + \
        alpha4*np.arctan((f-alpha5*fRD)/fdamp))#self.private_arctan(f,alpha5,fRD,fdamp))
    """Amplitude of Merger-Ringdown"""
    def amp_mr(self,f,gamma1,gamma2,gamma3,fRD,fdamp,M):
        numerator = (gamma1*gamma3*fdamp * M)*np.exp((-gamma2)*(f - fRD)/(gamma3*fdamp))
        denominator = (M**2*(f-fRD)**2 + M**2*(gamma3*fdamp)**2)
        return numerator/denominator
    ###########################################################################################################
    """Vectorized Waveform Functions"""
    """Break up full frequency range into respective regions and call regions separately -
    Removes the need for if statements, which would cause issues with vectorization -
    Standard techniques for coping with if statements (ie np.where etc) would not work with autograd"""
    def amp_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
        fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
        pn_amp = []
        for i in np.arange(len(self.pn_amp)):
            pn_amp.append(self.assign_pn_amp(symmratio,delta,chi_a,chi_s,i))
        rho0 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,0)
        rho1 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,1)
        rho2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,2)
        return np.sqrt(self.N_detectors)*self.amp_ins(f,M,rho0,rho1,rho2,pn_amp)*A0 *f**(-7/6)

    def amp_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
        fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
        deltas = []
        for i in np.arange(len(self.param_deltas)):
            deltas.append(self.assign_param_deltas(chirpm,symmratio,delta,chi_a,chi_s,fRD,fdamp,fpeak,i))
        return np.sqrt(self.N_detectors)*self.amp_int(f,deltas,M)*A0 *f**(-7/6)

    def amp_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
        fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
        gamma1=self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,4)
        gamma2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5)
        gamma3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6)
        return  np.sqrt(self.N_detectors)*self.amp_mr(f,gamma1,gamma2,gamma3,fRD,fdamp,M)*A0 *f**(-7/6)

    def phase_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
        sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
        sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
        pn_phase= []
        for i in [0,1,2,3,4,5,6,7]:
            pn_phase.append( self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f,i))
        return self.phi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)

    def phase_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s)
        beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        return self.phi_int(f,M,symmratio,beta0,beta1,beta2,beta3)

    def phase_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
        fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
        beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s)
        beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1)
        alpha1 = self.assign_alpha1(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1)
        alpha0 = self.assign_alpha0(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        return self.phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp)
    ###########################################################################################################
    """Split frequencies into ranges - inspiral, intermediate, Merger-Ringdown"""
    def split_freqs_amp(self,freqs):
        freqins = freqs[(freqs<=0.014/self.M)]
        # freqint = freqs[(freqs>0.014/self.M) & (freqs<self.fpeak)]
        freqint = np.asarray([x for x in freqs if x> 0.014/self.M and x <= self.fpeak])
        # freqmr = freqs[(freqs>self.fpeak)]
        freqmr = np.asarray([x for x in freqs if x > self.fpeak])
        return [freqins,freqint,freqmr]
    def split_freqs_phase(self,freqs):
        freqins = freqs[(freqs<=0.018/self.M)]
        freqint = freqs[(freqs>0.018/self.M) & (freqs<=self.fRD*0.5)]
        freqmr = freqs[(freqs>self.fRD*0.5)]
        return [freqins,freqint,freqmr]

    """Calculate the waveform - vectorized
    Outputs: amp vector, phase vector, (real) waveform vector"""
    def calculate_waveform_vector(self,freq):
        """Array of the functions used to populate derivative vectors"""
        ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
        phasefunc = [self.phase_ins_vector,self.phase_int_vector,self.phase_mr_vector]
        """Check to see if every region is sampled - if integration frequency
        doesn't reach a region, the loop is trimmed to avoid issues with unpopulated arrays"""
        famp = self.split_freqs_amp(freq)
        fphase = self.split_freqs_phase(freq)
        jamp=[0,1,2]
        jphase=[0,1,2]
        # if len(famp[0]) == 0:
        #     if len(famp[1])==0:
        #         jamp = [2]
        #     else:
        #         if len(famp[2])==0:
        #             jamp = [1]
        #         else:
        #             jamp = [1,2]
        # if len(famp[2])==0:
        #     if len(famp[1]) == 0:
        #         jamp = [0]
        #     else:
        #         if len(famp[0])==0:
        #             jamp = [1]
        #         else:
        #             jamp = [0,1]
        #######################################
        # if len(famp[0]) ==0 and len(famp[1]) and len(famp[2])==0:
        #     print( "error")
        # elif len(famp[0]) ==0 and len(famp[1])==0:
        #     jamp = [2]
        # elif len(famp[1])==0 and len(famp[2])==0:
        #     jamp = [0]
        # elif len(famp[0])==0:
        #     jamp = [1,2]
        # elif len(famp[1])==0:
        #     jamp = [2]
        # elif len(famp[0])==0 and len(famp[2])==0:
        #     jamp = [1]
        #################################################
        # if len(fphase[0]) == 0:
        #     if len(fphase[1])==0:
        #         jphase = [2]
        #     else:
        #         if len(fphase[2])==0:
        #             jphase = [1]
        #         else:
        #             jphase = [1,2]
        # if len(fphase[2])==0:
        #     if len(fphase[1]) == 0:
        #         jphase = [0]
        #     else:
        #         if len(fphase[0])==0:
        #             jphase = [1]
        #         else:
        #             jphase = [0,1]
        jamp = [0,1,2]
        for i in np.arange(len(famp)):
            if len(famp[i]) == 0:
                jamp[i] = -1
        jamp = [x for x in jamp if x != -1]
        jphase = [0,1,2]
        for i in np.arange(len(fphase)):
            if len(fphase[i]) == 0:
                jphase[i] = -1
        jphase = [x for x in jphase if x != -1]

        var_arr= self.var_arr[:]
        amp = [[],[],[]]
        phase = [[],[],[]]

        """Populate derivative vectors one region at a time"""
        for j in jamp:
            amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6])
        for j in jphase:
            phase[j]=phasefunc[j](fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6])

        """Concatenate the regions into one array"""
        ampout,phaseout =[],[]
        for j in jamp:
            ampout = np.concatenate((ampout,amp[j]))
        for j in jphase:
            phaseout = np.concatenate((phaseout,phase[j]))

        """Return the amplitude vector, phase vector, and real part of the waveform"""
        return ampout,phaseout, np.multiply(ampout,np.cos(phaseout))

    ###########################################################################################################
    """Stitch the amplitude together based on the critical frequencies - LOOP VERSION - Much slower than vectorized version"""
    def full_amp(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a):
        """Assign instance objects to local variables - to be passed to functions below
        improves ability to convert to c in cython and is needed for fisher calculation"""

        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
        fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
        freq = f*M

        if freq < 0.014:
            pn_amp = []
            for i in np.arange(len(self.pn_amp)):
                pn_amp.append(self.assign_pn_amp(symmratio,delta,chi_a,chi_s,i))
            rho0 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,0)
            rho1 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,1)
            rho2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,2)

            return np.sqrt(self.N_detectors)*self.amp_ins(f,M,rho0,rho1,rho2,pn_amp)*A0 *f**(-7/6)
        elif freq < fpeak*M:
            deltas = []
            for i in np.arange(len(self.param_deltas)):
                deltas.append(self.assign_param_deltas(chirpm,symmratio,delta,chi_a,chi_s,fRD,fdamp,fpeak,i))

            return np.sqrt(self.N_detectors)*self.amp_int(f,deltas,M)*A0 *f**(-7/6)
        else:
            gamma1=self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,4)
            gamma2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5)
            gamma3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6)
            return np.sqrt(self.N_detectors)*self.amp_mr(f,gamma1,gamma2,gamma3,fRD,fdamp,M)*A0 *f**(-7/6)

    """Stitch the phase together based on the critical frequencies - LOOP VERSION - much slower than vectorized version"""
    def full_phi(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)

        if f < 0.018/M:
            sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
            sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
            sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
            pn_phase= []
            for i in [0,1,2,3,4,5,6,7]:
                pn_phase.append( self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f,i))
            return self.phi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)
        elif f <  0.5 * fRD:
            beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s)
            beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1)
            beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
            beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
            return self.phi_int(f,M,symmratio,beta0,beta1,beta2,beta3)
        else:
            fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
            fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
            beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s)
            beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1)
            alpha1 = self.assign_alpha1(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1)
            alpha0 = self.assign_alpha0(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1)
            alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
            alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
            alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
            alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
            return self.phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp)

    """Calculate the full amplitude and phase for frequencies f (an array) - implements multiprocessing
    for space
    returns amp, phi, and full (real) waveform h (Amp*cos(phi)) -- LOOP VERSION - Slower than vectorized"""
    def full_waveform(self,f):
        pool = mp.Pool(processes=mp.cpu_count())
        amp_reduced = partial(self.full_amp,A0=self.A0,phic=self.phic,tc=self.tc,chirpm=self.chirpm,symmratio=self.symmratio,chi_s=self.chi_s,chi_a=self.chi_a)
        phase_reduced = partial(self.full_phi,A0=self.A0,phic=self.phic,tc=self.tc,chirpm=self.chirpm,symmratio=self.symmratio,chi_s=self.chi_s,chi_a=self.chi_a)
        amp = pool.map(amp_reduced,f)
        phi = pool.map(phase_reduced,f)
        return amp, phi,amp*np.cos(phi)
    ###########################################################################################################

    """Fisher Calculation functions - derivatives"""
    def calculate_derivatives(self):
        """Pre-calculate Derivatives here - (Order does matter - parameter functions may be functions of system constants)
        If the variable is instantiated as an array, a derivate array for each system variable is created and is cycled through
        (ie Lambda paramaters is parameters[i] and has derivate arrays parameters_deriv_symmratio etc).
        If the variable is a single value, the variable has one array of derivates, the elements of which are the derivatives wrt
        various system variables (ie M -> M_deriv[i] for symmratio and chripm etc)
        -M
        -m1
        -m2
        -Lambda parameters
        -pn_amp
        -pn_phase
        -delta parameters (intermediate amplitude parameters)
        -fRD
        -fdamp
        -fpeak
        -delta (mass parameter)
        -phase continuitiy variables (beta1,beta0,alpha1,alpha0)
        """

        self.total_mass_deriv = []
        for i in range(2):
            self.total_mass_deriv.append(grad(utilities.calculate_totalmass,i)(self.chirpm,self.symmratio))
        self.mass1_deriv = []
        for i in range(2):
            self.mass1_deriv.append(grad(self.calculate_mass1,i)(self.chirpm,self.symmratio))
        self.mass2_deriv = []
        for i in range(2):
            self.mass2_deriv.append(grad(self.calculate_mass2,i)(self.chirpm,self.symmratio))
        self.lambda_derivs_symmratio=[]
        self.lambda_derivs_chirpm = []
        self.lambda_derivs_chi_a = []
        self.lambda_derivs_chi_s = []
        for i in np.arange(len(Lambda)):
            self.lambda_derivs_chirpm.append(grad(self.calculate_parameter,0)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))
            self.lambda_derivs_symmratio.append(grad(self.calculate_parameter,1)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))
            self.lambda_derivs_chi_a.append(grad(self.calculate_parameter,2)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))
            self.lambda_derivs_chi_s.append(grad(self.calculate_parameter,3)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))

        self.pn_amp_deriv_symmratio = []
        self.pn_amp_deriv_delta = []
        self.pn_amp_deriv_chi_a = []
        self.pn_amp_deriv_chi_s = []
        for i in np.arange(len(self.pn_amp)):
            self.pn_amp_deriv_symmratio.append(grad(self.calculate_pn_amp,0)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_delta.append(grad(self.calculate_pn_amp,1)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_chi_a.append(grad(self.calculate_pn_amp,2)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_chi_s.append(grad(self.calculate_pn_amp,3)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
        self.pn_phase_deriv_chirpm = []
        self.pn_phase_deriv_symmratio = []
        self.pn_phase_deriv_delta = []
        self.pn_phase_deriv_chi_a = []
        self.pn_phase_deriv_chi_s = []
        for i in np.arange(len(self.pn_phase)):
            self.pn_phase_deriv_chirpm.append(grad(self.calculate_pn_phase,0)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_symmratio.append(grad(self.calculate_pn_phase,1)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_delta.append(grad(self.calculate_pn_phase,2)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_chi_a.append(grad(self.calculate_pn_phase,3)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_chi_s.append(grad(self.calculate_pn_phase,4)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
        """Delta Parameters take up ~50 percent of the total time"""
        start=time()
        # self.param_deltas_derivs_chirpm = []
        # self.param_deltas_derivs_symmratio = []
        # self.param_deltas_derivs_delta = []
        # self.param_deltas_derivs_chi_a = []
        # self.param_deltas_derivs_chi_s = []
        # self.param_deltas_derivs_fRD = []
        # self.param_deltas_derivs_fdamp = []
        # self.param_deltas_derivs_f3 = []
        # for i in np.arange(len(self.param_deltas)):
        #     self.param_deltas_derivs_chirpm.append(grad(self.calculate_delta_parameter,0)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i))
        #     self.param_deltas_derivs_symmratio.append(grad(self.calculate_delta_parameter,1)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i))
        #     self.param_deltas_derivs_delta.append(grad(self.calculate_delta_parameter,2)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i))
        #     self.param_deltas_derivs_chi_a.append(grad(self.calculate_delta_parameter,3)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i))
        #     self.param_deltas_derivs_chi_s.append(grad(self.calculate_delta_parameter,4)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i))
        #     self.param_deltas_derivs_fRD.append(grad(self.calculate_delta_parameter,5)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i))
        #     self.param_deltas_derivs_fdamp.append(grad(self.calculate_delta_parameter,6)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i))
        #     self.param_deltas_derivs_f3.append(grad(self.calculate_delta_parameter,7)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i))

        temp1 = grad(self.calculate_delta_parameter,0)
        temp2 = grad(self.calculate_delta_parameter,1)
        temp3 = grad(self.calculate_delta_parameter,2)
        temp4 = grad(self.calculate_delta_parameter,3)
        temp5 = grad(self.calculate_delta_parameter,4)
        temp6 = grad(self.calculate_delta_parameter,5)
        temp7 = grad(self.calculate_delta_parameter,6)
        temp8 = grad(self.calculate_delta_parameter,7)
        self.param_deltas_derivs_chirpm = list(map(lambda i:temp1(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        self.param_deltas_derivs_symmratio = list(map(lambda i:temp2(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        self.param_deltas_derivs_delta = list(map(lambda i:temp3(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        self.param_deltas_derivs_chi_a = list(map(lambda i:temp4(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        self.param_deltas_derivs_chi_s = list(map(lambda i:temp5(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        self.param_deltas_derivs_fRD = list(map(lambda i:temp6(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        self.param_deltas_derivs_fdamp = list(map(lambda i:temp7(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        self.param_deltas_derivs_f3 = list(map(lambda i:temp8(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))

        # vdelta = np.vectorize(self.calculate_delta_parameter)
        # temp1 = egrad(vdelta,0)
        # temp2 = egrad(vdelta,1)
        # temp3 = grad(vdelta,2)
        # temp4 = grad(vdelta,3)
        # temp5 = grad(vdelta,4)
        # temp6 = grad(vdelta,5)
        # temp7 = grad(vdelta,6)
        # temp8 = grad(vdelta,7)
        # print(temp1(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,1))
        # num_deltas = np.arange(5)
        # num_deltas, varrchirpm = np.broadcast_arrays(num_deltas,self.chirpm)
        # self.param_deltas_derivs_chirpm = temp1(varrchirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak, np.arange(5) )
        # num_deltas, varrsymmratio = np.broadcast_arrays(num_deltas,self.symmratio)
        # self.param_deltas_derivs_symmratio = temp2(self.chirpm,varrsymmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak, np.arange(5) )
        # print(self.param_deltas_derivs_symmratio)
        # self.param_deltas_derivs_delta = list(map(lambda i:temp3(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        # self.param_deltas_derivs_chi_a = list(map(lambda i:temp4(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        # self.param_deltas_derivs_chi_s = list(map(lambda i:temp5(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        # self.param_deltas_derivs_fRD = list(map(lambda i:temp6(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        # self.param_deltas_derivs_fdamp = list(map(lambda i:temp7(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        # self.param_deltas_derivs_f3 = list(map(lambda i:temp8(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
        # print("deltas: {}".format(time()-start))

        self.fRD_deriv = []
        for i in range(6):
            self.fRD_deriv.append(grad(self.calculate_postmerger_fRD,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
        self.fdamp_deriv = []
        for i in range(6):
            self.fdamp_deriv.append(grad(self.calculate_postmerger_fdamp,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
        self.fpeak_deriv = []
        for i in range(5):
            self.fpeak_deriv.append(grad(self.calculate_fpeak,i)(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6]))
        self.delta_deriv = grad(self.calculate_delta)(self.symmratio)
        self.beta1_deriv = []
        self.beta0_deriv = []
        self.alpha1_deriv = []
        self.alpha0_deriv = []
        for i in range(7):
            self.beta1_deriv.append(grad(self.phase_cont_beta1,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s))
        for i in range(8):
            self.beta0_deriv.append(grad(self.phase_cont_beta0,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1))
        for i in range(8):
            self.alpha1_deriv.append(grad(self.phase_cont_alpha1,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1))
        for i in range(9):
            self.alpha0_deriv.append(grad(self.phase_cont_alpha0,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1))

        """Populate array with variables for transformation from d/d(theta) to d/d(log(theta)) - begins with 0 because fisher matrix variables start at 1, not 0"""
        self.log_factors = [0,self.A0,1,1,self.chirpm,self.symmratio,1,1]

    """Function for actual element integrand - 4*Re(dh/dtheta_i* dh/dtheta_j)"""
    def calculate_element_integrand(self,f,i,j):
        elem1 = self.log_factors[i]*(grad(self.full_amp,i)(f,self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a)-\
            self.full_amp(f,self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a)* grad(self.full_phi,i)(f,self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a)*1j)
        elem2 = self.log_factors[j]*(grad(self.full_amp,j)(f,self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a)-\
            self.full_amp(f,self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a)* grad(self.full_phi,j)(f,self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a)*1j)
        prod = (elem1*np.conj(elem2)).real
        return 4*prod/self.noise_func(f)**2

    """Function to calculate array of waveform derivatives - LOOP VERSION - much slower"""
    def calculate_waveform_derivative(self,f,i):
        return grad(self.full_amp,i)(f,self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a)-\
            self.full_amp(f,self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a)* grad(self.full_phi,i)(f,self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a)*1j

    # """Analytic noise curve from Will '97, for testing - aLIGOAnalytic"""
    # def sym_noise_curve(self,f):
    #     S = 3e-48
    #     fknee = 70.
    #     return np.sqrt(S * ((fknee/f)**4 + 2 + 2*(f/fknee)**2)/5)
    # """Returns a fitted S**(1/2) curve of Hanford O(1) noise data - see 1603.08955 appendix C"""
    # def fitted_hanford_noise(self, f):
    #     a_vec = np.array([47.8466,-92.1896,35.9273,-7.61447,0.916742,-0.0588089,0.00156345],dtype=float)
    #     S0 = .8464
    #     x = np.log(f)
    #     return (np.sqrt(S0) * np.exp(a_vec[0] + a_vec[1]*x + a_vec[2]*x**2 +
    #             a_vec[3] * x**3 + a_vec[4]*x**4 + a_vec[5]*x**5 + a_vec[6]*x**6))
    def calculate_upper_freq(self,freq,detector):
        """Finds indicies of frequency that are elements of [800,10000] and trims freq and noise to match"""
        if detector != 'LISA':
            indicies = np.asarray(np.where(np.asarray(freq)>=800)[0],dtype=int)
            trimmed_freq = np.asarray(freq)[indicies]
            trimmed_noise = np.asarray(self.noise_curve)[indicies]
            """Calculate integrand quantity"""
            """Return first integrand item that is less than .1"""
            ratio_table = np.abs(np.divide(np.multiply(np.sqrt(trimmed_freq),self.calculate_waveform_vector(trimmed_freq)[0]),trimmed_noise))
            fup = trimmed_freq[np.where(ratio_table<.1)[0][0]]+100

        else:
            """I'm just going to use the entire array -
            I doubt these operations will take very long, even with ~5000 items"""
            trimmed_freq = np.asarray(freq[int(len(freq)/2):])
            trimmed_noise = np.asarray(self.noise_curve[int(len(freq)/2):])
            """Calculate integrand quantity"""
            """Return first integrand item that is less than .1"""
            ratio_table = np.abs(np.divide(np.multiply(np.sqrt(trimmed_freq),self.calculate_waveform_vector(trimmed_freq)[0]),trimmed_noise))
            fup = trimmed_freq[np.where(ratio_table<.1)[0][0]]

        ####################################################################
        #Testing
        ####################################################################
        #print(trimmed_freq)
        # plt.plot(np.log(trimmed_freq),np.log(np.divide(np.multiply(np.sqrt(trimmed_freq),self.calculate_waveform_vector(trimmed_freq)[0]),trimmed_noise)))
        # plt.show()
        # plt.close()
        ####################################################################
        # waveform = self.full_waveform(trimmed_freq)[0]
        if self.NSflag:
            Rcontact = 24./3e5
            fcontact = (1/np.pi)*(np.sqrt(self.totalMass_restframe/Rcontact**3))
            return np.amin(np.array([fcontact,fup]))
        else:
            return fup

    def calculate_lower_freq(self,freq,detector):
        """Trim lists s.t. frequencies are elements of [1,8]"""
        if detector != 'LISA':
            indicies = np.asarray(np.where(np.asarray(freq)<=8)[0],dtype=int)
            trimmed_freq = np.asarray(freq)[indicies]
            trimmed_noise = np.asarray(self.noise_curve)[indicies]
        else:
            """I'm just going to use the entire array -
            I doubt these operations will take very long, even with ~5000 items"""
            trimmed_freq = np.asarray(freq[:int(len(freq)/10)])
            trimmed_noise = np.asarray(self.noise_curve[:int(len(freq)/10)])
        """Calculate integrands"""
        ratio_table =  np.abs(np.divide(np.multiply(np.sqrt(trimmed_freq),self.calculate_waveform_vector(trimmed_freq)[0]),trimmed_noise))
        """Finds indexes where the integrand is <.1 - returns 1 if all elements are >.1"""
        index_list = np.where(ratio_table<0.1)[0]
        # print(index_list)
        # print(ratio_table[index_list])
        if len(index_list) == 0:
            if detector != 'LISA':
                return 1
            else:
                return trimmed_freq[0]
        else:
            index = index_list[-1]
        if detector != 'LISA':
            return 1+0.1*trimmed_freq[index]
        else:
            return 0.1*trimmed_freq[index]
    """Calcualtes the Fisher and the Inverse Fisher
    args: detector = 'aLIGO', 'aLIGOAnalytic' int_scheme = 'simps','trapz','quad', stepsize= float
    options aLIGOAnalytic and stepsize are purely for testing. The real data has these set.
    int_scheme should be left with simps - orders of magnitude faster than quad, and interpolating the noise data
    makes it debatable whether the accuracy is better than simps
    LOOP VERSION - this is much slower than the vectorized version"""
    def calculate_fisher_matrix(self,detector,int_scheme = 'simps',stepsize=None):
        if int_scheme == 'simps':
            int_func = integrate.simps
        elif int_scheme == 'trapz':
            int_func = integrate.trapz
        else:
            int_func = integrate.quad

        names = [ 'aLIGO', 'A+', 'A++', 'Vrt', 'Voyager', 'CE1', 'CE2 wide', 'CE2 narrow', 'ET-B', 'ET-D']
        freq = noise[0]
        if detector in names:
            self.noise_curve = noise[names.index(detector)+1]
            if int_scheme == 'quad':
                self.noise_func = CubicSpline(noise[0],self.noise_curve)
        # if detector == 'aLIGO':
        #     self.noise_curve = noise[1]
        #     if int_scheme == 'quad':
        #         self.noise_func = CubicSpline(noise[0],self.noise_curve)
        elif detector == 'LISA':
            self.noise_curve = noise_lisa[1]
            freq = noise_lisa[0]
            if int_scheme == 'quad':
                self.noise_func = CubicSpline(noise_lisa[0],self.noise_curve)
        elif detector == 'aLIGOAnalytic':
            """Purely for testing"""
            if stepsize != None:
                freq = np.arange(1,10000,stepsize)
            self.noise_curve = sym_noise_curve(np.asarray(freq))
            self.noise_func = sym_noise_curve
        elif detector == 'aLIGOFitted':
            if stepsize != None:
                freq = np.arange(1,10000,stepsize)
            self.noise_curve = fitted_hanford_noise(np.asarray(freq))
            self.noise_func = fitted_hanford_noise
        elif detector == 'DECIGO':
            if stepsize != None:
                freq = np.arange(1e-3,100,stepsize)
            else:
                freq = np.arange(1e-3,100,.1)
            self.noise_curve = decigo_noise(np.asarray(freq))
            self.noise_func = decigo_noise
        else:
            print('DETECTOR ISSUE - check to make sure the detector name is spelled exactly as in {},{},{},{}'.format(names,'aLIGOAnalytic','aLIGOFitted','DECIGO'))
            return 0,0,0

        self.lower_freq =self.calculate_lower_freq(freq,detector=detector)#self.flower
        self.upper_freq =self.calculate_upper_freq(freq,detector=detector)#self.fupper
        print("lower: {}".format(self.lower_freq),"upper: {}".format(self.upper_freq))
        """Pre-populate Derivative arrays for faster evaluation"""
        self.calculate_derivatives()

        variable_indicies = range(1,8)
        fisher = np.zeros((len(variable_indicies),len(variable_indicies)))
        relerror = np.zeros((len(variable_indicies),len(variable_indicies)))

        ##############################################################
        #Quad method with function
        if int_scheme == 'quad':
            for i in variable_indicies:
                for j in range(1,i+1):
                    if i == j:
                        el, err = int_func(self.calculate_element_integrand,self.lower_freq,self.upper_freq,args=(i,j),limit=1000,epsabs=1e-50,epsrel=1e-15)
                        fisher[i-1][j-1] = (1/2)*el
                        relerror[i-1][j-1] = (1/2)*err/el
                    else:
                        fisher[i-1][j-1], err = int_func(self.calculate_element_integrand,self.lower_freq,self.upper_freq,args=(i,j),limit=1000,epsabs=1e-50,epsrel=1e-15)
                        relerror[i-1][j-1] = err/el

        ##############################################################
        #Discrete methods
        else:
            """Trim frequency and noise curve down to [flower,fupper]"""
            ftemp = freq[0]
            i = 0
            while ftemp <self.lower_freq:
                i +=1
                ftemp = freq[i]
            flow_pos = i

            ftemp = freq[len(freq)-1]
            i = len(freq)-1
            while ftemp > self.upper_freq:
                i-= 1
                ftemp = freq[i]
            fup_pos = i

            int_freq = freq[flow_pos:fup_pos]
            noise_integrand = self.noise_curve[flow_pos:fup_pos]

            waveform_derivs = []
            for i in variable_indicies:
                waveform_derivs.append([self.log_factors[i]*self.calculate_waveform_derivative(f,i) for f in int_freq])
            for i in variable_indicies:
                for j in range(1,i+1):
                    integrand = [4*(waveform_derivs[i-1][f]*np.conj(waveform_derivs[j-1][f])).real/noise_integrand[f]**2 for f in np.arange(len(int_freq))]
                    if i == j:
                        fisher[i-1][j-1] = (1/2)*int_func(integrand,int_freq)
                    else:
                        fisher[i-1][j-1] = int_func(integrand,int_freq)
        ###############################################


        fisher = fisher + np.transpose(fisher)
        try:
            chol_fisher = np.linalg.cholesky(fisher)
            inv_chol_fisher = np.linalg.inv(chol_fisher)
            inv_fisher = np.dot(inv_chol_fisher.T,inv_chol_fisher)
            cholo = True
        except:
            inv_fisher = np.linalg.inv(fisher)
            cholo = False
        self.fisher = fisher
        self.inv_fisher = inv_fisher
        return fisher,inv_fisher,cholo

    """Function for actual element integrand - 4*Re(dh/dtheta_i* dh/dtheta_j) - Vectorized"""
    def calculate_waveform_derivative_vector(self,famp,fphase,i):
        """Array of the functions used to populate derivative vectors"""
        ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
        phasefunc = [self.phase_ins_vector,self.phase_int_vector,self.phase_mr_vector]
        """Check to see if every region is sampled - if integration frequency
        doesn't reach a region, the loop is trimmed to avoid issues with unpopulated arrays"""
        jamp=[0,1,2]
        jphase=[0,1,2]
        if len(famp[0]) == 0:
            if len(famp[1])==0:
                jamp = [2]
            else:
                if len(famp[2])==0:
                    jamp = [1]
                else:
                    jamp = [1,2]
        if len(famp[2])==0:
            if len(famp[1]) == 0:
                jamp = [0]
            else:
                if len(famp[0])==0:
                    jamp = [1]
                else:
                    jamp = [0,1]

        if len(fphase[0]) == 0:
            if len(fphase[1])==0:
                jphase = [2]
            else:
                if len(fphase[2])==0:
                    jphase = [1]
                else:
                    jphase = [1,2]
        if len(fphase[2])==0:
            if len(fphase[1]) == 0:
                jphase = [0]
            else:
                if len(fphase[0])==0:
                    jphase = [1]
                else:
                    jphase = [0,1]


        # jamp = [0,1,2]
        # for i in np.arange(len(famp)):
        #     if len(famp[i]) == 0:
        #         jamp[i] = -1
        # jamp = [x for x in jamp if x != -1]
        # jphase = [0,1,2]
        # for i in np.arange(len(fphase)):
        #     if len(fphase[i]) == 0:
        #         jphase[i] = -1
        # jphase = [x for x in jphase if x != -1]

        var_arr= self.var_arr[:]
        gamp = [[],[],[]]
        amp = [[],[],[]]
        phase = [[],[],[]]
        """Array of the functions used to populate derivative vectors"""
        ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
        phasefunc = [self.phase_ins_vector,self.phase_int_vector,self.phase_mr_vector]

        """Populate derivative vectors one region at a time"""
        for j in jamp:
            var_arr= self.var_arr[:]
            famp[j], var_arr[i-1] = np.broadcast_arrays(famp[j],var_arr[i-1])
            gamp[j]=( egrad(ampfunc[j],i)(famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6]))
            var_arr= self.var_arr[:]
            amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6])
        for j in jphase:
            var_arr= self.var_arr[:]
            fphase[j], var_arr[i-1] = np.broadcast_arrays(fphase[j],var_arr[i-1])
            phase[j]=( egrad(phasefunc[j],i)(fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6]))

        """Concatenate the regions into one array"""
        gampout,ampout,phaseout = [],[],[]
        for j in jamp:
            ampout = np.concatenate((ampout,amp[j]))
            gampout = np.concatenate((gampout,gamp[j]))
        for j in jphase:
            phaseout = np.concatenate((phaseout,phase[j]))

        """Return the complex waveform derivative"""
        return np.subtract(gampout,np.multiply(ampout,np.multiply(1j,phaseout)))

    def populate_noise(self, detector='aLIGO',int_scheme='simps',stepsize = None):
        names = [ 'aLIGO', 'A+', 'A++', 'Vrt', 'Voyager', 'CE1', 'CE2 wide', 'CE2 narrow', 'ET-B', 'ET-D']
        freq = noise[0]
        if detector in names:
            noise_curve = noise[names.index(detector)+1]
            if int_scheme == 'quad':
                noise_func = CubicSpline(noise[0],noise_curve)
            else:
                noise_func = None
        elif detector == 'LISA':
            noise_curve = noise_lisa[1]
            freq = noise_lisa[0]
            if int_scheme == 'quad':
                noise_func = CubicSpline(noise_lisa[0],noise_curve)
            else:
                noise_func = None
        elif detector == 'aLIGOAnalytic':
            """Purely for testing"""
            if stepsize != None:
                freq = np.arange(1,10000,stepsize)
            noise_curve = sym_noise_curve(np.asarray(freq))
            noise_func = sym_noise_curve
        elif detector == 'aLIGOFitted':
            if stepsize != None:
                freq = np.arange(1,10000,stepsize)
            noise_curve = fitted_hanford_noise(np.asarray(freq))
            noise_func = fitted_hanford_noise

        else:
            print('DETECTOR ISSUE - check to make sure the detector name is spelled exactly as in {},{},{}'.format(names,'aLIGOAnalytic','aLIGOFitted'))
            return [],[]
        return noise_curve,noise_func,freq


    """Calcualtes the Fisher and the Inverse Fisher - Vectorized
    args: detector = 'aLIGO', 'aLIGOAnalytic' int_scheme = 'simps','trapz','quad', stepsize= float
    options aLIGOAnalytic and stepsize are purely for testing. The real data has these set.
    int_scheme should be left with simps - orders of magnitude faster than quad, and interpolating the noise data
    makes it debatable whether the accuracy is better than simps"""
    def calculate_fisher_matrix_vector(self,detector,int_scheme = 'simps',stepsize=None):
        start = time()
        if int_scheme == 'simps':
            int_func = integrate.simps
        elif int_scheme == 'trapz':
            int_func = integrate.trapz
        else:
            int_func= integrate.quad
        freq = noise[0]

        self.noise_curve, self.noise_func, freq = self.populate_noise(detector, int_scheme, stepsize)


        #STARTING AT 20HZ BECASUE YYP DID IN THEIR PAPER - TESTING ONLY
        self.lower_freq =self.calculate_lower_freq(freq,detector=detector)#self.flower
        self.upper_freq = self.calculate_upper_freq(freq,detector=detector)#self.fupper

        # print("lower: {}".format(self.lower_freq),"upper: {}".format(self.upper_freq))

        ## Almost entire time is spent here ##
        """Pre-populate Derivative arrays for faster evaluation"""
        self.calculate_derivatives()

        variable_indicies = range(1,len(self.var_arr)+1)
        fisher = np.zeros((len(variable_indicies),len(variable_indicies)))
        relerror = np.zeros((len(variable_indicies),len(variable_indicies)))

        ##########################################################################################
        #Quad method with function
        if int_scheme == 'quad':
            for i in variable_indicies:
                for j in range(1,i+1):
                    if i == j:
                        el, err = int_func(self.calculate_element_integrand,self.lower_freq,self.upper_freq,args=(i,j),limit=1000,epsabs=1e-50,epsrel=1e-15)
                        fisher[i-1][j-1] = (1/2)*el
                        relerror[i-1][j-1] = (1/2)*err/el
                    else:
                        fisher[i-1][j-1], err = int_func(self.calculate_element_integrand,self.lower_freq,self.upper_freq,args=(i,j),limit=1000,epsabs=1e-50,epsrel=1e-15)
                        relerror[i-1][j-1] = err/el

        ##########################################################################################
        #Discrete methods
        else:
            """Trim frequency and noise curve down to [flower,fupper]"""
            ftemp = freq[0]
            i = 0
            while ftemp <self.lower_freq:
                i +=1
                ftemp = freq[i]
            flow_pos = i

            ftemp = freq[len(freq)-1]
            i = len(freq)-1
            while ftemp > self.upper_freq:
                i-= 1
                ftemp = freq[i]
            fup_pos = i

            """Trim Frequencies to seperate which stage to apply (ins,int,mr)"""
            int_freq = np.asarray(freq[flow_pos:fup_pos])
            noise_integrand = self.noise_curve[flow_pos:fup_pos]
            amp_freqs = self.split_freqs_amp(int_freq)
            phase_freqs = self.split_freqs_phase(int_freq)

            waveform_derivs = []
            for i in variable_indicies:
                waveform_derivs.append(self.log_factors[i]*self.calculate_waveform_derivative_vector(amp_freqs,phase_freqs,i))
            for i in variable_indicies:
                for j in range(1,i+1):
                    integrand = np.multiply(4,np.divide(np.real(np.multiply(waveform_derivs[i-1],np.conj(waveform_derivs[j-1]))),np.multiply(noise_integrand,noise_integrand)))
                    if i == j:
                        fisher[i-1][j-1] = (1/2)*int_func(integrand,int_freq)
                    else:
                        fisher[i-1][j-1]= int_func(integrand,int_freq)

        ##########################################################################################
        #TESTING - CONVERTS FROM CHIRPM TO CHIRPME
        # for i in np.arange(len(fisher)):
        #     for j in np.arange(len(fisher[0])):
        #         if i == 3 or j ==3:
        #             if i == 3 and j ==3:
        #                 fisher[i][j] = fisher[i][j]/(1+self.Z)**2
        #             else:
        #                 fisher[i][j] = fisher[i][j]/(1+self.Z)


        fisher = fisher + np.transpose(fisher)
        try:
            chol_fisher = np.linalg.cholesky(fisher)
            inv_chol_fisher = np.linalg.inv(chol_fisher)
            inv_fisher = np.dot(inv_chol_fisher.T,inv_chol_fisher)
            cholo = True
        except:
            inv_fisher = np.linalg.inv(fisher)
            cholo = False
        self.fisher = fisher
        self.inv_fisher = inv_fisher
        return fisher,inv_fisher,cholo


    """Calculate SNR defined to be integral(|h|**2/NOISE) = integral(2 A**2/NOISE)
    **NOTE** I'm using trimmed frequencies here. Should I be using the full 10000 Hz range?"""
    def calculate_snr(self,detector='aLIGO'):
        self.noise_curve, self.noise_func, freq = self.populate_noise(detector=detector)
        if len(self.noise_curve) == 0:
            return "ERROR in noise_curve population"
        """For SNR, should the frequency be trimmed or not? Currently, not"""
        self.lower_freq =self.calculate_lower_freq(freq,detector=detector)
        self.upper_freq =self.calculate_upper_freq(freq,detector=detector)
        """Trim frequency and noise curve down to [flower,fupper]"""
        ftemp = freq[0]
        i = 0
        while ftemp <self.lower_freq:
            i +=1
            ftemp = freq[i]
        flow_pos = i

        ftemp = freq[len(freq)-1]
        i = len(freq)-1
        while ftemp > self.upper_freq:
            i-= 1
            ftemp = freq[i]
        fup_pos = i

        """Trim Frequencies to seperate which stage to apply (ins,int,mr)"""
        int_freq = np.asarray(freq[flow_pos:fup_pos])#np.asarray(freq)#
        # print(int_freq[0],int_freq[-1])
        # print(freq[0],freq[-1])
        noise_integrand = self.noise_curve[flow_pos:fup_pos]#np.asarray(self.noise_curve)#
        amp,phase,wave = self.calculate_waveform_vector(int_freq)
        Asquared = np.multiply(amp,amp)
        SNR = np.sqrt(integrate.simps( np.divide( np.multiply(4,Asquared) ,np.multiply(noise_integrand,noise_integrand) ),int_freq ) )
        return SNR

    """Assignment helper functions - each must have a manually defined grad wrt each argument
    For element_wise_grad to work correcly (priority - using loops over vectors is VASTLY slower)
    these helper functions must have the option to return an array (which would just be an array of
    all the same values) - required the addition of the isinstance check - if one of the arguments
    is an array, the return is an array of the same length"""
    @primitive
    def assign_lambda_param(self,chirpm,symmratio,chi_a,chi_s,i):
        for j in [chirpm,symmratio,chi_a,chi_s]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.parameters[i]
        return self.parameters[i]

    defvjp(assign_lambda_param,None,
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,i: lambda g: g*self.lambda_derivs_chirpm[i],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,i: lambda g: g*self.lambda_derivs_symmratio[i],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,i: lambda g: g*self.lambda_derivs_chi_a[i],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,i: lambda g: g*self.lambda_derivs_chi_s[i],None)

    @primitive
    def assign_totalmass(self,chirpm,symmratio):
        for i in [chirpm,symmratio]:
            if isinstance(i,np.ndarray):
                return np.ones(len(i))*self.M
        return self.M
    defvjp(assign_totalmass,None,
                lambda ans,self,chirpm,symmratio: lambda g: g*self.total_mass_deriv[0],
                lambda ans,self,chirpm,symmratio: lambda g: g*self.total_mass_deriv[1])
    @primitive
    def assign_fRD(self,m1,m2,M,symmratio,chi_s,chi_a):
        for i in [m1,m2,M,symmratio,chi_a,chi_s]:
            if isinstance(i,np.ndarray):
                return np.ones(len(i))*self.fRD
        return self.fRD
    defvjp(assign_fRD,None,
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fRD_deriv[0],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fRD_deriv[1],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fRD_deriv[2],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fRD_deriv[3],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fRD_deriv[4],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fRD_deriv[5])
    @primitive
    def assign_fdamp(self,m1,m2,M,symmratio,chi_s,chi_a):
        for i in [m1,m2,M,symmratio,chi_a,chi_s]:
            if isinstance(i,np.ndarray):
                return np.ones(len(i))*self.fdamp
        return self.fdamp
    defvjp(assign_fdamp,None,
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fdamp_deriv[0],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fdamp_deriv[1],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fdamp_deriv[2],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fdamp_deriv[3],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fdamp_deriv[4],
                lambda ans,self,m1,m2,M,symmratio,chi_s,chi_a: lambda g: g*self.fdamp_deriv[5])
    @primitive
    def assign_fpeak(self,M,fRD,fdamp,gamma2,gamma3):
        for i in [M,fRD,fdamp,gamma2,gamma3]:
            if isinstance(i,np.ndarray):
                return np.ones(len(i))*self.fpeak
        return self.fpeak
    defvjp(assign_fpeak,None,
                lambda ans,self,M,fRD,fdamp,gamma2,gamma3: lambda g: g*self.fpeak_deriv[0],
                lambda ans,self,M,fRD,fdamp,gamma2,gamma3: lambda g: g*self.fpeak_deriv[1],
                lambda ans,self,M,fRD,fdamp,gamma2,gamma3: lambda g: g*self.fpeak_deriv[2],
                lambda ans,self,M,fRD,fdamp,gamma2,gamma3: lambda g: g*self.fpeak_deriv[3],
                lambda ans,self,M,fRD,fdamp,gamma2,gamma3: lambda g: g*self.fpeak_deriv[4])
    @primitive
    def assign_mass1(self,chirpm,symmratio):
        for i in [chirpm,symmratio]:
            if isinstance(i,np.ndarray):
                return np.ones(len(i))*self.m1
        return self.m1
    defvjp(assign_mass1,None,
                lambda ans,self,chirpm,symmratio: lambda g: g*self.mass1_deriv[0],
                lambda ans,self,chirpm,symmratio: lambda g: g*self.mass1_deriv[1])
    @primitive
    def assign_mass2(self,chirpm,symmratio):
        for i in [chirpm,symmratio]:
            if isinstance(i,np.ndarray):
                return np.ones(len(i))*self.m2
        return self.m2
    defvjp(assign_mass2,None,
                lambda ans,self,chirpm,symmratio: lambda g: g*self.mass2_deriv[0],
                lambda ans,self,chirpm,symmratio: lambda g: g*self.mass2_deriv[1])
    @primitive
    def assign_pn_amp(self,symmratio,massdelta,chi_a,chi_s,i):
        for j in [massdelta,symmratio,chi_a,chi_s]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.pn_amp[i]
        return self.pn_amp[i]
    defvjp(assign_pn_amp,None,
                lambda ans,self,symmratio,massdelta,chi_a,chi_s,i: lambda g: g*self.pn_amp_deriv_symmratio[i],
                lambda ans,self,symmratio,massdelta,chi_a,chi_s,i: lambda g: g*self.pn_amp_deriv_delta[i],
                lambda ans,self,symmratio,massdelta,chi_a,chi_s,i: lambda g: g*self.pn_amp_deriv_chi_a[i],
                lambda ans,self,symmratio,massdelta,chi_a,chi_s,i: lambda g: g*self.pn_amp_deriv_chi_s[i],None)

    """Slightly more complicated assignment function - two of the pn_phase elements depend on the frequency
    so that must be handled more in depth - pushes that to second layer of functions that determines if a calculation is needed"""
    @primitive
    def assign_pn_phase(self,chirpm,symmratio,massdelta,chi_a,chi_s,f,i):
        if i in [5,6]:
            return  self.calculate_pn_phase(chirpm,symmratio,massdelta,chi_a,chi_s,f,i)
        for j in [chirpm,massdelta,symmratio,chi_a,chi_s,f]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.pn_phase[i]
        return self.pn_phase[i]
    defvjp(assign_pn_phase,None,
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,f,i: lambda g: g*self.grad_pn_phase_sorter_chirpm(f,i),
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,f,i: lambda g: g*self.grad_pn_phase_sorter_symmratio(f,i),#grad(self.calculate_pn_phase,0)(symmratio,massdelta,chi_a,chi_s,f,i),#g*self.pn_phase_deriv_symmratio[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,f,i: lambda g: g*self.grad_pn_phase_sorter_delta(f,i),#grad(self.calculate_pn_phase,1)(symmratio,massdelta,chi_a,chi_s,f,i),#g*self.pn_phase_deriv_delta[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,f,i: lambda g: g*self.grad_pn_phase_sorter_chi_a(f,i),#grad(self.calculate_pn_phase,2)(symmratio,massdelta,chi_a,chi_s,f,i),#g*self.pn_phase_deriv_chi_a[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,f,i: lambda g: g*self.grad_pn_phase_sorter_chi_s(f,i),#grad(self.calculate_pn_phase,3)(symmratio,massdelta,chi_a,chi_s,f,i),#g*self.pn_phase_deriv_chi_s[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,f,i: lambda g: g*grad(self.calculate_pn_phase,5)(chirpm,symmratio,massdelta,chi_a,chi_s,f,i),None)
    """Sorter functions to handle the frequency dependent elements of pn_phase"""
    def grad_pn_phase_sorter_chirpm(self,f,i):
        if i in [0,1,2,3,4,7]:
            return self.pn_phase_deriv_chirpm[i]
        else:
            return egrad(self.calculate_pn_phase,0)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,f,i)
    def grad_pn_phase_sorter_symmratio(self,f,i):
        if i in [0,1,2,3,4,7]:
            return self.pn_phase_deriv_symmratio[i]
        else:
            return egrad(self.calculate_pn_phase,1)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,f,i)
    def grad_pn_phase_sorter_delta(self,f,i):
        if i in [0,1,2,3,4,7]:
            return self.pn_phase_deriv_delta[i]
        else:
            return egrad(self.calculate_pn_phase,2)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,f,i)
    def grad_pn_phase_sorter_chi_a(self,f,i):
        if i in [0,1,2,3,4,7]:
            return self.pn_phase_deriv_chi_a[i]
        else:
            return egrad(self.calculate_pn_phase,3)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,f,i)
    def grad_pn_phase_sorter_chi_s(self,f,i):
        if i in [0,1,2,3,4,7]:
            return self.pn_phase_deriv_chi_s[i]
        else:
            return egrad(self.calculate_pn_phase,4)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,f,i)

    @primitive
    def assign_param_deltas(self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i):
        for j in [chirpm,massdelta,symmratio,chi_a,chi_s,fRD,fdamp,f3]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.param_deltas[i]
        return self.param_deltas[i]
    defvjp(assign_param_deltas,None,
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i: lambda g: g*self.param_deltas_derivs_chirpm[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i: lambda g: g*self.param_deltas_derivs_symmratio[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i: lambda g: g*self.param_deltas_derivs_delta[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i: lambda g: g*self.param_deltas_derivs_chi_a[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i: lambda g: g*self.param_deltas_derivs_chi_s[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i: lambda g: g*self.param_deltas_derivs_fRD[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i: lambda g: g*self.param_deltas_derivs_fdamp[i],
                lambda ans,self,chirpm,symmratio,massdelta,chi_a,chi_s,fRD,fdamp,f3,i: lambda g: g*self.param_deltas_derivs_f3[i],None)
    @primitive
    def assign_delta(self,symmratio):
        if isinstance(symmratio,np.ndarray):
            return np.ones(len(symmratio))*self.delta
        return self.delta
    defvjp(assign_delta,None,
                lambda ans,self,symmratio: lambda g: g*self.delta_deriv)
    @primitive
    def assign_beta1(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s):
        for j in [chirpm,delta,symmratio,phic,tc,chi_a,chi_s]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.beta1
        return self.beta1
    defvjp(assign_beta1,None,
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s: lambda g: g*self.beta1_deriv[0],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s: lambda g: g*self.beta1_deriv[1],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s: lambda g: g*self.beta1_deriv[2],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s: lambda g: g*self.beta1_deriv[3],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s: lambda g: g*self.beta1_deriv[4],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s: lambda g: g*self.beta1_deriv[5],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s: lambda g: g*self.beta1_deriv[6])
    @primitive
    def assign_beta0(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1):
        for j in [chirpm,delta,symmratio,phic,tc,chi_a,chi_s,beta1]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.beta0
        return self.beta0
    defvjp(assign_beta0,None,
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1: lambda g: g*self.beta0_deriv[0],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1: lambda g: g*self.beta0_deriv[1],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1: lambda g: g*self.beta0_deriv[2],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1: lambda g: g*self.beta0_deriv[3],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1: lambda g: g*self.beta0_deriv[4],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1: lambda g: g*self.beta0_deriv[5],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1: lambda g: g*self.beta0_deriv[6],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1: lambda g: g*self.beta0_deriv[7])
    @primitive
    def assign_alpha1(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1):
        for j in [chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.alpha1
        return self.alpha1
    defvjp(assign_alpha1,None,
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1: lambda g: g*self.alpha1_deriv[0],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1: lambda g: g*self.alpha1_deriv[1],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1: lambda g: g*self.alpha1_deriv[2],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1: lambda g: g*self.alpha1_deriv[3],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1: lambda g: g*self.alpha1_deriv[4],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1: lambda g: g*self.alpha1_deriv[5],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1: lambda g: g*self.alpha1_deriv[6],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1: lambda g: g*self.alpha1_deriv[7])
    @primitive
    def assign_alpha0(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1):
        for j in [chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.alpha0
        return self.alpha0
    defvjp(assign_alpha0,None,
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1: lambda g: g*self.alpha0_deriv[0],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1: lambda g: g*self.alpha0_deriv[1],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1: lambda g: g*self.alpha0_deriv[2],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1: lambda g: g*self.alpha0_deriv[3],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1: lambda g: g*self.alpha0_deriv[4],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1: lambda g: g*self.alpha0_deriv[5],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1: lambda g: g*self.alpha0_deriv[6],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1: lambda g: g*self.alpha0_deriv[7],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1: lambda g: g*self.alpha0_deriv[8])


if __name__ == "__main__":
    """Example code that generates a model with the parameters below, calculates GR Fisher and modified Fisher,
    plots the allowed regions of the (lambda_g,screening radius) space, plots the phase,
    the amplitude, the full waveform, and an example derivative (wrt to the amplitude)"""
    #import objgraph
    #import gc
    # dl = 420*mpc
    # mass1 =36*s_solm
    # mass2 =29*s_solm
    # spin1 = 0.32
    # spin2 = 0.44
    # detect = 'aLIGOFitted'
    # # detect = 'aLIGO'
    # # dl = 16000*mpc
    # # mass1 =5e3*s_solm
    # # mass2 =4e3*s_solm
    # # spin1 = 0.7
    # # spin2 = 0.9
    # # detect = 'LISA'
    # show_plots = True
    # NSflag = False
    # # model1 = IMRPhenomD(mass1,mass2,spin1,spin2,0,0,dl,N_detectors = 1,NSflag=NSflag)
    # model2 = Modified_IMRPhenomD(mass1,mass2,spin1,spin2,0,0,dl,0.,N_detectors=2,NSflag=NSflag)
    # model2.calculate_derivatives()
    # print(model2.beta1_deriv)
    # print("{} SNR model2: {}".format(detect,model2.calculate_snr(detector=detect)))
    #
    # # print(model1.calculate_snr(detector=detect))
    # ################################################################
    # #Temp model to find fpeak of GW170817:
    # #assumed to be two black holes
    # #Will need to replace with more accurate representation
    # # m1 = (1.6+1.36)/2*s_solm
    # # m2 = (1.36+1.17)/2*s_solm
    # # s1 = 0
    # # s2 = 0
    # # lumd = 40*mpc
    # # print("fpeak value for GW170817: ",IMRPhenomD(m1,m2,s1,s2,0,0,lumd).fpeak)
    # ################################################################
    # ################################################################
    # #TESTING
    # # x =np.asarray( noise[0])
    # # noise_integrand = noise[1]
    # # model2.calculate_derivatives()
    # #
    # # waveform_derivs = []
    # # variable_indicies = range(8)
    # # for i in variable_indicies:
    # #     waveform_derivs.append(model2.log_factors[i]*model2.calculate_waveform_derivative_vector(model2.split_freqs_amp(x),model2.split_freqs_phase(x),i))
    # # for i in variable_indicies:
    # #     for j in range(1,i+1):
    # #         integrand = np.multiply(4,np.divide(np.real(np.multiply(waveform_derivs[i-1],np.conj(waveform_derivs[j-1]))),np.multiply(noise_integrand,noise_integrand)))
    # #         plt.loglog(x,integrand)
    # # plt.show()
    # # plt.close()
    # ################################################################
    # start =time()
    # modfish,modinvfish,modcholo = model2.calculate_fisher_matrix_vector(detector=detect)
    # print("Modified variances: " ,np.sqrt(np.diagonal(modinvfish)),"time to calculate: {}".format(time()-start))
    # print("Model2Beta (90%)= {}".format(np.sqrt(np.diagonal(modinvfish))[-1]*1.645))
    # print("Model2Beta (sigma)= {}".format(np.sqrt(np.diagonal(modinvfish))[-1]))
    # ################################################################
    # #Compute the lambda_g value with no screening - Testing purposes
    # H0=model2.cosmo_model.H0.to('Hz').value#self.cosmo_model.H(0).u()
    # model2D = (1+model2.Z)*(integrate.quad(lambda x: 1/(H0*(1+x)**2*np.sqrt(.3*(1+x)**3 + .7)),0,model2.Z )[0])
    # model2beta =  .1344444444#np.sqrt(np.diagonal(modinvfish))[-1]
    # model2lambda = (model2beta* (1+model2.Z)  / (model2D * np.pi**2 * model2.chirpm))**(-1/2)
    # print("Lambda_g calculated from model2: {}x10^16".format(model2lambda*c/10**16))
    # print("Mass_g calculated from model2: {}".format(hplanck * c / (model2lambda*c )))
    # # print('################################################################')
    # # gw15lambda = model2.degeneracy_function_lambda_GW150914(.001*420*mpc)*c
    # # print("Lambda_g calculated from GW15: {}".format(gw15lambda))
    # # print("m_g calculated from GW15: {}".format(hplanck * c / (gw15lambda )))
    # # print('################################################################')
    # # mathematicaBeta = .138141
    # # mathematicaLambda = (mathematicaBeta * (1+model2.Z) / (model2D * np.pi**2 * model2.chirpm))**(-1/2)*c
    # # print("Beta calculated from Mathematica (90 all): {}".format(mathematicaBeta))
    # # print("Lambda_g calculated from Mathematica: {}x10^16".format(mathematicaLambda/1e16))
    # # print("m_g calculated from Mathematica: {}".format(hplanck * c / (mathematicaLambda )))
    # ################################################################
    #
    # # fig = model2.create_degeneracy_plot(model2beta/1.645,comparison=True)
    # # fig = model2.create_degeneracy_plot(delta_beta=0.04797785341657554)
    # # plt.savefig("print('Testing_IMRPhenomD/sample_beta.png")
    # # if show_plots:
    # #    plt.show()
    # # plt.close()
    # #objgraph.show_growth()
    #
    #
    # # start = time()
    # # fishvec,invvec,cholovec = model1.calculate_fisher_matrix_vector(detect)
    # # vectortime = time()-start
    # # print("VecFisher Time",vectortime)
    #
    # # objgraph.show_growth()
    # # objgraph.show_backrefs([fishvec,invvec])
    # ##################################################################################################
    # #Compute the fisher in loop to compare speed
    # # start = time()
    # # fish,inv,cholo = model1.calculate_fisher_matrix('aLIGO')
    # # looptime = time()-start
    # # print("Loop Fisher Time",looptime)
    # # print("Speedup: {}x".format(looptime/vectortime))
    # #
    # # match =True
    # # for i in np.arange(len(fish)):
    # #     for j in np.arange(len(fish[0])):
    # #         if (fishvec[i][j]-fish[i][j]) != 0:
    # #             match = False
    # #             print("element {},{} is {}% off".format(i,j,(fishvec[i][j]-fish[i][j])/fish[i][j]*100))
    # # print("Do they match? {}".format(match))
    #
    # ##################################################################################################
    # #Plot Example Output
    # # frequencies = np.linspace(1,5000,1e6)
    # # frequencies = np.linspace(1e-4,.001,1e5)
    # # Amp,phase,h = model1.calculate_waveform_vector(frequencies)
    # #
    # # eta_deriv = model1.log_factors[5]*model1.calculate_waveform_derivative_vector(model1.split_freqs_amp(frequencies),model1.split_freqs_phase(frequencies),5)
    # # fig, axes = plt.subplots(2,2)
    # # axes[0,0].plot(frequencies,Amp)
    # # axes[0,0].set_xscale('log')
    # # axes[0,0].set_yscale('log')
    # # axes[0,0].set_title('Amplitude')
    # # axes[0,0].set_ylabel("Amplitude")
    # # axes[0,0].set_xlabel("Frequency (Hz)")
    # #
    # # axes[0,1].plot(frequencies,phase)
    # # axes[0,1].set_title('Phase')
    # # axes[0,1].set_ylabel("Phase")
    # # axes[0,1].set_xlabel("Frequency (Hz)")
    # #
    # # axes[1,0].plot(frequencies,h,linewidth=0.5)
    # # axes[1,0].set_title('Full Waveform')
    # # axes[1,0].set_ylabel("Waveform (s)")
    # # axes[1,0].set_xlabel("Frequency (Hz)")
    # # #axes[1,0].set_xlim(0,50)
    # #
    # # axes[1,1].plot(frequencies,eta_deriv)
    # # axes[1,1].set_title(r'$\partial{h}/\partial{log(\eta)}$')
    # # axes[1,1].set_ylabel("Waveform (s)")
    # # axes[1,1].set_xlabel("Frequency (Hz)")
    # # axes[1,1].set_xscale('log')
    # # axes[1,1].set_yscale('log')
    # #
    # # plt.suptitle("Example Plots for Sample Model",fontsize = 16)
    # # if show_plots:
    # #     plt.show()
    # # plt.close()
