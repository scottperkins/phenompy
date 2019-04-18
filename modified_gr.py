from phenompy.gr import IMRPhenomD
from phenompy.gr import Lambda
import autograd.numpy as np
import numpy
from scipy import integrate
from scipy.interpolate import CubicSpline,interp1d
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

from phenompy import utilities
from phenompy.noise_utilities import *


c = utilities.c
G = utilities.G
s_solm = utilities.s_solm
mpc = utilities.mpc

#Rewrite to accept array of parameters - then won't have to copy and paste for child classes
"""Child class of IMRPhenomD - adds ppE modification to the phase of the waveform in the entire frequency range-
extra arguments: phase_mod (=0), bppe (=-3) -> phi_ins = IMRPhenomD.phi_ins + phase_mod*(pi*chirpm*f)**(bppe/3), etc.
Calculation of the modification from Will '97 for the last, degeneracy calculations -> specific to massive gravity"""
class Modified_IMRPhenomD_Full_Freq(IMRPhenomD):
    def __init__(self, mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,phase_mod = 0,bppe = -3,
                    cosmo_model = cosmology.Planck15,NSflag = False,N_detectors = 1):
        """Populate model variables"""
        self.N_detectors = N_detectors
        self.NSflag = NSflag
        self.cosmo_model = cosmo_model
        self.DL = Luminosity_Distance
        self.tc = float(collision_time)
        self.phic = float(collision_phase)
        self.symmratio = (mass1 * mass2) / (mass1 + mass2 )**2
        self.chirpme =  (mass1 * mass2)**(3/5)/(mass1 + mass2)**(1/5)
        self.delta = utilities.calculate_delta(self.symmratio)
        self.Z =Distance(Luminosity_Distance/mpc,unit=u.Mpc).compute_z(cosmology = self.cosmo_model)#.01
        self.chirpm = self.chirpme*(1+self.Z)
        self.M = utilities.calculate_totalmass(self.chirpm,self.symmratio)
        self.m1 = utilities.calculate_mass1(self.chirpm,self.symmratio)
        self.m2 = utilities.calculate_mass2(self.chirpm,self.symmratio)
        self.totalMass_restframe = mass1+mass2
        #self.A0 =(np.pi/30)**(1/2)*self.chirpm**2/self.DL * (np.pi*self.chirpm)**(-7/6)
        self.A0 =(np.pi*40./192.)**(1/2)*self.chirpm**2/self.DL * (np.pi*self.chirpm)**(-7/6)

        """Spin Variables"""
        self.chi1 = spin1
        self.chi2 = spin2
        self.chi_s = (spin1 + spin2)/2
        self.chi_a = (spin1 - spin2)/2

        """Post Newtonian Phase"""
        self.pn_phase = np.zeros(8)
        for i in [0,1,2,3,4,7]:
            self.pn_phase[i] = utilities.calculate_pn_phase(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1,i)

        """Numerical Fit Parameters"""
        self.parameters =[]
        for i in np.arange(len(Lambda)):
            self.parameters.append(self.calculate_parameter(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))

        """Post Newtonian Amplitude"""
        self.pn_amp = np.zeros(7)
        for i in np.arange(7):
            self.pn_amp[i]=utilities.calculate_pn_amp(self.symmratio,self.delta,self.chi_a,self.chi_s,i)

        """Post Merger Parameters - Ring Down frequency and Damping frequency"""
        self.fRD = utilities.calculate_postmerger_fRD(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fdamp = utilities.calculate_postmerger_fdamp(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fpeak = utilities.calculate_fpeak(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6])



        """Calculating the parameters for the intermediate amplitude region"""
        self.param_deltas = np.zeros(5)
        for i in np.arange(5):
            self.param_deltas[i] = self.calculate_delta_parameter(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i)

        """Only modifications to the system variables are below:
        -beta1
        -beta0
        -phase_mod
        -bppe
        -var_arr"""
        #################################################################################
        """Phase continuity parameters"""
        """Must be done in order - beta1,beta0,alpha1, then alpha0"""
        self.phase_mod = float(phase_mod)
        self.bppe = bppe
        self.beta1 = self.phase_cont_beta1(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.phase_mod)
        self.beta0 = self.phase_cont_beta0(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1,self.phase_mod)
        self.alpha1 = self.phase_cont_alpha1(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.phase_mod)
        self.alpha0 = self.phase_cont_alpha0(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1,self.phase_mod)
        self.var_arr = [self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a,self.phase_mod]


    def phase_cont_beta1(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod):
        M = self.assign_totalmass(chirpm,symmratio)
        f1 = 0.018/M
        pn_phase =[]
        for x in np.arange(len(self.pn_phase)):
            pn_phase.append(self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f1,x))
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
        sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
        sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
        #ins_grad = egrad(self.phi_ins,0)(f1,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod)
        #int_grad = egrad(self.phi_int,0)(f1,M,symmratio,0,0,beta2,beta3,chirpm,phase_mod)
        ins_grad = self.Dphi_ins(f1,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod)
        int_grad = self.Dphi_int(f1,M,symmratio,0,0,beta2,beta3,chirpm,phase_mod)
        return ((1/M)*ins_grad*symmratio
            -symmratio/M*int_grad)

    def phase_cont_beta0(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod):
        M = self.assign_totalmass(chirpm,symmratio)
        f1 = 0.018/M
        pn_phase =[]
        for x in np.arange(len(self.pn_phase)):
            pn_phase.append(self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f1,x))
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
        sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
        sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
        return self.phi_ins(f1,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod)*symmratio \
        - symmratio*self.phi_int(f1,M,symmratio,0,beta1,beta2,beta3,chirpm,phase_mod)

    def phase_cont_alpha1(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod):
        M = self.assign_totalmass(chirpm,symmratio)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        f2 = fRD*0.5
        #int_grad = egrad(self.phi_int,0)(f2,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod)
        #mr_grad = egrad(self.phi_mr,0)(f2,chirpm,symmratio,0,0,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod)
        int_grad = self.Dphi_int(f2,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod)
        mr_grad = self.Dphi_mr(f2,chirpm,symmratio,0,0,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod)
        return (1/M)*int_grad*symmratio \
        -(symmratio/M)*mr_grad

    def phase_cont_alpha0(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod):
        M = self.assign_totalmass(chirpm,symmratio)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        f2 = fRD*0.5
        return self.phi_int(f2,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod) *symmratio \
        - symmratio*self.phi_mr(f2,chirpm,symmratio,0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod)

    ######################################################################################
    """Added Phase Modification"""
    def phi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).phi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase) + phase_mod*(chirpm*np.pi*f)**(self.bppe/3))
    def Dphi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).Dphi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)+phase_mod*(chirpm*np.pi)**(self.bppe/3)*f**(self.bppe/3-1)*(self.bppe/3))

    def phi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).phi_int(f,M,symmratio,beta0,beta1,beta2,beta3))+phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def Dphi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).Dphi_int(f,M,symmratio,beta0,beta1,beta2,beta3)+phase_mod*(chirpm*np.pi)**(self.bppe/3)*f**(self.bppe/3-1)*(self.bppe/3))

    def phi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp))+phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def Dphi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).Dphi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp)+phase_mod*(chirpm*np.pi)**(self.bppe/3)*f**(self.bppe/3-1)*(self.bppe/3))
    ######################################################################################
    """Added phase mod argument for derivatives - returns the same as GR"""
    def amp_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
        return super(Modified_IMRPhenomD_Full_Freq,self).amp_ins_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    def amp_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
        return super(Modified_IMRPhenomD_Full_Freq,self).amp_int_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    def amp_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
        return super(Modified_IMRPhenomD_Full_Freq,self).amp_mr_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    ######################################################################################

    """Only added phase_mod to the arguments of beta1 and beta0 - Otherwise, exact copy of GR model method"""
    def phase_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
        fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
        beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod)
        beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod)
        alpha1 = self.assign_alpha1(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod)
        alpha0 = self.assign_alpha0(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        return self.phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod)

    """Uses overriden phi_ins method - added phase_mod arg"""
    def phase_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
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
        return self.phi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod)

    """Added phase_mod arg to beta parameter calls"""
    def phase_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod)
        beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        return self.phi_int(f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod)

    """Added derivatives for all the mod phi arguments - """
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
            self.mass1_deriv.append(grad(utilities.calculate_mass1,i)(self.chirpm,self.symmratio))
        self.mass2_deriv = []
        for i in range(2):
            self.mass2_deriv.append(grad(utilities.calculate_mass2,i)(self.chirpm,self.symmratio))
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
            self.pn_amp_deriv_symmratio.append(grad(utilities.calculate_pn_amp,0)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_delta.append(grad(utilities.calculate_pn_amp,1)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_chi_a.append(grad(utilities.calculate_pn_amp,2)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_chi_s.append(grad(utilities.calculate_pn_amp,3)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
        self.pn_phase_deriv_chirpm = []
        self.pn_phase_deriv_symmratio = []
        self.pn_phase_deriv_delta = []
        self.pn_phase_deriv_chi_a = []
        self.pn_phase_deriv_chi_s = []
        for i in np.arange(len(self.pn_phase)):
            self.pn_phase_deriv_chirpm.append(grad(utilities.calculate_pn_phase,0)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_symmratio.append(grad(utilities.calculate_pn_phase,1)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_delta.append(grad(utilities.calculate_pn_phase,2)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_chi_a.append(grad(utilities.calculate_pn_phase,3)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_chi_s.append(grad(utilities.calculate_pn_phase,4)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
        """Delta Parameters take up ~50 percent of the total time"""

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
        # print("deltas: {}".format(time()-start))
        self.fRD_deriv = []
        for i in range(6):
            self.fRD_deriv.append(grad(utilities.calculate_postmerger_fRD,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
        self.fdamp_deriv = []
        for i in range(6):
            self.fdamp_deriv.append(grad(utilities.calculate_postmerger_fdamp,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
        self.fpeak_deriv = []
        for i in range(5):
            self.fpeak_deriv.append(grad(utilities.calculate_fpeak,i)(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6]))

        """Only deviations from original IMRPhenomD are below: extra derivative for beta1,beta0, and extra log_factor"""
        ########################################################################################################################
        self.delta_deriv = grad(utilities.calculate_delta)(self.symmratio)
        self.beta1_deriv = []
        self.beta0_deriv = []
        self.alpha1_deriv = []
        self.alpha0_deriv = []
        for i in range(8):
            self.beta1_deriv.append(grad(self.phase_cont_beta1,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.phase_mod))
        for i in range(9):
            self.beta0_deriv.append(grad(self.phase_cont_beta0,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1,self.phase_mod))
        for i in range(9):
            self.alpha1_deriv.append(grad(self.phase_cont_alpha1,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.phase_mod))
        for i in range(10):
            self.alpha0_deriv.append(grad(self.phase_cont_alpha0,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1,self.phase_mod))
        """Populate array with variables for transformation from d/d(theta) to d/d(log(theta)) - begins with 0 because fisher matrix variables start at 1, not 0"""
        self.log_factors = [0,self.A0,1,1,self.chirpm,self.symmratio,1,1,1]

    """Function for actual element integrand - 4*Re(dh/dtheta_i* dh/dtheta_j) - Vectorized
    -added extra mod_phi argument"""
    def calculate_waveform_derivative_vector(self,freq,i):

        famp = self.split_freqs_amp(freq)
        fphase = self.split_freqs_phase(freq)

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
        #

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
            gamp[j]=( egrad(ampfunc[j],i)(famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7]))
            var_arr= self.var_arr[:]
            amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
        for j in jphase:
            var_arr= self.var_arr[:]
            fphase[j], var_arr[i-1] = np.broadcast_arrays(fphase[j],var_arr[i-1])
            phase[j]=( egrad(phasefunc[j],i)(fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7]))

        """Concatenate the regions into one array"""
        gampout,ampout,phaseout = [],[],[]
        for j in jamp:
            ampout = np.concatenate((ampout,amp[j]))
            gampout = np.concatenate((gampout,gamp[j]))
        for j in jphase:
            phaseout = np.concatenate((phaseout,phase[j]))

        """Return the complex waveform derivative"""
        return np.subtract(gampout,np.multiply(ampout,np.multiply(1j,phaseout)))

    """Calculate the waveform - vectorized
    Outputs: amp vector, phase vector, (real) waveform vector
    -added extra mod_phi argument"""
    def calculate_waveform_vector(self,freq):

        """Array of the functions used to populate derivative vectors"""
        ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
        phasefunc = [self.phase_ins_vector,self.phase_int_vector,self.phase_mr_vector]
        """Check to see if every region is sampled - if integration frequency
        doesn't reach a region, the loop is trimmed to avoid issues with unpopulated arrays"""

        famp = self.split_freqs_amp(freq)
        fphase = self.split_freqs_phase(freq)
        

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
            amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
        for j in jphase:
            phase[j]=phasefunc[j](fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
        """Concatenate the regions into one array"""
        ampout,phaseout =[],[]
        for j in jamp:
            ampout = np.concatenate((ampout,amp[j]))
        for j in jphase:
            if phase[j] is not None:
                phaseout = np.concatenate((phaseout,phase[j]))

        """Return the amplitude vector, phase vector, and real part of the waveform"""
        return ampout,phaseout, np.multiply(ampout,np.cos(phaseout))

    """For expediated evaluation of snr"""
    def calculate_waveform_amplitude_vector(self,freq):

        """Array of the functions used to populate derivative vectors"""
        ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
        """Check to see if every region is sampled - if integration frequency
        doesn't reach a region, the loop is trimmed to avoid issues with unpopulated arrays"""

        famp = self.split_freqs_amp(freq)
        

        jamp = [0,1,2]
        for i in np.arange(len(famp)):
            if len(famp[i]) == 0:
                jamp[i] = -1
        jamp = [x for x in jamp if x != -1]

        var_arr= self.var_arr[:]
        amp = [[],[],[]]

        """Populate derivative vectors one region at a time"""
        for j in jamp:
            amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
        """Concatenate the regions into one array"""
        ampout=[]
        for j in jamp:
            ampout = np.concatenate((ampout,amp[j]))

        """Return the amplitude vector, phase vector, and real part of the waveform"""
        return ampout

    """Derivative Definitions - added phase_mod to derivatives"""
    @primitive
    def assign_beta1(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod):
        for j in [chirpm,delta,symmratio,phic,tc,chi_a,chi_s,phase_mod]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.beta1
        return self.beta1
    defvjp(assign_beta1,None,
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[0],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[1],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[2],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[3],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[4],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[5],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[6],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[7])
    @primitive
    def assign_beta0(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod):
        for j in [chirpm,delta,symmratio,phic,tc,chi_a,chi_s,beta1,phase_mod]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.beta0
        return self.beta0
    defvjp(assign_beta0,None,
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[0],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[1],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[2],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[3],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[4],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[5],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[6],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[7],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[8])
    @primitive
    def assign_alpha1(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod):
        for j in [chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.alpha1
        return self.alpha1
    defvjp(assign_alpha1,None,
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[0],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[1],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[2],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[3],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[4],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[5],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[6],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[7],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[8])
    @primitive
    def assign_alpha0(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod):
        for j in [chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.alpha0
        return self.alpha0
    defvjp(assign_alpha0,None,
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[0],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[1],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[2],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[3],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[4],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[5],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[6],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[7],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[8],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[9])


"""Same as Modified full Freq, but the arguments are detector frame"""
class Modified_IMRPhenomD_Full_Freq_detector_frame(Modified_IMRPhenomD_Full_Freq):
    def __init__(self, mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,phase_mod = 0,bppe = -3,
                    cosmo_model = cosmology.Planck15,NSflag = False,N_detectors = 1):
        """Populate model variables"""
        self.N_detectors = N_detectors
        self.NSflag = NSflag
        self.cosmo_model = cosmo_model
        self.DL = Luminosity_Distance
        self.tc = float(collision_time)
        self.phic = float(collision_phase)
        self.symmratio = (mass1 * mass2) / (mass1 + mass2 )**2
        #self.chirpme =  (mass1 * mass2)**(3/5)/(mass1 + mass2)**(1/5)
        self.delta = utilities.calculate_delta(self.symmratio)
        self.chirpm =  (mass1 * mass2)**(3/5)/(mass1 + mass2)**(1/5)
        #self.Z =Distance(Luminosity_Distance/mpc,unit=u.Mpc).compute_z(cosmology = self.cosmo_model)#.01
        #self.chirpm = self.chirpme*(1+self.Z)
        self.M = utilities.calculate_totalmass(self.chirpm,self.symmratio)
        self.m1 = utilities.calculate_mass1(self.chirpm,self.symmratio)
        self.m2 = utilities.calculate_mass2(self.chirpm,self.symmratio)
        self.totalMass_restframe = mass1+mass2
        #self.A0 =(np.pi/30)**(1/2)*self.chirpm**2/self.DL * (np.pi*self.chirpm)**(-7/6)
        self.A0 =(np.pi*40./192.)**(1/2)*self.chirpm**2/self.DL * (np.pi*self.chirpm)**(-7/6)

        """Spin Variables"""
        self.chi1 = spin1
        self.chi2 = spin2
        self.chi_s = (spin1 + spin2)/2
        self.chi_a = (spin1 - spin2)/2

        """Post Newtonian Phase"""
        self.pn_phase = np.zeros(8)
        for i in [0,1,2,3,4,7]:
            self.pn_phase[i] = utilities.calculate_pn_phase(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1,i)

        """Numerical Fit Parameters"""
        self.parameters =[]
        for i in np.arange(len(Lambda)):
            self.parameters.append(self.calculate_parameter(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))

        """Post Newtonian Amplitude"""
        self.pn_amp = np.zeros(7)
        for i in np.arange(7):
            self.pn_amp[i]=utilities.calculate_pn_amp(self.symmratio,self.delta,self.chi_a,self.chi_s,i)

        """Post Merger Parameters - Ring Down frequency and Damping frequency"""
        self.fRD = utilities.calculate_postmerger_fRD(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fdamp = utilities.calculate_postmerger_fdamp(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fpeak = utilities.calculate_fpeak(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6])



        """Calculating the parameters for the intermediate amplitude region"""
        self.param_deltas = np.zeros(5)
        for i in np.arange(5):
            self.param_deltas[i] = self.calculate_delta_parameter(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i)

        """Only modifications to the system variables are below:
        -beta1
        -beta0
        -phase_mod
        -bppe
        -var_arr"""
        #################################################################################
        """Phase continuity parameters"""
        """Must be done in order - beta1,beta0,alpha1, then alpha0"""
        self.phase_mod = float(phase_mod)
        self.bppe = bppe
        self.beta1 = self.phase_cont_beta1(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.phase_mod)
        self.beta0 = self.phase_cont_beta0(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1,self.phase_mod)
        self.alpha1 = self.phase_cont_alpha1(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.phase_mod)
        self.alpha0 = self.phase_cont_alpha0(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1,self.phase_mod)
        self.var_arr = [self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a,self.phase_mod]

    def fix_snr(self,snr_target,detector='aLIGO',lower_freq=None,upper_freq=None):
        snr_current = self.calculate_snr(detector=detector,lower_freq=lower_freq,upper_freq=upper_freq)
        #snr_current = self.calculate_snr_old(detector=detector,lower_freq=lower_freq,upper_freq=upper_freq)
        oldDL = self.DL
        self.DL = self.DL*snr_current/snr_target
        self.A0 = self.A0*oldDL/self.DL
        self.var_arr[0] = self.A0

    def fix_snr_series(self,snr_target,frequencies,detector='aLIGO'):
        snr_current = self.calculate_snr_series(detector=detector,frequencies=frequencies)
        oldDL = self.DL
        self.DL = self.DL*snr_current/snr_target
        self.A0 = self.A0*oldDL/self.DL
        self.var_arr[0] = self.A0


"""Child class of Modified_IMRPhenomD - adds modification to the phase of the waveform in the inspiral region -
extra arguments: phase_mod (=0), bppe (=-3) -> phi_ins = IMRPhenomD.phi_ins + phase_mod*(pi*chirpm*f)**(bppe/3)
Calculation of th modification from Will '97"""
class Modified_IMRPhenomD_Inspiral_Freq(Modified_IMRPhenomD_Full_Freq):
    def phi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).phi_int(f,M,symmratio,beta0,beta1,beta2,beta3))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def Dphi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).Dphi_int(f,M,symmratio,beta0,beta1,beta2,beta3))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def phi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def Dphi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).Dphi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)

"""Same as above, but for detector frame"""
class Modified_IMRPhenomD_Inspiral_Freq_detector_frame(Modified_IMRPhenomD_Full_Freq_detector_frame):
    def phi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).phi_int(f,M,symmratio,beta0,beta1,beta2,beta3))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def Dphi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).Dphi_int(f,M,symmratio,beta0,beta1,beta2,beta3))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def phi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def Dphi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq,self).Dphi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)

"""Class that includes ppE parameter in phase for Fisher analysis in entire frequency range- adjusted for correction to SPA, following
the derivation in arXiv:gr-qc/9901076 - includes ppE correction and GR correction term, see nb for the mod SPA coefficients"""
class Modified_IMRPhenomD_Full_Freq_SPA(Modified_IMRPhenomD_Full_Freq):
    def SPA_correction(self,f,chirpm,b,phase_mod):
        return 92/45 * (np.pi * chirpm * f)**(5/3)+ (160/1125)*(1431 + 676*b + 134*b**2 + 9*b**3) * phase_mod * (np.pi * chirpm* f)**((10+b)/3)
    def DSPA_correction(self,f,chirpm,b,phase_mod):
        return 92/45 * (np.pi * chirpm )**(5/3)*(5/3)*f**(5/3-1)+ (160/1125)*(1431 + 676*b + 134*b**2 + 9*b**3) * phase_mod * (np.pi * chirpm)**((10+b)/3)*((10+b)/3)*f**((10+b)/3-1)

    """Just need to add terms to the phase functions - call super method, and append correction term"""
    def phi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq_SPA,self).phi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod))+ self.SPA_correction(f,chirpm,self.bppe,phase_mod)
    def Dphi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq_SPA,self).Dphi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod))+ self.DSPA_correction(f,chirpm,self.bppe,phase_mod)

    def phi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq_SPA,self).phi_int(f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod))+ self.SPA_correction(f,chirpm,self.bppe,phase_mod)
    def Dphi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq_SPA,self).Dphi_int(f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod))+ self.DSPA_correction(f,chirpm,self.bppe,phase_mod)

    def phi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq_SPA,self).phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod))+ self.SPA_correction(f,chirpm,self.bppe,phase_mod)
    def Dphi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD_Full_Freq_SPA,self).Dphi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod))+ self.DSPA_correction(f,chirpm,self.bppe,phase_mod)

"""Class that includes ppE parameter in phase for Fisher analysis in inspiral frequency range- adjusted for correction to SPA, following
the derivation in arXiv:gr-qc/9901076 - includes ppE correction and GR correction term"""
class Modified_IMRPhenomD_Inspiral_Freq_SPA(Modified_IMRPhenomD_Inspiral_Freq):
    def SPA_correction(self,f,chirpm,b,phase_mod):
        return 92/45 * (np.pi * chirpm * f)**(5/3)+ (160/1125)*(1431 + 676*b + 134*b**2 + 9*b**3) * phase_mod * (np.pi * chirpm* f)**((10+b)/3)
    def DSPA_correction(self,f,chirpm,b,phase_mod):
        return 92/45 * (np.pi * chirpm )**(5/3)*(5/3)*f**(5/3-1)+ (160/1125)*(1431 + 676*b + 134*b**2 + 9*b**3) * phase_mod * (np.pi * chirpm)**((10+b)/3)*((10+b)/3)*f**((10+b)/3-1)

    """Just need to add terms to the phase functions - call super method, and append correction term"""
    def phi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        return (super(Modified_IMRPhenomD_Inspiral_Freq_SPA,self).phi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod))+ self.SPA_correction(f,chirpm,self.bppe,phase_mod)
    def Dphi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        return (super(Modified_IMRPhenomD_Inspiral_Freq_SPA,self).Dphi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod))+ self.DSPA_correction(f,chirpm,self.bppe,phase_mod)

"""Modified IMRPhenomD to treat the log of the transition frequency from intermediate to merger-rindown
as extra Fisher Variables - Full Variable List:
[lnA, phi_c, t_c, ln Chirpm Mass, ln symmetric mass ratio, chi_s, chi_a, ln f_trans_mr]"""
class Modified_IMRPhenomD_Transition_Freq(IMRPhenomD):
    def __init__(self, mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors = 1, f_int_mr = None):
        """Populate model variables"""
        self.N_detectors = N_detectors
        self.NSflag = NSflag
        self.cosmo_model = cosmo_model
        self.DL = Luminosity_Distance
        self.tc = float(collision_time)
        self.phic = float(collision_phase)
        self.symmratio = (mass1 * mass2) / (mass1 + mass2 )**2
        self.chirpme =  (mass1 * mass2)**(3/5)/(mass1 + mass2)**(1/5)
        self.delta = utilities.calculate_delta(self.symmratio)
        self.Z =Distance(Luminosity_Distance/mpc,unit=u.Mpc).compute_z(cosmology = self.cosmo_model)#.01
        self.chirpm = self.chirpme*(1+self.Z)
        self.M = utilities.calculate_totalmass(self.chirpm,self.symmratio)
        self.m1 = utilities.calculate_mass1(self.chirpm,self.symmratio)
        self.m2 = utilities.calculate_mass2(self.chirpm,self.symmratio)
        self.totalMass_restframe = mass1+mass2
        self.A0 =(np.pi/30)**(1/2)*self.chirpm**2/self.DL * (np.pi*self.chirpm)**(-7/6)

        """Spin Variables"""
        self.chi1 = spin1
        self.chi2 = spin2
        self.chi_s = (spin1 + spin2)/2
        self.chi_a = (spin1 - spin2)/2

        """Post Newtonian Phase"""
        self.pn_phase = np.zeros(8)
        for i in [0,1,2,3,4,7]:
            self.pn_phase[i] = utilities.calculate_pn_phase(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1,i)

        """Numerical Fit Parameters"""
        self.parameters =[]
        for i in np.arange(len(Lambda)):
            self.parameters.append(self.calculate_parameter(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))

        """Post Newtonian Amplitude"""
        self.pn_amp = np.zeros(7)
        for i in np.arange(7):
            self.pn_amp[i]=utilities.calculate_pn_amp(self.symmratio,self.delta,self.chi_a,self.chi_s,i)

        """Post Merger Parameters - Ring Down frequency and Damping frequency"""
        self.fRD = utilities.calculate_postmerger_fRD(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fdamp = utilities.calculate_postmerger_fdamp(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fpeak = utilities.calculate_fpeak(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6])



        """Calculating the parameters for the intermediate amplitude region"""
        self.param_deltas = np.zeros(5)
        for i in np.arange(5):
            self.param_deltas[i] = self.calculate_delta_parameter(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i)

        """Only modifications to the system variables are below:
        -beta1
        -beta0
        -f_transition
        -bppe
        -var_arr"""
        #################################################################################
        self.f_trans0 = 0.018/self.M
        if f_int_mr == None:
            self.f_transition = .5*self.fRD
        else:
            self.f_transition = f_int_mr

        """Phase continuity parameters"""
        """Must be done in order - beta1,beta0,alpha1, then alpha0"""

        self.beta1 = self.phase_cont_beta1(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s)
        self.beta0 = self.phase_cont_beta0(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1)
        self.alpha1 = self.phase_cont_alpha1(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.f_transition)
        self.alpha0 = self.phase_cont_alpha0(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1,self.f_transition)
        self.var_arr = [self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a,self.f_transition]


    def phase_cont_alpha1(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition):
        M = self.assign_totalmass(chirpm,symmratio)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        f1 = f_transition
        int_grad = self.Dphi_int(f2,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod)
        mr_grad = self.Dphi_mr(f2,chirpm,symmratio,0,0,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod)
        return (1/M)*int_grad*symmratio \
        -(symmratio/M)*mr_grad

    def phase_cont_alpha0(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition):
        M = self.assign_totalmass(chirpm,symmratio)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        f1 = f_transition
        return self.phi_int(f1,M,symmratio,beta0,beta1,beta2,beta3) *symmratio \
        - symmratio*self.phi_mr(f1,chirpm,symmratio,0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp)

    ######################################################################################
    """Added phase mod argument for derivatives - returns the same as GR"""
    def amp_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_transition):
        return super(Modified_IMRPhenomD_Transition_Freq,self).amp_ins_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    def amp_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_transition):
        return super(Modified_IMRPhenomD_Transition_Freq,self).amp_int_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    def amp_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_transition):
        return super(Modified_IMRPhenomD_Transition_Freq,self).amp_mr_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)

    ######################################################################################

    """Only added transition frequencies to the arguments of beta1 and beta0 - Otherwise, exact copy of GR model method"""
    def phase_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_transition):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
        fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
        beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s)
        beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1)
        alpha1 = self.assign_alpha1(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition)
        alpha0 = self.assign_alpha0(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        return self.phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp)

    """Uses overriden phi_ins method - added transition frequency args"""
    def phase_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a, f_transition):
        return super(Modified_IMRPhenomD_Transition_Freq,self).phase_ins_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    """Added transition frequency arg to beta parameter calls"""
    def phase_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a, f_transition):
        return super(Modified_IMRPhenomD_Transition_Freq,self).phase_int_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)

    def split_freqs_phase(self,freqs):
        freqins = freqs[(freqs<=self.f_trans0)]
        freqint = freqs[(freqs>self.f_trans0) & (freqs<=self.f_transition)]
        freqmr = freqs[(freqs>self.f_transition)]
        return [freqins,freqint,freqmr]

    """Added derivatives for all the mod phi arguments - """
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
            self.mass1_deriv.append(grad(utilities.calculate_mass1,i)(self.chirpm,self.symmratio))
        self.mass2_deriv = []
        for i in range(2):
            self.mass2_deriv.append(grad(utilities.calculate_mass2,i)(self.chirpm,self.symmratio))
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
            self.pn_amp_deriv_symmratio.append(grad(utilities.calculate_pn_amp,0)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_delta.append(grad(utilities.calculate_pn_amp,1)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_chi_a.append(grad(utilities.calculate_pn_amp,2)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_chi_s.append(grad(utilities.calculate_pn_amp,3)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
        self.pn_phase_deriv_chirpm = []
        self.pn_phase_deriv_symmratio = []
        self.pn_phase_deriv_delta = []
        self.pn_phase_deriv_chi_a = []
        self.pn_phase_deriv_chi_s = []
        for i in np.arange(len(self.pn_phase)):
            self.pn_phase_deriv_chirpm.append(grad(utilities.calculate_pn_phase,0)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_symmratio.append(grad(utilities.calculate_pn_phase,1)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_delta.append(grad(utilities.calculate_pn_phase,2)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_chi_a.append(grad(utilities.calculate_pn_phase,3)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_chi_s.append(grad(utilities.calculate_pn_phase,4)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
        """Delta Parameters take up ~50 percent of the total time"""

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
        # print("deltas: {}".format(time()-start))
        self.fRD_deriv = []
        for i in range(6):
            self.fRD_deriv.append(grad(utilities.calculate_postmerger_fRD,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
        self.fdamp_deriv = []
        for i in range(6):
            self.fdamp_deriv.append(grad(utilities.calculate_postmerger_fdamp,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
        self.fpeak_deriv = []
        for i in range(5):
            self.fpeak_deriv.append(grad(utilities.calculate_fpeak,i)(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6]))

        """Only deviations from original IMRPhenomD are below: extra derivative for beta1,beta0, and extra log_factor"""
        ########################################################################################################################
        self.delta_deriv = grad(utilities.calculate_delta)(self.symmratio)
        self.beta1_deriv = []
        self.beta0_deriv = []
        self.alpha1_deriv = []
        self.alpha0_deriv = []
        for i in range(7):
            self.beta1_deriv.append(grad(self.phase_cont_beta1,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s))
        for i in range(8):
            self.beta0_deriv.append(grad(self.phase_cont_beta0,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1))
        for i in range(9):
            self.alpha1_deriv.append(grad(self.phase_cont_alpha1,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.f_transition))
        for i in range(10):
            self.alpha0_deriv.append(grad(self.phase_cont_alpha0,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1,self.f_transition))
        """Populate array with variables for transformation from d/d(theta) to d/d(log(theta)) - begins with 0 because fisher matrix variables start at 1, not 0"""
        self.log_factors = [0,self.A0,1,1,self.chirpm,self.symmratio,1,1,self.f_transition]

    """Function for actual element integrand - 4*Re(dh/dtheta_i* dh/dtheta_j) - Vectorized
    -added extra mod_phi argument"""
    def calculate_waveform_derivative_vector(self,freq,i):
    
        famp = self.split_freqs_amp(freq)
        fphase = self.split_freqs_phase(freq)
        
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
        #

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
            gamp[j]=( egrad(ampfunc[j],i)(famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7]))
            var_arr= self.var_arr[:]
            amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
        for j in jphase:
            var_arr= self.var_arr[:]
            fphase[j], var_arr[i-1] = np.broadcast_arrays(fphase[j],var_arr[i-1])
            phase[j]=( egrad(phasefunc[j],i)(fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7]))

        """Concatenate the regions into one array"""
        gampout,ampout,phaseout = [],[],[]
        for j in jamp:
            ampout = np.concatenate((ampout,amp[j]))
            gampout = np.concatenate((gampout,gamp[j]))
        for j in jphase:
            phaseout = np.concatenate((phaseout,phase[j]))

        """Return the complex waveform derivative"""
        return np.subtract(gampout,np.multiply(ampout,np.multiply(1j,phaseout)))

    """Calculate the waveform - vectorized
    Outputs: amp vector, phase vector, (real) waveform vector
    -added extra mod_phi argument"""
    def calculate_waveform_vector(self,freq):

        """Array of the functions used to populate derivative vectors"""
        ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
        phasefunc = [self.phase_ins_vector,self.phase_int_vector,self.phase_mr_vector]
        """Check to see if every region is sampled - if integration frequency
        doesn't reach a region, the loop is trimmed to avoid issues with unpopulated arrays"""

        famp = self.split_freqs_amp(freq)
        fphase = self.split_freqs_phase(freq)

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
            amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
        for j in jphase:
            phase[j]=phasefunc[j](fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
        """Concatenate the regions into one array"""
        ampout,phaseout =[],[]
        for j in jamp:
            ampout = np.concatenate((ampout,amp[j]))
        for j in jphase:
            if phase[j] is not None:
                phaseout = np.concatenate((phaseout,phase[j]))

        """Return the amplitude vector, phase vector, and real part of the waveform"""
        return ampout,phaseout, np.multiply(ampout,np.cos(phaseout))


    """Derivative Definitions - added phase_mod to derivatives"""
    @primitive
    def assign_alpha1(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition):
        for j in [chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.alpha1
        return self.alpha1
    defvjp(assign_alpha1,None,
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition: lambda g: g*self.alpha1_deriv[0],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition: lambda g: g*self.alpha1_deriv[1],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition: lambda g: g*self.alpha1_deriv[2],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition: lambda g: g*self.alpha1_deriv[3],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition: lambda g: g*self.alpha1_deriv[4],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition: lambda g: g*self.alpha1_deriv[5],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition: lambda g: g*self.alpha1_deriv[6],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition: lambda g: g*self.alpha1_deriv[7],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_transition: lambda g: g*self.alpha1_deriv[8])
    @primitive
    def assign_alpha0(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition):
        for j in [chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.alpha0
        return self.alpha0
    defvjp(assign_alpha0,None,
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[0],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[1],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[2],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[3],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[4],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[5],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[6],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[7],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[8],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_transition: lambda g: g*self.alpha0_deriv[9])



"""Modified IMRPhenomD to treat the log of both of the transition frequencies as extra Fisher Variables
- Full Variable List: [lnA, phi_c, t_c, ln Chirpm Mass, ln symmetric mass ratio, chi_s, chi_a, ln f_trans_insp, ln f_trans_mr]"""
class Modified_IMRPhenomD_All_Transition_Freq(IMRPhenomD):
    def __init__(self, mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors = 1, f_ins_int = None, f_int_mr = None):
        """Populate model variables"""
        self.N_detectors = N_detectors
        self.NSflag = NSflag
        self.cosmo_model = cosmo_model
        self.DL = Luminosity_Distance
        self.tc = float(collision_time)
        self.phic = float(collision_phase)
        self.symmratio = (mass1 * mass2) / (mass1 + mass2 )**2
        self.chirpme =  (mass1 * mass2)**(3/5)/(mass1 + mass2)**(1/5)
        self.delta = utilities.calculate_delta(self.symmratio)
        self.Z =Distance(Luminosity_Distance/mpc,unit=u.Mpc).compute_z(cosmology = self.cosmo_model)#.01
        self.chirpm = self.chirpme*(1+self.Z)
        self.M = utilities.calculate_totalmass(self.chirpm,self.symmratio)
        self.m1 = utilities.calculate_mass1(self.chirpm,self.symmratio)
        self.m2 = utilities.calculate_mass2(self.chirpm,self.symmratio)
        self.totalMass_restframe = mass1+mass2
        #self.A0 =(np.pi/30)**(1/2)*self.chirpm**2/self.DL * (np.pi*self.chirpm)**(-7/6)
        self.A0 =(np.pi*40./192.)**(1/2)*self.chirpm**2/self.DL * (np.pi*self.chirpm)**(-7/6)

        """Spin Variables"""
        self.chi1 = spin1
        self.chi2 = spin2
        self.chi_s = (spin1 + spin2)/2
        self.chi_a = (spin1 - spin2)/2

        """Post Newtonian Phase"""
        self.pn_phase = np.zeros(8)
        for i in [0,1,2,3,4,7]:
            self.pn_phase[i] = utilities.calculate_pn_phase(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1,i)

        """Numerical Fit Parameters"""
        self.parameters =[]
        for i in np.arange(len(Lambda)):
            self.parameters.append(self.calculate_parameter(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))

        """Post Newtonian Amplitude"""
        self.pn_amp = np.zeros(7)
        for i in np.arange(7):
            self.pn_amp[i]=utilities.calculate_pn_amp(self.symmratio,self.delta,self.chi_a,self.chi_s,i)

        """Post Merger Parameters - Ring Down frequency and Damping frequency"""
        self.fRD = utilities.calculate_postmerger_fRD(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fdamp = utilities.calculate_postmerger_fdamp(\
            self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
        self.fpeak = utilities.calculate_fpeak(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6])



        """Calculating the parameters for the intermediate amplitude region"""
        self.param_deltas = np.zeros(5)
        for i in np.arange(5):
            self.param_deltas[i] = self.calculate_delta_parameter(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i)

        """Only modifications to the system variables are below:
        -beta1
        -beta0
        -phase_mod
        -bppe
        -var_arr"""
        #################################################################################
        """Phase continuity parameters"""
        """Must be done in order - beta1,beta0,alpha1, then alpha0"""
        if f_ins_int == None:
            self.f_trans0 = 0.018/self.M
        else:
            self.f_trans0 = f_ins_int
        if f_int_mr == None:
            self.f_trans1 = .5*self.fRD
        else:
            self.f_trans1 = f_int_mr

        self.beta1 = self.phase_cont_beta1(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.f_trans0)
        self.beta0 = self.phase_cont_beta0(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1,self.f_trans0)
        self.alpha1 = self.phase_cont_alpha1(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.f_trans1)
        self.alpha0 = self.phase_cont_alpha0(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1,self.f_trans1)
        self.var_arr = [self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a,self.f_trans0,self.f_trans1]


    def phase_cont_beta1(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,f_trans0):
        M = self.assign_totalmass(chirpm,symmratio)
        f0 = f_trans0
        pn_phase =[]
        for x in np.arange(len(self.pn_phase)):
            pn_phase.append(self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f0,x))
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
        sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
        sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
        ins_grad = egrad(self.phi_ins,0)
        return ((1/M)*ins_grad(f0,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)*symmratio
            -symmratio/M*(grad(self.phi_int,0)(f0,M,symmratio,0,0,beta2,beta3)))

    def phase_cont_beta0(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,f_trans0):
        M = self.assign_totalmass(chirpm,symmratio)
        f0 = f_trans0
        pn_phase =[]
        for x in np.arange(len(self.pn_phase)):
            pn_phase.append(self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f0,x))
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
        sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
        sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
        return self.phi_ins(f0,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)*symmratio \
        - symmratio*self.phi_int(f0,M,symmratio,0,beta1,beta2,beta3)

    def phase_cont_alpha1(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_trans1):
        M = self.assign_totalmass(chirpm,symmratio)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        f1 = f_trans1
        return (1/M)*egrad(self.phi_int,0)(f1,M,symmratio,beta0,beta1,beta2,beta3)*symmratio \
        -(symmratio/M)*egrad(self.phi_mr,0)(f1,chirpm,symmratio,0,0,alpha2,alpha3,alpha4,alpha5,fRD,fdamp)

    def phase_cont_alpha0(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_trans1):
        M = self.assign_totalmass(chirpm,symmratio)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        f1 = f_trans1
        return self.phi_int(f1,M,symmratio,beta0,beta1,beta2,beta3) *symmratio \
        - symmratio*self.phi_mr(f1,chirpm,symmratio,0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp)

    ######################################################################################
    """Added phase mod argument for derivatives - returns the same as GR"""
    def amp_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_trans0,f_trans1):
        return super(Modified_IMRPhenomD_Transition_Freq,self).amp_ins_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    def amp_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_trans0,f_trans1):
        return super(Modified_IMRPhenomD_Transition_Freq,self).amp_int_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    def amp_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_trans0,f_trans1):
        return super(Modified_IMRPhenomD_Transition_Freq,self).amp_mr_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)

    ######################################################################################

    """Only added transition frequencies to the arguments of beta1 and beta0 - Otherwise, exact copy of GR model method"""
    def phase_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_trans0,f_trans1):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
        fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
        beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,f_trans0)
        beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,f_trans0)
        alpha1 = self.assign_alpha1(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_trans1)
        alpha0 = self.assign_alpha0(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_trans1)
        alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
        alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
        alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
        alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
        return self.phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp)

    """Uses overriden phi_ins method - added transition frequency args"""
    def phase_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_trans0, f_trans1):
        return super(Modified_IMRPhenomD_Transition_Freq,self).phase_ins_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    """Added transition frequency arg to beta parameter calls"""
    def phase_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,f_trans0, f_trans1):
        M = self.assign_totalmass(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        delta = self.assign_delta(symmratio)
        fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
        beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,f_trans0)
        beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,f_trans0)
        beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
        beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
        return self.phi_int(f,M,symmratio,beta0,beta1,beta2,beta3)

    def split_freqs_phase(self,freqs):
        freqins = freqs[(freqs<=self.f_trans0)]
        freqint = freqs[(freqs>self.f_trans0) & (freqs<=self.f_trans1)]
        freqmr = freqs[(freqs>self.f_trans1)]
        return [freqins,freqint,freqmr]

    """Added derivatives for all the mod phi arguments - """
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
            self.mass1_deriv.append(grad(utilities.calculate_mass1,i)(self.chirpm,self.symmratio))
        self.mass2_deriv = []
        for i in range(2):
            self.mass2_deriv.append(grad(utilities.calculate_mass2,i)(self.chirpm,self.symmratio))
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
            self.pn_amp_deriv_symmratio.append(grad(utilities.calculate_pn_amp,0)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_delta.append(grad(utilities.calculate_pn_amp,1)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_chi_a.append(grad(utilities.calculate_pn_amp,2)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
            self.pn_amp_deriv_chi_s.append(grad(utilities.calculate_pn_amp,3)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
        self.pn_phase_deriv_chirpm = []
        self.pn_phase_deriv_symmratio = []
        self.pn_phase_deriv_delta = []
        self.pn_phase_deriv_chi_a = []
        self.pn_phase_deriv_chi_s = []
        for i in np.arange(len(self.pn_phase)):
            self.pn_phase_deriv_chirpm.append(grad(utilities.calculate_pn_phase,0)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_symmratio.append(grad(utilities.calculate_pn_phase,1)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_delta.append(grad(utilities.calculate_pn_phase,2)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_chi_a.append(grad(utilities.calculate_pn_phase,3)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
            self.pn_phase_deriv_chi_s.append(grad(utilities.calculate_pn_phase,4)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
        """Delta Parameters take up ~50 percent of the total time"""

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
        # print("deltas: {}".format(time()-start))
        self.fRD_deriv = []
        for i in range(6):
            self.fRD_deriv.append(grad(utilities.calculate_postmerger_fRD,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
        self.fdamp_deriv = []
        for i in range(6):
            self.fdamp_deriv.append(grad(utilities.calculate_postmerger_fdamp,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
        self.fpeak_deriv = []
        for i in range(5):
            self.fpeak_deriv.append(grad(utilities.calculate_fpeak,i)(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6]))

        """Only deviations from original IMRPhenomD are below: extra derivative for beta1,beta0, and extra log_factor"""
        ########################################################################################################################
        self.delta_deriv = grad(utilities.calculate_delta)(self.symmratio)
        self.beta1_deriv = []
        self.beta0_deriv = []
        self.alpha1_deriv = []
        self.alpha0_deriv = []
        for i in range(8):
            self.beta1_deriv.append(grad(self.phase_cont_beta1,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.f_trans0))
        for i in range(9):
            self.beta0_deriv.append(grad(self.phase_cont_beta0,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1,self.f_trans0))
        for i in range(9):
            self.alpha1_deriv.append(grad(self.phase_cont_alpha1,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.f_trans1))
        for i in range(10):
            self.alpha0_deriv.append(grad(self.phase_cont_alpha0,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1,self.f_trans1))
        """Populate array with variables for transformation from d/d(theta) to d/d(log(theta)) - begins with 0 because fisher matrix variables start at 1, not 0"""
        self.log_factors = [0,self.A0,1,1,self.chirpm,self.symmratio,1,1,self.f_trans0,self.f_trans1]

    """Function for actual element integrand - 4*Re(dh/dtheta_i* dh/dtheta_j) - Vectorized
    -added extra mod_phi argument"""
    def calculate_waveform_derivative_vector(self,freq,i):
        
        famp = self.split_freqs_amp(freq)
        fphase = self.split_freqs_phase(freq)

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
        #

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
            gamp[j]=( egrad(ampfunc[j],i)(famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7],var_arr[8]))
            var_arr= self.var_arr[:]
            amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7],var_arr[8])
        for j in jphase:
            var_arr= self.var_arr[:]
            fphase[j], var_arr[i-1] = np.broadcast_arrays(fphase[j],var_arr[i-1])
            phase[j]=( egrad(phasefunc[j],i)(fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7],var_arr[8]))

        """Concatenate the regions into one array"""
        gampout,ampout,phaseout = [],[],[]
        for j in jamp:
            ampout = np.concatenate((ampout,amp[j]))
            gampout = np.concatenate((gampout,gamp[j]))
        for j in jphase:
            phaseout = np.concatenate((phaseout,phase[j]))

        """Return the complex waveform derivative"""
        return np.subtract(gampout,np.multiply(ampout,np.multiply(1j,phaseout)))

    """Calculate the waveform - vectorized
    Outputs: amp vector, phase vector, (real) waveform vector
    -added extra mod_phi argument"""
    def calculate_waveform_vector(self,freq):

        """Array of the functions used to populate derivative vectors"""
        ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
        phasefunc = [self.phase_ins_vector,self.phase_int_vector,self.phase_mr_vector]
        """Check to see if every region is sampled - if integration frequency
        doesn't reach a region, the loop is trimmed to avoid issues with unpopulated arrays"""

        famp = self.split_freqs_amp(freq)
        fphase = self.split_freqs_phase(freq)

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
            amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7],var_arr[8])
        for j in jphase:
            phase[j]=phasefunc[j](fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7],var_arr[8])
        """Concatenate the regions into one array"""
        ampout,phaseout =[],[]
        for j in jamp:
            ampout = np.concatenate((ampout,amp[j]))
        for j in jphase:
            if phase[j] is not None:
                phaseout = np.concatenate((phaseout,phase[j]))

        """Return the amplitude vector, phase vector, and real part of the waveform"""
        return ampout,phaseout, np.multiply(ampout,np.cos(phaseout))


    """Derivative Definitions - added phase_mod to derivatives"""
    @primitive
    def assign_beta1(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,f_trans0):
        for j in [chirpm,delta,symmratio,phic,tc,chi_a,chi_s,f_trans0]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.beta1
        return self.beta1
    defvjp(assign_beta1,None,
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[0],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[1],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[2],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[3],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[4],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[5],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[6],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[7])
    @primitive
    def assign_beta0(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,f_trans0):
        for j in [chirpm,delta,symmratio,phic,tc,chi_a,chi_s,beta1,f_trans0]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.beta0
        return self.beta0
    defvjp(assign_beta0,None,
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[0],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[1],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[2],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[3],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[4],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[5],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[6],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[7],
                lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[8])
    @primitive
    def assign_alpha1(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_trans1):
        for j in [chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,f_trans1]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.alpha1
        return self.alpha1
    defvjp(assign_alpha1,None,
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[0],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[1],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[2],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[3],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[4],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[5],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[6],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[7],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,phase_mod: lambda g: g*self.alpha1_deriv[8])
    @primitive
    def assign_alpha0(self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_trans1):
        for j in [chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,f_trans1]:
            if isinstance(j,np.ndarray):
                return np.ones(len(j))*self.alpha0
        return self.alpha0
    defvjp(assign_alpha0,None,
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[0],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[1],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[2],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[3],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[4],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[5],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[6],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[7],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[8],
                lambda ans,self,chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1,phase_mod: lambda g: g*self.alpha0_deriv[9])


class dCS_IMRPhenomD(Modified_IMRPhenomD_Inspiral_Freq):
    def __init__(self, mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,phase_mod = 0,
                    cosmo_model = cosmology.Planck15,NSflag = False,N_detectors = 1):
        super(dCS_IMRPhenomD,self).__init__(mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,phase_mod = phase_mod,bppe = -1,
                    cosmo_model = cosmo_model,NSflag = NSflag,N_detectors = N_detectors)
    def phi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        m = utilities.calculate_totalmass(chirpm,symmratio)
        return (IMRPhenomD.phi_ins(
                self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)
                + 16*np.pi*phase_mod * utilities.dCS_g(chirpm,symmratio,chi_s,chi_a)/m**4 * (np.pi * chirpm* f)**(self.bppe/3))
    def Dphi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        m = utilities.calculate_totalmass(chirpm,symmratio)
        return (IMRPhenomD.Dphi_ins(
                self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)
                + 16*np.pi*phase_mod * utilities.dCS_g(chirpm,symmratio,chi_s,chi_a)/m**4 * (np.pi * chirpm)**(self.bppe/3)*(self.bppe/3)*f**(self.bppe/3 -1))



class dCS_IMRPhenomD_detector_frame(Modified_IMRPhenomD_Inspiral_Freq_detector_frame):
    def __init__(self, mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,phase_mod = 0,
                    cosmo_model = cosmology.Planck15,NSflag = False,N_detectors = 1):
        super(dCS_IMRPhenomD_detector_frame,self).__init__(mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,phase_mod = phase_mod,bppe = -1,
                    cosmo_model = cosmo_model,NSflag = NSflag,N_detectors = N_detectors)
    def phi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        m = utilities.calculate_totalmass(chirpm,symmratio)
        return (IMRPhenomD.phi_ins(
                self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)
                + 16*np.pi*phase_mod * utilities.dCS_g(chirpm,symmratio,chi_s,chi_a)/m**4 * (np.pi * chirpm* f)**(self.bppe/3))
    def Dphi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        m = utilities.calculate_totalmass(chirpm,symmratio)
        return (IMRPhenomD.Dphi_ins(
                self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)
                + 16*np.pi*phase_mod * utilities.dCS_g(chirpm,symmratio,chi_s,chi_a)/m**4 * (np.pi * chirpm)**(self.bppe/3)*(self.bppe/3)*f**(self.bppe/3 -1))


#Phase_mod = alpha**2, as defined by arXiv:1603.08955v2
class EdGB_IMRPhenomD_detector_frame(Modified_IMRPhenomD_Inspiral_Freq_detector_frame):
    def __init__(self, mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,phase_mod = 0,
                    cosmo_model = cosmology.Planck15,NSflag = False,N_detectors = 1):
        super(EdGB_IMRPhenomD_detector_frame,self).__init__(mass1, mass2,spin1,spin2, collision_time,
                    collision_phase,Luminosity_Distance,phase_mod = phase_mod,bppe = -7,
                    cosmo_model = cosmo_model,NSflag = NSflag,N_detectors = N_detectors)
    def phi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        m = utilities.calculate_totalmass(chirpm,symmratio)
        m1 = utilities.calculate_mass1(chirpm,symmratio)
        m2 = utilities.calculate_mass2(chirpm,symmratio)
        chi1 = chi_s+ chi_a
        chi2 = chi_s- chi_a
        temp1 = 2*(np.sqrt(1-chi1**2) - 1 + chi1**2)
        temp2 = 2*(np.sqrt(1-chi2**2) - 1 + chi2**2)
        chi1 = chi1 + 1e-10
        chi2 = chi2 + 1e-10
        s1 =temp1 /chi1**2
        s2 = temp2/chi2**2
        #if chi1 != 0:
        #    s1 = 2*(np.sqrt(1-chi1**2) - 1 + chi1**2)/chi1**2
        #else:
        #    s1 = 0
        #if chi2 != 0:
        #    s2 = 2*(np.sqrt(1-chi2**2) - 1 + chi2**2)/chi2**2
        #else:
        #    s2 =0
        return (IMRPhenomD.phi_ins(
                self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)
                + (-5/7168)*16*np.pi*phase_mod/m**4 * (m1**2 * s2 - m2**2 * s1)**2 / (m**4 * symmratio**(18/5)) * (np.pi * chirpm* f)**(self.bppe/3))
    def Dphi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
        m = utilities.calculate_totalmass(chirpm,symmratio)
        m1 = utilities.calculate_mass1(chirpm,symmratio)
        m2 = utilities.calculate_mass2(chirpm,symmratio)
        chi1 = chi_s+ chi_a
        chi2 = chi_s- chi_a
        temp1 = 2*(np.sqrt(1-chi1**2) - 1 + chi1**2)
        temp2 = 2*(np.sqrt(1-chi2**2) - 1 + chi2**2)
        chi1 = chi1 + 1e-10
        chi2 = chi2 + 1e-10
        s1 =temp1 /chi1**2
        s2 = temp2/chi2**2
        #if chi1 != 0:
        #    s1 = 2*(np.sqrt(1-chi1**2) - 1 + chi1**2)/chi1**2
        #else:
        #    s1 = 0
        #if chi2 != 0:
        #    s2 = 2*(np.sqrt(1-chi2**2) - 1 + chi2**2)/chi2**2
        #else:
        #    s2 =0
        return (IMRPhenomD.Dphi_ins(
                self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)
                + (-5/7168)*16*np.pi*phase_mod/m**4 * (m1**2 * s2 - m2**2 * s1)**2 / (m**4 * symmratio**(18/5)) * (np.pi * chirpm)**(self.bppe/3)*(self.bppe/3)*f**(self.bppe/3 -1))
