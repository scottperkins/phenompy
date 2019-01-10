import utilities as util
import numpy as np
import gr 
import astropy.cosmology as cosmology
from astropy.coordinates import Distance
from astropy import units as u
import astropy.constants as consts
import precession_utilities as p_util


c = util.c
G = util.G
s_solm = util.s_solm
mpc = util.mpc

"""Defined by the following papers: arXiv:1809.10113, arXiv:1703.03967, arXiv:1606.03117"""
class IMRPhenomPv3(gr.IMRPhenomD):

    def __init__(self, mass1, mass2,spin1,spin2, collision_time, \
                    collision_phase,Luminosity_Distance,cosmo_model = cosmology.Planck15,
                    NSflag = False,N_detectors=1):

        spin1_mag = np.sqrt(np.sum(np.asarray([x**2 for x in spin1])))
        spin2_mag = np.sqrt(np.sum(np.asarray([x**2 for x in spin2])))
        spin1_vec = spin1
        spin2_vec = spin2
        super(IMRPhenomPv3,self).__init__( mass1=mass1, mass2=mass2,spin1=spin1_mag,spin2=spin2_mag,
                                             collision_time=collision_time, 
                                                collision_phase=collision_phase,
                                                Luminosity_Distance=Luminosity_Distance, 
                                                cosmo_model=cosmology.Planck15, NSflag=False,
                                                N_detectors=1)
        self.S1 = self.calculate_spin1(self.chi_s, self.chi_a)
        self.S2 = self.calculate_spin2(self.chi_s, self.chi_a)
        
        xi = p_util.xi(m1 = m1, m2 = m2, S1=S1_vec, S2 = S2_vec, L = L_vec)
        self.delta_m = self.calculate_delta_m(self.chirpm,self.symmratio)

        #NEEDS TO CHANGE - just place holders - these parameters will need to be evaluated once
        self.S_plus_0 = 1.
        self.S_minus_0 = 1.
        

    #Will need to change
    def assign_S_plus_0(self):
        return self.S_plus_0
    def assign_S_minus_0(self):
        return self.S_minus_0

    def calculate_spin1(self,chi_s,chi_a):
        return chi_s + chi_a
    def calculate_spin2(self,chi_s,chi_a):
        return chi_s - chi_a
    def assign_spin1(self, chi_s,chi_a):
        return self.S1
    def assign_spin2(self, chi_s,chi_a):
        return self.S2

    def calculate_delta_m(self,chirpm,symmratio):
        m1 = util.calculate_mass1(chirpm,symmratio)
        m2 = util.calculate_mass2(chirpm,symmratio)
        return m1-m2
    def assign_delta_m(self,chirpm,symmratio):
        return self.delta_m


    def phi_z(self, f, chirpm, symmratio, chi_s, chi_a):
        M = self.assign_totalmass(chirpm, symmratio)
        S1 = self.assign_spin1(chi_s,chi_a)
        S2 = self.assign_spin2(chi_s,chi_a)
        delta_m = self.assign_delta_m(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        q = m2/m1
        xi = self.assign_xi(m1=m1,m2=m2,S1 = S1_vec_0, S2 = S2_vec_0, L=L_0)

        L =p_util.L(eta=symmratio, f=f, M = M) 

        theta_L = p_util.theta_L(J2 = J**2, L2 = L**2, S2 = S**2)
        L_vec = p_util.L_vec(L, theta_L)

        
        A = p_util.A(eta=symmratio, xi = xi, v = v)
        B = p_util.B(L=L,S1=S1, S2=S2, J=J, xi = xi, q = q)
        C = p_util.C(L=L,S1=S1, S2=S2, J=J, xi = xi,delta_m=delta_m, q = q, eta = symmratio)
        D = p_util.C(L=L,S1=S1, S2=S2, J=J, xi = xi,delta_m=delta_m, q = q, eta = symmratio)

        S_plus = np.sqrt( p_util.S_plus_2(A,B,C,D))
        S_minus = np.sqrt( p_util.S_minus_2(A,B,C,D))

        S_plus_0 = self.assign_S_plus_0()
        S_minus_0 = self.assign_S_minus_0()
        Sav = p_util.Sav(S_plus_0=S_plus_0, S_minus_0=S_minus_0)
        v = p_util.v(f=f, M=M)

        theta_prime = p_util.theta_prime(S, S1, S2)
        phi_prime = p_util.phi_prime(J, L, S, q, S1, S2)

        S1prime_vec(S1, theta_prime, phi_prime) 

        pz0 =p_util.phi_z_0(J, eta, c1,Sav, l1, v)
        pz1 =p_util.phi_z_1(J, eta, c1, L, Sav, l1)
        pz2 =p_util.phi_z_2(J, Sav, c1, eta, l1,l2)
        pz3 =p_util.phi_z_3(J, Sav, c1, eta,l1, l2,v)
        pz4 =p_util.phi_z_4(J, Sav, c1, eta,l1, l2,v)
        pz5 =p_util.phi_z_5(J, Sav, c1, eta,l1, l2,v)
        return phi_z 



######################################################################################################

if __name__=='__main__':
    m1 = 10*s_solm
    m2 = 5*s_solm
    spin1 = np.asarray([.3,.4,.5])
    spin2 = np.asarray([.6,.1,.1])
    dl = 100*mpc
    model = IMRPhenomPv3(mass1=m1,mass2=m2, spin1=spin1,spin2=spin2,collision_time=0,collision_phase=0, Luminosity_Distance=dl, NSflag = False, N_detectors=1)
    #model.phi_z(10., model.chirpm, model.symmratio, model.chi_s, model.chi_a)
