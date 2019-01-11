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
        self.S1_vec_0 = spin1*m1**2
        self.S2_vec_0 = spin2*m2**2
        
        #probably need to feed the projections of the spin to PhenomD, not the magnitudes..
        super(IMRPhenomPv3,self).__init__( mass1=mass1, mass2=mass2,spin1=spin1[2],spin2=spin2[2],
                                             collision_time=collision_time, 
                                                collision_phase=collision_phase,
                                                Luminosity_Distance=Luminosity_Distance, 
                                                cosmo_model=cosmology.Planck15, NSflag=False,
                                                N_detectors=1)
        self.S1 = spin1_mag*m1**2#self.calculate_spin1(self.chi_s, self.chi_a)
        self.S2 = spin2_mag*m2**2#self.calculate_spin2(self.chi_s, self.chi_a)
        
        self.delta_m = self.calculate_delta_m(self.chirpm,self.symmratio)
        self.q = self.m2/self.m1

        

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
    
    def calculate_waveform_vector(self, f):
        A_D, phi_D, h_D  = super(IMRPhenomPv3,self).calculate_waveform_vector(f)
        h_D_complex =  A_D*np.exp(1j* phi_D)
        
        ############################################################################
        #Find initial conditions to populate conserved quantities xi and c1
        f0 = f[0]
        L0 = p_util.L(self.symmratio, f0, self.M)
        theta_L_0 = np.arccos((self.S1_vec_0[0]+self.S2_vec_0[0])/L0)
        print(L0)
        L_vec_0 = p_util.L_vec(L0, theta_L_0)
        J0 = p_util.J0(L0,self.S1_vec_0,self.S2_vec_0)
        v0 = p_util.v(f0, self.M)
        ############################################################################

        self.xi = p_util.xi(m1 = self.m1, m2 = self.m2, S1=self.S1_vec_0, S2 = self.S2_vec_0, L = L_vec_0)

        ############################################################################
        #Calculate S_plus_0 and S_minus_0
        A0 = p_util.A(eta=self.symmratio, xi = self.xi, v = v0)
        B0 = p_util.B(L=L0,S1=self.S1, S2=self.S2, J=J0, xi = self.xi, q = self.q)
        C0 = p_util.C(L=L0,S1=self.S1, S2=self.S2, J=J0, xi = self.xi,delta_m=self.delta_m, 
                        q = self.q, eta = self.symmratio)
        D0 = p_util.C(L=L0,S1=self.S1, S2=self.S2, J=J0, xi = self.xi,delta_m=self.delta_m, 
                        q = self.q, eta = self.symmratio)
        self.S_plus_0, self.S_minus_0, self.S3_0 = p_util.calculate_S_roots(A0,B0,C0,D0)
        self.Sav = p_util.Sav(self.S_plus_0,self.S_minus_0)
        ############################################################################
        self.c1 = p_util.c1(L0,J0,self.Sav, v0)
    
        ############################################################################
        #NOTE: sigma4 depends on frequency - needs to be calculated every iteration
        self.mu = self.m1*self.m2/self.M
        self.beta3 = p_util.beta3(self.m1,self.m2,self.c1,self.q,self.symmratio,self.xi)
        self.beta5 = p_util.beta5(self.m1, self.m2, self.c1, self.q, self.symmratio, self.xi)
        self.beta6 = p_util.beta6(self.m1,self.m2,self.c1,self.qmself.symmratio,self.xi)
        self.beta7 = p_util.beta7(self.m1,self.m2,self.c1,self.qmself.symmratio,self.xi)
        self.sigma4 = p_util.sigma4(self.m1, self.m2, self.mu, self.Sav, self.S1, self.S2, self.c1, 
                                    self.q, self.symmratio,self.xi, self.S_plus, self.S_minus, self.v0)

        self.a0 = p_util.a0(self.symmratio)
        self.a1 = p_util.a1(self.symmratio) 
        self.a2 = p_util.a2(self.symmratio, self.xi)
        self.a3 = p_util.a3(self.beta3)
        self.a4 = p_util.a4(self.symmratio,self.sigma4)
        self.a5 = p_util.a5(self.symmratio,self.beta5)
        self.a6 = p_util.a6(self.symmratio, self.beta6)
        self.a7 = p_util.a7(self.symmratio,self.beta7)
        
        self.g0 = p_util.g0(a0)
        self.g2 = p_util.g2(a2,a0)
        self.g3 = p_util.g3(a3,a0)
        self.g4 = p_util.g4(a4,a2,a0)
        self.g5 = p_util.g5(a5,a3,a2,a0)
        self.g6 = p_util.g6(a6,a4,a3,a2,a0)
        self.g6_l = p_util.g6_l(a0)
        self.g7 = p_util.g7(a7,a5,a4,a3,a2,a0)
        
        g_vec = [self.g0,self.g2,self.g3,self.g4,self.g5,self.g6,self.g6_l,self.g7]
        a_vec = [self.a0,self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.a7]
        beta_vec = [self.beta3,self.beta5,self.beta6,self.beta7,self.sigma4]
        ############################################################################
        self.Delta = p_util.Delta(self.symmratio, self.q, self.delta_m,self.xi,self.S1,self.S2,self.c1)
        self.psi1 = p_util.psi1(self.xi,self.symmratio,self.c1, self.delta_m)
        self.psi2 = p_util.psi2(self.symmratio, self.delta_m,self.q,self.Delta,self.S1,self.S2,self.Sav,self.xi,
                                self.g0,self.g2,self.c1)
        return A_D
        
    def phi_z(self, f, chirpm, symmratio, S1,S2,xi, Sav, c1,psi1,psi2,g_vec):
        g0 = g_vec[0]
        g2 = g_vec[1]
        g3 = g_vec[2]
        g4 = g_vec[3]
        g5 = g_vec[4]
        M = self.assign_totalmass(chirpm, symmratio)
        #S1 = self.assign_spin1(chi_s,chi_a)
        #S2 = self.assign_spin2(chi_s,chi_a)
        delta_m = self.assign_delta_m(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        q = m2/m1

        v = p_util.v(f=f, M=M)
        L =p_util.L(eta=symmratio, f=f, M = M) 
        J = p_util.J(L,Sav,c1,v)
        J_vec = np.asarray([0,0,J])

        theta_L = p_util.theta_L(J2 = J**2, L2 = L**2, S2 = S**2)
        L_vec = p_util.L_vec(L, theta_L)

        
        A = p_util.A(eta=symmratio, xi = xi, v = v)
        B = p_util.B(L=L,S1=S1, S2=S2, J=J, xi = xi, q = q)
        C = p_util.C(L=L,S1=S1, S2=S2, J=J, xi = xi,delta_m=delta_m, q = q, eta = symmratio)
        D = p_util.C(L=L,S1=S1, S2=S2, J=J, xi = xi,delta_m=delta_m, q = q, eta = symmratio)

        #S_plus = np.sqrt( p_util.S_plus_2(A,B,C,D))
        #S_minus = np.sqrt( p_util.S_minus_2(A,B,C,D))
        S_plus, S_minus, S3 = p_util.calculate_S_roots(A,B,C,D)
        
        psi = p_util.psi(psi0,psi1,psi2,g0,delta_m,v)

        S = np.sqrt(S2(S_plus,S_minus,psi, m))

        #S_plus_0 = self.assign_S_plus_0()
        #S_minus_0 = self.assign_S_minus_0()
        #Sav = p_util.Sav(S_plus_0=S_plus_0, S_minus_0=S_minus_0)

        theta_prime = p_util.theta_prime(S, S1, S2)
        phi_prime = p_util.phi_prime(J, L, S, q, S1, S2)
        theta_s = p_util.theta_s(J,S,L)

        S1prime_vec = S1prime_vec(S1, theta_prime, phi_prime) 
        rotmat = p_util.rot_y(theta_s)
        
        #Spin vectors in non-inertial, co-precessing frame (about J_vec angle phi_z)
        S1_vec = np.matmul(rotmat, s1prime_vec)
        S2_vec = np.subtract( np.subtract(J_vec,L_vec,),S1_vec)
        

        ##################################################################
        l1 = p_util.l1(J,L,c1,symmratio)
        l2 = p_util.l2(J,Sav,c1,v)
        
        R_m = p_util.R_m(S_plus,S_minus)
        cp = p_util.cp(S_plus,c1,symmratio)
        cm = p_util.cm(S_minus,c1,symmratio)
        ad = p_util.ad(S1, S2, symmratio, delta_m, c1, xi, cp, cm)
        cd = p_util.cd(c1, eta, cp, cm, Rm)
        hd = p_util.hd(c1, eta, cp, cm)
        fd = p_util.fd(cp, cm, symmratio)

        Oz0 = p_util.Omega_z_0(a1, ad)
        Oz1 = p_util.Omega_z_1(a2, ad, xi, hd)
        Oz2 = p_util.Omega_z_2(ad, hd, xi, cd, fd)
        Oz3 = p_util.Omega_z_3(ad, fd, cd, hd, xi)
        Oz4 = p_util.Omega_z_4(cd, ad, hd, fd, xi)
        Oz5 = p_util.Omega_z_5(cd, ad, fd, hd, xi)

        Oz0avg = p_util.Omega_z_0_avg(g0, Oz0)
        Oz1avg = p_util.Omega_z_1_avg(g0, Oz1)
        Oz2avg = p_util.Omega_z_2_avg(g0,g2, Oz2, Oz0)
        Oz3avg = p_util.Omega_z_3_avg(g0, g2, g3,Oz3, Oz1, Oz0)
        Oz4avg = p_util.Omega_z_4_avg(g0,g2,g3,g4, Oz4, Oz2,  Oz1, Oz0)
        Oz5avg = p_util.Omega_z_5_avg(g0, g2, g3,g4,g5, Oz5, Oz3, Oz2, Oz1, Oz0)
        Ozavg_vec = [Oz0avg,Oz1avg,Oz2avg,Oz3avg,Oz4avg,Oz5avg]
    
        pz0 =p_util.phi_z_0(J = J, eta = symmratio, c1=c1,Sav=Sav, l1=l1, v=v)
        pz1 =p_util.phi_z_1(J = J, eta = symmratio, c1=c1, L=L, Sav=Sav, l1=l1)
        pz2 =p_util.phi_z_2(J = J, Sav=Sav, c1=c1, eta=symmratio,l1=l1, l2=l2)
        pz3 =p_util.phi_z_3(J = J, Sav=Sav, c1=c1, eta=symmratio,l1=l1, l2=l2,v=v)
        pz4 =p_util.phi_z_4(J = J, Sav=Sav, c1=c1, eta=symmratio,l1=l1, l2=l2,v=v)
        pz5 =p_util.phi_z_5(J = J, Sav=Sav, c1=c1, eta=symmratio,l1=l1, l2=l2,v=v)
        pz_vec = [pz0,pz1,pz2,pz3,pz4,pz5]
    
        ##################################################################
        #Calculate phi_z_0 
        psi_dot = p_util.psi_dot(A,S_plus,S3)
        c0 = p_util.c0(symmratio,xi,delta_m,J,S_plus,S_minus,S1,S2,v)
        c2 = p_util.c2(symmratio,xi,J,S_plus,S_minus,v)
        c4 = p_util.c4(symmratio, xi, S_plus, S_minus, v)

        d0 = p_util.d0(J,L, S_plus)
        d2 = p_util.d2(J,L,S_plus,S_minus )
        d4 = p_util.d4( S_plus,S_minus)

        sd = p_util.sd(d0,d2,d4)
        
        C1 = p_util.C1(c0,c2,c4,d0,d2,d4)
        C2 = p_util.C2(c0,c2,c4,d0,d2,d4,sd)
        
        Dphi = p_util.Dphi(C1,C2)
        Cphi = p_util.Cphi(C1,C2)
        
        nc = p_util.nc(d0,d2,d4,sd)
        nd = p_util.nd(d0,d2,sd)

        phi_z_0_0 = p_util.phi_z_0_0(Cphi,Dphi,nc,nd,psi_dot,psi)
        ##################################################################
        phi_z_minus1 = 0 
        for i in np.arange(len(pz_vec)):
            phi_z_minus1+= Ozavg_vec[i]*pz_vec[i]

        #integration constant?? 
        phi_z_minus1_0 =0
        phi_z_minus1+= phi_z_minus1_0 
        
        phi_z = phi_z_minus1 + phi_z_0_0
        return phi_z 



######################################################################################################

if __name__=='__main__':
    m1 = 10*s_solm
    m2 = 5*s_solm
    spin1 = m1**2 *np.asarray([.2,.1,.2])
    spin2 = m2**2 *np.asarray([.6,.1,.1])
    dl = 100*mpc
    model = IMRPhenomPv3(mass1=m1,mass2=m2, spin1=spin1,spin2=spin2,collision_time=0,collision_phase=0, Luminosity_Distance=dl, NSflag = False, N_detectors=1)
    frequencies = np.arange(1,1000,1)
    model.calculate_waveform_vector(frequencies)
    #model.phi_z(10., model.chirpm, model.symmratio, model.chi_s, model.chi_a)
