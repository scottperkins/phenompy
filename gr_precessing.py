import utilities as util
import numpy as np
import gr 
import astropy.cosmology as cosmology
from astropy.coordinates import Distance
from astropy import units as u
import astropy.constants as consts
import precession_utilities as p_util
import matplotlib.pyplot as plt


c = util.c
G = util.G
s_solm = util.s_solm
mpc = util.mpc

"""PhenomPv2 - specify three spin components at a reference frequency fref"""
class IMRPhenomPv2(gr.IMRPhenomD):

    def __init__(self, mass1, mass2,spin1,spin2, collision_time, \
                    collision_phase,Luminosity_Distance,fref,cosmo_model = cosmology.Planck15,
                    NSflag = False,N_detectors=1):

        spin1_mag = np.sqrt(np.sum(np.asarray([x**2 for x in spin1])))
        spin2_mag = np.sqrt(np.sum(np.asarray([x**2 for x in spin2])))
        self.S1_vec_0 = spin1*m1**2
        self.S2_vec_0 = spin2*m2**2
        
        #probably need to feed the projections of the spin to PhenomD, not the magnitudes..
        super(IMRPhenomPv2,self).__init__( mass1=mass1, mass2=mass2,spin1=spin1[2],spin2=spin2[2],
                                             collision_time=collision_time, 
                                                collision_phase=collision_phase,
                                                Luminosity_Distance=Luminosity_Distance, 
                                                cosmo_model=cosmology.Planck15, NSflag=False,
                                                N_detectors=1)
        self.S1 = spin1_mag*m1**2#self.calculate_spin1(self.chi_s, self.chi_a)
        self.S2 = spin2_mag*m2**2#self.calculate_spin2(self.chi_s, self.chi_a)
        
        self.chi2l = spin2[2]
        self.chi2 = spin2_mag
        
        self.delta_m = self.calculate_delta_m(self.chirpm,self.symmratio)
        self.q = self.m2/self.m1
    

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

    def calculate_waveform_vector(self,frequencies):
        A_D, phi_D, h_D  = super(IMRPhenomPv2,self).calculate_waveform_vector(frequencies)
        h_D_complex =  A_D*np.exp(-1j* phi_D)
        
        alpha = p_util.alpha(np.pi*frequencies,self.q,self.chi2l,self.chi2)
        beta = p_util.beta(frequencies)
        epsilon = p_util.epsilon(np.pi*frequencies,self.q,self.chi2l,self.chi2)
        
        
        
        #Assuming the formula in the LIGO internal is wrong... eq(12)
        #m= +2 or -2 are the same, with a minus sign due to wignerD
        waveform_plus2 = p_util.wignerD(2,2,2,-beta) * np.exp(1j*(-2*epsilon - 2*alpha)) * h_D_complex
        waveform_plus1 = p_util.wignerD(2,2,1,-beta) * np.exp(1j*(-2*epsilon - 1*alpha)) * h_D_complex
        waveform_0 = p_util.wignerD(2,2,0,-beta) * np.exp(1j*(-2*epsilon )) * h_D_complex
        
        #print(waveform_plus2,waveform_plus1,waveform_0)

        waveform_minus2 = waveform_plus2
        waveform_minus1 = waveform_plus1
        #waveform_minus2 = p_util.wignerD(2,2,-2,-beta) * np.exp(1j*(-2*epsilon - 2*alpha)) * h_D_complex
        return waveform_plus2

     
        
"""Defined by the following papers: arXiv:1809.10113, arXiv:1703.03967, arXiv:1606.03117"""
#Note: spin1 and spin2 are the chi parameters - DIMENSIONLESS
#Currently, spin1 and spin2 are defined in the CO-PRECESSING frame, so S1y and S2y MUST cancel - this will change
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
        
        #Convert from seconds to units M = 1
        chirpm = self.chirpm/self.M
        m1 = self.m1/self.M
        m2 = self.m2/ self.M
        delta_m = self.delta_m/self.M
        S1_vec_0 = self.S1_vec_0/self.M**2
        S2_vec_0 = self.S2_vec_0/self.M**2
        S1 = self.S1/self.M**2
        S2 = self.S2/self.M**2
        f = f*self.M
        
        
        ############################################################################
        #Find initial conditions to populate conserved quantities xi and c1
        f0 = f[0]
        L0 = p_util.L(self.symmratio, f0, M=1)
        theta_L_0 = np.arccos((S1_vec_0[0]+S2_vec_0[0])/L0)
        L_vec_0 = p_util.L_vec(L0, theta_L_0)
        J0 = p_util.J0(L0,S1_vec_0,S2_vec_0)
        v0 = p_util.v(f0, M=1)
        ############################################################################

        xi = p_util.xi(m1 = m1, m2 = m2, S1=S1_vec_0, S2 = S2_vec_0, L = L_vec_0)

        ############################################################################
        #Calculate S_plus_0 and S_minus_0
        A0 = p_util.A(eta=self.symmratio, xi =  xi, v = v0)
        B0 = p_util.B(L=L0,S1=S1, S2=S2, J=J0, xi =  xi, q = self.q)
        C0 = p_util.C(L=L0,S1=S1, S2=S2, J=J0, xi =  xi,delta_m=delta_m, 
                        q = self.q, eta = self.symmratio)
        D0 = p_util.D(L=L0,S1=S1, S2=S2, J=J0, xi =  xi,delta_m=delta_m, 
                        q = self.q, eta = self.symmratio)
        print("coeffs")
        print(A0,B0,C0,D0)
        S_plus_0,  S_minus_0,  S3_0 = p_util.calculate_S_roots(A0,B0,C0,D0)
        Sav = p_util.Sav( S_plus_0, S_minus_0)
        print("S plus, minus, 3:")
        print( S_plus_0, S_minus_0,  S3_0)
        print("Sav:")
        print( Sav)
        ############################################################################
        
        c1 = p_util.c1(L0,J0, Sav, v0)
    
        ############################################################################
        #NOTE: sigma4 depends on frequency - needs to be calculated every iteration - initialized with intital
        #values just for code structure
        mu = m1*m2 #In M=1 units
        beta3 = p_util.beta3( m1, m2, c1,self.q,self.symmratio, xi)
        beta5 = p_util.beta5( m1,  m2,  c1, self.q, self.symmratio,  xi)
        beta6 = p_util.beta6( m1, m2, c1,self.q,self.symmratio, xi)
        beta7 = p_util.beta7( m1, m2, c1,self.q,self.symmratio, xi)
        sigma4 = p_util.sigma4( m1,  m2,  mu,  Sav,  S1,  S2,  c1, 
                                   self.q, self.symmratio, xi,  S_plus_0,  S_minus_0, v0)

        a0 = p_util.a0(self.symmratio)
        a1 = p_util.a1(self.symmratio) 
        a2 = p_util.a2(self.symmratio,  xi)
        a3 = p_util.a3( beta3)
        a4 = p_util.a4(self.symmratio, sigma4)
        a5 = p_util.a5(self.symmratio, beta5)
        a6 = p_util.a6(self.symmratio,  beta6)
        a7 = p_util.a7(self.symmratio, beta7)
        
        g0 = p_util.g0( a0)
        g2 = p_util.g2( a2, a0)
        g3 = p_util.g3( a3, a0)
        g4 = p_util.g4( a4, a2, a0)
        g5 = p_util.g5( a5, a3, a2, a0)
        g6 = p_util.g6( a6, a4, a3, a2, a0)
        g6_l = p_util.g6_l( a0)
        g7 = p_util.g7( a7, a5, a4, a3, a2, a0)
        
        g_vec = [ g0, g2, g3, g4, g5, g6, g6_l, g7]
        a_vec = [ a0, a1, a2, a3, a4, a5, a6, a7]
        beta_vec = [ beta3, beta5, beta6, beta7, sigma4]
        ############################################################################
        Delta = p_util.Delta(self.symmratio, self.q,  delta_m, xi, S1, S2, c1)
        psi1 = p_util.psi1( xi,self.symmratio, c1,  delta_m)
        psi2 = p_util.psi2(self.symmratio,  delta_m, self.q, Delta, S1, S2, Sav, xi,
                                 g0, g2, c1)
        phi_z_vec = []
        for x in f:
            phi_z_vec.append(self.phi_z( x, chirpm, self.symmratio, S1,S2,xi, 
                            Sav, c1,psi1,psi2,g_vec))
        return A_D
        
    def phi_z(self, f, chirpm, symmratio, S1,S2,xi, Sav, c1,psi1,psi2,g_vec):
        g0 = g_vec[0]
        g2 = g_vec[1]
        g3 = g_vec[2]
        g4 = g_vec[3]
        g5 = g_vec[4]
        #M = self.assign_totalmass(chirpm, symmratio)
        #S1 = self.assign_spin1(chi_s,chi_a)
        #S2 = self.assign_spin2(chi_s,chi_a)
        delta_m = self.assign_delta_m(chirpm,symmratio)
        m1 = self.assign_mass1(chirpm,symmratio)
        m2 = self.assign_mass2(chirpm,symmratio)
        q = m2/m1

        v = p_util.v(f=f, M=1)
        L =p_util.L(eta=symmratio, f=f, M = 1) 
        J = p_util.J(L,Sav,c1,v)
        #J_vec = np.asarray([0,0,J])

        A = p_util.A(eta=symmratio, xi = xi, v = v)
        B = p_util.B(L=L,S1=S1, S2=S2, J=J, xi = xi, q = q)
        C = p_util.C(L=L,S1=S1, S2=S2, J=J, xi = xi,delta_m=delta_m, q = q, eta = symmratio)
        D = p_util.C(L=L,S1=S1, S2=S2, J=J, xi = xi,delta_m=delta_m, q = q, eta = symmratio)

        #S_plus = np.sqrt( p_util.S_plus_2(A,B,C,D))
        #S_minus = np.sqrt( p_util.S_minus_2(A,B,C,D))
        S_plus, S_minus, S3 = p_util.calculate_S_roots(A,B,C,D)
        
        #psi0 and integration constant?? Setting to 0
        psi0 = 0
        psi = p_util.psi(psi0,psi1,psi2,g0,delta_m,v)
        m = p_util.m(S_plus,S_minus,S3)

        S = np.sqrt(p_util.S2(S_plus,S_minus,psi, m))
    

        theta_L = p_util.theta_L(J2 = J**2, L2 = L**2, S2 = S**2)
        L_vec = p_util.L_vec(L, theta_L)

        

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
        #S2_vec = np.subtract( np.subtract(J_vec,L_vec,),S1_vec)
        

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
    m1 = 6*s_solm
    m2 = 4*s_solm
    #spin1 =  np.asarray([-.2,.1,.2])
    #spin2 = np.asarray([-.3,-.1,.1])
    spin1 =  np.asarray([0,0,.2])
    spin2 = np.asarray([0,0,.1])
    dl = 100*mpc
    model = IMRPhenomPv2(mass1=m1,mass2=m2, spin1=spin1,spin2=spin2,collision_time=0,collision_phase=0, 
            Luminosity_Distance=dl, NSflag = False, N_detectors=1,fref=10)
    frequencies = np.arange(10,200,.01)
    waveform = model.calculate_waveform_vector(frequencies)
    model2 = gr.IMRPhenomD(mass1=m1,mass2=m2, spin1=spin1[2],spin2=spin2[2],collision_time=0,collision_phase=0, 
            Luminosity_Distance=dl, NSflag = False, N_detectors=1)
    a,p , waveform2 = model2.calculate_waveform_vector(frequencies)
    waveform2 = a*np.exp(1j*p)
    #plt.loglog(frequencies,waveform.real,label="Pv2")
    #plt.loglog(frequencies,waveform2.real,label='D')
    plt.plot(frequencies,waveform.real,label="Pv2")
    plt.plot(frequencies,waveform2.real,label='D')
    plt.legend()
    plt.show()
    plt.close()
    
    #model.phi_z(10., model.chirpm, model.symmratio, model.chi_s, model.chi_a)
