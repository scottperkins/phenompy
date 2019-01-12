import autograd.numpy as np
import utilities as util
from scipy.special import ellipj

"""A variety of utilities for the construction of the precessing model IMRPhenomPv3:"""

def wignerD(l,m,m_prime,beta):
    k_sum = 0
    k = 0
    #factorial arguments - must be positive in sum
    check1 = l+m -k
    check2 = l-m-k
    check3 = m_prime-m+k
        
###############################################################################################
#Foundational quantities PHYSICAL REVIEW D 95, 104004
#Defining c1 from the initial parameters
def c1(L0,J0,Sav, v0):
    return (J0**2 -L0**2 - Sav**2)*(v0/2)

#All values are the initial values from the input parameters
#Returns the initial magnitude of total angular momentum
#form of S1_vec_0 = [S1_vec_0_x,S1_vec_0_y, S1_vec_0_z]
def J0(L0,S1_vec_0,S2_vec_0):
    theta_L_0 = np.arccos((S1_vec_0[0]+S2_vec_0[0])/L0)
    return L0*np.sin(theta_L_0) + S1_vec_0[2] + S2_vec_0[2]
   
#Calculate J magnitude 
def J(L,Sav,c1,v):
    return L**2 + (2/v)*c1 + Sav**2
#S1, S2, and L are 3D vectors - m1, m2 are scalars and in seconds
def xi(m1,m2,S1, S2,L): 
    S1 = np.asarray(S1)
    S2 = np.asarray(S2)
    L = np.asarray(L)
    sumL = 0
    L_mag = np.sqrt( np.sum(np.asarray( [x**2 for x in L]) ))
    L_hat = L/L_mag
    q = m2/m1
    return ((1 + q) * np.sum(np.asarray( [S1[i] * L_hat[i] for i in np.arange( len(L_hat) )] ) )+
             ( 1 + 1/q) *np.sum(np.asarray( [S2[i] * L_hat[i] for i in np.arange( len(L_hat) )] ) ))


#J2, L2, and S2 are  magnitudes
def theta_L(J2, L2, S2):
    return np.arccos((J2 + L2 - S2)/(2 * np.sqrt(J2*L2)))

#calculates the magnitude of the orbital angular momentum
def L(eta, f,M):
    return (2*np.pi*M*f)**(-1/3)*eta*M**2

#PN expansion coefficient defined in PhysRevD.95.104004 - dimensionless
def v(f,M):
    return (2 * np.pi*M * f)**(1/3)

#s1 and s2 are vectors
def S2(s1,s2):
    stotal= np.asarray([s1[x]+s2[x] for x in np.arange(len(s1))])
    return np.sum(np.asarray([x**2 for x in stotal]))
#overloaded method of above for calculating based on definitions in PhysRevD.95.104004 
def S2(S_plus, S_minus, psi, m):
    return S_plus**2 + (S_minus**2 - S_plus**2 )*ellipj(psi, m)

#spin parameter for calculating S**2
def m(S_plus,S_minus,S3):
    return (S_plus**2 - S_minus**2)/(S_plus**2 - S3**2)

def psi(psi0, psi1, psi2, g0, delta_m, v):
    return psi0 - (3*g0)/4 * delta_m * v**(-3) * (1 + psi1*v + psi2 * v**2)

def psi1(xi, eta, c1, delta_m):
    return 3 * (2*xi * eta**2 - c1)/(eta * delta_m**2)

def psi2(eta, delta_m, q, Delta, S1, S2, Sav, xi, g0, g2,c1):
    term1 = 3*g2/g0 
    term2 = 2*Delta
    term3 = -2*eta**2*Sav**2/delta_m**2
    term4 = -10 * eta * c1**2/delta_m**4
    term5 = 2*eta**2/delta_m**2 * (7 + 6*q + 7 *q**2)/(1-q)**2 * c1 * xi
    term6 = -eta**3/delta_m**2 * (3 + 4*q + 3*q**2)/(1-q)**2 * xi**2
    term7 = eta/(1-q)**2 * ( q * (2+q)*S1**2 + (1+2*q)*S2**2)
    return term1 + 3/(2*eta**3) * (term2 + term3 + term4 + term5 + term6 + term7) 

def Delta(eta, q, delta_m, xi, S1, S2, c1):
    term1 = c1**2* eta/(q*delta_m**4)
    term2 = -2* c1 * eta**3 *(1+q)*xi/(q*delta_m**4)
    term3 = -eta**2 / delta_m**4 * ( delta_m**2 * S1**2 - eta**2 * xi**2)
    term4 = c1**2*eta**2/delta_m**4
    term5 = -2* c1 * eta**3 *(1+q)*xi/(delta_m**4)
    term6 = -eta**2 / delta_m**4 * ( delta_m**2 * S2**2 - eta**2 * xi**2)
    return np.sqrt( (term1 + term2 + term3) * (term4 + term5 + term6) )

#L is a magnitude
def L_vec(L, theta_L):
    return np.asarray([L*np.sin(theta_L), 0, L*np.cos(theta_L)])     

#S1 is a magnitude
def S1prime_vec(S1, theta_prime, phi_prime):
    return S1*np.asarray(   [np.sin(theta_prime)*np.cos(phi_prime), 
                            np.sin(theta_prime)*np.sin(phi_prime),
                            np.cos(phi_prime)])

def theta_prime(S, S1, S2):
    return np.arccos((S**2 + S1**2 + S2**2)/(2*S*S1))

def phi_prime(J,L,S,q,S1,S2):
    A1 = np.sqrt(J**2 - (L-S)**2)
    A2 = np.sqrt((L+S)**2 - J**2)
    A3 = np.sqrt(S**2 - (S1-S2)**2)
    A4 = np.sqrt((S1+S2)**2 - S**2)
    term1 = J**2 - L**2 - S**2
    term2 = S**2 * (1+q)**2
    term3 = -(S1**2 - S2**2)*(1-q**2)
    term4 = -4*q*S**2 * L * xi
    term5 = (1-q**2)*A1*A2*A3*A4
    return np.arccos( (term1*(term2 + term3) + term4)/term5)

def theta_s(J, S, L):
    return np.arccos( (J**2 + S**2 - L**2)/(2*J*S) )

def rot_y(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.asarray(  [[c, 0, s],
                        [0,1,0],
                        [-s, 0, c]])
###############################################################################################
#Finding S+ and S- PHYSICAL REVIEW D 95, 104004
def S1_2(A,B,C,D):
    return (-B/3. - (2**0.3333333333333333*(-B**2 + 3*C))/
            (3.*(-2*B**3 + 9*B*C - 27*D + 3*np.sqrt(3)*
                  np.sqrt(-(B**2*C**2) + 4*C**3 + 4*B**3*D - 18*B*C*D + 27*D**2))**0.3333333333333333) + 
           (-2*B**3 + 9*B*C - 27*D + 3*np.sqrt(3)*
                np.sqrt(-(B**2*C**2) + 4*C**3 + 4*B**3*D - 18*B*C*D + 27*D**2))**0.3333333333333333/
            (3.*2**0.3333333333333333))
def S2_2(A,B,C,D):
    return (-B/3. + ((1 + 1j*np.sqrt(3))*(-B**2 + 3*C))/
        (3.*2**0.6666666666666666*(-2*B**3 + 9*B*C - 27*D + 
             3*np.sqrt(3)*np.sqrt(-(B**2*C**2) + 4*C**3 + 4*B**3*D - 18*B*C*D + 27*D**2))**
           0.3333333333333333) - ((1 - (1j)*np.sqrt(3.))*
          (-2*B**3 + 9*B*C - 27*D + 3*np.sqrt(3)*
              np.sqrt(-(B**2*C**2) + 4*C**3 + 4*B**3*D - 18*B*C*D + 27*D**2))**0.3333333333333333)/
        (6.*2**0.3333333333333333))
def S3_2(A,B,C,D):
    return (-B/3. + ((1 - 1j*np.sqrt(3))*(-B**2 + 3*C))/
        (3.*2**0.6666666666666666*(-2*B**3 + 9*B*C - 27*D + 
             3*np.sqrt(3)*np.sqrt(-(B**2*C**2) + 4*C**3 + 4*B**3*D - 18*B*C*D + 27*D**2))**
           0.3333333333333333) - ((1 + (1j)*np.sqrt(3))*
          (-2*B**3 + 9*B*C - 27*D + 3*np.sqrt(3)*
              np.sqrt(-(B**2*C**2) + 4*C**3 + 4*B**3*D - 18*B*C*D + 27*D**2))**0.3333333333333333)/
        (6.*2**0.3333333333333333))
#def S1_2(A,B,C,D):
#    return (-2*B + (2*np.power(2,0.3333333333333333)*(np.power(B,2) - 3*C))/
#      np.power(-2*np.power(B,3) + 9*B*C + 3*(-9*D + 
#           np.sqrt(-3*(np.power(B,2) - 4*C)*np.power(C,2) + 6*B*(2*np.power(B,2) - 9*C)*D + 81*np.power(D,2))),
#       0.3333333333333333) + np.power(2,0.6666666666666666)*
#      np.power(-2*np.power(B,3) + 9*B*C + 3*(-9*D + 
#           np.sqrt(-3*(np.power(B,2) - 4*C)*np.power(C,2) + 6*B*(2*np.power(B,2) - 9*C)*D + 81*np.power(D,2))),
#       0.3333333333333333))/6.
#def S2_2(A,B,C,D):
#    return (-4*B - (4*np.power(-2,0.3333333333333333)*(np.power(B,2) - 3*C))/
#      np.power(-2*np.power(B,3) + 9*B*C + 3*(-9*D + 
#           np.sqrt(-3*(np.power(B,2) - 4*C)*np.power(C,2) + 6*B*(2*np.power(B,2) - 9*C)*D + 81*np.power(D,2))),
#       0.3333333333333333) + 2*np.power(-2,0.6666666666666666)*
#      np.power(-2*np.power(B,3) + 9*B*C + 3*(-9*D + 
#           np.sqrt(-3*(np.power(B,2)  4*C)*np.power(C,2) + 6*B*(2*np.power(B,2) - 9*C)*D + 81*np.power(D,2))),
#       0.3333333333333333))/12.
#def S3_2(A,B,C,D):
#    return (-B/3. + (np.power(-1,0.6666666666666666)*np.power(2,0.3333333333333333)*(np.power(B,2) - 3*C))/
#    (3.*np.power(-2*np.power(B,3) + 9*B*C + 3*(-9*D + 
#           np.sqrt(-3*(np.power(B,2) - 4*C)*np.power(C,2) + 6*B*(2*np.power(B,2) - 9*C)*D + 81*np.power(D,2))),
#       0.3333333333333333)) - (np.power(-0.5,0.3333333333333333)*
#      np.power(-2*np.power(B,3) + 9*B*C + 3*(-9*D + 
#           np.sqrt(-3*(np.power(B,2)- 4*C)*np.power(C,2) + 6*B*(2*np.power(B,2) - 9*C)*D + 81*np.power(D,2))),
#       0.3333333333333333))/3.)
def S_plus_2(A,B,C,D):
    s1 = S1_2(A,B,C,D)
    s2 = S2_2(A,B,C,D)
    s3 = S3_2(A,B,C,D)
    return np.amax([s1,s2,s3])
def S_minus_2(A,B,C,D):
    s1 = S1_2(A,B,C,D)
    s2 = S2_2(A,B,C,D)
    s3 = S3_2(A,B,C,D)
    return np.amin([s1,s2,s3])
def calculate_S_roots(A,B,C,D):
    s1 = S1_2(A,B,C,D)
    s2 = S2_2(A,B,C,D)
    s3 = S3_2(A,B,C,D)
    roots = [s1,s2,s3]
    print("Roots")
    print(roots)
    S_plus2 = np.amax(roots)
    S_plus = np.sqrt(S_plus2)
    S_minus2 = np.amin(roots)
    S_minus = np.sqrt(S_minus2)
    #print(roots)
    #print(S_plus2,S_minus2)
    roots.remove(S_plus2)
    roots.remove(S_minus2)
    S3 = np.sqrt(roots[0])
    return S_plus, S_minus, S3
###############################################################################################
#a coefficients PHYSICAL REVIEW D 88, 063011
def a0(eta): return (96/5)*eta

def a1(eta): return 1/2 + 3/4 * eta

def a2 (eta, xi): return -3/(4*eta)*xi

def a3(beta3): return 4*np.pi - beta3

def a4(eta, sigma4): return 34103/18144 + 13661/2016 * eta + 56/18 * eta**2 - sigma4

def a5(eta, beta5): return -4159/672 * np.pi - 189/8 * np.pi *eta - beta5

def a6(eta, beta6): 
    return ( 16447322263/139708800 + 16/3 * np.pi**2 - 856/105 * np.log(16) - 
            1712/105 * util.gamma_E - beta6 + eta*(451/48 * np.pi**2 - 56198689/217728) +
            eta**2 * 541/896 - eta**3 * 5605/2592)

def a7(eta, beta7): return (-4415/4032 * np.pi + 358675/6048 * np.pi * eta + 91495/1512 * np.pi *eta**2
                            - beta7)

###############################################################################################
#Defining g parameters:PHYSICAL REVIEW D 95, 104004
def g0(a0):
    return 1/a0

def g2(a2,a0):
    return -a2/a0

def g3(a3,a0):
    return -a3/a0

def g4(a4,a2,a0):
    return -(a4-a2**2)/a0

def g5(a5,a3,a2,a0):
    return -(a5 - 2*a3*a2)/a0

def g6(a6,a4,a3,a2,a0):
    return -(a6-2*a4*a2-a3**2+a2**2)/a0

def g6_l(a0):
    b6 = -1712/315
    return -3*b6/a0

def g7(a7, a5,a4,a3,a2,a0):
    return -(a7 - 2*a5*a2 - 2*a4*a3+3*a3*a2**2)/a0

###############################################################################################
#Betas PHYSICAL REVIEW D 88, 063011
def beta3(m1, m2,c1,q,eta,xi): 
    M = m1 + m2
    spin_dot1 = S1_dot_L_avg(c1,q,eta,xi)
    spin_dot2 = S2_dot_L_avg(c1,q,eta,xi)
    term1 = (113/12 + 25/4 * m1/m2)*spin_dot1
    term2 = (113/12 + 25/4 * m2/m1)*spin_dot2
    return (1/M**2) * ( term1 + term2)

def beta5(m1, m2,c1,q,eta,xi): 
    M = m1 + m2
    spin_dot1 = S1_dot_L_avg(c1,q,eta,xi)
    spin_dot2 = S2_dot_L_avg(c1,q,eta,xi)
    term1 = (31319/1008 - 1159/24 * eta + m2/m1 * (809/84 - 281/8 * eta)) * spin_dot1
    term2 = (31319/1008 - 1159/24 * eta + m1/m2 * (809/84 - 281/8 * eta)) * spin_dot2
    return (1/M**2) * ( term1 + term2)

def beta6(m1, m2,c1,q,eta,xi): 
    M = m1 + m2
    spin_dot1 = S1_dot_L_avg(c1,q,eta,xi)
    spin_dot2 = S2_dot_L_avg(c1,q,eta,xi)
    term1 = (75/2 + 151/6 * m2/m1 ) * spin_dot1
    term2 = (75/2 + 151/6 * m1/m2 ) * spin_dot2
    return (np.pi/M**2) * (term1 + term2)

def beta7(m1, m2,c1,q,eta,xi): 
    M = m1 + m2
    spin_dot1 = S1_dot_L_avg(c1,q,eta,xi)
    spin_dot2 = S2_dot_L_avg(c1,q,eta,xi)
    term1 = ( 130325/756 - 796069/2016 * eta + 100019/864 * eta**2 + 
            m2/m1 * (1195759/18144 - 257023/1008 * eta +2903/32*eta**2) ) *spin_dot1
    term2 = ( 130325/756 - 796069/2016 * eta + 100019/864 * eta**2 + 
            m1/m2 * (1195759/18144 - 257023/1008 * eta +2903/32*eta**2) ) *spin_dot2
    return (1/M**2) * (term1 + term2)

#S1,S2 are magnitudes
def sigma4(m1, m2, mu, Sav, S1,S2, c1,q,eta,xi,S_plus,S_minus,v0 ):
    M = m1 +m2
    s1s2 = S1_dot_S2_avg(Sav,S1,S2)
    s1Ls2L = S1_dot_L_S2_dot_L_avg(c1,q,eta,xi,S_plus,S_minus,v0)
    S1L_squared = S1_dot_L_squared_avg(c1,q,eta,xi,S_plus,S_minus,v0)
    S2L_squared = S2_dot_L_squared_avg(c1,q,eta,xi,S_plus,S_minus,v0)
    term1 = (1/(mu*M**3)) * (247/48*s1s2 - 721/48 * s1Ls2L)
    term2 = 1/(M**2 * m1**2) * ( 233/96 * S1**2 - 719/96 * S1L_squared) 
    term3 = 1/(M**2 * m2**2) * ( 233/96 * S2**2 - 719/96 * S2L_squared) 
    return term1 + term2 + term3

###############################################################################################
#Orbit averaged angular momentum products - PHYSICAL REVIEW D 95, 104004
def S1_dot_L_avg(c1,q,eta,xi):
    num = c1*(1+q) - q*eta*xi
    denom = eta*(1-q**2)
    return num/denom

def S2_dot_L_avg(c1,q,eta,xi):
    num = c1*(1+q) - eta*xi
    denom = eta*(1-q**2)
    return -q*num/denom

#S1 and S2 are magnitudes
def S1_dot_S2_avg(Sav, S1,S2):
    return Sav**2/2 - (S1**2 + S2**2)/2

def S1_dot_L_squared_avg(c1,q,eta,xi,S_plus,S_minus,v0):
    term1 = S1_dot_L_avg(c1,q,eta,xi)
    term2 = (S_plus**2 - S_minus**2)**2 * v0**2 / (32*eta**2 * (1-q)**2 )
    return term1 + term2

def S2_dot_L_squared_avg(c1,q,eta,xi,S_plus,S_minus,v0):
    term1 = S1_dot_L_avg(c1,q,eta,xi)
    term2 = q**2*(S_plus**2 - S_minus**2)**2 * v0**2 / (32*eta**2 * (1-q)**2 )
    return term1 + term2

def S1_dot_L_S2_dot_L_avg(c1,q,eta,xi,S_plus,S_minus,v0):
    return (S1_dot_L_squared_avg(c1,q,eta,xi,S_plus,S_minus,v0)*
             S2_dot_L_squared_avg(c1,q,eta,xi,S_plus,S_minus,v0)
            - q*(S_plus**2 - S_minus**2)**2 * v0**2 / (32*eta**2 * (1-q)**2 ))

def Sav(S_plus_0, S_minus_0):
    return (1/2)*(S_plus_0**2 + S_minus_0**2)

###############################################################################################
#A,B,C,D coefficients for S_plus and S_minus -PHYSICAL REVIEW D 95, 104004
def A(eta, xi, v):
    return -(3/(2*np.sqrt(eta))) * v**6 * (1- xi*v)

#L,S1,S2,J are magnitudes
def B(L,S1,S2,J,xi,q):
    term1 = (L**2 + S1**2)*q
    term2 = 2*L*xi 
    term3 = -2*J**2 
    term4 = -(S1**2 + S2**2)
    term5 = (L**2 + S2**2)/q
    return term1+term2+term3+term4+term5

#L,S1,S2,J are magnitudes
def C(L,S1,S2,J,xi,delta_m, q, eta):
    term1 = (J**2 -L**2)**2
    term2 = -2*L*xi*(J**2-L**2)
    term3 = -2*(1-q)/q * (S1**2 - q* S2**2)*L**2
    term4 = 4*eta * L**2 *xi**2
    term5 = -2*delta_m * (S1**2 -S2**2)*xi*L 
    term6 = 2*(1-q)/q * (q*S1**2 - S2**2)*J**2
    return term1+term2 + term3 + term4 + term5 + term6

#L,S1,S2,J are magnitudes
def D(L,S1,S2,J,xi,delta_m, q, eta):
    term1 = (1-q)/q * (S2**2 -q*S1**2)*(J**2 - L**2)**2
    term2 = delta_m**2/eta * (S1**2 - S2**2)**2 * L**2
    term3 = 2 * delta_m *L * xi * (S1**2 - S2**2 )*(J**2 - L**2)
    return term1+term2+term3

###############################################################################################
#Formula from PHYSICAL REVIEW D 95, 104004
def R_m (S_plus, S_minus): return S_plus**2 + S_minus**2

def cp (S_plus,c1,eta): return (S_plus**2 * eta**2 - c1**2)

def cm ( S_minus, c1, eta): return (S_minus**2 * eta**2 - c1**2)

def ad (S1, S2, eta, delta_m, c1, xi, cp, cm) :
    t1 = -3*(S1**2 - S2**2 )* eta * delta_m
    t2 = 3 * (c1/eta)*(c1 - 2*eta**2 * xi)
    return (t1 + t2)/( 4 * np.sqrt(cp * cm))

def cd (c1, eta, cp, cm, Rm): return (3 / 128) * (Rm**2 / ( eta * np.sqrt(cp * cm)))

def hd(c1, eta, cp, cm): return (c1/eta**2) * (1 - (cp + cm)/(2 * np.sqrt(cp*cm))) 

def fd (cp, cm, eta): return (cp + cm)/(8*eta**4) * (1 - (cp+cm)/(2 * np.sqrt(cp*cm)))

###############################################################################################
#PHYSICAL REVIEW D 95, 104004
def Omega_z_0 (a1, ad): return a1+ad

def Omega_z_1 (a2, ad, xi, hd): return a2 - ad*xi - ad*hd

def Omega_z_2 (ad, hd, xi, cd, fd): return ad*hd*xi + cd - ad*fd + ad*hd**2

def Omega_z_3(ad, fd, cd, hd, xi): return (ad*fd - cd - ad*hd**2)*(xi + hd) + ad*fd*hd

def Omega_z_4 (cd, ad, hd, fd, xi): return (cd+ad*hd**2 - 2 *ad*fd)*(hd*xi + hd**2 -fd) - ad*fd**2

def Omega_z_5 (cd, ad, fd, hd, xi): 
    t1 = cd - ad*fd + ad*hd**2
    t2 = xi + 2 * hd
    t3 = cd + ad*hd**2 - 2*ad*fd
    t4 = xi + hd
    return t1*fd*t2 - t3*hd**2 * t4 - ad*fd**2 * hd

###############################################################################################
#PHYSICAL REVIEW D 95, 104004
def Omega_z_0_avg (g0, O_z_0): return 3 * g0 * O_z_0

def Omega_z_1_avg (g0, O_z_1): return 3 * g0 * O_z_1

def Omega_z_2_avg (g0,g2, O_z_2, O_z_0): return 3*(g0*O_z_2 + g2 * O_z_0)

def Omega_z_3_avg (g0, g2, g3,O_z_3, O_z_1, O_z_0): return 3*(g0*O_z_3 + g2*O_z_1 + g3*O_z_0)

def Omega_z_4_avg (g0,g2,g3,g4, O_z_4, O_z_2,  O_z_1, O_z_0): return 3*(g0*O_z_4 + g2*O_z_2 +g3 * O_z_1
                                                                        + g4*O_z_0)
def Omega_z_5_avg(g0, g2, g3,g4,g5, O_z_5, O_z_3, O_z_2, O_z_1, O_z_0):
    return 3*(g0*O_z_5 + g2*O_z_3 + g3*O_z_2 + g4*O_z_1 + g5 * O_z_0)

###############################################################################################
#Phi_z_n calculations for eq 66 in PHYSICAL REVIEW D 95, 104004
def phi_z_0(J, eta, c1,Sav, l1, v):
    term1 = (J/(eta**4)) * (c1**2/2 - c1* eta**2/ (6*v) - Sav**2 * eta**2/3 - eta**4/(3*v**2))
    term2 = -c1/(2*eta) * (c1**2/eta**4 - Sav**2/eta**2 )* l1
    return term1 + term2

def phi_z_1(J, eta, c1, L, Sav, l1):
    term1 = -J/(2*eta**2) * (c1 + eta*L)
    term2 = 1/(2*eta**3) * (c1 - eta**2 * Sav**2)*l1
    return term1+term2

def phi_z_2(J, Sav, c1, eta, l1,l2):
    return -J + np.sqrt( Sav**2 *l2) - c1/eta * l1

def phi_z_3(J, Sav, c1, eta,l1, l2,v):
    return J*v - eta*l1 + c1/np.sqrt(Sav**2) * l2

def phi_z_4(J, Sav, c1, eta,l1, l2,v):
    term1 = J/(2*Sav**2) * v * (c1 + v*Sav**2)
    term2 = -1/(2*(Sav**2)**(3/2) ) * (c1**2 - eta**2 * Sav**2)*l2
    return term1 + term2

def phi_z_5(J, Sav, c1, eta,l1, l2,v):
    term1 = -J*v*(c1**2/(2*Sav**4) - c1*v/(6*(Sav**2)**(3/2)) - v**2/3 - eta**2/(3*Sav**2))
    term2 = c1/(2*(Sav**2)**(5/2)) * (c1**2 - eta**2 * Sav**2)*l2
    return term1 + term2

def l1(J, L, c1, eta):
    return np.log(c1 + J*eta + L*eta)

def l2(J,Sav, c1, v):
    return np.log(c1 + J * np.sqrt(Sav**2) *v + Sav**2 * v)

###############################################################################################
#coefficients for Phi_0 in eq 67 in PHYSICAL REVIEW D 95, 104004
def psi_dot(A,S_plus,S3):
    return (A/2)*np.sqrt(S_plus**2 - S3**2)

def a(eta,xi,v):
    return (1/2) * v**6 * (1 + (3/(2*eta)) * (1-xi*v))

def c0(eta,xi,delta_m,J,S_plus,S_minus,S1,S2,v):
    term1 = (3/4) * (1-xi*v)*v**2 
    term2 = eta**3 + 4*eta**3 * xi*v
    term3 = -2*eta*(J**2 - S_plus**2 + 2*(S1**2 - S2**2)*delta_m)*v**2
    term4 = -4*eta*xi*(J**2 - S_plus**2) *v**3
    term5 = (J**2 - S_plus**2)**2 * v**4 /eta
    return term1*(term2 + term3 + term4 + term5)

def c2(eta,xi,J,S_plus,S_minus,v):
    term1 = (-3*eta/2) *(S_plus**2 - S_minus**2)*(1-xi*v)*v**4
    term2 = 1 + 2*xi*v - (J**2 - S_plus**2 ) /eta**2 * v**2
    return term1*term2

def c4(eta, xi, S_plus, S_minus, v):
    return (3/(4*eta))*(S_plus**2 - S_minus**2 )**2 * (1-xi*v)*v**6

def d0(J,L, S_plus):
    return -(J**2 -(L + S_plus)**2)*(J**2 - (L-S_plus)**2)

def d2(J,L,S_plus,S_minus ):
    return -2*(S_plus**2 - S_minus**2)*(J**2 +L**2 -S_plus**2)

def d4(S_plus,S_minus):
    return -(S_plus**2 - S_minus**2)**2

def C1(c0,c2,c4,d0,d2,d4):
    return (-1/2)*(c0/d0 - (c0+c2+c4)/(d0+d2+d4))
def C2(c0,c2,c4,d0,d2,d4,sd):
    term1 = c0*(-2*d0*d4+d2**2 +d2*d4)
    term2 = -c2*d0*(d2 + 2*d4)
    term3 = 2*d0*(d0+ d2 + d4)*sd
    term4 = c4*d0*(2*d0 + d2)
    term5 = 2*d0*(d0+d2+d4)*sd
    return (term1 + term2)/term3 + term4/term5

def sd(d0,d2,d4):
    return np.sqrt(d2**2 - 4*d0*d4)

def Cphi(C1, C2):
    return C1 + C2

def Dphi(C1,C2):
    return C1-C2

def nc(d0,d2,d4,sd):
    return 2*(d0 + d2 + d4) / (2*d0 + d2 + sd)

def nd(d0,d2,sd):
    return (2*d0 + d2 + sd)/(2*d0)

def phi_z_0_0(Cphi,Dphi,nc,nd,psi_dot,psi):
    term1 = (Cphi/psi_dot * (np.sqrt(nc)/(nc-1) ) *
            np.arctan( (1-np.sqrt(nc))*np.tan(psi)/(1+np.sqrt(nc)*np.tan(psi)**2)))
    term2 = (Dphi/psi_dot * (np.sqrt(nd)/(nd-1) ) *
            np.arctan( (1-np.sqrt(nd)) *np.tan(psi)/(1+ np.sqrt(nd) * np.tan(psi)**2 ) ))
    return term1 + term2
