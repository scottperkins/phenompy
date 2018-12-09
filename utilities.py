import autograd.numpy as np
import astropy.constants as consts
import astropy.cosmology as cosmology
from astropy.coordinates import Distance

"""Euler's Number (Not in SciPy or NumPy Constants)"""
gamma_E = 0.5772156649015328606065120900824024310421
"""NOTE: the values are commented out below to match the previous code
- should be uncommented before publication"""
c = consts.c.value #Speed of light m/s
G = consts.G.to('m3/(s2 kg)').value*consts.M_sun.value #Gravitational constant in m**3/(s**2 SolMass)=6.674*10**(-11)*(1.98855*10**30)
s_solm = G / consts.c.value**3#G/c**3#seconds per solar mass =492549095*10**(-14)
mpc = 1/consts.c.to('Mpc/s').value#consts.kpc.to('m')*1000/c#Mpc in sec =3085677581*10**(13)/c
H0 = cosmology.Planck15.H0#6780*10**(-2)/(3 * 10**5)#67.80/(3.086*10**19) #Hubble constant in [1/Mpc]
hplanck = consts.h.to('eV s').value #Planck Constant in eV s
#c = 299792458#consts.c #Speed of light m/s
#G = 6.674*10**(-11)*(1.98855*10**30)#consts.G.to('m**3/(s**2*solMass)') #Gravitational constant in m**3/(s**2 SolMass)
#s_solm =492549095*10**(-14) #G/c**3#seconds per solar mass
#mpc = 3085677581*10**(13)/c #consts.kpc.to('m')*1000/c#Mpc in sec
#H0 = cosmology.Planck15.H0#6780*10**(-2)/(3 * 10**5)#67.80/(3.086*10**19) #Hubble constant in [1/Mpc]
#hplanck = 4.135667662e-15 #ev s



"""Generic, short, simple functions that can be easily separated from a specific model"""
###########################################################################################
"""Calculates the total mass given symmratio and chirp mass"""
def calculate_totalmass(chirpm,symmratio):return chirpm*symmratio**(-3/5)


"""Calculates the individual masses given symmratio and chirp mass"""
def calculate_mass1(chirpm,symmratio):
    return 1/2*(chirpm / symmratio**(3/5) \
    + np.sqrt(1-4*symmratio)*chirpm / symmratio**(3/5))
def calculate_mass2(chirpm,symmratio):
    return 1/2*(chirpm / symmratio**(3/5) \
    - np.sqrt(1-4*symmratio)*chirpm / symmratio**(3/5))


"""calculate fpeak"""
def calculate_fpeak(M,fRD,fdamp,gamma2,gamma3):return abs(fRD + fdamp*gamma3*(np.sqrt(1-gamma2**2)-1)/gamma2)

"""Calculate delta parameter"""
def calculate_delta(symmratio):return np.sqrt(1-4*symmratio)


"""Post Newtonian approximation for the inspiral amplitude for the ith (\el {0,1,2,3,4,5,6}) parameter"""
def calculate_pn_amp(symmratio,massdelta,chi_a,chi_s,i):
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
def calculate_pn_phase( chirpm,symmratio,delta,chi_a,chi_s,f,i):
    """5 and 6 depend on the given freq."""
    M = calculate_totalmass(chirpm,symmratio)
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


###########################################################################################################
"""Caclulate the post merger paramters fRD and fdamp
from the kerr parameter a = J/M**2 - the relationship between a and fRD,f_damp
is numerical - interpolated data from http://www.phy.olemiss.edu/~berti/ringdown/ - 0905.2975
a has a numerical fit from the khan phenomD paper and is a function of the
symmetric mass ratio and the total initial spin
the final parameters are then determined from omega = omegaM/(M - energry radiated)"""
def calculate_postmerger_fRD(m1,m2,M,symmratio,chi_s,chi_a):
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

def calculate_postmerger_fdamp(m1,m2,M,symmratio,chi_s,chi_a):
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
###########################################################################################
"""Analytic functions for amplitude continutity"""
"""Solved the system analytically in mathematica and transferred the functions --
  probably quicker to evaluate the expressions below than solve the system with matrices and needed for autograd to function properly

  Tried to define derivates manually, but stopped because too little of an effect. *(Actually, something's not working with the second
  function anyway, so just won't use it)"""
def calculate_delta_parameter_0(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return -((dd3*f1**5*f2**2*f3 - 2*dd3*f1**4*f2**3*f3 + dd3*f1**3*f2**4*f3 - dd3*f1**5*f2*f3**2 + \
   dd3*f1**4*f2**2*f3**2 - dd1*f1**3*f2**3*f3**2 + dd3*f1**3*f2**3*f3**2 + \
   dd1*f1**2*f2**4*f3**2 - dd3*f1**2*f2**4*f3**2 + dd3*f1**4*f2*f3**3 + \
   2*dd1*f1**3*f2**2*f3**3 - 2*dd3*f1**3*f2**2*f3**3 - dd1*f1**2*f2**3*f3**3 + \
   dd3*f1**2*f2**3*f3**3 - dd1*f1*f2**4*f3**3 - dd1*f1**3*f2*f3**4 - \
   dd1*f1**2*f2**2*f3**4 + 2*dd1*f1*f2**3*f3**4 + dd1*f1**2*f2*f3**5 - \
   dd1*f1*f2**2*f3**5 + 4*f1**2*f2**3*f3**2*v1 - 3*f1*f2**4*f3**2*v1 - \
   8*f1**2*f2**2*f3**3*v1 + 4*f1*f2**3*f3**3*v1 + f2**4*f3**3*v1 + 4*f1**2*f2*f3**4*v1 + \
   f1*f2**2*f3**4*v1 - 2*f2**3*f3**4*v1 - 2*f1*f2*f3**5*v1 + f2**2*f3**5*v1 - \
   f1**5*f3**2*v2 + 3*f1**4*f3**3*v2 - 3*f1**3*f3**4*v2 + f1**2*f3**5*v2 - \
   f1**5*f2**2*v3 + 2*f1**4*f2**3*v3 - f1**3*f2**4*v3 + 2*f1**5*f2*f3*v3 - \
   f1**4*f2**2*f3*v3 - 4*f1**3*f2**3*f3*v3 + 3*f1**2*f2**4*f3*v3 - 4*f1**4*f2*f3**2*v3 + \
   8*f1**3*f2**2*f3**2*v3 - 4*f1**2*f2**3*f3**2*v3)/\
 ((f1 - f2)**2*(f1 - f3)**3*(-f2 + f3)**2))
def calculate_delta_parameter_1(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return -((-(dd3*f1**5*f2**2) + 2*dd3*f1**4*f2**3 - dd3*f1**3*f2**4 - dd3*f1**4*f2**2*f3 + \
   2*dd1*f1**3*f2**3*f3 + 2*dd3*f1**3*f2**3*f3 - 2*dd1*f1**2*f2**4*f3 - \
   dd3*f1**2*f2**4*f3 + dd3*f1**5*f3**2 - 3*dd1*f1**3*f2**2*f3**2 - \
   dd3*f1**3*f2**2*f3**2 + 2*dd1*f1**2*f2**3*f3**2 - 2*dd3*f1**2*f2**3*f3**2 + \
   dd1*f1*f2**4*f3**2 + 2*dd3*f1*f2**4*f3**2 - dd3*f1**4*f3**3 + dd1*f1**2*f2**2*f3**3 + \
   3*dd3*f1**2*f2**2*f3**3 - 2*dd1*f1*f2**3*f3**3 - 2*dd3*f1*f2**3*f3**3 + \
   dd1*f2**4*f3**3 + dd1*f1**3*f3**4 + dd1*f1*f2**2*f3**4 - 2*dd1*f2**3*f3**4 - \
   dd1*f1**2*f3**5 + dd1*f2**2*f3**5 - 8*f1**2*f2**3*f3*v1 + 6*f1*f2**4*f3*v1 + \
   12*f1**2*f2**2*f3**2*v1 - 8*f1*f2**3*f3**2*v1 - 4*f1**2*f3**4*v1 + 2*f1*f3**5*v1 + \
   2*f1**5*f3*v2 - 4*f1**4*f3**2*v2 + 4*f1**2*f3**4*v2 - 2*f1*f3**5*v2 - 2*f1**5*f3*v3 + \
   8*f1**2*f2**3*f3*v3 - 6*f1*f2**4*f3*v3 + 4*f1**4*f3**2*v3 - 12*f1**2*f2**2*f3**2*v3 + \
   8*f1*f2**3*f3**2*v3)/((f1 - f2)**2*(f1 - f3)**3*(f2 - f3)**2*M))
def calculate_delta_parameter_2(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return  -((dd3*f1**5*f2 - dd1*f1**3*f2**3 - 3*dd3*f1**3*f2**3 + dd1*f1**2*f2**4 + \
    2*dd3*f1**2*f2**4 - dd3*f1**5*f3 + dd3*f1**4*f2*f3 - dd1*f1**2*f2**3*f3 + \
    dd3*f1**2*f2**3*f3 + dd1*f1*f2**4*f3 - dd3*f1*f2**4*f3 - dd3*f1**4*f3**2 + \
    3*dd1*f1**3*f2*f3**2 + dd3*f1**3*f2*f3**2 - dd1*f1*f2**3*f3**2 + dd3*f1*f2**3*f3**2 - \
    2*dd1*f2**4*f3**2 - dd3*f2**4*f3**2 - 2*dd1*f1**3*f3**3 + 2*dd3*f1**3*f3**3 - \
    dd1*f1**2*f2*f3**3 - 3*dd3*f1**2*f2*f3**3 + 3*dd1*f2**3*f3**3 + dd3*f2**3*f3**3 + \
    dd1*f1**2*f3**4 - dd1*f1*f2*f3**4 + dd1*f1*f3**5 - dd1*f2*f3**5 + 4*f1**2*f2**3*v1 - \
    3*f1*f2**4*v1 + 4*f1*f2**3*f3*v1 - 3*f2**4*f3*v1 - 12*f1**2*f2*f3**2*v1 + \
    4*f2**3*f3**2*v1 + 8*f1**2*f3**3*v1 - f1*f3**4*v1 - f3**5*v1 - f1**5*v2 - \
    f1**4*f3*v2 + 8*f1**3*f3**2*v2 - 8*f1**2*f3**3*v2 + f1*f3**4*v2 + f3**5*v2 + \
    f1**5*v3 - 4*f1**2*f2**3*v3 + 3*f1*f2**4*v3 + f1**4*f3*v3 - 4*f1*f2**3*f3*v3 + \
    3*f2**4*f3*v3 - 8*f1**3*f3**2*v3 + 12*f1**2*f2*f3**2*v3 - 4*f2**3*f3**2*v3)/\
  ((f1 - f2)**2*(f1 - f3)**3*(f2 - f3)**2*M**2))
def calculate_delta_parameter_3(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return -((-2*dd3*f1**4*f2 + dd1*f1**3*f2**2 + 3*dd3*f1**3*f2**2 - dd1*f1*f2**4 - dd3*f1*f2**4 + \
   2*dd3*f1**4*f3 - 2*dd1*f1**3*f2*f3 - 2*dd3*f1**3*f2*f3 + dd1*f1**2*f2**2*f3 - \
   dd3*f1**2*f2**2*f3 + dd1*f2**4*f3 + dd3*f2**4*f3 + dd1*f1**3*f3**2 - dd3*f1**3*f3**2 - \
   2*dd1*f1**2*f2*f3**2 + 2*dd3*f1**2*f2*f3**2 + dd1*f1*f2**2*f3**2 - \
   dd3*f1*f2**2*f3**2 + dd1*f1**2*f3**3 - dd3*f1**2*f3**3 + 2*dd1*f1*f2*f3**3 + \
   2*dd3*f1*f2*f3**3 - 3*dd1*f2**2*f3**3 - dd3*f2**2*f3**3 - 2*dd1*f1*f3**4 + \
   2*dd1*f2*f3**4 - 4*f1**2*f2**2*v1 + 2*f2**4*v1 + 8*f1**2*f2*f3*v1 - 4*f1*f2**2*f3*v1 - \
   4*f1**2*f3**2*v1 + 8*f1*f2*f3**2*v1 - 4*f2**2*f3**2*v1 - 4*f1*f3**3*v1 + 2*f3**4*v1 + \
   2*f1**4*v2 - 4*f1**3*f3*v2 + 4*f1*f3**3*v2 - 2*f3**4*v2 - 2*f1**4*v3 + \
   4*f1**2*f2**2*v3 - 2*f2**4*v3 + 4*f1**3*f3*v3 - 8*f1**2*f2*f3*v3 + 4*f1*f2**2*f3*v3 + \
   4*f1**2*f3**2*v3 - 8*f1*f2*f3**2*v3 + 4*f2**2*f3**2*v3)/\
 ((f1 - f2)**2*(f1 - f3)**3*(f2 - f3)**2*M**3))
def calculate_delta_parameter_4(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return  -((dd3*f1**3*f2 - dd1*f1**2*f2**2 - 2*dd3*f1**2*f2**2 + dd1*f1*f2**3 + dd3*f1*f2**3 - \
   dd3*f1**3*f3 + 2*dd1*f1**2*f2*f3 + dd3*f1**2*f2*f3 - dd1*f1*f2**2*f3 + \
   dd3*f1*f2**2*f3 - dd1*f2**3*f3 - dd3*f2**3*f3 - dd1*f1**2*f3**2 + dd3*f1**2*f3**2 - \
   dd1*f1*f2*f3**2 - 2*dd3*f1*f2*f3**2 + 2*dd1*f2**2*f3**2 + dd3*f2**2*f3**2 + \
   dd1*f1*f3**3 - dd1*f2*f3**3 + 3*f1*f2**2*v1 - 2*f2**3*v1 - 6*f1*f2*f3*v1 + \
   3*f2**2*f3*v1 + 3*f1*f3**2*v1 - f3**3*v1 - f1**3*v2 + 3*f1**2*f3*v2 - 3*f1*f3**2*v2 + \
   f3**3*v2 + f1**3*v3 - 3*f1*f2**2*v3 + 2*f2**3*v3 - 3*f1**2*f3*v3 + 6*f1*f2*f3*v3 - \
   3*f2**2*f3*v3)/((f1 - f2)**2*(f1 - f3)**3*(f2 - f3)**2*M**4))


###########################################################################################################
#dCS g func for phase modification - \delta \phi = Zeta *g(eta,chirp mass, chi_s, chi_a) (\pi chirpm f)**(-1/3)
###########################################################################################

def dCS_g(chirpm,symmratio,chi_s,chi_a): 
    coeff = 1549225/11812864 
    m1 = calculate_mass1(chirpm,symmratio) 
    m2 = calculate_mass2(chirpm,symmratio) 
    m = calculate_totalmass(chirpm,symmratio) 
    delta = (m1-m2)/m 
    return coeff/symmratio**(14/5) * ( ( 1-231808/61969 * symmratio)*chi_s**2 + 
            (1 - 16068/61969 *symmratio)*chi_a**2 - 2 * delta * chi_s * chi_a)
