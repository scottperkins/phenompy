import autograd.numpy as np
import astropy.cosmology as cosmology
from scipy.optimize import fsolve
from phenompy.gr import IMRPhenomD
from phenompy import utilities
import os
import matplotlib.pyplot as plt
from astropy.coordinates import Distance
from astropy import units as u
import csv
from scipy.interpolate import interp1d


"""Useful functions for analysis of waveforms. NOT used for waveform production, like the utilities.py file.
These functions usually require creating or passing an already created model."""

"""Path variables"""
IMRPD_dir = os.path.dirname(os.path.realpath(__file__))
IMRPD_tables_dir = IMRPD_dir + '/Data_Tables'
"""Useful Functions from IMRPhenomD"""
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
"""Constants from IMRPhenomD"""
mpc = utilities.mpc
c = utilities.c
s_solm = utilities.s_solm
hplanck = utilities.hplanck

"""Calculate luminositiy distance for a desired SNR and model"""
###########################################################################################
def LumDist_SNR(mass1, mass2,spin1,spin2,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors=1,detector='aLIGO',SNR_target=10,lower_freq=None,upper_freq=None):
    D_L_target = fsolve(
            lambda l: SNR_target - LumDist_SNR_assist(mass1=mass1, mass2=mass2,spin1=spin1,spin2=spin2,DL=l,cosmo_model = cosmo_model,NSflag = NSflag,N_detectors=N_detectors,detector=detector,lower_freq=lower_freq,upper_freq=upper_freq),
            100*imr.mpc)[0]
    return D_L_target/imr.mpc
def LumDist_SNR_assist(mass1, mass2,spin1,spin2,DL,cosmo_model,NSflag ,N_detectors,detector,lower_freq,upper_freq):
    temp_model = IMRPhenomD(mass1=mass1, mass2=mass2,spin1=spin1,spin2=spin2, collision_time=0, \
                    collision_phase=0,Luminosity_Distance=DL,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors=1)
    SNR_temp = temp_model.calculate_snr(detector=detector,lower_freq=lower_freq,upper_freq=upper_freq)
    return SNR_temp
###########################################################################################



###########################################################################################
###########################################################################################
"""Functions specific to constraining th graviton mass in screened gravity - see arXiv:1811.02533"""
###########################################################################################
###########################################################################################

"""Function for plotting degeneracy of Delta Beta, which depends on both the wavelength
of the graviton and the screening radius
Delta beta = pi^2 *self.chirpm*D/((1+Z)*lambda^2)
- D depends on the screening radius (D proportional to the integral_r1^r2 a dt)
returns D as a function of lambda

Second method returns lambda_g as a function of screening radius -
Assumptions: the mass of the two galaxies are approximately equal
arguments: delta_beta, Rvs - List of Vainshtein radii (cannot be larger than DL/2)"""

def degeneracy_function_D(model, delta_beta, lambda_g ):
    return delta_beta* lambda_g**2 *(1+ model.Z)/(np.pi**2 *model.chirpm)
def degeneracy_function_lambda(model,delta_beta, Rvs):
    lambdas = []
    H0=model.cosmo_model.H0.to('Hz').value
    Z1 = Zfunc(Rvs/mpc)
    Z2 = Zfunc((model.DL-Rvs)/mpc)
    D = (1+model.Z)*(Dfunc(Z2)*mpc- Dfunc(Z1)*mpc)
    return (D*np.pi**2*model.chirpm/((1+model.Z)*delta_beta))**(1/2)


"""Add the Mass unit axis to plot"""
def convert_wavelength_mass(wavelength):
    return hplanck * c / (wavelength )
def convert_lambda_mass_axis( model,ax):
    l1, l2 = ax.get_ylim()
    model.ax_ev.set_ylim(convert_wavelength_mass(l1),convert_wavelength_mass(l2))
    model.ax_ev.figure.canvas.draw()

"""Plot the constraint on Beta from the TOA difference of the peak GW signal and peak gamma ray detection
measured for GW170817 -
arguments: screening radius array (seconds)"""
def degeneracy_function_lambda_GW170817(model,Rvs):
    lambdas = []
    H0=model.cosmo_model.H0.to('Hz').value
    DL_GW170817 = 40*mpc
    Delta_T_measured = 1.7 #seconds
    fpeak_measured =3300 #3300#REF 1805.11579

    Z = Distance(DL_GW170817/mpc,unit=u.Mpc).compute_z(cosmology = model.cosmo_model)

    Z1 = Zfunc(Rvs/mpc)
    Z2 = Zfunc((DL_GW170817-Rvs)/mpc)
    D = (1+Z)*(Dfunc(Z2)*mpc - Dfunc(Z1)*mpc)
    return (2*Delta_T_measured*fpeak_measured**2/((1+Z)*D))**(-1/2)


"""Results of Fisher analysis on GW150914 -  Refs 1602.03840, PhysRevLett.116.221101"""
def degeneracy_function_lambda_GW150914(model,Rvs):
    lambdas = []
    H0=model.cosmo_model.H0.to('Hz').value
    GW15DL = 420*mpc # MPC
    GW15chirpm = 30.4 *s_solm#Solar Masses
    GW15Z = .088
    GW15lambdag = 1e16/c # seconds
    GW15D = (1+GW15Z)*(integrate.quad(lambda x: 1/(H0*(1+x)**2*np.sqrt(.3*(1+x)**3 + .7)),0,GW15Z )[0])

    GW15DeltaBeta = (GW15D*np.pi**2*GW15chirpm)/((1+GW15Z)*GW15lambdag**2)
    print("GW15LIGO (90): {}".format(GW15DeltaBeta))

    Z1 = Zfunc(Rvs/mpc)
    Z2 = Zfunc((GW15DL-Rvs)/mpc)
    D = (1+GW15Z)*(Dfunc(Z2)*mpc - Dfunc(Z1)*mpc)
    return (D*np.pi**2*GW15chirpm/((1+GW15Z)*GW15DeltaBeta))**(1/2)


"""Returns figure object that is the shaded plot of allowable graviton wavelengths vs vainshtein radius
-args:
    delta_beta -> if empty, trys to use a previously calculated fisher and returns exception if no
fisher exists - it will accept any delta_beta as an argument. Including this as an argument allows for the
calculation of different confidence, ie 1.645 sigma is 90%
    comparison -> boolean for comparing the model in question to previous GW detections \el [GW150914,GW151226,GW170104,GW170814,GW170817,GW170608]
plots 5000 points from r_V \el [0.0001 DL/2,DL/2]
- Compares constructed model with the models """
def create_degeneracy_plot(model,delta_beta = None,comparison = True):
    colors = ['r','g','b','c','m','y','k','w']
    alpha_param = .2
    fig,ax = plt.subplots(nrows=1,ncols=1)
    model.ax_ev = ax.twinx()
    ax.callbacks.connect("ylim_changed",lambda ax: convert_lambda_mass_axis(model,ax))

    if delta_beta == None:
        try:
            DB = np.sqrt(np.diagonal(model.inv_fisher))[-1]
        except:
            print("Issue with fisher - please calculate fisher before calling this function")
            return 0
    else:
        DB = delta_beta

    points = 5000
    print("DB: {}".format(DB))
    x,y = create_degeneracy_data(model,delta_beta=DB, points= points)
    lower = np.zeros(len(x))
    ax.fill_between(np.divide(x,mpc),np.multiply(y,c),lower,alpha = alpha_param+.2,color= colors[0],label="Model - Disallowed Region")

    if comparison:
        """Bounds from GW170817 - Propagation speed"""
        try:
            with open(IMRPD_tables_dir+'/GW170817_prop_speed.csv','r') as f:
                reader = csv.reader(f, delimiter=',')
                x = []
                y =[]
                for row in reader:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
        except:
            print('No stored data for {}, writing data to file'.format('GW170817 - Propagation speed'))
            DL_GW170817 = 40*mpc
            x = np.linspace(10e-5*DL_GW170817/2,.9999999*DL_GW170817/2,points)
            y = degeneracy_function_lambda_GW170817(model, x)
            with open(IMRPD_tables_dir+'/GW170817_prop_speed.csv','w') as f:
                writer = csv.writer(f,delimiter=',')
                output = [[x[i],y[i]] for i in np.arange(len(x))]
                for i in output:
                    writer.writerow(list(i))
        lower = np.zeros(len(x))
        ax.fill_between(np.divide(x,mpc),np.multiply(y,c),lower,alpha = alpha_param,color= 'blue')#colors[1],label="Disallowed Region - GW170817 Bound - Speed Constraint")

        """Bounds from GW150914 - Bayesian 90%"""
        try:
            with open(IMRPD_tables_dir+'/GW150914_bayes_90.csv','r') as f:
                reader = csv.reader(f, delimiter=',')
                x = []
                y =[]
                for row in reader:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
        except:
            print('No stored data for {}, writing data to file'.format('GW150914 - Bayesian 90%'))
            GW150914DL = 410*mpc
            x = np.linspace(10e-5*GW150914DL/2,.9999999*GW150914DL/2,points)
            y = degeneracy_function_lambda_GW150914(model,x)
            with open(IMRPD_tables_dir+'/GW150914_bayes_90.csv','w') as f:
                writer = csv.writer(f,delimiter=',')
                output = [[x[i],y[i]] for i in np.arange(len(x))]
                for i in output:
                    writer.writerow(list(i))
        lower = np.zeros(len(x))
        ax.fill_between(np.divide(x,mpc),np.multiply(y,c),lower,alpha=alpha_param, color='grey')#colors[2],label = "Disallowed Region - GW150914 Bound - Bayesian 90% confidence")

        """The rest of the bounds will be calculated and tabulated separately, then read in.
        If plots do not show up, just run the population script in ./Data_Tables"""
        names = ['GW150914','GW151226','GW170104','GW170814','GW170817','GW170608']
        i= 3
        for name in names:
            try:
                with open(IMRPD_tables_dir+'/'+name+'_fisher_deg.csv','r') as f:
                    reader = csv.reader(f, delimiter=',')
                    x =[]
                    y =[]
                    for row in reader:
                        x.append(float(row[0]))
                        y.append(float(row[1]))
                lower = np.zeros(len(x))
                ax.fill_between(np.divide(x,mpc),np.multiply(y,c),lower,alpha = alpha_param,color= 'grey')#np.random.rand(3,),label="Disallowed Region - {} Bound - 1-sigma Fisher Constraint".format(name))
                i+=i
            except:
                print("Data table for {} not populated. Rerun observational_models_data_generation.py".format(name))

    ax.set_ylabel(r'Bound on $\lambda_g$ (meters)')
    ax.set_xlabel(r'Bound on Vainshtein Radius (Mpc)')
    ax.set_title(r'Degeneracy of $\Delta \beta$')
    ax.text(.7,.9,s='Detector: {} \n Masses: {},{} \n Spin: {},{} \n LumDist: {}'.format('aLIGO',model.m1/s_solm,model.m2/s_solm,model.chi1,model.chi2,model.DL/mpc),horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
    model.ax_ev.set_ylabel('Mass (eV)')
    model.ax_ev.grid(False)
    ax.set_yscale('log')
    model.ax_ev.set_yscale('log')
    ax.legend()
    return fig

"""Helper function to populate the data for plotting
Returns: array of Rv, and array of lambda values"""
def create_degeneracy_data(model,delta_beta = None,points = 5000):
    if delta_beta == None:
        try:
            DB = np.sqrt(np.diagonal(model.inv_fisher))[-1]
        except:
            print("Issue with fisher - please calculate fisher before calling this function")
            return 0
    else:
        DB = delta_beta
    x = np.linspace(10e-5*model.DL/2,.9999999*model.DL/2,points)
    y = degeneracy_function_lambda(model,DB,x)
    return x,y

###########################################################################################