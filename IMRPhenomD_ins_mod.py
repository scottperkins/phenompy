from IMRPhenomD_full_mod import *

"""Child class of Modified_IMRPhenomD - adds modification to the phase of the waveform in the inspiral region -
extra arguments: phase_mod (=0), bppe (=-3) -> phi_ins = IMRPhenomD.phi_ins + phase_mod*(pi*chirpm*f)**(bppe/3)
Calculation of th modification from Will '97"""
class insModified_IMRPhenomD(Modified_IMRPhenomD):
    # def __init__(self, mass1, mass2,spin1,spin2, collision_time, \
    #                 collision_phase,Luminosity_Distance,phase_mod = 0,bppe = -3,cosmo_model = cosmology.Planck15,NSflag = False,N_detectors = 1):
    #     """Populate model variables"""
    #     self.N_detectors = N_detectors
    #     self.NSflag = NSflag
    #     self.cosmo_model = cosmo_model
    #     self.DL = Luminosity_Distance
    #     self.tc = float(collision_time)
    #     self.phic = float(collision_phase)
    #     self.symmratio = (mass1 * mass2) / (mass1 + mass2 )**2
    #     self.chirpme =  (mass1 * mass2)**(3/5)/(mass1 + mass2)**(1/5)
    #     self.delta = utilities.calculate_delta(self.symmratio)
    #     self.Z =Distance(Luminosity_Distance/mpc,unit=u.Mpc).compute_z(cosmology = self.cosmo_model)#.01
    #     self.chirpm = self.chirpme*(1+self.Z)
    #     self.M = utilities.calculate_totalmass(self.chirpm,self.symmratio)
    #     self.m1 = utilities.calculate_mass1(self.chirpm,self.symmratio)
    #     self.m2 = utilities.calculate_mass2(self.chirpm,self.symmratio)
    #     self.totalMass_restframe = mass1+mass2
    #     self.A0 =(np.pi/30)**(1/2)*self.chirpm**2/self.DL * (np.pi*self.chirpm)**(-7/6)
    #
    #     """Spin Variables"""
    #     self.chi1 = spin1
    #     self.chi2 = spin2
    #     self.chi_s = (spin1 + spin2)/2
    #     self.chi_a = (spin1 - spin2)/2
    #
    #     """Post Newtonian Phase"""
    #     self.pn_phase = np.zeros(8)
    #     for i in [0,1,2,3,4,7]:
    #         self.pn_phase[i] = utilities.calculate_pn_phase(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1,i)
    #
    #     """Numerical Fit Parameters"""
    #     self.parameters =[]
    #     for i in np.arange(len(Lambda)):
    #         self.parameters.append(self.calculate_parameter(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))
    #
    #     """Post Newtonian Amplitude"""
    #     self.pn_amp = np.zeros(7)
    #     for i in np.arange(7):
    #         self.pn_amp[i]=utilities.calculate_pn_amp(self.symmratio,self.delta,self.chi_a,self.chi_s,i)
    #
    #     """Post Merger Parameters - Ring Down frequency and Damping frequency"""
    #     self.fRD = utilities.calculate_postmerger_fRD(\
    #         self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
    #     self.fdamp = utilities.calculate_postmerger_fdamp(\
    #         self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a)
    #     self.fpeak = utilities.calculate_fpeak(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6])
    #
    #     """Calculating the parameters for the intermediate amplitude region"""
    #     self.param_deltas = np.zeros(5)
    #     for i in np.arange(5):
    #         self.param_deltas[i] = self.calculate_delta_parameter(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i)
    #
    #     """Only modifications to the system variables are below:
    #     -beta1
    #     -beta0
    #     -phase_mod
    #     -bppe
    #     -var_arr"""
    #     #################################################################################
    #
    #     """Phase continuity parameters"""
    #     """Must be done in order - beta1,beta0,alpha1, then alpha0"""
    #     self.phase_mod = float(phase_mod)
    #     self.bppe = bppe
    #     self.beta1 = self.phase_cont_beta1(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.phase_mod)
    #     self.beta0 = self.phase_cont_beta0(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1,self.phase_mod)
    #     self.alpha1 = self.phase_cont_alpha1(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1)
    #     self.alpha0 = self.phase_cont_alpha0(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1)
    #     self.var_arr = [self.A0,self.phic,self.tc,self.chirpm,self.symmratio,self.chi_s,self.chi_a,self.phase_mod]
    #
    # def phase_cont_beta1(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod):
    #     M = self.assign_totalmass(chirpm,symmratio)
    #     f1 = 0.018/M
    #     pn_phase =[]
    #     for x in np.arange(len(self.pn_phase)):
    #         pn_phase.append(self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f1,x))
    #     beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
    #     beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
    #     sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
    #     sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
    #     sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
    #     ins_grad = egrad(self.phi_ins,0)
    #     return ((1/M)*ins_grad(f1,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod)*symmratio\
    #         - symmratio/M*egrad(self.phi_int,0)(f1,M,symmratio,0,0,beta2,beta3))
    #      #- (beta2*(1/(M*f1)) + beta3*(M*f1)**(-4)))
    #
    # def phase_cont_beta0(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod):
    #     M = self.assign_totalmass(chirpm,symmratio)
    #     f1 = 0.018/M
    #     pn_phase =[]
    #     for x in np.arange(len(self.pn_phase)):
    #         pn_phase.append(self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f1,x))
    #     beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
    #     beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
    #     sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
    #     sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
    #     sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
    #     return (self.phi_ins(f1,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod)*symmratio
    #         - symmratio*self.phi_int(f1,M,symmratio,0,beta1,beta2,beta3))
    #     #- ( beta1*f1*M + beta2*np.log(M*f1) - beta3/3 *(M*f1)**(-3)))
    #
    # def phi_ins(self,f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod):
    #     return (super(insModified_IMRPhenomD,self).phi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase)+phase_mod*(chirpm*np.pi*f)**(self.bppe/3))
    #
    # def amp_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
    #     return super(insModified_IMRPhenomD,self).amp_ins_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    # def amp_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
    #     return super(insModified_IMRPhenomD,self).amp_int_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    # def amp_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
    #     return super(insModified_IMRPhenomD,self).amp_mr_vector(f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a)
    #
    # """Only added phase_mod to the arguments of beta1 and beta0 - Otherwise, exact copy of GR model method"""
    # def phase_mr_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
    #     M = self.assign_totalmass(chirpm,symmratio)
    #     m1 = self.assign_mass1(chirpm,symmratio)
    #     m2 = self.assign_mass2(chirpm,symmratio)
    #     delta = self.assign_delta(symmratio)
    #     fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
    #     fdamp = self.assign_fdamp(m1,m2,M,symmratio,chi_s,chi_a)
    #     fpeak = self.assign_fpeak(M,fRD,fdamp,self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,5),self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,6))
    #     beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod)
    #     beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod)
    #     alpha1 = self.assign_alpha1(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1)
    #     alpha0 = self.assign_alpha0(chirpm,symmratio,chi_a,chi_s,fRD,fdamp,beta0,beta1,alpha1)
    #     alpha2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,15)
    #     alpha3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,16)
    #     alpha4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,17)
    #     alpha5 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,18)
    #     return self.phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp)
    #
    # """Uses overriden phi_ins method - added phase_mod arg"""
    # def phase_ins_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
    #     M = self.assign_totalmass(chirpm,symmratio)
    #     m1 = self.assign_mass1(chirpm,symmratio)
    #     m2 = self.assign_mass2(chirpm,symmratio)
    #     delta = self.assign_delta(symmratio)
    #     fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
    #     sigma2 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,8)
    #     sigma3 =self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,9)
    #     sigma4 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,10)
    #     pn_phase= []
    #     for i in [0,1,2,3,4,5,6,7]:
    #         pn_phase.append( self.assign_pn_phase(chirpm,symmratio,delta,chi_a,chi_s,f,i))
    #     return self.phi_ins(f,phic,tc,chirpm,symmratio,delta,chi_a,chi_s,sigma2,sigma3,sigma4,pn_phase,phase_mod)
    #
    # """Added phase_mod arg to beta parameter calls"""
    # def phase_int_vector(self,f,A0,phic,tc,chirpm,symmratio,chi_s,chi_a,phase_mod):
    #     M = self.assign_totalmass(chirpm,symmratio)
    #     m1 = self.assign_mass1(chirpm,symmratio)
    #     m2 = self.assign_mass2(chirpm,symmratio)
    #     delta = self.assign_delta(symmratio)
    #     fRD = self.assign_fRD(m1,m2,M,symmratio,chi_s,chi_a)
    #     beta1 = self.assign_beta1(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod)
    #     beta0 = self.assign_beta0(chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod)
    #     beta2 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,12)
    #     beta3 = self.assign_lambda_param(chirpm,symmratio,chi_a,chi_s,13)
    #     return self.phi_int(f,M,symmratio,beta0,beta1,beta2,beta3)
    #
    # """Added derivatives for all the mod phi arguments - """
    # def calculate_derivatives(self):
    #     """Pre-calculate Derivatives here - (Order does matter - parameter functions may be functions of system constants)
    #     If the variable is instantiated as an array, a derivate array for each system variable is created and is cycled through
    #     (ie Lambda paramaters is parameters[i] and has derivate arrays parameters_deriv_symmratio etc).
    #     If the variable is a single value, the variable has one array of derivates, the elements of which are the derivatives wrt
    #     various system variables (ie M -> M_deriv[i] for symmratio and chripm etc)
    #     -M
    #     -m1
    #     -m2
    #     -Lambda parameters
    #     -pn_amp
    #     -pn_phase
    #     -delta parameters (intermediate amplitude parameters)
    #     -fRD
    #     -fdamp
    #     -fpeak
    #     -delta (mass parameter)
    #     -phase continuitiy variables (beta1,beta0,alpha1,alpha0)
    #     """
    #
    #     self.total_mass_deriv = []
    #     for i in range(2):
    #         self.total_mass_deriv.append(grad(utilities.calculate_totalmass,i)(self.chirpm,self.symmratio))
    #     self.mass1_deriv = []
    #     for i in range(2):
    #         self.mass1_deriv.append(grad(utilities.calculate_mass1,i)(self.chirpm,self.symmratio))
    #     self.mass2_deriv = []
    #     for i in range(2):
    #         self.mass2_deriv.append(grad(utilities.calculate_mass2,i)(self.chirpm,self.symmratio))
    #     self.lambda_derivs_symmratio=[]
    #     self.lambda_derivs_chirpm = []
    #     self.lambda_derivs_chi_a = []
    #     self.lambda_derivs_chi_s = []
    #     for i in np.arange(len(Lambda)):
    #         self.lambda_derivs_chirpm.append(grad(self.calculate_parameter,0)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))
    #         self.lambda_derivs_symmratio.append(grad(self.calculate_parameter,1)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))
    #         self.lambda_derivs_chi_a.append(grad(self.calculate_parameter,2)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))
    #         self.lambda_derivs_chi_s.append(grad(self.calculate_parameter,3)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,i))
    #
    #     self.pn_amp_deriv_symmratio = []
    #     self.pn_amp_deriv_delta = []
    #     self.pn_amp_deriv_chi_a = []
    #     self.pn_amp_deriv_chi_s = []
    #     for i in np.arange(len(self.pn_amp)):
    #         self.pn_amp_deriv_symmratio.append(grad(utilities.calculate_pn_amp,0)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
    #         self.pn_amp_deriv_delta.append(grad(utilities.calculate_pn_amp,1)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
    #         self.pn_amp_deriv_chi_a.append(grad(utilities.calculate_pn_amp,2)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
    #         self.pn_amp_deriv_chi_s.append(grad(utilities.calculate_pn_amp,3)(self.symmratio,self.delta,self.chi_a,self.chi_s,i))
    #     self.pn_phase_deriv_chirpm = []
    #     self.pn_phase_deriv_symmratio = []
    #     self.pn_phase_deriv_delta = []
    #     self.pn_phase_deriv_chi_a = []
    #     self.pn_phase_deriv_chi_s = []
    #     for i in np.arange(len(self.pn_phase)):
    #         self.pn_phase_deriv_chirpm.append(grad(utilities.calculate_pn_phase,0)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
    #         self.pn_phase_deriv_symmratio.append(grad(utilities.calculate_pn_phase,1)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
    #         self.pn_phase_deriv_delta.append(grad(utilities.calculate_pn_phase,2)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
    #         self.pn_phase_deriv_chi_a.append(grad(utilities.calculate_pn_phase,3)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
    #         self.pn_phase_deriv_chi_s.append(grad(utilities.calculate_pn_phase,4)(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,1.1,i))
    #     """Delta Parameters take up ~50 percent of the total time"""
    #
    #     temp1 = grad(self.calculate_delta_parameter,0)
    #     temp2 = grad(self.calculate_delta_parameter,1)
    #     temp3 = grad(self.calculate_delta_parameter,2)
    #     temp4 = grad(self.calculate_delta_parameter,3)
    #     temp5 = grad(self.calculate_delta_parameter,4)
    #     temp6 = grad(self.calculate_delta_parameter,5)
    #     temp7 = grad(self.calculate_delta_parameter,6)
    #     temp8 = grad(self.calculate_delta_parameter,7)
    #
    #     self.param_deltas_derivs_chirpm = list(map(lambda i:temp1(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
    #     self.param_deltas_derivs_symmratio = list(map(lambda i:temp2(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
    #     self.param_deltas_derivs_delta = list(map(lambda i:temp3(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
    #     self.param_deltas_derivs_chi_a = list(map(lambda i:temp4(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
    #     self.param_deltas_derivs_chi_s = list(map(lambda i:temp5(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
    #     self.param_deltas_derivs_fRD = list(map(lambda i:temp6(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
    #     self.param_deltas_derivs_fdamp = list(map(lambda i:temp7(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
    #     self.param_deltas_derivs_f3 = list(map(lambda i:temp8(self.chirpm,self.symmratio,self.delta,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.fpeak,i), np.arange(5) ))
    #     # print("deltas: {}".format(time()-start))
    #     self.fRD_deriv = []
    #     for i in range(6):
    #         self.fRD_deriv.append(grad(utilities.calculate_postmerger_fRD,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
    #     self.fdamp_deriv = []
    #     for i in range(6):
    #         self.fdamp_deriv.append(grad(utilities.calculate_postmerger_fdamp,i)(self.m1,self.m2,self.M,self.symmratio,self.chi_s,self.chi_a))
    #     self.fpeak_deriv = []
    #     for i in range(5):
    #         self.fpeak_deriv.append(grad(utilities.calculate_fpeak,i)(self.M,self.fRD,self.fdamp,self.parameters[5],self.parameters[6]))
    #
    #     """Only deviations from original IMRPhenomD are below: extra derivative for beta1,beta0, and extra log_factor"""
    #     ########################################################################################################################
    #     self.delta_deriv = grad(utilities.calculate_delta)(self.symmratio)
    #     self.beta1_deriv = []
    #     self.beta0_deriv = []
    #     self.alpha1_deriv = []
    #     self.alpha0_deriv = []
    #     for i in range(8):
    #         self.beta1_deriv.append(grad(self.phase_cont_beta1,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.phase_mod))
    #     for i in range(9):
    #         self.beta0_deriv.append(grad(self.phase_cont_beta0,i)(self.chirpm,self.symmratio,self.delta,self.phic,self.tc,self.chi_a,self.chi_s,self.beta1,self.phase_mod))
    #     for i in range(8):
    #         self.alpha1_deriv.append(grad(self.phase_cont_alpha1,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1))
    #     for i in range(9):
    #         self.alpha0_deriv.append(grad(self.phase_cont_alpha0,i)(self.chirpm,self.symmratio,self.chi_a,self.chi_s,self.fRD,self.fdamp,self.beta0,self.beta1,self.alpha1))
    #
    #     """Populate array with variables for transformation from d/d(theta) to d/d(log(theta)) - begins with 0 because fisher matrix variables start at 1, not 0"""
    #     self.log_factors = [0,self.A0,1,1,self.chirpm,self.symmratio,1,1,1]
    #
    # """Function for actual element integrand - 4*Re(dh/dtheta_i* dh/dtheta_j) - Vectorized
    # -added extra mod_phi argument"""
    # def calculate_waveform_derivative_vector(self,famp,fphase,i):
    #     """Array of the functions used to populate derivative vectors"""
    #     ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
    #     phasefunc = [self.phase_ins_vector,self.phase_int_vector,self.phase_mr_vector]
    #     """Check to see if every region is sampled - if integration frequency
    #     doesn't reach a region, the loop is trimmed to avoid issues with unpopulated arrays"""
    #
    #     jamp=[0,1,2]
    #     jphase=[0,1,2]
    #     if len(famp[0]) == 0:
    #         if len(famp[1])==0:
    #             jamp = [2]
    #         else:
    #             if len(famp[2])==0:
    #                 jamp = [1]
    #             else:
    #                 jamp = [1,2]
    #     if len(famp[2])==0:
    #         if len(famp[1]) == 0:
    #             jamp = [0]
    #         else:
    #             if len(famp[0])==0:
    #                 jamp = [1]
    #             else:
    #                 jamp = [0,1]
    #     if len(fphase[0]) == 0:
    #         if len(fphase[1])==0:
    #             jphase = [2]
    #         else:
    #             if len(fphase[2])==0:
    #                 jphase = [1]
    #             else:
    #                 jphase = [1,2]
    #     if len(fphase[2])==0:
    #         if len(fphase[1]) == 0:
    #             jphase = [0]
    #         else:
    #             if len(fphase[0])==0:
    #                 jphase = [1]
    #             else:
    #                 jphase = [0,1]
    #
    #     # jamp = [0,1,2]
    #     # for i in np.arange(len(famp)):
    #     #     if len(famp[i]) == 0:
    #     #         jamp[i] = -1
    #     # jamp = [x for x in jamp if x != -1]
    #     # jphase = [0,1,2]
    #     # for i in np.arange(len(fphase)):
    #     #     if len(fphase[i]) == 0:
    #     #         jphase[i] = -1
    #     # jphase = [x for x in jphase if x != -1]
    #     #
    #
    #     var_arr= self.var_arr[:]
    #     gamp = [[],[],[]]
    #     amp = [[],[],[]]
    #     phase = [[],[],[]]
    #     """Array of the functions used to populate derivative vectors"""
    #     ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
    #     phasefunc = [self.phase_ins_vector,self.phase_int_vector,self.phase_mr_vector]
    #
    #     """Populate derivative vectors one region at a time"""
    #     for j in jamp:
    #         var_arr= self.var_arr[:]
    #         famp[j], var_arr[i-1] = np.broadcast_arrays(famp[j],var_arr[i-1])
    #         gamp[j]=( egrad(ampfunc[j],i)(famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7]))
    #         var_arr= self.var_arr[:]
    #         amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
    #     for j in jphase:
    #         var_arr= self.var_arr[:]
    #         fphase[j], var_arr[i-1] = np.broadcast_arrays(fphase[j],var_arr[i-1])
    #         phase[j]=( egrad(phasefunc[j],i)(fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7]))
    #
    #     """Concatenate the regions into one array"""
    #     gampout,ampout,phaseout = [],[],[]
    #     for j in jamp:
    #         ampout = np.concatenate((ampout,amp[j]))
    #         gampout = np.concatenate((gampout,gamp[j]))
    #     for j in jphase:
    #         phaseout = np.concatenate((phaseout,phase[j]))
    #
    #     """Return the complex waveform derivative"""
    #     return np.subtract(gampout,np.multiply(ampout,np.multiply(1j,phaseout)))
    #
    # """Calculate the waveform - vectorized
    # Outputs: amp vector, phase vector, (real) waveform vector
    # -added extra mod_phi argument"""
    # def calculate_waveform_vector(self,freq):
    #     """Array of the functions used to populate derivative vectors"""
    #     ampfunc = [self.amp_ins_vector,self.amp_int_vector,self.amp_mr_vector]
    #     phasefunc = [self.phase_ins_vector,self.phase_int_vector,self.phase_mr_vector]
    #     """Check to see if every region is sampled - if integration frequency
    #     doesn't reach a region, the loop is trimmed to avoid issues with unpopulated arrays"""
    #
    #     famp = self.split_freqs_amp(freq)
    #     fphase = self.split_freqs_phase(freq)
    #
    #     # jamp=[0,1,2]
    #     # jphase=[0,1,2]
    #     # if len(famp[0]) == 0:
    #     #     if len(famp[1])==0:
    #     #         jamp = [2]
    #     #     else:
    #     #         if len(famp[2])==0:
    #     #             jamp = [1]
    #     #         else:
    #     #             jamp = [1,2]
    #     # if len(famp[2])==0:
    #     #     if len(famp[1]) == 0:
    #     #         jamp = [0]
    #     #     else:
    #     #         if len(famp[0])==0:
    #     #             jamp = [1]
    #     #         else:
    #     #             jamp = [0,1]
    #     #
    #     # if len(fphase[0]) == 0:
    #     #     if len(fphase[1])==0:
    #     #         jphase = [2]
    #     #     else:
    #     #         if len(fphase[2])==0:
    #     #             jphase = [1]
    #     #         else:
    #     #             jphase = [1,2]
    #     # if len(fphase[2])==0:
    #     #     if len(fphase[1]) == 0:
    #     #         jphase = [0]
    #     #     else:
    #     #         if len(fphase[0])==0:
    #     #             jphase = [1]
    #     #         else:
    #     #             jphase = [0,1]
    #     jamp = [0,1,2]
    #     for i in np.arange(len(famp)):
    #         if len(famp[i]) == 0:
    #             jamp[i] = -1
    #     jamp = [x for x in jamp if x != -1]
    #     jphase = [0,1,2]
    #     for i in np.arange(len(fphase)):
    #         if len(fphase[i]) == 0:
    #             jphase[i] = -1
    #     jphase = [x for x in jphase if x != -1]
    #
    #     var_arr= self.var_arr[:]
    #     amp = [[],[],[]]
    #     phase = [[],[],[]]
    #
    #     """Populate derivative vectors one region at a time"""
    #     for j in jamp:
    #         amp[j]= ampfunc[j](famp[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
    #     for j in jphase:
    #         phase[j]=phasefunc[j](fphase[j],var_arr[0],var_arr[1],var_arr[2],var_arr[3],var_arr[4],var_arr[5],var_arr[6],var_arr[7])
    #
    #     """Concatenate the regions into one array"""
    #     ampout,phaseout =[],[]
    #     for j in jamp:
    #         ampout = np.concatenate((ampout,amp[j]))
    #     for j in jphase:
    #         phaseout = np.concatenate((phaseout,phase[j]))
    #
    #     """Return the amplitude vector, phase vector, and real part of the waveform"""
    #     return ampout,phaseout, np.multiply(ampout,np.cos(phaseout))
    #
    # """Derivative Definitions - added phase_mod to derivatives"""
    # @primitive
    # def assign_beta1(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod):
    #     for j in [chirpm,delta,symmratio,phic,tc,chi_a,chi_s,phase_mod]:
    #         if isinstance(j,np.ndarray):
    #             return np.ones(len(j))*self.beta1
    #     return self.beta1
    # defvjp(assign_beta1,None,
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[0],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[1],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[2],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[3],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[4],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[5],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[6],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,phase_mod: lambda g: g*self.beta1_deriv[7])
    # @primitive
    # def assign_beta0(self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod):
    #     for j in [chirpm,delta,symmratio,phic,tc,chi_a,chi_s,beta1,phase_mod]:
    #         if isinstance(j,np.ndarray):
    #             return np.ones(len(j))*self.beta0
    #     return self.beta0
    # defvjp(assign_beta0,None,
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[0],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[1],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[2],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[3],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[4],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[5],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[6],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[7],
    #             lambda ans,self,chirpm,symmratio,delta,phic,tc,chi_a,chi_s,beta1,phase_mod: lambda g: g*self.beta0_deriv[8])
    #
    # """Function for plotting degeneracy of Delta Beta, which depends on both the wavelength
    # of the graviton and the screening radius
    # Delta beta = pi^2 *self.chirpm*D/((1+Z)*lambda^2)
    # - D depends on the screening radius (D proportional to the integral_r1^r2 a dt)
    # returns D as a function of lambda
    #
    # Second method returns lambda_g as a function of screening radius -
    # Assumptions: the mass of the two galaxies are approximately equal
    # arguments: delta_beta, Rvs - List of Vainshtein radii (cannot be larger than DL/2)"""
    #
    # def degeneracy_function_D(self, delta_beta, lambda_g ):
    #     return delta_beta* lambda_g**2 *(1+ self.Z)/(np.pi**2 *self.chirpm)
    # def degeneracy_function_lambda(self,delta_beta, Rvs):
    #     lambdas = []
    #     H0=self.cosmo_model.H0.to('Hz').value#self.cosmo_model.H(0).u()
    #     Z1 = Zfunc(Rvs/mpc)
    #     Z2 = Zfunc((self.DL-Rvs)/mpc)
    #     D = (1+self.Z)*(Dfunc(Z2)*mpc- Dfunc(Z1)*mpc)
    #     return (D*np.pi**2*self.chirpm/((1+self.Z)*delta_beta))**(1/2)
    #
    #
    # """Add the Mass unit axis to plot"""
    # def convert_wavelength_mass(self, wavelength):
    #     return hplanck * c / (wavelength )
    # def convert_lambda_mass_axis(self, ax):
    #     l1, l2 = ax.get_ylim()
    #     self.ax_ev.set_ylim(self.convert_wavelength_mass(l1),self.convert_wavelength_mass(l2))
    #     self.ax_ev.figure.canvas.draw()
    #
    # """Plot the constraint on Beta from the TOA difference of the peak GW signal and peak gamma ray detection
    # measured for GW170817 -
    # arguments: screening radius array (seconds)"""
    # def degeneracy_function_lambda_GW170817(self,Rvs):
    #     lambdas = []
    #     H0=self.cosmo_model.H0.to('Hz').value#self.cosmo_model.H(0).u()
    #     DL_GW170817 = 40*mpc
    #     Delta_T_measured = 1.7 #seconds
    #     fpeak_measured =3300 #3300#REF 1805.11579
    #
    #     Z = Distance(DL_GW170817/mpc,unit=u.Mpc).compute_z(cosmology = self.cosmo_model)
    #
    #     Z1 = Zfunc(Rvs/mpc)
    #     Z2 = Zfunc((DL_GW170817-Rvs)/mpc)
    #     D = (1+Z)*(Dfunc(Z2)*mpc - Dfunc(Z1)*mpc)
    #     return (2*Delta_T_measured*fpeak_measured**2/((1+Z)*D))**(-1/2)
    #
    #
    # """Results of Fisher analysis on GW150914 -  Refs 1602.03840, PhysRevLett.116.221101"""
    # def degeneracy_function_lambda_GW150914(self,Rvs):
    #     lambdas = []
    #     H0=self.cosmo_model.H0.to('Hz').value#self.cosmo_model.H(0).u()
    #     GW15DL = 410*mpc # MPC
    #     GW15chirpm = 30.4 *s_solm#Solar Masses
    #     GW15Z = .088
    #     GW15lambdag = 1e16/c # seconds
    #     GW15D = (1+GW15Z)*(integrate.quad(lambda x: 1/(H0*(1+x)**2*np.sqrt(.3*(1+x)**3 + .7)),0,GW15Z )[0])
    #
    #     GW15DeltaBeta = (GW15D*np.pi**2*GW15chirpm)/((1+GW15Z)*GW15lambdag**2)
    #
    #     Z1 = Zfunc(Rvs/mpc)
    #     Z2 = Zfunc((GW15DL-Rvs)/mpc)
    #     D = (1+GW15Z)*(Dfunc(Z2)*mpc - Dfunc(Z1)*mpc)
    #     return (D*np.pi**2*GW15chirpm/((1+GW15Z)*GW15DeltaBeta))**(1/2)
    #
    #
    # """Returns figure object that is the shaded plot of allowable graviton wavelengths vs vainshtein radius
    # -args:
    #     delta_beta -> if empty, trys to use a previously calculated fisher and returns exception if no
    # fisher exists - it will accept any delta_beta as an argument
    #     comparison -> boolean for comparing the model in question to previous GW detections \el [GW150914,GW151226,GW170104,GW170814,GW170817,GW170608]
    # plots 5000 points from r_V \el [0.0001 DL/2,DL/2]
    # - Compares constructed model with the models """
    # def create_degeneracy_plot(self,delta_beta = None,comparison = True):
    #     colors = ['r','g','b','c','m','y','k','w']
    #     alpha_param = .2
    #     fig,ax = plt.subplots(nrows=1,ncols=1)
    #     self.ax_ev = ax.twinx()
    #     ax.callbacks.connect("ylim_changed",self.convert_lambda_mass_axis)
    #
    #     if delta_beta == None:
    #         try:
    #             DB = np.sqrt(np.diagonal(self.inv_fisher))[-1]
    #         except:
    #             print("Issue with fisher - please calculate fisher before calling this function")
    #             return 0
    #     else:
    #         DB = delta_beta
    #
    #     points = 5000
    #     x = np.linspace(10e-5*self.DL/2,.9999999*self.DL/2,points)
    #     y = self.degeneracy_function_lambda(DB,x)
    #     lower = np.zeros(len(x))
    #     ax.fill_between(np.divide(x,mpc),np.multiply(y,c),lower,alpha = alpha_param+.2,color= colors[0],label="Model - Disallowed Region")
    #
    #     if comparison:
    #         """Bounds from GW170817 - Propagation speed"""
    #         try:
    #             with open(IMRPD_tables_dir+'/GW170817_prop_speed.csv','r') as f:
    #                 reader = csv.reader(f, delimiter=',')
    #                 x = np.array([])
    #                 y =np.array([])
    #                 for row in reader:
    #                     x.append(float(row[0]))
    #                     y.append(float(row[1]))
    #         except:
    #             DL_GW170817 = 40*mpc
    #             x = np.linspace(10e-5*DL_GW170817/2,.9999999*DL_GW170817/2,points)
    #             y = self.degeneracy_function_lambda_GW170817(x)
    #             with open(IMRPD_tables_dir+'/GW170817_prop_speed.csv','w') as f:
    #                 writer = csv.writer(f,delimiter=',')
    #                 output = [[x[i],y[i]] for i in np.arange(len(x))]
    #                 writer.writerow(output)
    #         lower = np.zeros(len(x))
    #         ax.fill_between(np.divide(x,mpc),np.multiply(y,c),lower,alpha = alpha_param,color= 'blue')#colors[1],label="Disallowed Region - GW170817 Bound - Speed Constraint")
    #
    #         """Bounds from GW150914 - Bayesian 90%"""
    #         try:
    #             with open(IMRPD_tables_dir+'/GW150914_bayes_90.csv','r') as f:
    #                 reader = csv.reader(f, delimiter=',')
    #                 x = np.array([])
    #                 y =np.array([])
    #                 for row in reader:
    #                     x.append(float(row[0]))
    #                     y.append(float(row[1]))
    #         except:
    #             GW150914DL = 410*mpc
    #             x = np.linspace(10e-5*GW150914DL/2,.9999999*GW150914DL/2,points)
    #             y = self.degeneracy_function_lambda_GW150914(x)
    #             with open(IMRPD_tables_dir+'/GW150914_bayes_90.csv','w') as f:
    #                 writer = csv.writer(f,delimiter=',')
    #                 output = [[x[i],y[i]] for i in np.arange(len(x))]
    #                 writer.writerow(output)
    #         lower = np.zeros(len(x))
    #         ax.fill_between(np.divide(x,mpc),np.multiply(y,c),lower,alpha=alpha_param, color='grey')#colors[2],label = "Disallowed Region - GW150914 Bound - Bayesian 90% confidence")
    #
    #         """The rest of the bounds will be calculated and tabulated separately, then read in.
    #         If plots do not show up, just run the population script in ./Data_Tables"""
    #         names = ['GW150914','GW151226','GW170104','GW170814','GW170817','GW170608']
    #         i= 3
    #         for name in names:
    #             try:
    #                 with open(IMRPD_tables_dir+'/'+name+'_fisher_deg.csv','r') as f:
    #                     reader = csv.reader(f, delimiter=',')
    #                     x =[]
    #                     y =[]
    #                     for row in reader:
    #                         x.append(float(row[0]))
    #                         y.append(float(row[1]))
    #                 lower = np.zeros(len(x))
    #                 ax.fill_between(np.divide(x,mpc),np.multiply(y,c),lower,alpha = alpha_param,color= 'grey')#np.random.rand(3,),label="Disallowed Region - {} Bound - 1-sigma Fisher Constraint".format(name))
    #                 i+=i
    #             except:
    #                 print("Data table for {} not populated. Rerun observational_models_data_generation.py".format(name))
    #
    #     ax.set_ylabel(r'Bound on $\lambda_g$ (meters)')
    #     ax.set_xlabel(r'Bound on Vainshtein Radius (Mpc)')
    #     ax.set_title(r'Degeneracy of $\Delta \beta$')
    #     ax.text(.7,.9,s='Detector: {} \n Masses: {},{} \n Spin: {},{} \n LumDist: {}'.format('aLIGO',self.m1/s_solm,self.m2/s_solm,self.chi1,self.chi2,self.DL/mpc),horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
    #     self.ax_ev.set_ylabel('Mass (eV)')
    #     self.ax_ev.grid(False)
    #     ax.set_yscale('log')
    #     self.ax_ev.set_yscale('log')
    #     ax.legend()
    #     return fig
    #
    # def create_degeneracy_data(self,delta_beta = None,points = 5000):
    #     if delta_beta == None:
    #         try:
    #             DB = np.sqrt(np.diagonal(self.inv_fisher))[-1]
    #         except:
    #             print("Issue with fisher - please calculate fisher before calling this function")
    #             return 0
    #     else:
    #         DB = delta_beta
    #     x = np.linspace(10e-5*self.DL/2,.9999999*self.DL/2,points)
    #     y = self.degeneracy_function_lambda(DB,x)
    #     return x,y
    def phi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD,self).phi_int(f,M,symmratio,beta0,beta1,beta2,beta3))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def phi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD,self).phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
