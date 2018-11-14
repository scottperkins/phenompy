from IMRPhenomD_full_mod import *

"""Child class of Modified_IMRPhenomD - adds modification to the phase of the waveform in the inspiral region -
extra arguments: phase_mod (=0), bppe (=-3) -> phi_ins = IMRPhenomD.phi_ins + phase_mod*(pi*chirpm*f)**(bppe/3)
Calculation of th modification from Will '97"""
class insModified_IMRPhenomD(Modified_IMRPhenomD):
    def phi_int(self,f,M,symmratio,beta0,beta1,beta2,beta3,chirpm,phase_mod):
        return (super(Modified_IMRPhenomD,self).phi_int(f,M,symmratio,beta0,beta1,beta2,beta3))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
    def phi_mr(self,f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp,phase_mod):
        return (super(Modified_IMRPhenomD,self).phi_mr(f,chirpm,symmratio,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,fRD,fdamp))#-phase_mod*(chirpm*np.pi*f)**(self.bppe/3)
