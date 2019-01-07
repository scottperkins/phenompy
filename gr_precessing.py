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



######################################################################################################

if __name__=='__main__':
    m1 = 10*s_solm
    m2 = 5*s_solm
    spin1 = np.asarray([.3,.4,.5])
    spin2 = np.asarray([.6,.1,.1])
    dl = 100*mpc
    model = IMRPhenomPv3(mass1=m1,mass2=m2, spin1=spin1,spin2=spin2,collision_time=0,collision_phase=0, Luminosity_Distance=dl, NSflag = False, N_detectors=1)
