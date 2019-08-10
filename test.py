from phenompy.gr import IMRPhenomD as imr                                                              
from phenompy.modified_gr import Modified_IMRPhenomD_Full_Freq as imr_mod                              

from phenompy.modified_gr import Modified_IMRPhenomD_Inspiral_Freq as imr_mod_ins
from phenompy.modified_gr import Modified_IMRPhenomD_Full_Freq_SPA as imr_mod_SPA
from phenompy.modified_gr import Modified_IMRPhenomD_Inspiral_Freq_SPA as imr_mod_SPAins
from phenompy.modified_gr import dCS_IMRPhenomD as imr_dcs
from phenompy.utilities import mpc,s_solm                                                              
#import IMRPhenomD as imrcomp                                                                           
#import IMRPhenomD_full_mod as imrcompmod                                                               
import matplotlib.pyplot as plt                                                                        
import numpy as np                                                                                     
                                                                                                       
#mass1 = np.asarray([20,40,100])*s_solm  
#mass2 = np.asarray([2,40,50])*s_solm                                                                                       
#spin1 = np.asarray([.1,.8,0])
#spin2 = np.asarray([.8,.8,0])                                                                                              
#dl = 100*mpc                                                                                           
#bppe =[3,5,6]
#phase_mod =0                                                                                    
                                                                                                       
mass1 = np.asarray([20,31,40])*s_solm  
mass2 = np.asarray([2,30])*s_solm                                                                                       
spin1 = np.asarray([.7,.9,.05])
spin2 = np.asarray([.8,0])                                                                                              
dl = 100*mpc                                                                                           
bppe =[-3]
phase_mod =0                                                                                    
#classes = [imr,imr_mod,imr_mod_ins,imr_mod_SPA,imr_mod_SPAins,imr_dcs]
#classes = [imr_mod,imr_mod_ins,imr_dcs]
classes = [imr,imr_mod_SPA,imr_mod_SPAins]
#model1 = imr(mass1=mass1,mass2=mass2,spin1=spin1,spin2=spin2,collision_phase=0,collision_time=0,Luminosity_Distance=dl) 
#model2 = imrcomp.IMRPhenomD(mass1=mass1,mass2=mass2,spin1=spin1,spin2=spin2,collision_phase=0,collision_time=0,Luminosity_Distance=dl)
fout = open("testing.txt",'w')
for method in classes:
    for m1 in mass1:
        for m2 in mass2:
            for s1 in spin1:
                for s2 in spin2:
                    for b in bppe: 
                        if method in [imr,imr_mod_SPA,imr_mod_SPAins]:
                            if m1> m2:
                                model1 = method(mass1=m1,mass2=m2,spin1=s1,spin2=s2,collision_phase=0,collision_time=0,Luminosity_Distance=dl)
                            if m2> m1:
                                model1 = method(mass1=m2,mass2=m1,spin1=s2,spin2=s1,collision_phase=0,collision_time=0,Luminosity_Distance=dl)
                        elif method is imr_dcs:
                            if m1> m2:
                                model1 = method(mass1=m1,mass2=m2,spin1=s1,spin2=s2,collision_phase=0,collision_time=0,Luminosity_Distance=dl,phase_mod=phase_mod)
                            if m2> m1:
                                model1 = method(mass1=m2,mass2=m1,spin1=s2,spin2=s1,collision_phase=0,collision_time=0,Luminosity_Distance=dl,phase_mod=phase_mod)
                        else:    
                            if m1> m2:
                                model1 = method(mass1=m1,mass2=m2,spin1=s1,spin2=s2,collision_phase=0,collision_time=0,Luminosity_Distance=dl,phase_mod=phase_mod,bppe=b)
                            if m2> m1:
                                model1 = method(mass1=m2,mass2=m1,spin1=s2,spin2=s1,collision_phase=0,collision_time=0,Luminosity_Distance=dl,phase_mod=phase_mod,bppe=b)
                        print(model1.chi_a,model1.chi_s)
                        #model2 = imrcompmod.Modified_IMRPhenomD(mass1=mass1,mass2=mass2,spin1=spin1,spin2=spin2,collision_phase=0,collision_time=0,Luminosity_Distance=dl,phase_mod=phase_mod,bppe=bppe)                                    
                        #freq = np.linspace(1,2000,1000)
                        try: 
                            fisher1,invfisher1 = model1.calculate_fisher_matrix_vector(detector='aLIGO')
                        except:
                            print(m1/s_solm,m2/s_solm,s1,s2,b)
                            print(method)
                    
                        #fisher2,invfisher2,cholo =model2.calculate_fisher_matrix_vector(detector='aLIGO')
                        for x in np.arange(len(fisher1)):
                            for y in np.arange(len(fisher1[0])):
                                fout.write(str(invfisher1[x][y]))
                                fout.write("\n")
                                #fout.write((invfisher1[x][y]-invfisher2[x][y])/invfisher2[x][y])

#a1,p1,h1 = model1.calculate_waveform_vector(freq)                                                      
#a2, p2,h2 = model2.calculate_waveform_vector(freq)                                                     
#p1 = p1+100
#p2 = p2+100                                                                                            
#plt.plot(freq,(p1-p2)/p2,label='phase')
#plt.plot(freq,(a1-a2)/a2,label='amplitude')
#plt.legend()                                                                                           
#plt.show()

