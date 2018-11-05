import numpy as np
import IMRPhenomD_full_mod as imrmod
# import IMRPhenomD_ins_mod as imrmod
import csv
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import seaborn as sns
import pandas
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import integrate
import matplotlib.lines as mlines

hplanck = imrmod.hplanck
hplanckJs = 6.62607004e-34 #Js
G = 6.674e-11 #N kg^-2 m^2
c = imrmod.c
mpc = imrmod.mpc
s_solm = imrmod.s_solm
Mpl = np.sqrt(hplanckJs*c / ( G)) * (G/c**3) # Planck mass * 4 pi in seconds -> [s/kg]
sns.set(style='whitegrid')
sns.set_palette('colorblind')
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
# plt.rc('text',usetex=True)

points = 5000
num_detect = 1
data_filetree = os.path.dirname(os.path.realpath(__file__))
lower = np.zeros(points)
colors = sns.color_palette('colorblind')#['black','firebrick', 'olivedrab','darkcyan','slateblue','lightblue']
a = .4 #alpha for plots
aline = 1
figdim = [6,3] # in inches


linestyles = {'solid':               (0, ()),
     'loosely dotted':      (0, (1, 10)),
     'dotted':              (0, (1, 5)),
     'densely dotted':      (0, (1, 1)),

     'loosely dashed':      (0, (5, 10)),
     'dashed':              (0, (5, 5)),
     'densely dashed':      (0, (5, 1)),

     'loosely dashdotted':  (0, (3, 10, 1, 10)),
     'dashdotted':          (0, (3, 5, 1, 5)),
     'densely dashdotted':  (0, (3, 1, 1, 1)),

     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}
styles_paper = ['solid','densely dotted','densely dashed','densely dashdotted','densely dashdotdotted','dashdotted']
linestyles_paper = [linestyles[x] for x in styles_paper]
axis_digits = 1
line_width=1

"""Add the Mass unit axis to plot"""
def convert_wavelength_mass(wavelength):
    return hplanck * c / (wavelength )
def convert_lambda_mass_axis(ax, ax_ev):
    l1, l2 = ax.get_ylim()
    ax_ev.set_ylim(convert_wavelength_mass(l1),convert_wavelength_mass(l2))
    ax_ev.figure.canvas.draw()


#return the vainshtein radius for dRGT given wavelength in seconds
gal_mass = 5.8e11*s_solm
def rv_drgt(lam):
    return (gal_mass*lam**2/(2*np.pi)**2)**(1/3)
mass1 =np.array([20,35,1e6],dtype=float)*s_solm
mass2 = np.array([15,10,1e5],dtype=float)*s_solm
DL = np.array([700,1e3,1.5e3],dtype=float)*mpc
spin1 = np.array([.8,.5,.4],dtype=float)
spin2 = np.array([.9,.3,.7],dtype=float)
detectors = ['aLIGO','ET-D','LISA']
fig,axes = plt.subplots(figsize=figdim)
ax_ev = axes.twinx()
legend =[]
E = 1e1
L = 1e4
# names = [r'aLIGO $A$ = $1$',r'ET $A$ = $10$',r'LISA $A$ = $10^4$']
names = [r'aLIGO',r'ET',r'LISA']

for i in np.arange(len(mass1)):
    model = imrmod.Modified_IMRPhenomD(mass1=mass1[i],mass2=mass2[i],spin1=spin1[i],spin2=spin2[i],\
        collision_time=0,collision_phase=0,Luminosity_Distance=DL[i],NSflag=False,N_detectors=1)
    fish,invfish,cholo = model.calculate_fisher_matrix_vector(detectors[i])
    x,y= model.create_degeneracy_data(points = points)
    if detectors[i] == 'aLIGO':
        x = np.divide(x,mpc)
        y = np.multiply(y,c)
    if detectors[i] == 'ET-D':
        x = np.divide(x,mpc)
        y = np.multiply(y,c/E)
    if detectors[i] == 'LISA':
        x = np.divide(x,mpc)
        y = np.multiply(y,c/L)
    lower = np.zeros(points)
    axes.fill_between(x,y,lower,alpha = a,color=colors[i])
    #legend.append(mlines.Line2D([],[],color=colors[i],
#                    linestyle=linestyles_paper[i]))
    axes.plot(x,y,alpha=aline, linestyle=linestyles_paper[i],color=colors[i],linewidth=line_width,label=names[i])




axes.set_ylabel(r'$\lambda_g$/$A$ ($10^{16} $ meters)')
axes.set_xlabel(r'Vainshtein Radius (Mpc)')
axes.set_xlim([0,axes.get_xlim()[1]])
axes.set_ylim([0,round(axes.get_ylim()[1],0)])

#########################################################################
ticks = 7
axes.set_yticks([x for x in np.linspace(axes.get_ylim()[0],axes.get_ylim()[1],ticks)])
axes.set_yticklabels([round(x/10**16,1) for x in axes.get_yticks()])
ax_ev_ticks = [round(convert_wavelength_mass(x)/10**-22,1) for x in axes.get_yticks() if x!= 0]
ax_ev_ticks.insert(0,r'$\infty$')
ax_ev.set_yticklabels(ax_ev_ticks)
ax_ev.set_yticks(axes.get_yticks())
ax_ev.set_ylabel(r'$m_g/A$ ($10^{-22}$ eV)')
#########################################################################
ax_ev.grid(False)

# axes_ligo.legend(prop={'size':5})
plt.tight_layout()
# axes.legend(legend,names)
axes.legend()
# plt.show()
plt.savefig('../../../Figures/Scott/illustrative_plot.pdf')
plt.close()
