import numpy as np
import multiprocessing as mp
import csv
import astropy.cosmology as cosmos
import astropy.units as u
from astropy.coordinates import Distance
from scipy.integrate import quad
import matplotlib.pyplot as plt
from IMRPhenomD import mpc
from scipy.interpolate import interp1d,CubicSpline
import os
from time import time
if __name__=="__main__":
    data_filetree = os.path.dirname(os.path.realpath(__file__))
    cosmo = cosmos.Planck15

    # def helper(y):
    #     return Distance(y,unit=u.Mpc).compute_z(cosmology = cosmos.Planck15)
    # def helper2(zfinal):
    #     return quad(lambda x: 1/(H0*(1+x)**2*np.sqrt(.3*(1+x)**3 + .7)),0,zfinal )[0]
    def H(z):
        return cosmo.H(z).to('Hz').value
    def helperold(y):
        return Distance(y,unit=u.Mpc).compute_z(cosmology = cosmo)
    def helper2(zfinal):
        return quad(lambda x: 1/((1+x)**2*H(x)),0,zfinal )[0]
    def helper(y):
        return (1+y)*quad(lambda x: 1/H(x),0,y)[0]
    pool = mp.Pool(processes=mp.cpu_count())

    # dl = np.linspace(1e-3,500,1000)
    # y1 = helper(dl)
    # y2 = list(map(helperold,dl))
    # # plt.plot(dl,helper(dl),label='interpolated')
    # # plt.plot(dl,list(map(helperold,dl)),label='astropy')
    # plt.plot(dl,(y1-y2)/y2,label='astropy')
    # plt.legend()
    # plt.show()
    # plt.close()
    # H0=cosmos.Planck15.H0.to('Hz').value
    #############################################################
    #This code is used to tabulate data for the mapping of redshift to luminosity Distance
    #to be interpolated later for speed
    #File has the form: LumD , Z
    #LumD ranges from 1 to 50000 MPC
    #############################################################

    # ld = np.linspace(0,50000,1e5)
    #
    # z = pool.map(helper,ld)

    z = np.linspace(0,20,1e5)
    # dl = np.asarray([(1+zi)*quad(lambda x: 1/H(x),0,zi)[0] for zi in z])/mpc
    ld = np.asarray(pool.map(helper, z))/mpc

    with open(data_filetree+'/tabulated_LumD_Z.csv','w') as file:
        writer = csv.writer(file, delimiter=',')
        row = [[ld[t],z[t]] for t in np.arange(len(ld))]
        for i in row:
            writer.writerow(list(i))

    #############################################################
    #This code is used to tabulate data for the mapping of redshift to cosmological distance D/(1+Z)
    # defined in Will '97 to be interpolated later for speed
    # D/(1+Z) = integral^Z_0 dz/((1+Z)**2*np.sqrt(.3(1+Z)**3 + .7))
    #using the Lambda CDM universe with Omega_M = 0.3 and Omega_Lambda =0.7
    #File has the form: Z , D
    #Z ranges from 0 to 1.5
    #############################################################
    z = np.linspace(0,10,1e5)
    d = np.asarray(pool.map(helper2, z))
    with open(data_filetree+'/tabulated_Z_D.csv','w') as file:
        writer = csv.writer(file, delimiter=',')
        row = [[z[t],d[t]/mpc] for t in np.arange(len(d))]
        for i in row:
            writer.writerow(list(i))

    #TESTING the accuracy of the second set of data
    # Zcheck = np.linspace(0,.5,100)
    # start = time()
    # d1 = np.asarray(list(map(helper2,Zcheck)))
    # print(time()-start)
    # start = time()
    # dfunc = interp1d(z,d/mpc)
    # d2 = np.asarray(dfunc(Zcheck))
    # print(time()-start)
    # print((d1/mpc-d2))
    #
    # plt.plot(Zcheck,dfunc(Zcheck))
    # plt.scatter(z,np.divide(d,mpc))
    # plt.show()
    # plt.close()
