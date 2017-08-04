# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:38:23 2016

@author: HP
"""

"""This code creates artificial ground motion time-history based on the works
of Zentner, Vanmarcke & Gasparini

Research Paper
Zentner, I. (2014). A procedure for simulating synthetic accelerograms compatible with 
correlated and conditional probabilistic response spectra. Soil Dynamics and Earthquake Engineering(63), 226-233."""

import os
import pandas as pd
import numpy as np
from scipy import integrate
from scipy.optimize import fmin
import matplotlib.pyplot as plt
#from datetime import datetime
from numba import jit

import montecarlosim_spectra as simulated


class artificial_timehistory():

    def __init__(self,w,sa): #initialize by introducing w and sa
        self.w = w
        self.sa = sa
        #sampling = np.array([0.002,0.003,0.004])   #sampling frequencies for artificial time history
        #self.dt = sampling[np.random.randint(0,3)]
        self.dt = 0.02
        #self.seeds = int(str(datetime.now()).split('.')[-1])
        self.seeds = np.random.randint(234324)

    def PSD(self,sa): #calculate the PSD given spectral accelerations
        wk = 30
        Tsm = 17
        zi = 0.05
        delta = np.sqrt(4*zi/np.pi)
        n0 = 1.4427*wk/(2*np.pi)*Tsm
        n2tsm = 2*np.log(2*n0*(1-np.exp(-delta**1.2*np.sqrt(np.pi*np.log(2*n0)))))
        s_init = np.array([2*zi*sa[0]**2/(1/(4*np.pi)*np.pi*n2tsm)])
        Sx = np.ndarray(len(sa))
        Sx[0] = 1/(self.w[0]*(np.pi/(2*zi)-2))*(sa[0]**2/n2tsm-\
                2*integrate.trapz(s_init,dx=0.05))
        for i in range(1,len(sa)):
            Sx[i] = 1/(self.w[i]*(np.pi/(2*zi))-2)*(sa[i]**2/n2tsm-\
                2*integrate.trapz(Sx[:i],self.w[:i]))
        return Sx

    def gamma(self,x): #Seed is crucial or else it throw an error
        t = np.arange(0,50.02,0.02)
        q = t**(x[0]-1)*np.exp(-x[1]*t)
        Ia = np.pi/(2*9.81)*integrate.trapz(q**2,t)
        Ia_cum = np.pi/(2*9.81)*integrate.cumtrapz(q**2,t,initial = 0)
        t1 = t[np.where(np.logical_and(Ia_cum/Ia>=0.05,Ia_cum/Ia<=0.95))][0]
        t2 = t[np.where(np.logical_and(Ia_cum/Ia>=0.05,Ia_cum/Ia<=0.95))][-1]
        tsm = t2-t1
        np.random.seed(self.seeds)
        t1_ = np.random.uniform(1,5)
        tsm_= np.random.lognormal(18.54,0.31)
        return np.sqrt((tsm-np.log(tsm_))**2+(t1-t1_)**2)

    def modulating(self): #gamma modulating function to calculate the value of a1,a2,a3
        res = fmin(self.gamma,[2,.27])
        a2,a3 = res[0], res[1]
        t = np.arange(0,50.02,0.02)
        q = t**(a2-1)*np.exp(-a3*t)
        Ia = np.pi/(2*9.81)*integrate.trapz(q**2,t)
        Ia_cum = np.pi/(2*9.81)*integrate.cumtrapz(q**2,t,initial = 0)
        t1 = t[np.where(np.logical_and(Ia_cum/Ia>=0.05,Ia_cum/Ia<=0.95))][0]
        t2 = t[np.where(np.logical_and(Ia_cum/Ia>=0.05,Ia_cum/Ia<=0.95))][-1]
        Tsm = t2-t1
        a1 = Tsm/(integrate.trapz(q**2,t))
        q =  a1*t**(a2-1)*np.exp(-a3*t)
        return a1,a2,a3

    #method to generate artifical accelerogram through response spectrum compatible
    #Power Spectral Density
    def accelerogram(self,Sx):
        Ns = 2**11
        dt = self.dt
        ohm = np.pi/dt
        dw = 2*ohm/Ns
        n = np.arange(0,Ns)
        k = np.arange(0,Ns)
        tk = k*dt
        wn = -ohm+dw*(0.5+n)
        a1,a2,a3 = self.modulating()
        Sx_wn = np.ndarray(Ns)
        Sx_wn[np.where(wn>=0)] = np.interp(wn[np.where(wn>=0)],2*np.pi*self.w,Sx)
        Sx_wn[np.where(wn<0)] = Sx_wn[np.where(wn>=0)][::-1]
        #introducing the random variables
        np.random.seed(self.seeds)
    #    std_nor = 1/np.sqrt(2)*(np.random.normal(0,1,Ns)+1j*np.random.normal(0,1,Ns))
        std_nor = np.random.normal(0,1,Ns)
        phi = np.random.uniform(0,2*np.pi)
        #simulating the the first accelerogram
        Vk = np.fft.ifft(np.sqrt(Sx_wn)*std_nor*np.exp(phi*1j))
        sk = np.real(Vk*np.exp(1j*k*np.pi*(1-1/Ns)))*np.sqrt(dw)
        qt = a1*tk**(a2-1)*np.exp(-a3*tk)
        yt = qt*sk
        return tk,yt

    @jit
    def response_Newmark(self,wf,t,ag): # average acceleration method
        Y, B = 1/2, 1/4
        u_t = np.ndarray(len(ag), dtype = 'float')
        v_t = np.ndarray(len(ag), dtype = 'float')
        a_t = np.ndarray(len(ag), dtype = 'float')
        dt = t[1] - t[0]
        m = 1
        zi = 0.05
        w = 2*np.pi*wf
        k = w**2 * m
        c = 2*m*w*zi
        #initial values
        u_t[0], v_t[0] = 0., 0.
        a_t[0] = (m*ag[0] - c*v_t[0] - k * u_t[0])/m
        k_ = k + Y * c/(B * dt) + m/(B*dt**2)
        a = m/(B*dt) + Y/B *c
        b = m/(2*B) + dt*(Y/(2*B) - 1) * c

    # main calculations start here
        for i in range(1,len(ag)):
            delta_p = m * (ag[i]-ag[i-1]) + a * v_t[i-1] + b * a_t[i-1]
            delta_u = delta_p/k_
            delta_v = Y/(B*dt)*delta_u - Y/B*v_t[i-1] + dt * (1-Y/(2*B)) * a_t[i-1]
            delta_a = 1/(B*dt**2) * delta_u - v_t[i-1] / (B*dt) - a_t[i-1]/(2*B)
            u_t[i] = u_t[i-1] + delta_u
            v_t[i] = v_t[i-1] + delta_v
            a_t[i] = a_t[i-1] + delta_a
        return max(abs(u_t))*w**2

    #generate the time-history and evaluate its response spectra

    @jit
    def timehistory(self):
        Sx = self.PSD(self.sa)
        tk,yt = self.accelerogram(Sx)
        vfunc = np.vectorize(self.response_Newmark, excluded = [1,2])
        Sa = vfunc(self.w,tk,yt)
        for i in range(20):
            Sx = (self.sa/Sa)**2*Sx
            tk,yt = self.accelerogram(Sx)
            Sa = vfunc(self.w,tk,yt)
        return Sa,tk,yt

    def rms(self,sa,sa_):
        sa_ = self.sa
        return np.sum((sa-sa_)**2)/len(sa_)

    def write_to_file(self,fileid,values):
        fid = open(fileid,'w')
        fid.write("DT= {0:.3f}\n".format(self.dt))
        fid.write("\n")
        for i in range(1,len(values)):
            fid.write(("{0:.7E}\n".format(values[i])))
        fid.close()


class counter(): #class to get the file id counter once the time history is selected

    n = 0
    def __call__(self,x):
        return self.n+x

    def count(self):
        self.n = self.n+ 1
        return self.n

#function to create a database
    def database(self,df,sa):
        #df.rename(columns={0:'F(Hz)'},inplace = True)
        return pd.concat([pd.DataFrame(df),pd.DataFrame(sa)],axis = 1, \
                ignore_index = True)

#function to write to file once the appropriate time history has been chosen
    def write_to_file(self,fileid,dt,values):
        filename = "synth_timehist_" + str(fileid) + ".txt"
        path = "F:\Books\Thesis\python scripts\Synthetic Record"
        truefile = os.path.join(path,filename)
        fid = open(truefile,'w')
        fid.write("DT= {0:.3f}\n".format(dt))
        for i in range(1,len(values)):
            fid.write(("{0:.7E}\n".format(values[i])))


def main():
    path = "F:\Books\Thesis\python scripts\RealSpectra.csv"
    a = simulated.simulated_spectra(path)
    w = a.w
    #median = a.median
    target_sa = a.simulate()
    b = artificial_timehistory(w,target_sa)
    sa,tk,yt = b.timehistory()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_xlabel('Frequency(Hz)',fontsize = 13)
    ax1.set_ylabel('Psa(g)', fontsize = 13)
    ax1.set_title('Simulated Spectra',fontsize = 15)
    ax1.loglog(w,target_sa, label = 'Simulated')
    ax1.loglog(w,sa,label ='Artificial')
    ax1.legend(loc = 4)
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Time(secs)',fontsize = 13)
    ax2.set_ylabel('Acceleration(g)',fontsize = 13)
    ax2.plot(tk,yt)
    fig.tight_layout()
    plt.show()
    return sa,tk,yt,b.dt

#sa,tk,yt,dt = main()





if __name__== '__main__':
    main()

