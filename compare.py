#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:30:39 2019

@author: evanfrank
"""

import pandas as pd
from numpy import array,arange
import pylab as plt
import numpy as np
# Constants
m = 9.1094e-31     # Mass of electron
hbar = 1.0546e-34  # Planck's constant over 2*pi
e = 1.6022e-19     # Electron charge
a = 2.05e-10     # distance in meters
N = 100
YAS=N
h = a/N
k=8.98755179e9
plank=4.135667662e-15

#potential function for the AHO
def V_a(x):
    top = k*e*e
    return top/(x-1)+top/(1+a-x)

#helper fucntion
def f(r,x,E):
    psi = r[0]
    phi = r[1]
    fpsi = phi
    fphi = (2*m/hbar**2)*(V_a(x)-E)*psi
    return array([fpsi,fphi],float)

# Calculate the wavefunction for a particular energy
def solve(E):
    psi = 0.0
    phi = 1.0
    r = array([psi,phi],float)
    for x in arange(1+1e-14,1+a,h):
        k1 = h*f(r,x,E)
        k2 = h*f(r+0.5*k1,x+0.5*h,E)
        k3 = h*f(r+0.5*k2,x+0.5*h,E)
        k4 = h*f(r+k3,x+h,E)
        r += (k1+2*k2+2*k3+k4)/6
    return r[0]

# Calculate the wavefunction for the energy level
def wave_f(E):
    psi = 0.0
    phi = 1.0
    r = array([psi,phi],float)
    phi_list=np.array([])
    for x in arange(1+1e-14,1+a,h):
        k1 = h*f(r,x,E)
        k2 = h*f(r+0.5*k1,x+0.5*h,E)
        k3 = h*f(r+0.5*k2,x+0.5*h,E)
        k4 = h*f(r+k3,x+h,E)
        r += (k1+2*k2+2*k3+k4)/6
        phi_list=np.append(phi_list,r[0])
   # plt.plot(np.linspace(0,a,N),phi_list,'-r')
    #plt.show()
    return phi_list

#to find the energy using the secant method
def energy_level_a(N):
    E1 = 0.0 +70*N*e
    E2 = e +70*N*e
    psi2 = solve(E1)
    target = e/1000000
    while abs(E1-E2)>target:
        psi1,psi2 = psi2,solve(E2)
        E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
    #print('QAHO-E', N, '=' ,E2/e,'eV')
    phi_list=wave_f(E2)
    return phi_list, (E2/e)*.001
    
#potenal energy of HO
def V_o(x):
    return 6.2e3*(x-(1+a/2))**2

def f_o(r,x,E):
    psi = r[0]
    phi = r[1]
    fpsi = phi
    fphi = (2*m/hbar**2)*(V_o(x)-E)*psi
    return array([fpsi,fphi],float)

# Calculate the wavefunction for a particular energy
def solve_o(E):
    psi = 0.0
    phi = 1.0
    r = array([psi,phi],float)
    for x in arange(1+1e-14,1+a,h):
        k1 = h*f_o(r,x,E)
        k2 = h*f_o(r+0.5*k1,x+0.5*h,E)
        k3 = h*f_o(r+0.5*k2,x+0.5*h,E)
        k4 = h*f_o(r+k3,x+h,E)
        r += (k1+2*k2+2*k3+k4)/6
    return r[0]

# Calculate the wavefunction for the energy level
def wave_f_o(E):
    psi = 0.0
    phi = 1.0
    r = array([psi,phi],float)
    phi_list=np.array([])
    for x in arange(1+1e-14,1+a,h):
        k1 = h*f_o(r,x,E)
        k2 = h*f_o(r+0.5*k1,x+0.5*h,E)
        k3 = h*f_o(r+0.5*k2,x+0.5*h,E)
        k4 = h*f_o(r+k3,x+h,E)
        r += (k1+2*k2+2*k3+k4)/6
        phi_list=np.append(phi_list,r[0])
    #plt.plot(np.linspace(0,a,N),phi_list,'-r')
    #plt.show()
    return phi_list

#to find the energy using the secant method
def energy_level_h(N):
    E1 = 0.0 +N*e
    E2 = e +100*N*e
    psi2 = solve_o(E1)
    target = e/10000000
    while abs(E1-E2)>target:
        psi1,psi2 = psi2,solve_o(E2)
        E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
    #print('QHO-E', N, '=' ,E2/e,'eV')
    phi_list=wave_f_o(E2)
    return phi_list, (E2/e)*.001

#Now the main event, comapring the QAHO and QHO wavefunction and energy levels.

def compare(N):
    E_as=[]
    E_os=[]
    phi_list_as=[]
    phi_list_hs=[]
    #finds wave function and energy values from N values
    for n in range(N):
        phi_list_a, E_a= energy_level_a(n)
        phi_list_h, E_h= energy_level_h(n)
        E_as.append(E_a)
        E_os.append(E_h)
        phi_list_hs.append([phi_list_h])
        phi_list_as.append([phi_list_a])
    plt.xlabel('Distance in meters')
    #plots the wave functions for each model at a given N
    for n in range(N):
        plt.plot(np.linspace(0,2.05e-10,YAS), phi_list_as[n][0]*2, 'b-', label='Aharmoic ocilator wave function')
        plt.plot(np.linspace(0,2.05e-10,YAS), phi_list_hs[n][0], 'g-', label='Harmoic ocilator wave function')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('Distance (m)')
        plt.ylabel('Psi(x)')
        plt.title('Comparing the Wavefuction of QAHO and QHO at a Given Energy Level', y=1.05)
        plt.grid()
        plt.show()
    #plots the energy values against each other over all Ns
    plt.plot(E_as, 'bx', label='Energy levels of Aharmoic ocilator')
    plt.plot(E_os, 'gx',label='Energy levels of Harmoic ocilator')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Quantum number (n)')
    plt.ylabel('Energy (eV)')
    #plt.ylim(0,500)
    plt.grid()
    plt.title('Comparing the Energy Levels of QAHO and QHO')
    plt.show()
    energy_theory=pd.DataFrame([])
    energy_theory['QHO']=E_os
    energy_theory['QAHO']=E_as
    energy_theory['energy_diff_QHO']=abs(energy_theory['QHO']-energy_theory['QHO'].shift(1))
    energy_theory['energy_diff_QAHO']=abs(energy_theory['QAHO']-energy_theory['QAHO'].shift(1))
    energy_theory=energy_theory.dropna(axis=0)
    energy_theory=energy_theory.reset_index()
    return energy_theory

def experimental_step():
    df1=pd.read_excel(open('emerald_sodalite.xlsx', 'rb'))
    plt.plot(df1['wavelength_nm'], df1['intensity'], 'bx', label='spectra')
    plt.legend(loc='upper left')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intesity')
    plt.suptitle('Flurencses of Emerald Sodalite at 77K', fontsize=16)
    plt.ylim(top=2600) 
    plt.ylim(bottom=0)
    plt.xlim(left=500) 
    plt.xlim(right=790)
    plt.show()
    bounds=[[540, 557], [560, 575], [575, 590], [590, 615], [620, 640], [640, 655], [660, 675], [680, 715], [715, 745], [745, 760]]
    Max=pd.DataFrame()
    for bound in bounds:
        df2 = df1[(df1['wavelength_nm'] >= bound[0]) & (df1['wavelength_nm'] <= bound[1])]
        Max0= df2[df2['intensity']==df2['intensity'].max()]
        Max= Max.append(Max0)
    Max['energy_at_spike(eV)']=(3e8*plank)/(Max['wavelength_nm']*1e-9)
    Max['energy_diff']=abs(Max['energy_at_spike(eV)']-Max['energy_at_spike(eV)'].shift(1))
    Max=Max.dropna(axis=0)
    Max=Max.reset_index()
    return Max

def final():
    ex=experimental_step()
    th=compare(6)
    plt.plot(th['energy_diff_QHO'],label='energy_diff_QHO' )
    plt.plot(th['energy_diff_QAHO'], label='energy_diff_QAHO')
    plt.plot(ex['energy_diff'], label='energy_diff_experimental')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Energy (eV)')
    plt.xlim(0,4)
    plt.ylim(0,0.1) 
    plt.title('Comparing the Energy Levels Differences of QAHO, QHO, and Experimental')

def main(): 
    final()
if __name__ == "__main__":
    main()