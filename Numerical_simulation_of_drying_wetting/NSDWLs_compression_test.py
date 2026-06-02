# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:25:34 2024

@author: tomdegast
"""

" ## local files##" 
#from main_program import *
from Input_parameters_definitiions import *
from build_domain import *
from NSDWLs_errors import *
from NSDWLs_states import *
from NSDWLs_equations import *
from NSDWLs_potential import *

from NSDWLs_calc import *
from NSDWLs_flux import *

from scipy import special

" ## global files ##"
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sci
import math as math
import pandas as pd
import timeit as timeit
import pandas as pd

" #### input parameters "

file_path = 'H:/test/zegveld.csv'
try: 
    df = pd.read_csv(file_path,sep=';')
except FileNotFoundError:
    print ("file doesn't exist")

file_path = 'H:/test/zegveld1.csv'
try: 
    df1 = pd.read_csv(file_path,sep=';')
except FileNotFoundError:
    print ("file doesn't exist")

file_path = 'H:/test/zegveld2.csv'
try: 
    df2 = pd.read_csv(file_path,sep=';')
except FileNotFoundError:
    print ("file doesn't exist")
    
file_path = 'H:/test/zegveld3.csv'
try: 
    df3 = pd.read_csv(file_path,sep=';')
except FileNotFoundError:
    print ("file doesn't exist")    
    
file_path = 'H:/test/zegveld4.csv'
try: 
    df4 = pd.read_csv(file_path,sep=';')
except FileNotFoundError:
    print ("file doesn't exist")    

file_path = 'H:/test/zegveld5.csv'
try: 
    df5 = pd.read_csv(file_path,sep=';')
except FileNotFoundError:
    print ("file doesn't exist")    

file_path = 'H:/test/zegveld6.csv'
try: 
    df6 = pd.read_csv(file_path,sep=';')
except FileNotFoundError:
    print ("file doesn't exist")        

load_increments = [1.9809, 3.0183, 1.9867, 2.992, 4.9896, 5.0048, 19.8894, -19.8379, 19.9315, 40.0327, 59.9423]

load = np.cumsum(load_increments)

time_incr = [0, 1.00393, 8.02939, 9.03358, 10.0375, 11.04165,12.04558119, 13.05007096, 14.05408645, 15.05812487, 22.08338878]
time_incr_step = np.zeros(len(time_incr))


time_max = 22

simulation_time = np.linspace(0, time_max,2500)
time_step = simulation_time[1]
max_time_step = time_step 
rtol =1e-4 #relative tolerance



dload_increment = load_increments/simulation_time[1]

Compression_vector = Compression_vector()
Compression_vector.Lambda = 0.22   # plastic strain
Compression_vector.Kappa = 0.15    # elastic strain
Compression_vector.Mu = 0.01     # creep strain
Compression_vector.ccott = 15
Compression_vector.Tau_ref = 1
Compression_vector.unloadreload = 0.25

Water_retention_curve_node = Water_retention_curve_vectors()
Water_retention_curve_node.vG_alpha = 0.11
Water_retention_curve_node.vG_n = 1.23
Water_retention_curve_node.vG_m = (1-1/1.23)
Water_retention_curve_node.vG_a = 50000
Water_retention_curve_node.vG_mod = 1

PP = 9
e0 = 5.2
h0 = 0.02
S_e =1

j=0
for i in range(len(simulation_time)):
   if time_incr[j] < simulation_time[i]: 
       time_incr_step [j] = simulation_time[i]
       j=j+1
       if j== len(time_incr) : break


dsigmaeff_dt = 0 
sigma_tot = 12*(h0/2)
u = 10*(h0/2)
sigma_eff = sigma_tot-u


def equations(simulation_time, in_in):
    global load_inc
    global load
    global time_incr
    global Compression_vector
    global dload_increment 
    global time_incr_step
    global time_step
    global h0
    global e0
    
    " get "
    PP_temp = in_in.copy
    eps = in_in[0]
    PP = in_in[1]
    sigma_eff = in_in[2]
    u = in_in[3]
    simga_tot = in_in[4]
    S_e = in_in[5]
    
    
    " calculate void ratio"
    e = (1+e0)*(1+eps)-1
    theta = S_e*e
    
    " get suction forom u"    
    psi_m = np.where(u <= 0, u, 0)
    
    " get boundary conditions"
    t_interval = len(time_incr)-1
    
    for i in range(len(time_incr)):
        #print(i)
        if simulation_time <time_incr[i]:
            t_interval = t_interval-1   
         
    dsigmaeff_dt = 0
    
    if simulation_time <= time_incr[t_interval] + time_step:
            dsigmaeff_dt = -dload_increment[t_interval]

    #if simulation_time >= time_incr[t_interval] + time_step:
            #sigma_eff = load[t_interval]
     
            
     
    dsimga_tot_dt = dsigmaeff_dt

    "Calculate permeability based on void ratio"
    #S_e = 1
    
    K= 10**(0.4*e-4)
    
    
    Water_retention_curve_vector = 1
     
    " set change in pore pressure"
    du_dt = K*d2f_dz2_non_uniform_loc(u,-0.5*h0*(1+eps)*10,1.5*h0*(1+eps)*10,h0)
    
    "check compression or extension"
    compression =1 
    
    if dsigmaeff_dt < 0 : compression=0
    
    "check if saturated"
    
    saturated = 1
    
    #if S_e <1 : saturated =0
    
    " add total boundary load to water pressure"
    du_dt = du_dt - S_e*dsimga_tot_dt #(-1/K)*compression*saturated*deps_dt
    
    "calculate saturated effective stress"
    
    dsigmaeff_dt = dsimga_tot_dt + du_dt

    [deps_dt, dPP_dt] = deSSc_dt_loc(Compression_vector, sigma_eff,dsigmaeff_dt,Water_retention_curve_vector, PP)

    dS_e_dpsi = derrivative_van_genuchten_S_e_loc (Water_retention_curve_node, theta,e, psi_m) 

    #dSat_dt = dtheta_dt/e - unsaturated*de_dt/e
    dSat_dt = -dS_e_dpsi*du_dt 

    return [deps_dt,dPP_dt,-dsigmaeff_dt,du_dt,-dsimga_tot_dt,dSat_dt]

input_in = [0,PP,sigma_eff,u,sigma_tot,S_e]

sol = solve_ivp(equations, [simulation_time[0], simulation_time[-1]],input_in,method='RK45',t_eval=simulation_time,max_step = max_time_step, rtol = rtol)#, atol = atol)



plt.figure()
plt.plot(sol.t,sol.y[1,:])

plt.figure()
plt.plot(sol.t,sol.y[2,:])

Linearstrain = df['Linear strain'].to_numpy()
time_values = df['Time'].to_numpy()
load_measured = df['Step'].to_numpy()

Linearstrain1 = df1['Linear strain'].to_numpy()
time_values1 = df1['Time'].to_numpy()
load_measured1 = df1['Step'].to_numpy()
time_values1 = time_values1/(60*60*24)

Linearstrain2 = df2['Linear strain'].to_numpy()
time_values2 = df2['Time'].to_numpy()
load_measured2 = df2['Step'].to_numpy()
time_values2 = time_values2/(60*60*24)

Linearstrain3 = df3['Linear strain'].to_numpy()
time_values3 = df3['Time'].to_numpy()
load_measured3 = df3['Step'].to_numpy()
time_values3 = time_values3/(60*60*24)

Linearstrain4 = df4['Linear strain'].to_numpy()
time_values4 = df4['Time'].to_numpy()
load_measured4 = df4['Step'].to_numpy()
time_values4 = time_values4/(60*60*24)

Linearstrain5 = df5['Linear strain'].to_numpy()
time_values5 = df5['Time'].to_numpy()
load_measured5 = df5['Step'].to_numpy()
time_values5 = time_values5/(60*60*24)

Linearstrain6 = df6['Linear strain'].to_numpy()
time_values6 = df6['Time'].to_numpy()
load_measured6 = df6['Step'].to_numpy()
time_values6 = time_values6/(60*60*24)


load_measured = load[load_measured-1]

plt.figure()
plt.plot(sol.t,50*sol.y[5,:]-50,'--')
plt.plot(sol.t,sol.y[3,:],'-.')

plt.figure()
#df.plot(kind='line', x='Time', y='Linear strain', title='Time vs Linear Strain')
#plt.plot(time_values,(1-y_values)*(1+e0)-1)
plt.plot(sol.t,sol.y[4,:],'--')
plt.plot(sol.t,sol.y[3,:],'-.')
plt.plot(sol.t,sol.y[2,:])


plt.figure()
plt.plot(load_measured,Linearstrain)
plt.plot(load_measured1,Linearstrain1)
plt.plot(load_measured2,Linearstrain2)
plt.plot(load_measured3,Linearstrain3)
plt.plot(load_measured5,Linearstrain5)
plt.plot(load_measured6,Linearstrain6)
plt.plot(sol.y[2,:],-sol.y[0,:],'--')
plt.ylim([max(-sol.y[0,:]),min(-sol.y[0,:])])
plt.xlim(1,100)
plt.xscale('log')



plt.figure()
#df.plot(kind='line', x='Time', y='Linear strain', title='Time vs Linear Strain')
#plt.plot(time_values,(1-y_values)*(1+e0)-1)
plt.plot(time_values,Linearstrain)
plt.plot(time_values1,Linearstrain1)
plt.plot(time_values2,Linearstrain2)
plt.plot(time_values3,Linearstrain3)
#plt.plot(time_values4,-Linearstrain4)
plt.plot(time_values5,Linearstrain5)
plt.plot(time_values6,Linearstrain6)
plt.plot(sol.t,-sol.y[0,:],'--')
plt.vlines(time_incr, 0, 1, colors = 'gray', linestyles = 'dotted')
plt.ylim([max(-sol.y[0,:]),min(-sol.y[0,:])])
plt.xlim(0,time_max)
plt.xlabel('Time [days]')
plt.ylabel('Linear Strain [-]')
