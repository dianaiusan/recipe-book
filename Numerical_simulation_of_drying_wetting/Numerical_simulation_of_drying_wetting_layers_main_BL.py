# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:27:28 2021

@author: tomdegast

This model links the change in effective stress due to transient flow to a consitutitve modelincluding creep (Modelled after Soft Soil creep). 
The permeability is changed based on watercontent and void ratio. 
Additionally this model can be used to calculate GHG-production rated based on saturation, temperature and available source material


"""
"The following function libraries are used in the programme and are local"
" ## local files##" 

#from main_program import *
from Input_parameters_definitiions import *
from build_domain import *
from NSDWLs_errors import *
from NSDWLs_states import *
from NSDWLs_equations import *
from NSDWLs_potential import *

#from NSDWLs_calc import *
#from NSDWLs_flux import *

"The following function libraries are used in the programme and are global"

" ## global files ##"
from scipy import special
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sci
import math as math
import pandas as pd
import timeit as timeit




" #### input parameters "
tic = timeit.default_timer()

" Initiation of he soil domain, give top, bottom and precision"
" precision directly influences calculation time. If the precision is small, the relative error required for accuracy will increase calulation time"
Soil_dimension = Soil_domain()
Soil_dimension.Depth_top = 0.0                                                    # m size
Soil_dimension.Depth_bottom = -5.4                                               # m size
Soil_dimension.precision = 0.1                                               # m node size

" Layers are added to the domain."
" Soil properties are collected in the library Input_parameters_definitiions "

Soil_dimension.Depth_o_soils = [-0.4, -0.8,-0.9,-2.3,-2.7,-3.6,-3.7,-4.1, Soil_dimension.Depth_bottom ]                 # bottom of soil layer
#Soil_dimension.soils = ["Test_soil", "Test_soil", "Test_soil"]                                      # soil types used
Soil_dimension.soils = ["Clay_BL_KL", "Clay_BL_KM","Clay_BL_KL", "Clay_BL_KM", "ZPeat_BL","Clay_BL_KZ", "Clay_BL_KM", "Clay_BL_KZ","ZPeat_BL"]                                      # soil types used


""

Soil_dimension.Number_o_soils = len(Soil_dimension.Depth_o_soils)                                                 # number of soils 
Soil_dimension.domainsize = Soil_dimension.Depth_top-Soil_dimension.Depth_bottom




#TEST MATERIALS BELOW
#Clay_BL
#Peat

#Soil_parameters = Soil_parameters()

#Soil_states = Soil_states()
Soil_states = Soil_states()

#Soil_stresses = Soil_stresses()
Soil_stresses = Soil_stresses()

Soil_calc = Soil_calc()

" ## Set calculation time, simulation steps to be reported, maximum time step and relative tolerance ##"
minimum_accuracy_time_step = 1               #days
end_time = 1000
simssteps = 1000
simulation_time =  np.linspace (0,end_time,simssteps+1)                                                         # days
max_time_step = 1.0  
rtol =1e-4 #relative tolerance
#atol =1e-16 #relative tolerance

"## Set boundary conditions"

"Top boundary conditions"
Bctop = Bc_top()
Bctop.time =     [0, 10, 200, 300, 400, 500, 600, 700, 800,1500,2000]
Bctop.pressure = [0,-10, -100, -300,-200,-100, 100,-100, 100, -100,   0]    
Bctop.trans =    [1,  10,   1,   1,   1,  1,   1, 1 ,  1,   1,   1]
Bctop.load_increments = [1.9809, 3.0183, 1.9867, 2.992, 4.9896, 5.0048, 19.8894, -19.8379, 19.9315, 40.0327, 59.9423]
Bctop.load = np.cumsum(Bctop.load_increments)
Bctop.time_incr = [0, 1.00393, 8.02939, 9.03358, 10.0375, 11.04165,12.04558119, 13.05007096, 14.05408645, 15.05812487, 22.08338878]


"Top boundary conditions in equation form <-> the choice to use an equation or steps is made further on in the code"
#Make a boundary condition
def Boundary_equation(time):
    t = time 
    P = 5.0*np.sin(np.pi*t*(1/100)+0)-10.0*np.sin(np.pi*t*(1/50)+1)-2.5*np.sin(np.pi*t*(1/500))
    T = np.sin(t*(6/365)-1.8)+7.5
    #P=-1.25
    return (P,T)

"Bottom boundary conditions"        
                               # kPa
Bctop.devaporation = np.zeros(len(Bctop.pressure))

Bcbottom = Bc_bottom()
Bcbottom.time = [0,500,750]
Bcbottom.pressure = [10 ,10, 10]  
Bcbottom.Temp = [10, 10, 10]                                                        # kPa
Bcbottom.dflow = [0.00,0.00, 0.00]
Bcbottom.trans = [0,0,0]

"## check for errors ##"

checkerror(Soil_dimension,Soil_parameters)

" Build the model domain and fill vectors with properties set initial conditions" 

(Soil_states,Soil_states.theta_ini,Soil_stresses.psi_ini) = set_ini(Soil_dimension,Soil_parameters,Soil_states)
Soil_states.e_ini = Soil_states.e
#Soil_states.theta_ini[0]=Soil_states.theta_ini[0]*0.9
Soil_states.theta = Soil_states.theta_ini
Soil_states.e_ini = Soil_states.theta
Soil_dimension.internode_vector = modeldomain (Soil_dimension,Soil_parameters)
(material_vector,gamma_vector,Permeability_curve_Perm_vector,Shrinkage_curve_vector,Water_retention_curve_vector, Compression_vector,Emission_vector) = soilbuilder(Soil_dimension,Soil_parameters,Soil_dimension.node_vector,Soil_dimension.internode_vector)

Temperature = 10+0.0*Soil_states.theta_ini

S_e_ini = Soil_states.theta_ini/Soil_states.e_ini

#required for GHG emission
(Emission_ini,dEmission_ini) = respiration_rate_curve (S_e_ini, Soil_states.e, Soil_states.theta,0*Soil_states.theta , 0*Temperature,Temperature, Water_retention_curve_vector,Emission_vector)

Soil_dimension.node_initial=get_node_distance(Soil_dimension)

" ### RUN PROGRAM ###" 

(Soil_stresses) = calc_stress (Soil_dimension,gamma_vector,Soil_states,Water_retention_curve_vector,Soil_stresses)

Soil_stresses.sigma_effective_node_ini = np.array(Soil_stresses.sigma_effective_node)
Soil_states.Plastic_point =  Soil_stresses.sigma_effective_node_ini+Compression_vector.Pg

  
def equations(simulation_time, in_in):
    #global time_step
    #global Soil_calc
    #global Results_states
    global gamma_vector
    global Compression_vector
    global Shrinkage_curve_vector
    global Water_retention_curve_vector
    global Permeability_cruve_Perm_vector
    global Emission_vector
    global Soil_dimension
    global Bctop
    global Bcbottom
    
    #print("~~~~~~~~~~~~~~~~~~")
    print ("time", "%.2f" % simulation_time)
    "~~~~~~~setup_local_parameters~~~~~~"    
    
    Shrinkage_curve_node = Shrinkage_curve_vector

    Water_retention_curve_node = Water_retention_curve_vector
    "~~~~ retreive parameters from loop~~~~~~~"
    
    " here the / 4 means that there are three sets of parameters changing in dx/dt"
    
    nr_elements = int(np.size(in_in)/6)
         
    Pressure,theta,e,Plastic_point,T,emm = untangle_parameters(in_in.copy(),6)

    Soil_states.theta = theta
    Soil_states.e = e
    
    S_e                   = theta/e #update_S_e(Water_retention_curve_node,theta) #
    
    saturated = (S_e >= 1).astype(int)
    unsaturated = (S_e < 1).astype(int)
    #print(saturated)
    #print("Se","%.1f" % S_e[3])    
      
    "~~~~ Set boundary ~~~~~"
    
    t_interval = len(Bctop.time)-1
    for i in range(len(Bctop.time)):
        #print(i)
        if simulation_time < Bctop.time[i]:
            t_interval = t_interval-1         

    Bctop_in = Bc_top()
    Bctop_in.pressure =  Bctop.pressure[t_interval]
    
    if t_interval == len(Bctop.time)-1 :
        t_next = Bctop.time[t_interval]
        P_next = Bctop.pressure[t_interval]
        trans_next = Bctop.trans[t_interval]
    else: 
        t_next = Bctop.time[t_interval+1]
        P_next = Bctop.pressure[t_interval+1]
        trans_next = Bctop.trans[t_interval+1]
        P_average = 0.5*(Bctop.pressure[t_interval] + P_next)
        P_range = P_next - Bctop.pressure[t_interval]
        Bctop_in.pressure = P_average+0.5*P_range*special.erf((simulation_time-t_next+trans_next)*(3/trans_next))
    
    #print(Bctop_in.pressure)
                                           # kPa
    Bctop_in.devaporation = Bctop.devaporation[t_interval]
    Bctop_in.trans = Bctop.trans[t_interval]

    t_interval = len(Bcbottom.time)-1
    for i in range(len(Bcbottom.time)):
        if simulation_time < Bcbottom.time[i]:
            t_interval = t_interval-1       

    Bcbottom_in = Bc_bottom()
    Bcbottom_in.pressure = Bcbottom.pressure[t_interval]  
    Bcbottom_in.Temp = Bcbottom.Temp[t_interval]                                                         # kPa
    Bcbottom_in.dflow = Bcbottom.dflow[t_interval]
    Bcbottom_in.trans = Bcbottom.trans[t_interval]
    
    
    Bctop_in.pressure,Bctop_in.Temp = Boundary_equation(simulation_time)
    
    "~~~~~~~Update soil_dimension~~~~~~"
    Soil_dimension.node_vector = get_node_coordinates(Soil_dimension)
    Soil_dimension.node_distance = get_node_distance(Soil_dimension)  
    Soil_dimension.node_distance = (1-((Soil_states.e_ini-e)/(1+Soil_states.e_ini)))*Soil_dimension.node_initial
         
    "~~~~~~Update_soil_stresses~~~~~~~"
    sigma_total = np.zeros(len(e))
    sigma_total = update_sigma_total (Soil_dimension.node_distance,gamma_vector,Soil_states,Water_retention_curve_vector,Bctop_in)
    sigma_eff = np.abs(sigma_total - Pressure*S_e)
        
    #print(Pressure[0])
    #print(sigma_total[0])
    #print(sigma_eff[0])

    "Matrix potential "
    psi_m = np.where(Pressure <= 0, Pressure, 0)
    
    "~~~~ Calculate permeability ~~~~~"
    Soil_calc.K                     = update_K (theta,e,S_e,Permeability_curve_Perm_vector,Water_retention_curve_node)
    K= Soil_calc.K/Soil_dimension.node_distance
         
    
    
    "set differentials"
    # First set is pressure 
    d2psi_dz2 = d2f_dz2_non_uniform(Pressure,Bctop_in.pressure,Bcbottom_in.pressure,Soil_dimension) #Pressure*0+np.sin(simulation_time)
    dpsi_dt = K*d2psi_dz2
    
    #print("Pressure", Pressure[0:4])
    dS_e_dpsi = derrivative_van_genuchten_S_e (Water_retention_curve_node, theta,e, psi_m)
    
    
    dtheta_dpsi = dS_e_dpsi*e
    # third set is void ratio
    
    de_dt,dPP_dt = deSSc_dt(Compression_vector, S_e, sigma_eff,dpsi_dt,Water_retention_curve_vector, Plastic_point)
    
    
    
    d2T_dz2 = d2f_dz2_non_uniform(T,Bctop_in.Temp,Bcbottom_in.Temp,Soil_dimension)   
    dT_dt = 0.044064*d2T_dz2     

    # second set is theta
    dtheta_dt = dpsi_dt*dtheta_dpsi  
      
    
    compression = (de_dt < 0).astype(int)    
    extension = (de_dt > 0).astype(int)  
    
    positive_pressure = (dpsi_dt > 0).astype(int)
    negative_pressure = (dpsi_dt < 0).astype(int)
    
    dtheta_dt
  
    dpsi_dt = dpsi_dt + (-1/K)*compression*saturated*de_dt        
  
    #print((positive_pressure*extension-1)*-1)
    #print((positive_pressure*extension))
    
    dtheta_dt = -(dtheta_dt - positive_pressure*extension*saturated*(de_dt) )
    
    
    for i in range(len(S_e)):
        if S_e[i] >= 1:
            if dtheta_dt[i] > de_dt[i] :  
                dtheta_dt[i] = de_dt[i]    

    dSat_dt = dtheta_dt/e - unsaturated*de_dt/e
    
    #dSat_dt = dtheta_dt/e - de_dt/e
    
    #print (dtheta_dt-de_dt)
    #print (de_dt)

    # emission
    
    #(Emission,dEmission) = respiration_rate_curve (S_e,e, theta,dSat_dt, dT_dt, T, Water_retention_curve_vector,Emission_vector)

    demm_dt = 0*dSat_dt
    
    for i in range(len(emm)):
        if emm[i] <= 0 : 
            if demm_dt[i] <= 0:
                demm_dt[i] = 0
                
                
            
    
    out_out = np.concatenate([dpsi_dt,dtheta_dt,de_dt,dPP_dt,dT_dt,demm_dt],axis=0)

    return [out_out]

"Set input" 
# First set is pressure #
input_in = Soil_stresses.psi+Soil_stresses.sigma_pressure_sum_node
# Second set is water content #
input_in = np.append(input_in,Soil_states.theta)
# Third set is void ratio #
input_in = np.append(input_in,Soil_states.e_ini) 
# Fourth set is max effective stress #
input_in = np.append(input_in,Soil_states.Plastic_point)
# Fifth set is Temperature#
input_in = np.append(input_in,Temperature)
# Sixth set is emmision#
input_in = np.append(input_in,Emission_ini)

"This is the code to get the numerical differential equation"
sol = solve_ivp(equations, [simulation_time[0], simulation_time[-1]],input_in,method='DOP853',t_eval=simulation_time,max_step = max_time_step, rtol = rtol)#, atol = atol)

#sol = RK4_mine(equations, [simulation_time[0], simulation_time[-1]],input_in,method='RK45',t_eval=simulation_time,max_step = max_time_step, rtol = rtol)#, atol = atol)

#sol = solve_ivp(equations, [simulation_time[0], simulation_time[-1]],input_in,method='RK45',t_eval=simulation_time,max_step = max_time_step, rtol = rtol)#, atol = atol)
#sol = solve_ivp(equations, [simulation_time[0], simulation_time[-1]],input_in,method='LSODA',t_eval=simulation_time)
#sol = solve_ivp(equations, [simulation_time[0], simulation_time[-1]],input_in,method='DOP853',t_eval=simulation_time)

toc = timeit.default_timer()  

#print("time",toc-tic)
#

"Present the results - get the results"

nr_elements = int(np.size(input_in)/6)
    # First set is pressure #
Pressure_results = sol.y[0:nr_elements]
    # Second set is water content #
theta_results = sol.y[nr_elements:2*nr_elements]
    # Third set is void ratio #
e_results = sol.y[2*nr_elements:3*nr_elements]
    # Fourth set is max effective stress #
Plastic_point_results = sol.y[3*nr_elements:4*nr_elements]
    # Fifth set is Temperature#
Temperature_results = sol.y[4*nr_elements:5*nr_elements]
    # Fifth set is Emmision#
Emission_results = sol.y[5*nr_elements:6*nr_elements]
" Extra "

#diffrence in pressure
dPressure_results = 0*Pressure_results
dtheta_results = 0*theta_results
de_results = 0*e_results


for i in range(len(Pressure_results[0,:])):
    dPressure_results[:,i] = Pressure_results[:,i]-Pressure_results[:,0]
    dtheta_results[:,i] = theta_results[:,i]-theta_results[:,0]
    de_results[:,i] = e_results[:,i]-e_results[:,0]

#ground water level
gwl = 0*dPressure_results

for i in range(len(Pressure_results[0,:])):
    for j in range(len(Pressure_results[:,0])-1):
        if Pressure_results[j,i]/Pressure_results[j+1,i] < 0:
            gwl[j,i] = 1
        #if theta_results[j,i]/e_results[j+1,i] < 1:
        #    gwl[j,i] = 1


print (Soil_states.e_ini)
"calculate settlement"

colum_heigth=np.zeros(len(e_results[0,:]))
node_heigth=0*(e_results)

for i in range (len(e_results[0,:])):
    colum_heigth[i] = np.sum((1-((Soil_states.e_ini-e_results[:,i])/(1+Soil_states.e_ini)))*Soil_dimension.node_initial)
    node_heigth[:,i] = ((1-((Soil_states.e_ini-e_results[:,i])/(1+Soil_states.e_ini)))*Soil_dimension.node_initial)

colum_node_heigth = node_heigth*0

colum_node_heigth[-1,:] = node_heigth[-1,:]

for i in range(len(e_results[:,0])-2,-1,-1):
    colum_node_heigth[i,:] = node_heigth[i,:]+colum_node_heigth[i+1,:]

pressure_top = sol.t*0
Temperature_top =  sol.t*0

for i in range(len(sol.t)):
    pressure_top[i], Temperature_top[i] = Boundary_equation(sol.t[i])



" ~~~~~~~~~~~ "
Se_results = theta_results/e_results

AU_results = 0*Emission_results

AU_results = ((0.05*(Temperature_results-(-10)))**2)*sci.stats.beta.pdf(Se_results,2.59,1.84)



" ~~~~~~~~~~~~ "
"Plot results"

plt.figure()
plt.plot(sol.t,pressure_top)
plt.xlabel('time [days]')
plt.ylabel('pressure_top [kPa]')
plt.title('Pressure')


plt.figure()
plt.plot(sol.t,colum_heigth+Soil_dimension.Depth_bottom)
plt.xlabel('time [days]')
plt.ylabel('displacement [m]')
plt.title('displacement')

plt.figure()
for i in range(len(e_results[:,0])):  
    plt.plot(sol.t,colum_node_heigth[i,:]+Soil_dimension.Depth_bottom)
plt.xlabel('time [days]')
plt.ylabel('displacement [m]')
plt.title('displacement')

plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,Pressure_results, cmap='RdBu', vmin=np.min(Pressure_results), vmax=np.max(Pressure_results))
#plt.pcolormesh(sol.t,Soil_dimension.node_vector,Pressure_results, cmap='RdBu', vmin=-10, vmax=100)
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("pressure")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("kPa")

plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,dPressure_results, cmap='RdBu', vmin=np.min(dPressure_results), vmax=np.max(dPressure_results))
#plt.pcolormesh(sol.t,Soil_dimension.node_vector,Pressure_results, cmap='RdBu', vmin=-10, vmax=100)
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("delta pressure")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("kPa")

plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,dtheta_results, cmap='RdBu', vmin=np.min(dtheta_results), vmax=np.max(dtheta_results))
#plt.pcolormesh(sol.t,Soil_dimension.node_vector,Pressure_results, cmap='RdBu', vmin=-10, vmax=100)
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("delta theta")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("-")


plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,de_results, cmap='RdBu', vmin=np.min(de_results), vmax=np.max(de_results))
#plt.pcolormesh(sol.t,Soil_dimension.node_vector,Pressure_results, cmap='RdBu', vmin=-10, vmax=100)
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("delta e")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("-")

plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,dtheta_results-de_results, cmap='RdBu')
#plt.pcolormesh(sol.t,Soil_dimension.node_vector,Pressure_results, cmap='RdBu', vmin=-10, vmax=100)
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("delta theta-de")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("-")


plt.figure()
#plt.pcolormesh(sol.t,Soil_dimension.node_vector,theta_results, cmap='RdBu', vmin=np.min(theta_results), vmax=np.max(theta_results))
plt.pcolormesh(sol.t,Soil_dimension.node_vector,theta_results, cmap='RdBu', vmin=np.min(theta_results), vmax=np.max(theta_results))
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("theta")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("[-]")

plt.figure()
#plt.pcolormesh(sol.t,Soil_dimension.node_vector,theta_results, cmap='RdBu', vmin=np.min(theta_results), vmax=np.max(theta_results))



plt.pcolormesh(sol.t,Soil_dimension.node_vector,theta_results/e_results, cmap='RdBu')
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("Se")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("[-]")

plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,e_results, cmap='RdBu')
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("void ratio")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("[-]")

plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,Plastic_point_results, cmap='RdBu')
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("Plastic point")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("[-]")

plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,Temperature_results, cmap='RdBu')
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("Temperature []")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("[-]")


plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,Emission_results, cmap='RdBu')
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("Emission")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("[-]")


plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,AU_results, cmap='RdBu')
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("Emission")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("[-]")


plt.figure()
plt.pcolormesh(sol.t,Soil_dimension.node_vector,gwl, cmap='RdBu', vmin=0, vmax=5)
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
plt.title("ground water level")
plt.xlabel("time [days]")
plt.ylabel("depth [m]")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((0.85,0.1,0.075,0.8))
plt.colorbar(cax=cax)
plt.title("[-]")



plt.figure()
plt.plot(Se_results[:,[0,10,20,30,40,50,60,70,80,90,100]],Soil_dimension.node_vector)
for i in range(len(Soil_dimension.Depth_o_soils)-1):
    plt.axhline(y=Soil_dimension.Depth_o_soils[i], color='k', linestyle='--')
#plt.pcolormesh(sol.t,Soil_dimension.node_vector,theta_results, cmap='RdBu', vmin=0, vmax=10)
plt.title("theta")
plt.xlabel("Se [-]")
plt.ylabel("depth [m]")




#plt.style.use('_mpl-gallery')
#fig, ax = plt.subplots()
#for i in range(Results_states.theta[0,:].size):
#    ax.plot(Results_states.time, Results_states.theta[:,i]-i, linewidth=1.0)
#   
#plt.show()
#
#plt.style.use('_mpl-gallery')
#fig, ax = plt.subplots()
#for i in range(Results_states.theta[0,:].size):
#    ax.plot(Results_states.time, Results_states.e[:,i]-i, linewidth=1.0)
#   
#plt.show()

#sol  = diffusion(sigma_sumsoil_ini, simulation_time, e_ini, Soil_dimension.precision, Soil_dimension.domainsize, Bctop, Bcbottom)
