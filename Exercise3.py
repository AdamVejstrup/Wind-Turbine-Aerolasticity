# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:30:30 2023

@author: adamv
"""
import numpy as np
import math


A = 0.2#m
omega = 5 #rad/s
a_0 = np.deg2rad(10) #deg
V_0 = 10 #m/s
c = 1 #m, cord length
rho = 1.225 #kg/m^3


T = 2*np.pi / omega #one period T
cyc = 10 # number of full cycles
N = 50 #number of timesteps
dt = cyc * T / N #timestep

time_arr = np.zeros(N)
x = np.zeros(N)
F_x = np.zeros(N)


air_foil = np.loadtxt("FFA-W3-241.txt")
aoa_tab, cl_stat_tab, cd_stat_tab, _, _, _, _ = air_foil.T

theta_step = 5 #deg
W = np.zeros(len(range(0, 360, theta_step)))
theta_deg =  np.zeros(len(range(0, 360, theta_step)))

for idx,i in enumerate(range(0, 360, theta_step)):
    
    theta = np.deg2rad(i) #rad
    
    for n in range(1, N):
        
        time_arr[n] =  n * dt  #time
        
        x[n] = A * np.sin(omega* time_arr[n])
        
        x_dot = (x[n] - x[n-1]) / dt
        
        V_y = V_0 * np.cos(a_0) + x_dot * np.cos(theta)
        V_z = V_0 * np.sin(a_0) + x_dot * np.sin(theta)
        
        V_rel = np.sqrt((V_y)**2 + (V_z)**2)
    
        alpha = np.arctan(V_z / V_y)
        
        C_l = np.interp(np.rad2deg(alpha), aoa_tab, cl_stat_tab)
        C_d = np.interp(np.rad2deg(alpha), aoa_tab, cd_stat_tab)
        
        F_x[n] = (0.5 * rho * V_rel**2 * c *
                    (C_l * np.sin(alpha - theta)
                    - C_d * np.cos(alpha - theta)))
        
        
        
    theta_deg[idx] = i
    W[idx] = A * omega * np.trapz(F_x* np.cos(omega * time_arr), time_arr)
    
    




