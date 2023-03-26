# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:15:31 2023

@author: Toke Schäffer
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Airfoil name
a_name = 'FFA-W3-241'

# Airfoil file name
a_file_name = f'{a_name}.txt'

# Airfoil data
# Columns: alpha, cl, cd, cm, f_stat, cl_inv, cl_fs
a_data = np.loadtxt(a_file_name)

# Extractig force coefficients
(alpha_tab, cl_tab, cd_tab, cm_tab,
f_stat_tab, cl_inv_tab, cl_fs_tab) = a_data.T

# Given system parameters
m = 1 # kg
k = 61.7 # N/m
c = 0.2 # m
V_0 = 2 # m/s
alpha_g = np.deg2rad(20) # rad
rho = 1.225 # kg/m**3
s = 1 # m

# Use dynamic stall
use_stall = True

def pend(y, t, m, k, c, V_0, alpha_g, rho, s, use_stall):
    
    # Defining z1, z2 and z3 from y in this manner is just a "copy"
    # of the example from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    
    if use_stall:
        z1, z2, z3 = y
    
    else:
        z1, z2 = y
    
    # Follwing the algorithm from NII_1 slide 12
    phi = np.arctan(z2/V_0)
    
    alpha = alpha_g + phi
    
    V_rel = np.sqrt(z2**2 + V_0**2)
    
    if use_stall:
        
        # Follwing the algorithm from NII_1 slide 13
        f_stat = np.interp(np.rad2deg(alpha), alpha_tab, f_stat_tab)
        cl_inv = np.interp(np.rad2deg(alpha), alpha_tab, cl_inv_tab)
        cl_fs = np.interp(np.rad2deg(alpha), alpha_tab, cl_fs_tab)
        
        tau = 4*c/V_rel
        dfdt = (f_stat - z3)/tau
        cl = z3 * cl_inv + (1-z3) * cl_fs
        
    else:
        cl = np.interp(np.rad2deg(alpha), alpha_tab, cl_tab)
    
    input_force = 0.5*rho * V_rel**2 * c*s*cl * np.cos(phi)
    
    # Not sure why the input force is subtracted and not added
    if use_stall:
        dydt = [z2, (-k*z1 - input_force) / m, (f_stat-z3)/tau]
    else:
        dydt = [z2, (-k*z1 - input_force) / m]
        
    return dydt

# Given initial conditions

if use_stall:
    y0 = [0.02, 0, 0]
else:
    y0 = [0.02, 0]

# Number of time steps
nots = 1000

# Time vector
t = np.linspace(0, 20, nots)

# Solving the problem
sol = odeint(pend, y0, t, args=(m, k, c, V_0, alpha_g, rho, s, use_stall))

# Plotting the results
plt.figure()

if use_stall:
    stall_str = 'with dynamic stall'
else:
    stall_str = 'without dynamic stall'

alpha_g_str = f'$\\alpha_g$ = {np.rad2deg(alpha_g)} $\degree$'

plt.title(f'x-position, {alpha_g_str} ({stall_str})')
plt.plot(t, sol[:, 0], label='x')
plt.xlabel('Time [s]')
plt.ylabel('x [m]')
plt.grid()
