# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:52:17 2023

@author: Toke Schäffer
"""

import numpy as np
import matplotlib.pyplot as plt

L = 2 # [m]
m1 = 1 # [kg] mass of cart
m2 = 0.5 # [kg/m] mass of beam
sm = 0.5*m2 * L**2 # Coefficient for coordinate [0, 1] and [1, 0] of the mass matrix.
# Note that the actual cordinate is sin(theta) * Coefficient from above

im = (m2 * L**3) / 3 # Coordinate [1, 1] of mass matrix
g = 9.81 # [m/s**2]

# Given parameters
beta = 0.25
gamma = 0.50

h = 0.0001 # Timestep

tstart = 0 # Start time 
tend = 50 # End time

time = np.arange(tstart, tend, h)

# x[0, :] is the position of the cart
# x[1, :] is the angular position of the beam
# dx[0, :] is the velocity of the cart
# dx[1, :] is the angular velocity of beam

x = np.zeros([2, len(time)])

dx = np.zeros(x.shape)


M = np.array([[m2*L + m1, -sm*np.sin(x[1, 0])],
              [-sm*np.sin(x[1, 0]), im]])

C = np.zeros(M.shape)

K = np.zeros(M.shape)

gf = np.zeros(x.shape)
gf[0, 0] = dx[1, 0]**2 * np.cos(x[1, 0]) * sm
gf[1, 0] = g*sm * np.cos(x[1, 0])


# Step 1: System matrices
M_star = M + gamma*h*C + beta * h**2 * K

# Step 2: Initial conditions

ddx = np.zeros(x.shape)
ddx[:, 0] = np.linalg.inv(M_star) @ (gf[:, 0] - C@dx[:, 0] - K@x[:, 0])



for n in range(1, len(time)):
    # Step 3: Prediction step
    
    dx_up = dx[:, n-1] + (1-gamma)*h * ddx[:, n-1]
    
    x_up = x[:, n-1] + h*dx[:, n-1] + (0.5-beta)*h**2*ddx[:, n-1]
    
    # Step 4: Correction step
    
    M_up = np.array([[m2*L + m1, -sm*np.sin(x_up[1])],
                  [-sm*np.sin(x_up[1]), im]])
    
    gf_up = np.array([dx_up[1]**2 * np.cos(x_up[1]) * sm,
                      g*sm * np.cos(x_up[1])])
    
    M_star_up = M_up + gamma*h*C + beta*h**2*K
    
    ddx_correct = np.linalg.inv(M_star_up) @ (gf_up - C@dx[:, n-1] - K@x[:, n-1])
    
    dx_correct = dx_up + gamma*h*ddx_correct
    
    x_correct = x_up + beta*h**2*ddx_correct
    
    x[:, n] = x_correct
    
    dx[:, n] = dx_correct

#%%
plt.figure()
plt.grid()
plt.title(f'Cart position and beam angular positon, timestep = {h} s')
plt.plot(time, x[0, :],label='$x_{newmark}$')
plt.plot(time, x[1, :],label='$\Theta_{newmark}$')
plt.xlabel('Time [s]')
plt.ylabel('x, $\Theta$')
plt.legend()
plt.show()
    

