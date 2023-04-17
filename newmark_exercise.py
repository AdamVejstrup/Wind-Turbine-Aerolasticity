# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:52:17 2023

@author: Toke Schäffer
"""

import numpy as np
import matplotlib.pyplot as plt

linear_newmark = False
non_linear_newmark = True

L = 2 # [m]
m1 = 1 # [kg] mass of cart
m2 = 0.5 # [kg/m] mass of beam
sm = 0.5*m2 * L**2 # Coefficient for coordinate [0, 1] and [1, 0] of the mass matrix.
# Note that the actual cordinate is sin(theta) * Coefficient from above

im = (m2 * L**3) / 3 # Coordinate [1, 1] of mass matrix
g = 9.81 # [m/s**2]

# Given parameters
if linear_newmark:
    beta = 0.25
    gamma = 0.50
elif non_linear_newmark:
    beta = 0.27
    gamma = 0.51
    eps = 0.001
else:
    raise ValueError('Please choose between linear or non-linear newmark')

h = 0.001 # Timestep

tstart = 0 # Start time 
tend = 50 # End time

time = np.arange(tstart, tend, h)

# x[0, :] is the position of the cart
# x[1, :] is the angular position of the beam
# dx[0, :] is the velocity of the cart
# dx[1, :] is the angular velocity of beam

x = np.zeros([2, len(time)])

dx = np.zeros(x.shape)


M = np.array([[m2*L + m1,           -sm*np.sin(x[1, 0])],
              [-sm*np.sin(x[1, 0]),  im]])

C = np.zeros(M.shape)

K = np.zeros(M.shape)

gf = np.zeros(2)
gf[0] = dx[1, 0]**2 * np.cos(x[1, 0]) * sm
gf[1] = g*sm * np.cos(x[1, 0])


if linear_newmark:
    # Step 1: System matrices
    M_star = M + (gamma*h*C) + (beta * h**2 * K)

    # Step 2: Initial conditions

    ddx = np.zeros(x.shape)
    ddx[:, 0] = np.linalg.inv(M_star)@(gf[:, 0] - C@dx[:, 0] - K@x[:, 0])
    
    for n in range(1, len(time)):
        # Step 3: Prediction step
        
        dx_up = dx[:, n-1] + (1-gamma)*h * ddx[:, n-1]
        
        x_up = x[:, n-1] + h*dx[:, n-1] + (0.5-beta)*h**2*ddx[:, n-1]
        
        # Step 4: Correction step
        
        M_up = np.array([[m2*L + m1,      -sm*np.sin(x_up[1])],
                    [-sm*np.sin(x_up[1]),  im]])
        
        gf_up = np.array([dx_up[1]**2 * np.cos(x_up[1]) * sm,
                          g*sm * np.cos(x_up[1])])
        
        M_star_up = M_up + gamma*h*C + beta*(h**2)*K
        
        ddx_correct = np.linalg.inv(M_star_up) @ (gf_up - (C @ dx[:, n-1]) - (K @ x[:, n-1]))
        
        dx_correct = dx_up + gamma*h*ddx_correct
        
        x_correct = x_up + beta*h**2*ddx_correct
        
        # Updating position, velocity and acceleration
        x[:, n] = x_correct
        dx[:, n] = dx_correct
        ddx[:, n] = ddx_correct

elif non_linear_newmark:
    
    ddx = np.zeros(x.shape)
    ddx[:, 0] = np.linalg.inv(M) @ (gf - C @ dx[:, 0] - K @ x[:, 0])
    
    
    for n in range(1, len(time)):
        
        # Step 2: Predictions of position, velocity and acceleration
        x_up = x[:, n-1] + h*dx[:, n-1] + 0.5* h**2 *ddx[:, n-1]
        dx_up = dx[:, n-1] + h*ddx[:, n-1]
        ddx_up = ddx[:, n-1]


        #Step 3: Residual calculation
        counter = 0
        r = np.array([1, 1])
        
        while max(abs(r)) > eps and counter < 600:
            
            M_up = np.array([[m2*L + m1,      -sm*np.sin(x_up[1])],
                        [-sm*np.sin(x_up[1]),  im]])
            
            gf_up = np.array([dx_up[1]**2 * np.cos(x_up[1]) * sm,
                              g*sm * np.cos(x_up[1])])
            
            #Calculate residual
            r = gf_up - M_up @ ddx_up - C @ dx_up - K @ x_up
            
            K_star = K + gamma/(beta*h) * C + (1/(beta * h**2)) * M_up
            
            delta_x = np.linalg.inv(K_star) @ r
            
            #Update dof
            x_up = x_up + delta_x
            dx_up = dx_up + gamma / (beta*h) * delta_x
            ddx_up = ddx_up + 1 / (beta*h**2) * delta_x

            # Update counter
            counter = counter + 1
        
        #Save updated dof
        x[:, n] = x_up
        dx[:, n] = dx_up
        ddx[:, n] = ddx_up
            

# Plotting the results

plt.figure()
plt.grid()
plt.title(f'Cart position and beam angular positon, timestep = {h} s')
plt.plot(time, x[0, :],label='$x_{newmark}$')
plt.plot(time, x[1, :],label='$\Theta_{newmark}$')
plt.xlabel('Time [s]')
plt.ylabel('x, $\Theta$')
plt.legend()
plt.show()
    

