# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:52:17 2023

@author: Toke Schäffer

This script solves the 2 DOF system with a pendulum on a cart using the Newmark
method. The system is solved using either linear or non-linear Newmark.

The two degrees of freedom are:
x: Position of the cart named x[0, :] in the code
theta: Angular position of the pendulum/beam named x[1, :] in the code

The linear Newmark method requires a small timestep to be stable i.e. h = 0.0001 s
The non-linear Newmark method is stable for larger timesteps i.e. h = 0.01 s

"""

import numpy as np
import matplotlib.pyplot as plt

# Choose between linear or non-linear newmark
linear_newmark = True
non_linear_newmark = False

# Setting simulation time
h = 0.0001    # Timestep [s]
tstart = 0  # Start time [s]
tend = 50   # End time [s]

# Time vector
time = np.arange(tstart, tend, h)

# Number of degrees of freedom
dof = 2     # x and theta

# System parameters
L = 2       # Length of pendulum/beam [m]
m1 = 1      # Mass of cart [kg]
m2 = 0.5    # Mass of pendulum/beam [kg]

# Mass matrix given on slide 8 Newmark part 1

# Coefficient for coordinate [0, 1] and [1, 0] of the mass matrix.
# Note that the actual cordinate is sin(theta) * coeeficient
sm = 0.5*m2 * L**2

im = (m2 * L**3) / 3    # Coordinate [1, 1] of mass matrix
g = 9.81    # Gravitational acceleration [m/s^2]

# Newmark parameters:
# In order for the method to be stable the following conditions must be met:
# gamma >= 0.5
# beta >= 0.25 * (gamma + 0.5)**2

# We use the following parameters:
if linear_newmark:
    # Linear Newmark parameters (slide 7 Newmark part 1)
    beta = 0.25
    gamma = 0.50
    
elif non_linear_newmark:
    # Non-linear Newmark parameters (slide 17 Newmark part 1)
    beta = 0.27
    gamma = 0.51
    eps = 0.000001
else:
    # Raise error if neither linear or non-linear newmark is chosen
    raise ValueError('Please choose between linear or non-linear newmark')

# Initilization of position, velocity and acceleration vectors

# x[0, :] is the position of the cart
# x[1, :] is the angular position of the beam
x = np.zeros([dof, len(time)])

# dx[0, :] is the velocity of the cart
# dx[1, :] is the angular velocity of beam
dx = np.zeros(x.shape)

# System matrices

# Mass matrix given on slide 8 Newmark part 1
M = np.array([[m2*L + m1, -sm * np.sin(x[1, 0])],
              [-sm * np.sin(x[1, 0]), im]])

# Damping matrix
C = np.zeros(M.shape)

# Stiffness matrix
K = np.zeros(M.shape)

# Generalized force vector
gf = np.zeros(dof)

# Initial conditions (slide 8 Newmark part 1)
gf[0] = dx[1, 0]**2 * np.cos(x[1, 0]) * sm
gf[1] = g*sm * np.cos(x[1, 0])

if linear_newmark:
    # Step 1: System matrices
    M_star = M + (gamma*h*C) + (beta * h**2 * K)

    # Step 2: Initial conditions
    ddx = np.zeros(x.shape)
    ddx[:, 0] = np.linalg.inv(M_star)@(gf - C@dx[:, 0] - K@x[:, 0])
    
    for n in range(1, len(time)):
        # Step 3: Prediction step
        dx_up = dx[:, n-1] + (1-gamma)*h * ddx[:, n-1]
        
        x_up = x[:, n-1] + h*dx[:, n-1] + (0.5-beta)*h**2*ddx[:, n-1]
        
        # Mass matrix depends on the angular position of the beam
        # Therefore we need to update the mass matrix
        M_up = np.array([[m2*L + m1, -sm*np.sin(x_up[1])],
                    [-sm*np.sin(x_up[1]), im]])
        
        # Generalized force vector depends on the angular position of the beam
        gf_up = np.array([dx_up[1]**2 * np.cos(x_up[1]) * sm,
                          g*sm * np.cos(x_up[1])])
        
        M_star_up = M_up + gamma*h*C + beta*(h**2)*K
        
        # Step 4: Correction step
        ddx_correct = np.linalg.inv(M_star_up) @ (gf_up - (C @ dx[:, n-1]) - (K @ x[:, n-1]))
        
        dx_correct = dx_up + gamma*h*ddx_correct
        
        x_correct = x_up + beta*h**2*ddx_correct
        
        # Saving position, velocity and acceleration
        x[:, n] = x_correct
        dx[:, n] = dx_correct
        ddx[:, n] = ddx_correct

elif non_linear_newmark:
    
    # Step 1: Initial conditions
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
        
        # Step 4: System matrices and increment correction
        # The residual is calculated until it is smaller than eps or the
        # counter is larger than 600
        while max(abs(r)) > eps and counter < 600:
            
            # Update mass matrix and generalized force vector
            M_up = np.array([[m2*L + m1, -sm*np.sin(x_up[1])],
                        [-sm*np.sin(x_up[1]), im]])
            
            gf_up = np.array([dx_up[1]**2 * np.cos(x_up[1]) * sm,
                              g*sm * np.cos(x_up[1])])
            
            # Calculate residual
            r = gf_up - M_up @ ddx_up - C @ dx_up - K @ x_up
            
            K_star = K + gamma/(beta*h) * C + (1/(beta * h**2)) * M_up
            
            delta_x = np.linalg.inv(K_star) @ r
            
            # Update dof
            x_up = x_up + delta_x
            dx_up = dx_up + gamma / (beta*h) * delta_x
            ddx_up = ddx_up + 1 / (beta*h**2) * delta_x

            # Update counter
            counter = counter + 1
        
        # Saving position, velocity and acceleration
        x[:, n] = x_up
        dx[:, n] = dx_up
        ddx[:, n] = ddx_up
            

# Cart position and beam angular position
plt.figure()
plt.grid()
plt.title(f'Cart position and beam angular positon, timestep = {h} s')
plt.plot(time, x[0, :], label='$x_{newmark}[m]$')
plt.plot(time, x[1, :], label='$\Theta_{newmark}[rad]$')
plt.xlabel('Time [s]')
plt.ylabel('Cart position and beam angle')
plt.xlim(tstart,tend)
plt.legend()
plt.show()

# Beam angular position
plt.figure()
plt.grid()
plt.title(f'Beam angular position, timestep = {h} s')
plt.plot(time, np.rad2deg(x[1, :]), label='$\Theta_{newmark}$')
plt.xlabel('Time [s]')
plt.ylabel('Beam angle $\Theta$ [deg]')
plt.xlim(tstart,tend)
plt.legend()
plt.show()

