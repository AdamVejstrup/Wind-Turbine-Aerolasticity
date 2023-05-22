# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:09:29 2023

@author: leael
"""
# %% 

import numpy as np
import matplotlib.pyplot as plt
from assignment_functions import x_mask


H = 119                     # Hub height [m]
L_s = 7.1                   # Length of shaft [m]
R = 89.15                   # Radius [m]
theta_cone = np.deg2rad(0)  # Cone angle [rad]
theta_yaw = np.deg2rad(20)  # Yaw angle [rad]
theta_tilt = np.deg2rad(0)  # Tilt angle [rad]
omega = 0.62                # Rotational speed [rad/s]
delta_t = 0.05              # Time step [s]
timerange = 200             # Time range [s]
r = 70                      # Distance from hub to airfoil [m]
v0 = 10                     # Wind speed at hub height [m/s]
B = 3                       # Number of blades [-]

use_tower_shadow = True     # Enable/disable tower shadow
use_wind_shear = True       # Wind shear (true=wind shear 0.2, else 0)

if use_wind_shear:
    wind_shear = 0.2        # Wind shear exponent [-]
else:
    wind_shear = 0


# Initialising arrays
time_arr = np.zeros(timerange)
omega_arr = np.full(timerange, omega)
theta_blade_arr = np.zeros([B, timerange])
x1_arr = np.zeros([B, timerange])
y1_arr = np.zeros([B, timerange])
z1_arr = np.zeros([B, timerange])

V0x_arr = np.zeros([B, timerange])
V0y_arr = np.zeros([B, timerange])
V0z_arr = np.zeros([B, timerange])

V_rel_y_arr = np.zeros([B, timerange])
V_rel_z_arr = np.zeros([B, timerange])

# Transformation matrices
a1 = np.array([[1, 0, 0],
               [0, np.cos(theta_yaw), np.sin(theta_yaw)],
               [0, -np.sin(theta_yaw), np.cos(theta_yaw)]])

a2 = np.array([[np.cos(theta_tilt), 0, -np.sin(theta_tilt)],
               [0, 1, 0],
               [np.sin(theta_tilt), 0, np.cos(theta_tilt)]])

a3 = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

a12 = a3@a2@a1

a21 = np.transpose(a12)

a34 = np.array([[np.cos(theta_cone), 0, -np.sin(theta_cone)],
                [0, 1, 0],
                [np.sin(theta_cone), 0, np.cos(theta_cone)]])


rt1 = np.array([H, 0, 0])

rs1 = a21@np.array([0, 0, -L_s])


for n in range(1, timerange):

    time_arr[n] = n*delta_t

    for i in range(B):

        # If statements fortæller hvordan azimutten (theta_blade) skal sættes
        # afhængigt af hvad nummer vinge, vi kigger på

        if i == 0:            
            theta_blade_arr[i, n] = theta_blade_arr[0, n-1] + omega_arr[n-1] * delta_t
        elif i == 1:
            theta_blade_arr[i, n] = theta_blade_arr[0, n] + omega_arr[n-1] * delta_t + 0.666 * np.pi
        elif i == 2:
            theta_blade_arr[i, n] = theta_blade_arr[0, n] + omega_arr[n-1] * delta_t + 1.333 * np.pi

        a23 = np.array([[np.cos(theta_blade_arr[i, n]), np.sin(theta_blade_arr[i, n]), 0],
                        [-np.sin(theta_blade_arr[i, n]),
                         np.cos(theta_blade_arr[i, n]), 0],
                        [0, 0, 1]])

        a14 = a34@a23@a12

        a41 = np.transpose(a14)

        rb1 = a41@np.array([r, 0, 0])

        r1 = rt1 + rs1 + rb1

        x1_arr[i, n] = r1[0]
        y1_arr[i, n] = r1[1]
        z1_arr[i, n] = r1[2]
        
        if use_tower_shadow:
            # Tower shadow gælder kun når x<H
            if x1_arr[i, n] <= H:
                a = 3.32
            elif x1_arr[i, n] > H:
                a = 0
        
        r_til_punkt = (y1_arr[i, n]**2 + z1_arr[i, n]**2)**(1/2)
        
        V0_array = np.array([0, 0, v0*(x1_arr[i, n]/H)**wind_shear])

        # Radial velocity
        Vr = z1_arr[i, n]/r_til_punkt * V0_array[2] *(1-(a/r_til_punkt)**2)

        # Tangential velocity
        Vtheta = y1_arr[i, n]/r_til_punkt * V0_array[2] * (1+(a/r_til_punkt)**2)
        
        if use_tower_shadow:
            V0_array = np.array([0, y1_arr[i, n]/r_til_punkt*Vr -
                                            z1_arr[i, n]/r_til_punkt*Vtheta,
                                            z1_arr[i, n]/r_til_punkt*Vr +
                                            y1_arr[i, n]/r_til_punkt*Vtheta])

        # Omregner vindhastigheden til system 4
        V0_4 = a14@V0_array

        V0x_arr[i, n] = V0_4[0]
        V0y_arr[i, n] = V0_4[1]
        V0z_arr[i, n] = V0_4[2]


def plot_xy(r, blade_number=0):
    plt.figure()
    plt.title('Position in space (x,y)')
    plt.plot(y1_arr[blade_number, 1:], x1_arr[blade_number, 1:])
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.grid()
    plt.axis('scaled')
    plt.xlim(-r, r)
    plt.ylim(H - r, H + r)
    plt.show()
    
    return

def plot_Vy_Vz(blade_number=0):
    plt.figure()
    plt.title('Wind speed system 4')
    plt.plot(np.rad2deg(theta_blade_arr[blade_number, 1:]),
             V0y_arr[blade_number, 1:], label='Vy')
    plt.plot(np.rad2deg(theta_blade_arr[blade_number, 1:]),
             V0z_arr[blade_number, 1:], label='Vz')
    plt.xlim(0, 360)
    plt.xlabel('Azimuth angle [degree]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid()
    plt.show()

plot_xy(r)
plot_Vy_Vz()
