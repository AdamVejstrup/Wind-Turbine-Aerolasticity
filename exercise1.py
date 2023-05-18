# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:09:29 2023

@author: leael
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import unravel_index


H=119  #Hub height m
L_s=7.1  #Length of shaft m
R=89.15 #Radius m
tilt_deg=-5 #grader  

theta_cone=0 #radianer
theta_yaw=np.deg2rad(20) #radianer
theta_tilt=0 #radianer


omega=0.62 #rad/s
delta_t=0.15 #sek
timerange=200


r=70 #m

wind_shear=0.2

v0=10 #mean windspeed at hub height m/s

a=3.32 #m tower radius
#For mere præcist bør der tilføjes for x<H then a=3.32, else a=0. Til spg 4

theta_blade1=[omega*delta_t]
theta_blade2=[omega*delta_t+2*np.pi/3]
theta_blade3=[omega*delta_t+4*np.pi/3]


time=[0]


a1=np.array([[1,0,0],
          [0,np.cos(theta_yaw),np.sin(theta_yaw)],
          [0,-np.sin(theta_yaw),np.cos(theta_yaw)]])

a2=np.array([[np.cos(theta_tilt), 0, -np.sin(theta_tilt)],
          [0,1,0],
          [np.sin(theta_tilt),0, np.cos(theta_tilt)]])

a3=np.array([[1,0,0],
          [0,1,0],
          [0,0,1]])

a12=a3@a2@a1    #Udregner transformation matrice a12

a21=np.transpose(a12)


a34=np.array([[np.cos(theta_cone),0,-np.sin(theta_cone)],
              [0,1,0],
              [np.sin(theta_cone), 0, np.cos(theta_cone)]])


rt1=np.array([H,0,0])

rs1=a21@np.array([0,0,-L_s])

x1=[]
y1=[]
z1=[]

V0y=[]
V0z=[]

Vz=[]
Vy=[]
#################################   Opgave 1   ###########################


for i in range(timerange):
    time.append(i*delta_t)
    
    theta_blade1.append(theta_blade1[i]+omega*delta_t)
    theta_blade2.append(theta_blade1[i]+omega*delta_t+2*np.pi/3)
    theta_blade3.append(theta_blade1[i]+omega*delta_t+4*np.pi/3)
    
    a23=np.array([[np.cos(theta_blade1[i+1]),np.sin(theta_blade1[i+1]),0],
              [-np.sin(theta_blade1[i+1]),np.cos(theta_blade1[i+1]),0],
              [0,0,1]])
    
    a14=a34@a23@a12
    
    a41=np.transpose(a14)
    
    rb1=a41@np.array([r,0,0])
    
    r1=rt1+rs1+rb1
    
    x1.append(r1[0])
    y1.append(r1[1])
    z1.append(r1[2])
    
    r_til_punkt=( y1[i]**2+z1[i]**2   )**(1/2)
    
    V0_array=np.array([0,0,v0*(x1[i]/H)**wind_shear])
    
    #Opgave 3
    
    V0_4=a14@V0_array
    
    
    # V0 uden tower (system 4)
    V0y.append(V0_4[1])
    V0z.append(V0_4[2])
    
    
    #Opgave 4
    if x1[i]<=H:
        a=3.32
    elif x1[i]>H:
        a=0
        
    Vr=z1[i]/r_til_punkt*V0z[i]*(1-(a/r_til_punkt)**2)
    Vtheta=y1[i]/r_til_punkt*V0z[i]*(1+(a/r_til_punkt)**2)
    
    # V0 med tower (system 4)
    Vz.append((z1[i]/r_til_punkt*Vr+y1[i]/r_til_punkt*Vtheta))
    Vy.append((y1[i]/r_til_punkt*Vr-z1[i]/r_til_punkt*Vtheta))

    
    
    
#Calculating x and y position
theta_blade1=np.array(theta_blade1)

x1_pos=np.cos(theta_blade1)*r+H
y1_pos=np.sin(theta_blade1)*r

#Opgave 1
plt.figure()
plt.title('Position in space (x,y)')
plt.plot(time,x1_pos,color='blue', label='x')
plt.plot(time,y1_pos,color='red', label='y')
plt.xlim(0, 30)
plt.xlabel('time [s]')
plt.ylabel('Positon[m]')
plt.legend()
plt.grid()
plt.show()


#Opgave 2
plt.figure()
plt.title('Position in space (x,y)')
plt.plot(time[:-1],x1,color='blue', label='x')
plt.plot(time[:-1],y1,color='red', label='y')
plt.xlim(0, 30)
plt.xlabel('time [s]')
plt.ylabel('Positon[m]')
plt.legend()
plt.grid()
plt.show()

##Opgave 3
plt.figure()
plt.title('Wind velocity (V1)')
plt.plot(theta_blade1[:-1],V0y,color='red', label='V0_y')
plt.plot(theta_blade1[:-1],V0z,color='blue', label='V0_z')
plt.xlabel('Azimuth angle [rad]')
plt.ylabel('Velocity[m/s]')
plt.legend()
plt.grid()
plt.show()


##Opgave 4
plt.figure()
plt.title('Wind velocity (V1)')
plt.plot(np.rad2deg(theta_blade1[:-1]),Vy[:],color='blue', label='Vy')
plt.plot(np.rad2deg(theta_blade1[:-1]),Vz[:],color='red', label='Vz')
plt.xlabel('Azimuthal angle [degree]')
plt.ylabel('Velocity[m/s]')
plt.xlim(0,360)
plt.legend()
plt.grid()
plt.show()






