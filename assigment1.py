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


#%% Force coeff files

files=['FFA-W3-241.txt','FFA-W3-301.txt','FFA-W3-360.txt','FFA-W3-480.txt','FFA-W3-600.txt']

# Reading files
# for i in range(np.size(files)):
#     aoa,cl_tab[:,i],cd_tab[:,i],cm_tab[:,i] = np.loadtxt(files[i], skiprows=0).T






H=119  #Hub height m
L_s=7.1  #Length of shaft m
R=89.17 #Radius m
tilt_deg=-5 #grader   (bruges ikke i uge 1)

theta_cone=0 #radianer
theta_yaw=np.deg2rad(0) #radianer
theta_tilt=0 #radianer
theta_pitch=0 # radianer

rho=1.225 # kg/m**3

omega= 7.229*2*np.pi/60 #rad/s
delta_t=0.15 #sek
timerange=200


r=70 #m

wind_shear=0

v0=9 #mean windspeed at hub height m/s

a=3.32 #m tower radius
#For mere præcist bør der tilføjes for x<H then a=3.32, else a=0. Til spg 4

theta_blade=[omega*delta_t]
W_y = [0]
W_z = [0]


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
    
    theta_blade.append(theta_blade[i]+omega*delta_t)
    
    a23=np.array([[np.cos(theta_blade[i+1]),np.sin(theta_blade[i+1]),0],
              [-np.sin(theta_blade[i+1]),np.cos(theta_blade[i+1]),0],
              [0,0,1]])
    
    a14=a34@a23@a12
    
    a41=np.transpose(a14)
    
    rb1=a41@np.array([r,0,0])
    
    r1=rt1+rs1+rb1
    
    x1.append(r1[0])
    y1.append(r1[1])
    z1.append(r1[2])
    
    r_til_punkt = (y1[i]**2+z1[i]**2)**(1/2)
    
    V0_array = np.array([0,0,v0*(x1[i]/H)**wind_shear])
    
    #Opgave 3
    
    V0_4=a14@V0_array
    
    # V0 uden tower (system 4)
    V0y.append(V0_4[1])
    V0z.append(V0_4[2])
    
    
    # Vi bruger W_y[i] og ikke W_y[i-1], da W_y opdateres sidst i loopet
    
    # Vi bruger r i nedenstående fordi den allerede er givet i system 4,
    # hvilket vores relative hastigheder også er
    V_rel_y = V0y[i] + W_y[i] - omega*r*np.cos(theta_cone)
    V_rel_z = V0z[i] + W_z[i]
    
    phi = np.arctan(V_rel_z/-V_rel_y)
    
    aoa = phi - ()
    
    
    
    
    
    
    
    


