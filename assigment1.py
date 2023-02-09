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
from interpolation import force_coeffs_10MW


#%% Force coeff files


files=['FFA-W3-241.txt','FFA-W3-301.txt','FFA-W3-360.txt','FFA-W3-480.txt','FFA-W3-600.txt','cylinder.txt']

# Loader én fil, så vi har længden, som skal bruges til at lave vores np.zeros
# for de forskellige force coeff tabeller
first_airfoil = np.loadtxt(files[0])

# Laver tabeller til power coeffs
cl_stat_tab = np.zeros([len(first_airfoil),len(files)])
cd_stat_tab = np.zeros([len(first_airfoil),len(files)])
cm_stat_tab = np.zeros([len(first_airfoil),len(files)])
f_stat_tab = np.zeros([len(first_airfoil),len(files)])
cl_inv_tab = np.zeros([len(first_airfoil),len(files)])
cl_fs_tab=np.zeros([len(first_airfoil),len(files)])

# Indlæser tabellerne. Tabellen for cl inderholder 5 kolonner fordi der er 5 filer
# med hver sin cl kolonne. Tilsvarende for de andre koefficienter.
for i in range(np.size(files)):
    aoa_tab,cl_stat_tab[:,i],cd_stat_tab[:,i],cm_stat_tab[:,i],f_stat_tab[:,i],cl_inv_tab[:,i],cl_fs_tab[:,i] = np.loadtxt(files[i], skiprows=0).T


# Airfoil data
airfoils = np.loadtxt('bladedat.txt',skiprows=0)
r,beta,c,tc = airfoils.T


#%%

# NB: ALLE VINKLER ER RADIANER MED MINDRE DE HEDDER _DEG SOM F.EKS. AOA

B = 3 # Number of blades
H=119  # Hub height m
L_s=7.1  # Length of shaft m
R=89.17 # Radius m
tilt_deg=-5 # grader   (bruges ikke i uge 1)

theta_cone=0 # radianer
theta_yaw=np.deg2rad(0) # radianer
theta_tilt=0 # radianer
theta_pitch=0 # radianer

rho=1.225 # kg/m**3

omega= 7.229*2*np.pi/60 # rad/s
delta_t=0.15 # s
timerange=200


wind_shear=0

V_0=9 # mean windspeed at hub height m/s

a=3.32 # m tower radius
# For mere præcist bør der tilføjes for x<H then a=3.32, else a=0. Til spg 4

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

P=[]
T=[]

p_t=np.array([[0]*B]*len(airfoils))
p_n=np.array([[0]*B]*len(airfoils))
#################################   Opgave 1   ###########################


# for n in range(timerange):
for n in range(1):
    time.append(n*delta_t)
    
    for i in range(B):
        
        # If statements fortæller hvordan azimutten (theta_blade) skal sættes
        # afhængigt af hvad nummer vinge, vi kigger på
        
        if i == 1:
            theta_blade.append(theta_blade[i] + omega * delta_t)
        elif i == 2:
            theta_blade.append(theta_blade[i] + omega * delta_t + 0.66 * np.pi)
        else:
            theta_blade.append(theta_blade[i] + omega * delta_t+ 1.33 * np.pi)
        
        a23 = np.array([[np.cos(theta_blade[i+1]),np.sin(theta_blade[i+1]),0],
                  [-np.sin(theta_blade[i+1]),np.cos(theta_blade[i+1]),0],
                  [0,0,1]])
        
        a14 = a34 @ a23 @ a12
        
        a41=np.transpose(a14)
        
        for k in range(len(r)):
        # for k in range(1):
        
            rb1 = a41 @ np.array([r[k],0,0])
            
            r1 = rt1 + rs1 + rb1
            
            x1.append(r1[0])
            y1.append(r1[1])
            z1.append(r1[2])
            
            # Wind sheer til opgave 2
            V0_array = np.array([0,0,V_0*(x1[k]/H)**wind_shear])
            
            #Opgave 3
            
            V0_4 = a14 @ V0_array
            
            # V0 uden tower (system 4)
            V0y.append(V0_4[1])
            V0z.append(V0_4[2])
            
            #print(k)
            
            # Vi bruger W_y[k] og ikke W_y[k-1], da W_y opdateres sidst i loopet
            
            # Vi bruger r i nedenstående fordi den allerede er givet i system 4,
            # hvilket vores relative hastigheder også er
            V_rel_y = V0y[k] + W_y[k] - omega * r[k] * np.cos(theta_cone)
            V_rel_z = V0z[k] + W_z[k]
            
            phi = np.arctan(V_rel_z/-V_rel_y)
            
            # Index skal ændres til nummer airfoil
            aoa_deg = np.rad2deg(phi) - (beta[k] + np.rad2deg(theta_pitch))
                        
            # cl, cd, cm skal opdateres til cl, cd, cm, _, _, _ senere
            # Index af tc skal sættes til nummer airfoil
            cl, cd, cm = force_coeffs_10MW(aoa_deg, tc[k], aoa_tab, cl_stat_tab, cd_stat_tab, cm_stat_tab)
            
            V_rel_abs = np.sqrt(V_rel_y**2 + V_rel_z**2)
            
            # V_0 er konstant nu, men skal opdateres til en liste når turbulens tages med
            a = W_z[k]/V_0
            
            if a <= 0.33:
                f_g=1
            else:
                f_g=0.25*(5-3*a)
            
            V_f_W = np.sqrt(V0y[k]**2 + (V0z[k] + f_g * W_z[k])**2)
            
            l = 0.5 * rho * V_rel_abs**2 * c[k] * cl
            d = 0.5 * rho * V_rel_abs**2 * c[k] * cd
           
            if k==len(airfoils)-1:
                p_z=0
                p_y=0
            else:
                p_z = l * np.cos(phi) + d * np.sin(phi)      
                p_y = l * np.sin(phi) - d * np.cos(phi)
            
            #Appender normal og tangential loads for hvert blad
            p_t[k,i]=p_y
            
            p_n[k,i]=p_z
            
            F = (2/math.pi) * np.arccos(np.exp(-(B/2) * ((R-r[k])/(r[k] * np.sin(abs(phi))))))
            
            W_z_qs = (B * l * np.cos(phi))/(4 * np.pi * rho * r[k] * F * V_f_W)
            
            W_y_qs = (B * l * np.sin(phi))/(4 * np.pi * rho * r[k] * F * V_f_W)
        
        
            # NÆSTE SKRIDT: LAV W_Y OG APPEND
            # W_y er en list med ét element. Skal vi appende (W_y_qs)
            W_y.append(W_y_qs)
            W_z.append(W_z_qs)
    
    #Udregner thrsut og power disse skal appendes så vi kan plotte dem til tiden
    #M_r=B*np.trapz(np.array(p_y)*airfoils['r[m]'].values,airfoils['r[m]'].values)
    #P.append(omega*M_r)    
    #T.append(B*np.trapz(np.array(p_z),airfoils['r[m]'].values))
        


