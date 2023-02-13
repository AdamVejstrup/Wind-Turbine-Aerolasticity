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

# Giver figurer i bedre kvalitet når de vises i Spyder og når de gemmes (kan evt. sættes op til 500)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# Giver skriftstørrelse 12 som standard på plots i stedet for 10 som er default
plt.rcParams.update({'font.size': 12})

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
# omega = 9.6*2*np.pi/60 # rad/s gammel opgave
delta_t=0.15 # s
timerange=200
wind_shear=0
V_0=9 # mean windspeed at hub height m/s

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


#%% Array initializations

# Time array: 1D array (time)
time_arr = np.zeros([timerange])

# T, P, C_P and C_T
# 1D array (time)
T_arr = np.zeros([timerange])
P_arr = np.zeros([timerange])
CT_arr = np.zeros([timerange])
CP_arr = np.zeros([timerange])

# Blade position: 2D array (blade number, time)
theta_blade_arr = np.zeros([B,timerange])

# All V's, W's, p's
# 3D array (airfoil number, blade number, time)
x1_arr = np.zeros([len(airfoils), B, timerange])
y1_arr = np.zeros([len(airfoils), B, timerange])
z1_arr = np.zeros([len(airfoils), B, timerange])

V0x_arr = np.zeros([len(airfoils), B, timerange])
V0y_arr = np.zeros([len(airfoils), B, timerange])
V0z_arr = np.zeros([len(airfoils), B, timerange])

V_rel_y_arr = np.zeros([len(airfoils), B, timerange])
V_rel_z_arr = np.zeros([len(airfoils), B, timerange])

Wy_arr = np.zeros([len(airfoils), B, timerange])
Wz_arr = np.zeros([len(airfoils), B, timerange])

pt_arr = np.zeros([len(airfoils), B, timerange])
pn_arr = np.zeros([len(airfoils), B, timerange])


#%% Loop

for n in range(timerange):
    
    time_arr[n] = n*delta_t
    
    for i in range(B):
        
        # If statements fortæller hvordan azimutten (theta_blade) skal sættes
        # afhængigt af hvad nummer vinge, vi kigger på
        
        # Initialisering af første blade til 0 grader når tiden starter
        if n == 0 and i == 0:
            theta_blade_arr[i,n] = omega * delta_t
        # Udregning for de andre blades
        elif i == 0:
            theta_blade_arr[i,n] = theta_blade_arr[0,n-1] + omega * delta_t
        elif i == 1:
            theta_blade_arr[i,n] = theta_blade_arr[0,n] + omega * delta_t + 0.66 * np.pi
        elif i == 2:
            theta_blade_arr[i,n] = theta_blade_arr[0,n] + omega * delta_t + 1.33 * np.pi
        
        
        a23 = np.array([[np.cos(theta_blade_arr[i,n]),np.sin(theta_blade_arr[i,n]),0],
                  [-np.sin(theta_blade_arr[i,n]),np.cos(theta_blade_arr[i,n]),0],
                  [0,0,1]])
        
        
        a14 = a34 @ a23 @ a12
        
        a41=np.transpose(a14)
        
        for k in range(len(r)):
        
            rb1 = a41 @ np.array([r[k],0,0])
            
            r1 = rt1 + rs1 + rb1
            
            x1_arr[k, i, n] = r1[0]
            y1_arr[k, i, n] = r1[1]
            z1_arr[k, i, n] = r1[2]
            
            # Wind sheer. V_0 skal erstattes med et array af windspeeds i sidste opgave
            V0_array = np.array([0,0,V_0 * (x1_arr[k, i, n]/H)**wind_shear])
            
            # Går til system 4
            V0_4 = a14 @ V0_array
            
            V0x_arr[k, i, n] = V0_4[0]
            V0y_arr[k, i, n] = V0_4[1]
            V0z_arr[k, i, n] = V0_4[2]
            
            
            
            # If statement: Ved første tidsskridt er den forrige værdi af W'erne lig 0
            # Kommentar til r: Vi bruger r i nedenstående fordi den allerede er givet i system 4,
            # hvilket vores relative hastigheder også er
            if n == 0:
                V_rel_y_arr[k, i, n] = V0y_arr[k, i, n] - omega * r[k] * np.cos(theta_cone)
                V_rel_z_arr[k, i, n] = V0z_arr[k, i, n]
            else:
                V_rel_y_arr[k, i, n] = V0y_arr[k, i, n] + Wy_arr[k, i, n-1] - omega * r[k] * np.cos(theta_cone)
                V_rel_z_arr[k, i, n] = V0z_arr[k, i, n] + Wz_arr[k, i, n-1]
                
            # if k==17:
                # print(V_rel_z_arr[k, i, n]/(-V_rel_y_arr[k, i, n]))
                # print(-V_rel_y_arr[k, i, n])
            
            
            phi = np.arctan(V_rel_z_arr[k, i, n]/(-V_rel_y_arr[k, i, n]))
            
            
            aoa_deg = np.rad2deg(phi) - (beta[k] + np.rad2deg(theta_pitch))
            
            # cl, cd, cm skal opdateres til cl, cd, cm, _, _, _ senere
            # Index af tc skal sættes til nummer airfoil
            cl, cd, cm = force_coeffs_10MW(aoa_deg, tc[k], aoa_tab, cl_stat_tab, cd_stat_tab, cm_stat_tab)
            
            V_rel_abs = np.sqrt(V_rel_y_arr[k, i, n]**2 + V_rel_y_arr[k, i, n]**2)
            
            # V_rel_abs = np.sqrt(V_rel_y_arr[k, i, n]**2 + V_rel_z_arr[k, i, n]**2)
                        
            # V_0 er konstant nu, men skal opdateres til en liste når turbulens tages med
            a = Wz_arr[k, i, n]/V_0
            
            if a <= 0.33:
                f_g=1
            else:
                f_g=0.25*(5-3*a)
            
            V_f_W = np.sqrt(V0y_arr[k, i, n]**2 + (V0z_arr[k, i, n] + f_g * Wz_arr[k, i, n])**2)
                        
            l = 0.5 * rho * V_rel_abs**2 * c[k] * cl
            d = 0.5 * rho * V_rel_abs**2 * c[k] * cd
           
            if k==len(airfoils)-1:
                p_z=0
                p_y=0
            else:
                p_z = l * np.cos(phi) + d * np.sin(phi)      
                p_y = l * np.sin(phi) - d * np.cos(phi)
            
            # Gemmer normal og tangential loads for hvert blad
            
            pt_arr[k, i, n]=p_y
            pn_arr[k, i, n]=p_z
                        
            F = (2/np.pi) * np.arccos(np.exp(-(B/2) * ((R-r[k])/(r[k] * np.sin(abs(phi))))))
                        
            
            W_z_qs = (-B * l * np.cos(phi))/(4 * np.pi * rho * r[k] * F * V_f_W)
            
            W_y_qs = (-B * l * np.sin(phi))/(4 * np.pi * rho * r[k] * F * V_f_W)
            
            # Når vi ikke har timefilter på endnu
            
            Wz_arr[k, i, n] = W_z_qs
            Wy_arr[k, i, n] = W_y_qs
            
       
    # OBS: i stedet for at gange op til 3 blades så prøv at summér med de faktiske værdier
    # M_r = B * np.trapz(pt_arr[:,0,n]*r,r)
    M_r = np.trapz(np.sum(pt_arr,axis=1)[:,n]*r,r)
    
    P_arr[n] = omega*M_r
    # T = B*np.trapz(pn_arr[:,0,n],r)
    T = np.trapz(np.sum(pn_arr,axis=1)[:,n],r)
    T_arr[n] = T
    
    
    
    
#%% Hvordan theta udvikler sig i tid, nok ikke så brugbart    

plt.figure()
plt.grid()
plt.title('Blade position')
for i in range(B):
    plt.plot(time_arr, np.rad2deg(theta_blade_arr[i,:]),label='Blade {}'.format(i+1))
plt.xlabel('Time [s]')
plt.ylabel('$\Theta_b$ [deg]')
plt.legend()
plt.show()

#%% x og y position sammmen for en given airfoil
blade_element = 17
plt.figure()
plt.grid()
plt.title('Blade position, airfoil {} (1-based indexing)'.format(blade_element+1))
# For de tre vinger
for i in range(B):
    plt.plot(x1_arr[blade_element,i,:], y1_arr[blade_element,i,:],linewidth=7-3*i,label='Blade {}'.format(i+1))
# Tilføjer r
plt.plot([H,H+r[blade_element]],[0,0],label='r = {:.2f} m'.format(r[blade_element]))
# Symmetriske axer for cirkel i stedet for oval form
plt.axis('scaled')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='lower right')
plt.show()


#%% Plot af x positionen for en given airfoil. Hvis man tager airfoil nr. 17 kan man se at
# forskellen på top of bund svarer til rotordiameteren
blade_element = 17
plt.figure()
plt.grid()
plt.title('System 4 x-position for blade element {} (1-based indexing)'.format(blade_element+1))
# For de tre vinger
for i in range(B):
    plt.plot(time_arr, x1_arr[blade_element,i,:],label='Blade {}'.format(i+1))
# Tilføjer r
plt.plot([max(time_arr)*2/3,max(time_arr)*2/3],[H-r[blade_element],H+r[blade_element]],'--',label='r = {:.2f} m'.format(r[blade_element]))
plt.xlabel('Time [s]')
plt.ylabel('x [m]')
plt.legend()
plt.show()


#%% Creating figure with subplots of T and P
fig, ax1 = plt.subplots()

fig.suptitle('Thrust and power')
color = 'tab:orange'
ax1.grid()
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Thrust [MN]', color=color)
ax1.plot(time_arr, T_arr/(10**6), color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Power [MW]', color=color)  # we already handled the x-label with ax1
ax2.plot(time_arr, P_arr/(10**6), color=color)
# Man skal af en eller anden grund have denne her linje med for at de to y-akser bliver aligned
ax2.set_yticks(np.linspace(ax2.get_yticks()[0],ax2.get_yticks()[-1],len(ax1.get_yticks())))
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%%

blade_number = 0
plt.figure()
plt.grid()
plt.title('Load distribution for blade {} (1-based indexing)'.format(blade_number+1))
plt.plot(r,pn_arr[:, blade_number, -1],label='$p_n$',marker='o')
plt.plot(r,pt_arr[:, blade_number, -1],label='$p_t$',marker='x')
plt.xlabel('r [m]')
plt.ylabel('p [N]')
plt.legend()
plt.show()


#%% Creating figure with subplots of C_T and C_P
# fig, ax1 = plt.subplots()

# fig.suptitle('Thrust and power')
# color = 'tab:orange'
# ax1.grid()
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel('$C_T$ [-]', color=color)
# ax1.plot(time_arr, CT_arr, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('$C_P$ [-]', color=color)  # we already handled the x-label with ax1
# ax2.plot(time_arr, CP_arr, color=color)
# # Man skal af en eller anden grund have denne her linje med for at de to y-akser bliver aligned
# ax2.set_yticks(np.linspace(ax2.get_yticks()[0],ax2.get_yticks()[-1],len(ax1.get_yticks())))
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()















