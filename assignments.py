# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:09:29 2023

@author: leael
"""

import numpy as np
import matplotlib.pyplot as plt
from interpolation import force_coeffs_10MW
from load_turbulence_box import load
from scipy.interpolate import interp2d
from scipy import signal
from assignment_functions import (x_mask, make_gen_char,
                                  make_position_sys1)

# Giver figurer i bedre kvalitet når de vises i Spyder og når de gemmes (kan evt. sættes op til 500)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# Giver skriftstørrelse 12 som standard på plots i stedet for 10 som er default
plt.rcParams.update({'font.size':12})

#%% Indstillinger til de forskellige spørgsmål

# if use_wind_shear = False then wind_shear = 0
# if use_wind_shear = True then wind_shear = 0.2
use_wind_shear = False # Wind sheer

if use_wind_shear:
    wind_shear = 0.2
else:
    wind_shear = 0

# if use_pitch = True then pitch is changed in time interval (see assignment1 description)
# if use_pitch = False then the pitch is always 0 (except if pitch controller is used)
use_pitch = False

use_dwf = True # Dynamic wake filter
use_stall = True # Dynamic stall
use_turbulence = False # Turbulent data
use_pitch_controller = True # Pitch controller.

# NB hvis man skal se gode resultater for pds, skal man kører 4000 steps eller over
delta_t=0.1 # s
timerange=2000
# timerange=200*3

#for the plots, plots from xlim_min and forward
xlim_min = 50  #s
xlim_max = 120 #s

# xlim_max = time_arr[-1] #s

if use_turbulence and timerange < 4000:
    raise ValueError('Timerange < 4000 does not work for turbulent wind')


# %% Choose plots

plot_gen_char = False # Generator characteristic
plot_omega = True # Omega against time
plot_theta_p = True # Pitch against time
plot_position_sys1 = False # (y, x)-coordinates in system 1 of given blade element
plot_thrust_power = False # Thrust and power
plot_induced_wind = False # Induced wind y and z
plot_load_distribution = False # Load distribution and dtu 9 m/s load distribution
plot_thrust_per_blade = False # Thrust for each blade and total thrust
plot_pn_specific_element = False # Normal loading for specific blade and specific blade element
plot_thrust_psd = False # PSD of total thrust
plot_turbulence_contour = False # Contour plot of turbulence

# %% Force coeff files


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
r,beta_deg,c,tc = airfoils.T


#%% Konstanter

# NB: ALLE VINKLER ER RADIANER MED MINDRE DE HEDDER _DEG SOM F.EKS. AOA

V_0 = 9 # mean windspeed at hub height m/s

B = 3 # Number of blades
H = 119  # Hub height m
L_s = 7.1  # Length of shaft m
R = 89.17 # Radius m
A = R**2 *np.pi #m^2
tilt_deg = -5 # grader   (bruges ikke i uge 1)
lam_opt = 8 #
P_rated = 10.64*10**6 #W
rho = 1.225 # kg/m**3
I_rotor = 1.6*10**8 #kg*m^2  inertia moment of the drivetrain

K1 = np.deg2rad(14) # rad
K_I = 0.64 # no unit
K_P = 1.5 # s

C_p_opt = 0.47 #optimal C_p for DTU 10MW
K = 0.5*rho*A*R**3 * (C_p_opt/lam_opt**3) #konstant der bruges til at regne M_g
omega_rated = (P_rated/K)**(1/3)  
M_g_max = K * omega_rated**2  #Vores max generator torque
# M_g_max = K * 1.01**2  #Vores max generator torque
# omega_ref = 1.02 * omega_rated #tommelfingerregel fra Martin
omega_ref = 1.01#tommelfingerregel fra Martin

theta_cone = 0 # radianer
theta_yaw = np.deg2rad(0) # radianer
theta_tilt = 0 # radianer
theta_p = 0 # radianer
theta_p_I = 0 # radianer

theta_p_max_ang = np.deg2rad(45) #radianer, max pitch vinkel
theta_p_min_ang = 0 #radianer, min pitch vinkel
theta_p_max_vel = np.deg2rad(9) #radianer, max pitch vinkel ændring per sekund

# Dynamic wake filter constant
k_dwf = 0.6



#%% Turbulence box

if use_turbulence:
  
    # Reading in parameters for the turbulent box
    turbulence_parameters = np.genfromtxt('turbulence/inputEx3.INP')
    
    # Dimensionerne skal være ints. Disse er punkterne i boxen
    n1, n2, n3 = turbulence_parameters[2:5].astype(int)
    
    # Disse er længdedimensionerne af boxen
    Lx, Ly, Lz = turbulence_parameters[5:8]
    
    # Middel wind speed fra boxen. Skal matche middel wind speed fra scriptet her
    # Hvis disse to ikke ens, gives der en fejlmeddelelse
    umean = turbulence_parameters[9]
    if not np.isclose(umean,V_0):
        raise ValueError('The mean wind speed umean from the turbulent box does not match the V_0 in this script')

    deltay = Ly/(n2-1)
    deltax = Lx/(n1-1)
    deltaz = Lz/(n3-1)
    deltat = deltax/umean
    
    # Ligesom for middelvinden skal der være overensstemmelse mellem deltat fra turbulent
    # box og delta_t fra dette script
    deltat_threshold = 3 # digits (Denne threshold er sat for at undgå ValueError ved f.eks. 0.16666 og 0.16666666666666666)
    if not np.isclose(round(deltat,deltat_threshold),round(delta_t,deltat_threshold)):
        raise ValueError('Mismatch between deltat from turbulent box and delta_t in this script')
    
    # Load in the files and reshape them into 3D
    # turbulence er fordi filen ligger i en undermappe der hedder turbulence
    u=load('turbulence/sim1.bin',  N=(n1, n2, n3))
    
    ushp = np.reshape(u, (n1, n2, n3))
    
    # Vi ændrer på dimensionerne: fra box til aflvering
    # x bliver til z (tid)
    # z bliver til x
    # y bliver til y
    
    X_turb = np.arange(0,n2)*deltaz + (H - (n2-1)*deltaz/2) # Height
    Y_turb = np.arange(0,n3)*deltay - ((n3-1) * deltay)/2 # Width
    Z_turb = np.arange(0,n1)*deltax # Depth (Time)
    

#%% Transformation matrices

a1 = np.array([[1,0,0],
          [0,np.cos(theta_yaw),np.sin(theta_yaw)],
          [0,-np.sin(theta_yaw),np.cos(theta_yaw)]])

a2 = np.array([[np.cos(theta_tilt), 0, -np.sin(theta_tilt)],
          [0,1,0],
          [np.sin(theta_tilt),0, np.cos(theta_tilt)]])

a3 = np.array([[1,0,0],
          [0,1,0],
          [0,0,1]])

a12 = a3@a2@a1    #Udregner transformation matrice a12

a21 = np.transpose(a12)


a34 = np.array([[np.cos(theta_cone),0,-np.sin(theta_cone)],
              [0,1,0],
              [np.sin(theta_cone), 0, np.cos(theta_cone)]])


rt1 = np.array([H,0,0])

rs1 = a21@np.array([0,0,-L_s])


#%% Array initializations

# Time array: 1D array (time)
time_arr = np.zeros([timerange])

# T, P, C_P and C_T
# 1D array (time)
T_arr = np.zeros([timerange])
P_arr = np.zeros([timerange])
CT_arr = np.zeros([timerange])
CP_arr = np.zeros([timerange])
T_all_arr = np.zeros([B, timerange])

# Blade position: 2D array (blade number, time)
theta_blade_arr = np.zeros([B,timerange])
theta_blade_arr[1,0] = 2*np.pi/3
theta_blade_arr[2,0] = 4*np.pi/3


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
Wy_qs_arr = np.zeros([len(airfoils), B, timerange])
Wz_qs_arr = np.zeros([len(airfoils), B, timerange])
Wy_int_arr = np.zeros([len(airfoils), B, timerange])
Wz_int_arr = np.zeros([len(airfoils), B, timerange])

pt_arr = np.zeros([len(airfoils), B, timerange])
pn_arr = np.zeros([len(airfoils), B, timerange])

fs_arr = np.zeros([len(airfoils), B, timerange])
cl_arr = np.zeros([len(airfoils), B, timerange])

if use_pitch_controller:
    omega = (lam_opt*V_0)/R 
    # omega= 7.229*2*np.pi/60 # rad/s
    omega_arr = np.zeros(timerange)
    omega_arr[0] = omega
    omega_arr[1] = omega
    
else:
    omega= 7.229*2*np.pi/60 # rad/s
    omega_arr = np.full(timerange, omega)

theta_p_arr = np.zeros(timerange)
theta_p_arr[0] = np.deg2rad(25) 

theta_p_I_arr = np.zeros(timerange)

#%% Looping over time, blades, airfoils
for n in range(1,timerange):
    #%% Time loop
    
    time_arr[n] = n*delta_t
    
    if use_pitch:
        if 100 <= time_arr[n] <= 150:
            theta_p= np.deg2rad(2)    
        elif 150 < time_arr[n]:
            theta_p= 0
    
    if use_turbulence:
        
        # Turbulent box har tiden som første koordinat
        # og ikke som sidste koordinat som vi plejer
        f2d = interp2d(X_turb,Y_turb,ushp[n,:,:],kind='linear')
        
    
    for i in range(B):
        #%% Blade loop
        
        # If statements fortæller hvordan azimutten (theta_blade) skal sættes
        # afhængigt af hvad nummer vinge, vi kigger på
        
        if i == 0:
            theta_blade_arr[i,n] = theta_blade_arr[0,n-1] + omega_arr[n-1] * delta_t
        elif i == 1:
            theta_blade_arr[i,n] = theta_blade_arr[0,n] + omega_arr[n-1] * delta_t + 0.666 * np.pi
        elif i == 2:
            theta_blade_arr[i,n] = theta_blade_arr[0,n] + omega_arr[n-1] * delta_t + 1.333 * np.pi
        
        
        a23 = np.array([[np.cos(theta_blade_arr[i,n]),np.sin(theta_blade_arr[i,n]),0],
                  [-np.sin(theta_blade_arr[i,n]),np.cos(theta_blade_arr[i,n]),0],
                  [0,0,1]])
        
        
        a14 = a34 @ a23 @ a12
        
        a41=np.transpose(a14)
        
        for k in range(len(r)):
            #%% Airfoil loop
                        
            rb1 = a41 @ np.array([r[k],0,0])
            
            r1 = rt1 + rs1 + rb1
            
            x1_arr[k, i, n] = r1[0]
            y1_arr[k, i, n] = r1[1]
            z1_arr[k, i, n] = r1[2]
            
            # Wind shear. V_0 skal erstattes med et array af windspeeds i sidste opgave
            
            if use_turbulence:
                    
                turb = f2d([x1_arr[k, i, n]],[y1_arr[k, i, n]])[0]
                # turb = f2d(x1_arr[k, i, n],y1_arr[k, i, n])
                
                # v_arr[n] = f([x_arr[n]],[y_arr[n]])
                # v_arr_point[n] = f(point_x,point_y)
                
                V0_array = np.array([0,0,turb + V_0 * (x1_arr[k, i, n]/H)**wind_shear])
                
            else:
                V0_array = np.array([0,0,V_0 * (x1_arr[k, i, n]/H)**wind_shear])
            
            # Går til system 4
            V0_4 = a14 @ V0_array
            
            V0x_arr[k, i, n] = V0_4[0]
            V0y_arr[k, i, n] = V0_4[1]
            V0z_arr[k, i, n] = V0_4[2]
            
            
            # Kommentar til r: Vi bruger r i nedenstående fordi den allerede er givet i system 4,
            # hvilket vores relative hastigheder også er
            V_rel_y_arr[k, i, n] = V0y_arr[k, i, n] + Wy_arr[k, i, n-1] - omega_arr[n-1] * r[k] * np.cos(theta_cone)
            V_rel_z_arr[k, i, n] = V0z_arr[k, i, n] + Wz_arr[k, i, n-1]

            phi = np.arctan(V_rel_z_arr[k, i, n]/(-V_rel_y_arr[k, i, n]))
            
            if use_pitch_controller:
                aoa_deg = np.rad2deg(phi) - (beta_deg[k] + np.rad2deg(theta_p_arr[n-1]))
            else:
                aoa_deg = np.rad2deg(phi) - (beta_deg[k] + np.rad2deg(theta_p))
            
            # cl, cd, cm skal opdateres til cl, cd, cm, _, _, _ senere
            # Index af tc skal sættes til nummer airfoil
            cl, cd, cm, f_stat, cl_inv, cl_fs = force_coeffs_10MW(aoa_deg, tc[k], aoa_tab, cl_stat_tab, cd_stat_tab, cm_stat_tab, f_stat_tab, cl_inv_tab, cl_fs_tab)
            
            V_rel_abs = np.sqrt(V_rel_y_arr[k, i, n]**2 + V_rel_z_arr[k, i, n]**2)
            
            if use_stall:
                tau_stall = 4 * c[k] / V_rel_abs
                
                fs_arr[k, i, n] = f_stat + (fs_arr[k, i, n-1]-f_stat) * np.exp(-delta_t/tau_stall)
                
                cl_arr[k, i, n] = f_stat * cl_inv + (1-fs_arr[k, i, n]) * cl_fs
            
            else:
                cl_arr[k, i, n] = cl
            
                        
            # V_0 er konstant nu, men skal opdateres til en liste når turbulens tages med
                        
            a = -Wz_arr[k, i, n-1]/V_0
            
            if a <= 0.33:
                f_g=1
            else:
                f_g=0.25*(5-3*a)
            
            V_f_W = np.sqrt(V0y_arr[k, i, n]**2 + (V0z_arr[k, i, n] + f_g * Wz_arr[k, i, n-1])**2)
            
            l = 0.5 * rho * V_rel_abs**2 * c[k] * cl_arr[k, i, n]
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
            
            # F = 1
            # Når man laver 
            if np.sin(abs(phi)) <= 0.01 or R-r[k] <= 0.005:
                F = 1
            else:
                F = (2/np.pi) * np.arccos(np.exp(-(B/2) * ((R-r[k])/(r[k] * np.sin(abs(phi))))))
            
            Wz_qs_arr[k, i, n] = (-B * l * np.cos(phi))/(4 * np.pi * rho * r[k] * F * V_f_W)
            Wy_qs_arr[k, i, n] = (-B * l * np.sin(phi))/(4 * np.pi * rho * r[k] * F * V_f_W)
            
            
            # Dynamic wave filter
            if use_dwf:
                tau_1 = 1.1/(1-1.3*a)*R/V_0
                tau_2 = (0.39 - 0.26 * (r[k]/R)**2)*tau_1
                
                Hy_dwf = Wy_qs_arr[k, i, n] + k_dwf * tau_1 * (Wy_qs_arr[k, i, n] - Wy_qs_arr[k, i, n-1])/delta_t
                Hz_dwf = Wz_qs_arr[k, i, n] + k_dwf * tau_1 * (Wz_qs_arr[k, i, n] - Wz_qs_arr[k, i, n-1])/delta_t
                
                Wy_int_arr[k, i, n] = Hy_dwf + (Wy_int_arr[k, i, n-1] - Hy_dwf)*np.exp(-delta_t/tau_1)
                Wz_int_arr[k, i, n] = Hz_dwf + (Wz_int_arr[k, i, n-1] - Hz_dwf)*np.exp(-delta_t/tau_1)
                
                Wy_arr[k, i, n] = Wy_int_arr[k, i, n] + (Wy_arr[k, i, n-1] - Wy_int_arr[k, i, n])*np.exp(-delta_t/tau_2)
                Wz_arr[k, i, n] = Wz_int_arr[k, i, n] + (Wz_arr[k, i, n-1] - Wz_int_arr[k, i, n])*np.exp(-delta_t/tau_2)
            
            # Uden dynamic wave filter
            else:
                Wz_arr[k, i, n] = Wz_qs_arr[k, i, n]
                Wy_arr[k, i, n] = Wy_qs_arr[k, i, n]
    
    #%% Power and Thrust
       
    # OBS: i stedet for at gange op til 3 blades så summeres de faktiske værdier
    M_r = np.trapz(np.sum(pt_arr,axis=1)[:,n]*r,r)
    
    P_arr[n] = omega_arr[n-1]*M_r


    T_all_arr[0,n] = np.trapz(pn_arr[:,0,n],r)
    T_all_arr[1,n] = np.trapz(pn_arr[:,1,n],r)
    T_all_arr[2,n] = np.trapz(pn_arr[:,2,n],r)
    
    T = np.trapz(np.sum(pn_arr,axis=1)[:,n],r)
    T_arr[n] = T
    
    #%% update omega and pitch til pitch controller
    
    if use_pitch_controller:

        #Region 1
        if omega_arr[n-1] < omega_ref: 
            #update omega
            M_g = K * omega_arr[n-1]**2
            
        #region 2+3
        else:
            #update omega 
            # M_g = M_g_max
            M_g = 1.0545* 10**7
        
        #update theta_pitch
        GK = (1/ (1 + (theta_p_arr[n-1]/K1)))
        theta_p_P = GK * K_P * ( omega_arr[n-1] -omega_ref)
        theta_p_I_arr[n] = theta_p_I_arr[n-1] + GK * K_I * (omega_arr[n-1]-omega_ref) * delta_t
        
        #limit på theta_p_I angle
        if theta_p_I_arr[n] > theta_p_max_ang:
            theta_p_I_arr[n] = theta_p_max_ang
        elif theta_p_I_arr[n] < theta_p_min_ang:
            theta_p_I_arr[n] = theta_p_min_ang
        
        theta_p_arr[n] = theta_p_P + theta_p_I_arr[n]
        
        #hvis theta_p skal ændres hurtigere end den må (stigende i grader)
        if (theta_p_arr[n] > theta_p_arr[n-1] + theta_p_max_vel * delta_t):
            theta_p_arr[n] = theta_p_arr[n-1] + theta_p_max_vel * delta_t
            
        #hvis theta_p skal ændres hurtigere end den må (falende i grader)
        elif (theta_p_arr[n] < theta_p_arr[n-1] - theta_p_max_vel * delta_t):
            theta_p_arr[n] = theta_p_arr[n-1] - theta_p_max_vel * delta_t
            
        #theta_p må max være = theta_p_max_ang
        if (theta_p_arr[n] > theta_p_max_ang):
            theta_p_arr[n] = theta_p_max_ang

        #theta_p må min være = theta_p_min_ang
        elif (theta_p_arr[n] < theta_p_min_ang):
            theta_p_arr[n] = theta_p_min_ang
            
        #update omega
        omega_arr[n] = omega_arr[n-1] + ((M_r - M_g)/ I_rotor) * delta_t


#%% PLot af M_g mod omega (generator torque mod roational speed)
mask = x_mask(time_arr, xlim_min, xlim_max)

# Plotting generator characteristic
if plot_gen_char:
    make_gen_char(omega_rated, K)

#%% omega of pitch plot
mask = x_mask(time_arr, xlim_min, xlim_max)

if plot_omega:
    
    plt.figure()
    plt.grid()
    plt.title('Rotational speed $\omega$')
    plt.plot(time_arr[mask], omega_arr[mask], label = '$\omega$ rad/s')
    plt.xlabel('Time [s]')
    plt.ylabel('$\omega$ [rad/s]')
    plt.xlim(time_arr[mask][0], time_arr[mask][-1])
    plt.legend()
    plt.show()


if plot_theta_p:
    
    plt.figure()
    plt.grid()
    plt.title('Pitch angle $\Theta_p$')
    plt.plot(time_arr[mask], np.rad2deg(theta_p_arr)[mask], label = 'Pitch angle [deg]')
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch angle [deg]')
    plt.xlim(time_arr[mask][0], time_arr[mask][-1])
    plt.legend()
    plt.show()

    print('For V0=', V_0, 'theta_pitch=', np.rad2deg(theta_p_arr[-1]), 'deg')
          
#%% Plot x og y position sammmen for en given airfoil
mask = x_mask(time_arr, xlim_min, xlim_max)

# Last blade element
blade_element = 17


# Plot 1: Blade position of the last blade element (system 1)
# Plot 2: Blade position of the last blade element (system 1)
if plot_position_sys1:
    make_position_sys1(blade_element,
                       time_arr,
                       y1_arr,
                       x1_arr,
                       mask, r, B, H)

#%% Creating figure with subplots of T and P
mask = x_mask(time_arr, xlim_min, xlim_max)


if plot_thrust_power:
    
    fig, ax1 = plt.subplots(1,1)
    
    color = 'tab:orange'
    ax1.grid()
    ax1.set_title('Thrust and power')
    ax1.set_ylabel('Thrust [MN]', color=color)
    ax1.plot(time_arr[mask], (T_arr/(10**6))[mask], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim([time_arr[mask][0], time_arr[mask][-1]])
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Power [MW]', color=color)  # we already handled the x-label with ax1
    ax2.plot(time_arr[mask], (P_arr/(10**6))[mask], color=color)
    
    # Man skal af en eller anden grund have denne her linje med for at de to y-akser bliver aligned
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0],ax2.get_yticks()[-1],len(ax1.get_yticks())))
    ax2.tick_params(axis='y', labelcolor=color)


#%% Plotting induced wind
mask = x_mask(time_arr, xlim_min, xlim_max)


if plot_induced_wind:
    
    fig, ax1 = plt.subplots(1,1)
    
    blade_element = 8
    ax1.set_title('Induced wind')
    ax1.plot(time_arr,Wy_arr[blade_element, 0, :],label='$W_y$')
    ax1.plot(time_arr,Wz_arr[blade_element, 0, :],label='$W_z$')
    ax1.grid()
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('W [m/s]')
    ax1.set_xlim([xlim_min, xlim_max])
    ax1.legend(loc='center left')


#%% Plot af loadings


if plot_load_distribution:
    
    # DTU rapport loadings
    dtu_r, dtu_pt, dtu_pn = np.loadtxt('wsp_9_spanwise_loads.DAT',unpack=True,usecols=(0,1,2))
    
    blade_number = 0
    plt.figure()
    plt.grid()
    plt.title('Load distribution for blade {}'.format(blade_number+1))
    # plt.plot(r,pn_arr[:, blade_number, -1], label='$p_{n,calculated}$',marker='o')
    # plt.plot(r,pt_arr[:, blade_number, -1], label='$p_{t,calculated}$',marker='o')
    
    plt.plot(r, np.mean(pn_arr[:, blade_number, mask], axis=1), label='$p_{n,calculated}$',marker='o')
    plt.plot(r, np.mean(pt_arr[:, blade_number, mask], axis=1), label='$p_{t,calculated}$',marker='o')
    
    plt.plot(dtu_r,dtu_pt, label='$p_{t,dtu, 9 \; m/s}$', linestyle='--')
    plt.plot(dtu_r,dtu_pn, label='$p_{n,dtu, 9 \; m/s}$', linestyle='--')
    plt.xlim(0)
    plt.xlabel('r [m]')
    plt.ylabel('p [N/m]')
    plt.legend()
    plt.show()

#%% Plot af thrust for hver blade og for alle blades lagt sammen


if plot_thrust_per_blade:
    
    plt.figure()
    plt.grid()
    plt.title('Thrust')
    for i in range(B):
        plt.plot(time_arr[mask], (T_all_arr[i,:]/10**6)[mask],label='Blade {}'.format(i+1))
    plt.plot(time_arr[mask], (T_arr/(10**6))[mask],label = 'Total')
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust [MN]')
    plt.xlim(time_arr[mask][0], time_arr[mask][-1])
    plt.legend()
    plt.show()

#%% Turbulens plots

if plot_turbulence_contour:
    
    if not use_turbulence:
        raise ValueError('Set use turbulence to True to plot turbulence contour')
    
    # Plot a contour
    plane_number = 1000
    plane_time = deltat * plane_number
    
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(Y_turb,X_turb, ushp[plane_number,:,:])
    fig.colorbar(cp,label=f'Turbulence [m/s] at t = {plane_time:.1f} s') # Add a colorbar to a plot
    lwd = 3
    ax.scatter([0],[H],color='white',s=100)
    ax.plot([0,0],[H,H+R],color='white',linewidth = lwd)
    ax.plot([0,R*np.sin(2*np.pi/3)],[H,H + R*np.cos(2*np.pi/3)],color='white',linewidth = lwd)
    ax.plot([0,R*np.sin(4*np.pi/3)],[H,H + R*np.cos(4*np.pi/3)],color='white',linewidth = lwd)
    ax.plot([0,0],[X_turb[0],H],color='white',linewidth = lwd)
    ax.axis('scaled')
    ax.set_title(f'Wind speed = {umean} m/s + turbulence')
    ax.set_xlabel('y [m]')
    ax.set_ylabel('x [m]')
    plt.show()

# Plotting p_n in time with turbulence for a given blade and a given airfoil

if plot_pn_specific_element:
    
    # Need to discard the first few seconds to avoid the transcient part which
    # has a hight impact on the psd. Seconds to discard:
    sec_to_dis = 5
    # observations to discard
    obs_to_dis = int(sec_to_dis/deltat)
    
    # NB: Check on the time plot, that the transient part is gone
    
    blade_element = 8
    blade_number = 0
    plt.figure()
    plt.grid()
    plt.title('Loading blade {}, r = {:.2f} (turbulent wind)'.format(blade_number+1, r[blade_element]))
    plt.plot(time_arr[obs_to_dis:], pn_arr[blade_element,blade_number,obs_to_dis:])
    plt.xlabel('Time [s]')
    plt.xlim(obs_to_dis,time_arr[-1])
    plt.ylabel('$P_{n}$ [N/m]')
    plt.show()
    
    # Power spectral density for P_n
    
    # Frequency for psd
    fs=1/(time_arr[1]-time_arr[0])
    
    #Compute and plot the power spectral density using welch
    pn_freq, pn_psd = signal.welch(pn_arr[blade_element,blade_number,obs_to_dis:], fs, nperseg=1024)
    
    fig,ax=plt.subplots(1,1)
    ax.plot(pn_freq*2*np.pi/omega, pn_psd/(10**6))
    ax.set(xlim = [0,10], xlabel = '$2 \pi f / \omega}$ [-]', ylabel = 'PSD [$(MN/m)^{2} / Hz$]')
    
    # Sætter y lim, så den er lidt højere end peaket
    ylim_filter = (pn_freq*2*np.pi/omega) > 1
    ax.set_ylim(0,pn_psd[ylim_filter].max()*1.1/10**6)
    
    ax.set_title('Power spectral density of $p_n$ for one blade')
    ax.grid()
    plt.show()


# PSD of total thrust

if plot_thrust_psd:
    
    #Compute and plot the power spectral density. 
    T_freq, T_psd = signal.welch(T_arr[obs_to_dis:], fs, nperseg=1024)
    fig,ax=plt.subplots(1,1)
    plt.plot(T_freq*2*np.pi/omega, T_psd/(10**6), label='Blade')
    ax.set(xlim = [0,10], xlabel = '$2 \pi f / \omega}$ [-]', ylabel = 'PSD [$(MN)^{2} / Hz$]')
    # Sætter y lim, så den er lidt højere end peaket
    ylim_filter = (T_freq*2*np.pi/omega) > 1
    ax.set_ylim(0,T_psd[ylim_filter].max()*1.1/10**6)
    ax.set_title('Power spectral density of total thrust')
    ax.grid()
    plt.show()

incomingwind=[4,6,8,10,11,11.4, 12,14,16,18,20,22,24]
pitch_res=[0,0,0,0,0,2.4185394375943012,  5.0873878101, 9.4601400689, 12.5208309587, 15.1300994765, 17.4882184001, 19.6766815741, 21.7365520621]
rated_pitch=[2.4185394375943012, 2.4185394375943012, 2.4185394375943012,2.4185394375943012, 2.4185394375943012, 2.4185394375943012, 2.4185394375943012, 2.4185394375943012, 2.4185394375943012, 2.4185394375943012, 2.4185394375943012, 2.4185394375943012, 2.4185394375943012]

plt.figure()
plt.grid()
plt.title('Pitch angle as function of wind speed')
plt.plot(incomingwind,pitch_res,color='royalblue')
plt.plot(incomingwind,pitch_res, '.', color='blue')
plt.axvline(11.4, ls='--',color = 'Cornflowerblue', label='Rated wind speed')
plt.xlabel('Windspeed [m/s]')
plt.xlim(4,24)
plt.ylabel('$\Theta_p$ [deg]')
plt.legend()
plt.show()



