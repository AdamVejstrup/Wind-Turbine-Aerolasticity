# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:10:51 2023
This is an example of how to load in a Mann box file and plot the contour and corresponding PSD. 
Use with care! I make many errors!
@author: cgrinde
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# Load routines stolen from Wind Energy Toolbox on Gitlab: 
#    https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/-/blob/master/wetb/wind/turbulence/mann_turbulence.py

def load(filename, N=(32, 32)):
    """Load mann turbulence box

    Parameters
    ----------
    filename : str
        Filename of turbulence box
    N : tuple, (ny,nz) or (nx,ny,nz)
        Number of grid points

    Returns
    -------
    turbulence_box : nd_array

    Examples
    --------
    >>> u = load('turb_u.dat')
    """
    data = np.fromfile(filename, np.dtype('<f'), -1)
    if len(N) == 2:
        ny, nz = N
        nx = len(data) / (ny * nz)
        assert nx == int(nx), "Size of turbulence box (%d) does not match ny x nz (%d), nx=%.2f" % (
            len(data), ny * nz, nx)
        nx = int(nx)
    else:
        nx, ny, nz = N
        assert len(data) == nx * ny * \
            nz, "Size of turbulence box (%d) does not match nx x ny x nz (%d)" % (len(data), nx * ny * nz)
    return data.reshape(nx, ny * nz)


#Example! Make sure to change numbers according to your setup!
#Defining size of box
n1=4096
n2=32
n3=32

Lx=6142.5
Ly=180
Lz=180

umean=9

deltay=Ly/(n2-1)
deltax=Lx/(n1-1)
deltaz=Lz/(n3-1)
deltat=deltax/umean

time=np.arange(deltat, n1*deltat+deltat, deltat)

# Load in the files and reshape them into 3D
u=load('sim1.bin',  N=(n1, n2, n3))

ushp = np.reshape(u, (n1, n2, n3)) 


# Plot a countour
fig,ax=plt.subplots(1,1)
cp = ax.contourf(ushp[1000,:,:])
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
ax.set_xlabel('y (point number in box system)')
ax.set_ylabel('z (point number in box system)')
plt.show()

#%%

# Vi ændrer på dimensionerne: fra box til aflvering
# x bliver til z (tid)
# z bliver til x
# y bliver til y

H = 119 # m

X_turb = np.arange(0,n2)*deltaz + (H - (n2-1)*deltaz/2) # Height
Y_turb = np.arange(0,n3)*deltay - ((n3-1) * deltay)/2 # Width
Z_turb = np.arange(0,n1)*deltax # Depth (Time)


# Plot a countour
fig,ax=plt.subplots(1,1)
cp = ax.contourf(Y_turb,X_turb, ushp[1000,:,:])
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
ax.set_xlabel('y [m]')
ax.set_ylabel('x [m]')
plt.show()

#%% Rotating point

omega= 7.229*2*np.pi/60 # rad/s

timerange=4000

# delta_t=0.15 # s
delta_t = deltat # s

theta_arr = np.zeros(timerange)
time_arr = np.zeros(timerange)
x_arr = np.zeros(timerange)
y_arr = np.zeros(timerange)
v_arr = np.zeros(timerange)
v_arr_point = np.zeros(timerange)

point_x, point_y = [10,2]

r = 20 # m

for n in range(1,timerange):
    
    time_arr[n] = n*delta_t
    
    theta_arr[n] = omega * time_arr[n]
    
    x_arr[n] = r * np.sin(theta_arr[n]) + H
    
    y_arr[n] = r * np.cos(theta_arr[n])
    
    f = interp2d(X_turb,Y_turb,ushp[n,:,:],kind='linear')
    
    v_arr[n] = f([x_arr[n]],[y_arr[n]])
    v_arr_point[n] = f(point_x,point_y)
    
    
    
plt.figure()
plt.grid()
plt.title('Title')
plt.plot(y_arr[1:],x_arr[1:],label='Label')
plt.axis('scaled')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()

plt.figure()
plt.grid()
plt.title('Title')
plt.plot(time_arr,v_arr,label='Label')
plt.plot(time_arr,v_arr_point,label='Label')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()

#%%


# Picking a point on the velocity plane
# sig=ushp[:,point_x,point_y]

fs=1/(time_arr[1]-time_arr[0])

#Compute and plot the power spectral density.
# Check out this site for inputs to the Welch
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html 
f_point, Pxx_den_point = signal.welch(v_arr_point, fs, nperseg=1024)
f, Pxx_den = signal.welch(v_arr, fs, nperseg=1024)
fig,ax=plt.subplots(1,1)
plt.plot(f_point, Pxx_den_point,label='Point')
plt.plot(f, Pxx_den, label='Blade')
plt.yscale('log')
plt.xlim([0,1])
plt.axvline(omega/(2*np.pi),color='black')
plt.axvline(omega/np.pi,color='black')
plt.grid()
plt.legend()
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD ')
plt.show()

# Det vi har fået er kun afvigelsen fra middelvinden grundet turbulens

std = np.std(ushp,axis=0)

ti = std/umean*100
# ti = std/(np.mean(ushp,axis=0)+umean)*100



#%% Christians script
"""

# Picking a point on the velocity plane
sig=ushp[:,20,20]

fs=1/(time[1]-time[0])

#Compute and plot the power spectral density.
# Check out this site for inputs to the Welch
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html 
f, Pxx_den = signal.welch(sig, fs, nperseg=1024)
fig,ax=plt.subplots(1,1)
plt.loglog(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD ')
plt.show()

# Det vi har fået er kun afvigelsen fra middelvinden grundet turbulens

std = np.std(ushp,axis=0)

ti = std/umean*100
"""


