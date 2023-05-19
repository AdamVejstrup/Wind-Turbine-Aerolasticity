# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:30:30 2023

@author: Toke Schäffer

Note: make sure that the time step is low enough:
Parameters that work well:
    A = 0.2
    cyc = 20
    num_of_steps = 1000
    omega = 5
    dt = cyc * T / num_of_steps = 0.125

This aerodynamic damping exercise simulates an airfoil in a wind tunnel
moving with a sinusoidal motion along the axis x, which has the angle theta
with the horizontal axis (Module 5, slide 23). The amplitude of the motion
is A and the frequency is omega. The chord length [m] of the airfoil is the
class attribute c=1. The airfoil has a geometric angle of attack [rad] a_0 and
the incoming wind has the speed V_0 [m/s] along the axis x.

The script is used by creating an instance of the class Airfoil and then
calling the function calc_work with the desired input parameters e.g.:

airfoil = Airfoil()
airfoil.calc_work(rho=1.225,
                    a_0=np.deg2rad(10),
                    theta=np.deg2rad(0),
                    A=0.2,
                    omega=5,
                    V_0=10,
                    cyc=20,
                    num_of_steps=2000,
                    use_stall=True)

The most important output is the accumulated work done by the aerodynamic force
and the power (slope of the accumulated work). The accumulated work is calculated
for each timestep and the power for the full simulation.

The class Airfoil contains functions to plot:
- work and power: plot_work
- angle of attack: plot_alpha
- lift coefficient: plot_cl
- drag coefficient: plot_cd
- position along x-axis: plot_x
- relative velocity: plot_V_rel
- aerodynamic force: plot_F_x
- aerodynamic force in the lift direction: plot_F_x_l
- aerodynamic force in the drag direction: plot_F_x_d

Note that the plots can be limited to a certain time interval by using the
arguments x_lim_low and x_lim_high. The plots can also be limited to a certain
range of y-values by using the arguments y_lim_low and y_lim_high.

If use_mask is set to True, the data will be filtered using the function x_mask
from assignment_functions.py and the y-axis will be automatically adjusted.
If use_mask is set to False, the data will not be filtered and the y-axis
must be adjusted manually.

Sweeping the angle of attack and the angle of the x-axis can be done using the
function sweep_alpha. The function returns a dictionary with the structure:
{alpha: {wo_stall: work_arr_wo_stall, with_stall: work_arr_with_stall}}
Sweeping refers to running the code with different angles of attack and angles
of the x-axis and storing the work done by the aerodynamic force for each
combination of angles of attack and angles of the x-axis e.g.:

    # List of angles of attack [deg] to sweep
    alpha_list = np.array([5, 10, 15, 20])

    # List of angles of the x-axis [deg] to sweep
    theta_list = np.linspace(0, 180, 180)

This is only done if the boolean with_swep is set to True.
Afterwards the results are plottet using the function plot_sweep_alpha.

Things that can be improved:
- Missing plot of delta work (slide 6, Module 5)

"""

import numpy as np
import matplotlib.pyplot as plt
from assignment_functions import x_mask

class Airfoil:
    """Class only contain chord length and airfoil data as attributes.
    """
    
    def __init__(self, c=1, file_name="FFA-W3-241.txt"):
        self.c = c # Chord length [m]
        self.file_name = file_name # Airfoil data file name
        self.data = np.loadtxt(file_name) # Load airfoil data
        
         # Unpack data
        (self.alpha_tab, self.cl_tab, self.cd_tab, self.cm_tab,
        self.f_stat_tab, self.cl_inv_tab, self.cl_fs_tab) = self.data.T
    
    def calc_work(self,
                  A=0.2,                # Vibration amplitude [m]
                  omega=5,              # Vibration frequency [rad/s]
                  a_0=np.deg2rad(10),   # Angle of attack [rad]
                  theta=np.deg2rad(0),  # Angle of the x-axis [rad]
                  V_0=10,               # Wind speed [m/s]
                  rho=1.225,            # Air density [kg/m^3]
                  cyc=10,               # Number of full cycles
                  num_of_steps=50,      # Number of timesteps
                  use_stall=False):     # Use dynamic stall
        
        """Function to calculate the work done by the aerodynamic force.
        """
        
        T = 2*np.pi / omega                 # Period [s]
        dt = cyc * T / num_of_steps         # Calculate timestep [s]
        
        # Initialize arrays
        time_arr = np.zeros(num_of_steps)       # Time array
        x = np.zeros(time_arr.shape)            # Position array
        F_x = np.zeros(time_arr.shape)          # Aerodynamic force array
        F_x_l = np.zeros(time_arr.shape)        # Aerodynamic force array (lift)
        F_x_d = np.zeros(time_arr.shape)        # Aerodynamic force array (drag)
        accum_work = np.zeros(time_arr.shape)   # Accumulated work array
        fs_arr = np.zeros(time_arr.shape)       # Degree of stall array
        cl_arr = np.zeros(time_arr.shape)       # Lift coefficient array
        cd_arr = np.zeros(time_arr.shape)       # Drag coefficient array
        alpha_arr = np.zeros(time_arr.shape)    # Angle of attack array
        V_rel_arr = np.zeros(time_arr.shape)    # Relative velocity array
        
        # Loop over timesteps
        for n in range(1, num_of_steps):
            
            time_arr[n] =  n * dt  # Time
            
            x[n] = A * np.sin(omega * time_arr[n]) # Position along x-axis
            
            dxdt = A * omega * np.cos(omega * time_arr[n]) # Velocity along x-axis
            
            # Relative velocity (calculating and storing for plotting)
            V_y = V_0 * np.cos(a_0) + dxdt * np.cos(theta)
            
            V_z = V_0 * np.sin(a_0) + dxdt * np.sin(theta)
            
            V_rel = np.sqrt((V_y)**2 + (V_z)**2)
            V_rel_arr[n] = V_rel
            
            # Angle of attack [rad]
            alpha = np.arctan(V_z / V_y)
            alpha_arr[n] = alpha
            
            # Lift coefficient (will be overwritten if stall is used)
            cl = np.interp(np.rad2deg(alpha), self.alpha_tab, self.cl_tab)
            
            # Drag coefficient
            cd = np.interp(np.rad2deg(alpha), self.alpha_tab, self.cd_tab)
            cd_arr[n] = cd
            
            # Static value that fs_arr will try to get back to
            f_stat = np.interp(np.rad2deg(alpha), self.alpha_tab, self.f_stat_tab)
            
            # Lift coefficient for inviscid flow (without any separation)
            cl_inv = np.interp(np.rad2deg(alpha), self.alpha_tab, self.cl_inv_tab)
            
            # Fully separated lift coefficient
            cl_fs = np.interp(np.rad2deg(alpha), self.alpha_tab, self.cl_fs_tab)
            
            if use_stall:
                
                # Time constant for stall
                tau_stall = 4 * self.c / V_rel
                
                # Degree of stall
                fs_arr[n] = f_stat + (fs_arr[n-1] - f_stat) * np.exp(-dt/tau_stall)
                
                # Lift coefficient for stall conditions
                # cl = f_stat * cl_inv + (1-fs_arr[n]) * cl_fs
                cl = fs_arr[n] * cl_inv + (1-fs_arr[n]) * cl_fs

            # Store lift coefficient
            cl_arr[n] = cl
            
            # Aerodynamic force split in three parts
            # Start
            F_x_start = 0.5 * rho * V_rel**2 * self.c
            
            # Lift contribution
            F_x_l[n] = cl * np.sin(alpha-theta) * F_x_start
            # Drag contribution
            F_x_d[n] = - cd * np.cos(alpha-theta) * F_x_start
            
            # Old way of calculating the aerodynamic force
            # F_x_end = cl * np.sin(alpha-theta) - cd * np.cos(alpha-theta)
            # F_x[n] = F_x_start * F_x_end
            
            # According to Module 5, slide 24, the drag force should be
            # subtracted from the lift force, but here the drag force is
            # already negative, so it is added.
            F_x[n] = F_x_l[n] + F_x_d[n]
            
            # Accumulated work: work so far + work done by the aerodynamic force
            # Work = force * distance
            accum_work[n] = accum_work[n-1] + F_x[n] * dxdt * dt
            
        # Note that accumulated work was calculated for each timestep
        # The work for the full simulation can now be calculated.
        # Work done by the aerodynamic force split in two parts:
        work_coeff = A * omega
        work_integral = np.trapz(F_x * np.cos(omega * time_arr), time_arr)
        
        # Collecting the two parts
        work = work_coeff * work_integral
        
        # Store simulation results in class attributes
        # Note that power and work per ccyle are a floats and all other attributes are arrays
        self.sim_time = time_arr
        self.sim_accum_work = accum_work
        self.sim_x = x
        self.sim_alpha = alpha_arr
        self.sim_cl = cl_arr
        self.sim_cd = cd_arr
        self.sim_V_rel = V_rel_arr
        self.sim_F_x = F_x
        self.sim_F_x_l = F_x_l
        self.sim_F_x_d = F_x_d
        
        # Power (float)
        self.sim_power = work / (time_arr[-1]-time_arr[0])
        
        # Work per cycle
        self.sim_work_per_cycle = work / cyc
        
        return
    
    def plot_alpha(self, x_lim_low=None, x_lim_high=None, y_lim_low=None, y_lim_high=None, use_mask=True):
        """Plot angle of attack in degrees in time.
        """
        if use_mask:
            mask = x_mask(self.sim_time, x_lim_low, x_lim_high)
        
        plt.figure()
        plt.grid()
        plt.title("Angle of attack")
        plt.xlabel("Time [s]")
        plt.ylabel("Angle of attack [deg]")
        if use_mask:
            plt.plot(self.sim_time[mask], np.rad2deg(self.sim_alpha[mask]))
        else:
            plt.plot(self.sim_time, np.rad2deg(self.sim_alpha))
        plt.xlim(x_lim_low, x_lim_high)
        plt.ylim(y_lim_low, y_lim_high)
        plt.show()
        return
    
    def plot_cl(self, x_lim_low=None, x_lim_high=None, y_lim_low=None, y_lim_high=None, use_mask=True):
        """Plot lift coefficient in time.
        """
        if use_mask:
            mask = x_mask(self.sim_time, x_lim_low, x_lim_high)
            
        plt.figure()
        plt.grid()
        plt.title("Lift coefficient")
        plt.xlabel("Time [s]")
        plt.ylabel("Lift coefficient")
        if use_mask:
            plt.plot(self.sim_time[mask], self.sim_cl[mask])
        else:
            plt.plot(self.sim_time, self.sim_cl)
        plt.xlim(x_lim_low, x_lim_high)
        plt.ylim(y_lim_low, y_lim_high)
        plt.show()
        return

    def plot_cd(self, x_lim_low=None, x_lim_high=None, y_lim_low=None, y_lim_high=None, use_mask=True):
        """Plot drag coefficient in time.
        """
        if use_mask:
            mask = x_mask(self.sim_time, x_lim_low, x_lim_high)
        
        plt.figure()
        plt.grid()
        plt.title("Drag coefficient")
        plt.xlabel("Time [s]")
        plt.ylabel("Drag coefficient")
        if use_mask:
            plt.plot(self.sim_time[mask], self.sim_cd[mask])
        else:
            plt.plot(self.sim_time, self.sim_cd)
        plt.xlim(x_lim_low, x_lim_high)
        plt.ylim(y_lim_low, y_lim_high)
        plt.show()
        return
    
    def plot_x(self, x_lim_low=None, x_lim_high=None, y_lim_low=None, y_lim_high=None, use_mask=True):
        """Plot position along x-axis in time.
        """
        if use_mask:
            mask = x_mask(self.sim_time, x_lim_low, x_lim_high)
            
        plt.figure()
        plt.grid()
        plt.title("Position along x-axis")
        plt.xlabel("Time [s]")
        plt.ylabel("Position along x-axis [m]")
        if use_mask:
            plt.plot(self.sim_time[mask], self.sim_x[mask])
        else:
            plt.plot(self.sim_time, self.sim_x)
        plt.xlim(x_lim_low, x_lim_high)
        plt.ylim(y_lim_low, y_lim_high)
        plt.show()
        return
    
    def plot_V_rel(self, x_lim_low=None, x_lim_high=None, y_lim_low=None, y_lim_high=None, use_mask=True):
        """Plot relative velocity in time.
        """
        if use_mask:
            mask = x_mask(self.sim_time, x_lim_low, x_lim_high)
            
        plt.figure()
        plt.grid()
        plt.title("Relative velocity")
        plt.xlabel("Time [s]")
        plt.ylabel("Relative velocity [m/s]")
        if use_mask:
            plt.plot(self.sim_time[mask], self.sim_V_rel[mask])
        else:
            plt.plot(self.sim_time[1:], self.sim_V_rel[1:])
        plt.xlim(x_lim_low, x_lim_high)
        plt.ylim(y_lim_low, y_lim_high)
        plt.show()
        return
    
    def plot_F_x(self, x_lim_low=None, x_lim_high=None, y_lim_low=None, y_lim_high=None, use_mask=True):
        """Plot aerodynamic force in time.
        """
        if use_mask:
            mask = x_mask(self.sim_time, x_lim_low, x_lim_high)
            
        plt.figure()
        plt.grid()
        plt.title("Aerodynamic force")
        plt.xlabel("Time [s]")
        plt.ylabel("Aerodynamic force [N]")
        if use_mask:
            plt.plot(self.sim_time[mask], self.sim_F_x[mask])
        else:
            plt.plot(self.sim_time, self.sim_F_x)
        plt.xlim(x_lim_low, x_lim_high)
        plt.ylim(y_lim_low, y_lim_high)
        plt.show()
        return

    def plot_F_x_l(self, x_lim_low=None, x_lim_high=None, y_lim_low=None, y_lim_high=None, use_mask=True):
        """Plot aerodynamic force in the lift direction in time.
        """
        if use_mask:
            mask = x_mask(self.sim_time, x_lim_low, x_lim_high)
        
        plt.figure()
        plt.grid()
        plt.title("Lift force")
        plt.xlabel("Time [s]")
        plt.ylabel("Lift force [N]")
        if use_mask:
            plt.plot(self.sim_time[mask], self.sim_F_x_l[mask])
        else:
            plt.plot(self.sim_time, self.sim_F_x_l)
        plt.xlim(x_lim_low, x_lim_high)
        plt.ylim(y_lim_low, y_lim_high)
        plt.show()
        return
    
    def plot_F_x_d(self, x_lim_low=None, x_lim_high=None, y_lim_low=None, y_lim_high=None, use_mask=True):
        """Plot aerodynamic force in the drag direction in time.
        """
        if use_mask:
            mask = x_mask(self.sim_time, x_lim_low, x_lim_high)
        
        plt.figure()
        plt.grid()
        plt.title("Drag force")
        plt.xlabel("Time [s]")
        plt.ylabel("Drag force [N]")
        if use_mask:
            plt.plot(self.sim_time[mask], self.sim_F_x_d[mask])
        else:
            plt.plot(self.sim_time, self.sim_F_x_d)
        plt.xlim(x_lim_low, x_lim_high)
        plt.ylim(y_lim_low, y_lim_high)
        plt.show()
        return
    
    def plot_work(self, x_lim_low=None, x_lim_high=None, y_lim_low=None, y_lim_high=None, use_mask=True):
        """Plot accumulated work and power in time and slope (power).
        """
        if use_mask:
            mask = x_mask(self.sim_time, x_lim_low, x_lim_high)
            
        plt.figure()
        plt.grid()
        plt.title("Accumulated work")
        plt.xlabel("Time [s]")
        plt.ylabel("Accumulated work [J]")
        if use_mask:
            plt.plot(self.sim_time[mask], self.sim_accum_work[mask], label="Accumulated work")
            plt.plot(self.sim_time[mask], self.sim_power * self.sim_time[mask],
                     label=f"Power = {self.sim_power:.3f} W")
        else:
            plt.plot(self.sim_time, self.sim_accum_work, label="Accumulated work")
            plt.plot(self.sim_time, self.sim_power * self.sim_time,
                    label=f"Power = {self.sim_power:.3f} W")
        plt.xlim(x_lim_low, x_lim_high)
        plt.ylim(y_lim_low, y_lim_high)
        plt.legend()
        plt.show()
        return


def sweep_alpha(airfoil,
                alpha_list,
                theta_list,
                rho=1.225,
                A=0.2,
                omega=5,
                V_0=10,
                cyc=20,
                num_of_steps=1000):
    
    """Function to sweep the angle of attack and the angle of the x-axis.
    
    Returns
    -------
    sweep_dict : dict
        Dictionary with structure:
        {alpha: {wo_stall: work_arr_wo_stall, with_stall: work_arr_with_stall}}
    
    Examples
    --------
    sweep_dict = sweep_alpha(airfoil,
                         alpha_list,
                         theta_list,
                         rho=1.225,
                         A=0.2,
                         omega=5,
                         V_0=10,
                         cyc=20,
                         num_of_steps=1000
                         )
    
    """
    
    # Initialize dictionary
    sweep_dict = {}
    
    # Loop over angles of attack
    for alpha_idx, alpha_val in enumerate(alpha_list):
        
        print(f"Calculating work for alpha = {alpha_val} deg")
        
        # Create a dictionary for each angle of attack
        sweep_dict[alpha_val] = {}
        
        # Array to store work for each angle of the x-axis
        # Without stall
        work_arr_wo_stall = np.zeros(len(theta_list))
        # With stall
        work_arr_with_stall = np.zeros(len(theta_list))
        
        # Loop over angles of the x-axis
        for theta_idx, theta_val in enumerate(theta_list):
            
            # Calculate work for each angle of the x-axis without stall
            airfoil.calc_work(rho=rho,
                        a_0=np.deg2rad(alpha_val),
                        theta=np.deg2rad(theta_val),
                        A=A,
                        omega=omega,
                        V_0=V_0,
                        cyc=cyc,
                        num_of_steps=num_of_steps,
                        use_stall=False)
            
            work_arr_wo_stall[theta_idx] = airfoil.sim_work_per_cycle
            
            # Calculate work for each angle of the x-axis with stall
            airfoil.calc_work(rho=rho,
                        a_0=np.deg2rad(alpha_val),
                        theta=np.deg2rad(theta_val),
                        A=A,
                        omega=omega,
                        V_0=V_0,
                        cyc=cyc,
                        num_of_steps=num_of_steps,
                        use_stall=True)
            
            work_arr_with_stall[theta_idx] = airfoil.sim_work_per_cycle
        
        # Store work arrays in dictionary for each angle of attack
        sweep_dict[alpha_val]['wo_stall'] = work_arr_wo_stall
        sweep_dict[alpha_val]['with_stall'] = work_arr_with_stall
    
    return sweep_dict
    
def plot_sweep_dict(sweep_dict, alpha_list, theta_list):
    """Function to plot the sweep dictionary.
    
    Examples
    --------
    plot_sweep_dict(sweep_dict, alpha_list, theta_list)
    
    """
    
    # List of angles of attack [deg]
    alpha_list = np.array([5, 10, 15, 20])
    
    # List of angles of the x-axis [deg]
    theta_list = np.linspace(0, 180, 180)
    
    # Plotting
    plt.figure()
    plt.grid()
    plt.title("Work per cycle")
    plt.xlabel("Angle of the x-axis, theta [deg]")
    plt.ylabel("Work per cycle [J]")
    color_list = ['black', 'r', 'b', 'g']
    
    for alpha_idx, alpha_val in enumerate(alpha_list):
        plt.plot(theta_list, sweep_dict[alpha_val]['wo_stall'],
                    label=f"alpha = {alpha_val} deg",
                    color=color_list[alpha_idx])
        plt.plot(theta_list, sweep_dict[alpha_val]['with_stall'],
                    label=f"alpha = {alpha_val} deg (with stall)",
                    color=color_list[alpha_idx],
                    linestyle='--')
    plt.ylim(-30, 15)
    # Legend on the right side of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    return

# Instantiate airfoil
airfoil = Airfoil()

# Calculate work
airfoil.calc_work(rho=1.225,
                    a_0=np.deg2rad(10),
                    theta=np.deg2rad(0),
                    A=0.2,
                    omega=5,
                    V_0=10,
                    cyc=20,
                    num_of_steps=10000,
                    use_stall=False)

# Quick plots
airfoil.plot_work()
airfoil.plot_F_x_l(x_lim_low=1)
airfoil.plot_F_x_d(x_lim_low=1)
airfoil.plot_cd(x_lim_low=1)
airfoil.plot_cl(x_lim_low=1)
airfoil.plot_alpha(x_lim_low=1)
airfoil.plot_V_rel(x_lim_low=1)

# Setting limits manually
airfoil.plot_F_x_l(x_lim_low=20, x_lim_high=25, y_lim_low=14, y_lim_high=17, use_mask=False)

# Calculations take about 1 minute with 1000 timesteps and 20 full cycles
# angles of attack 5, 10, 15, 20
# angles of the x-axis 1, 2, 3, ..., 180
with_sweep = False

if with_sweep:
    # List of angles of attack [deg] to sweep
    alpha_list = np.array([5, 10, 15, 20])

    # List of angles of the x-axis [deg] to sweep
    theta_list = np.linspace(0, 180, 180)

    # Sweep angles of attack and angles of the x-axis
    sweep_dict = sweep_alpha(airfoil,
                            alpha_list,
                            theta_list,
                            rho=1.225,
                            A=0.2,
                            omega=5,
                            V_0=10,
                            cyc=20,
                            num_of_steps=1000
                            )

    # Plot sweep dictionary
    plot_sweep_dict(sweep_dict, alpha_list, theta_list)


