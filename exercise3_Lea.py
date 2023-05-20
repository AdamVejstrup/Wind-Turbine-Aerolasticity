# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:30:30 2023

@author: adamv
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation import force_coeffs_10MW


class Airfoil:
    def __init__(self, c=1, file_name="FFA-W3-241.txt"):
        self.c = c
        self.file_name = file_name
        self.data = np.loadtxt(file_name)
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
        
        T = 2*np.pi / omega                 # Period [s]
        dt = cyc * T / num_of_steps         # Calculate timestep [s]
        
        time_arr = np.zeros(num_of_steps)   # Initialize time array
        x = np.zeros(time_arr.shape)        # Initialize x array
        F_x = np.zeros(time_arr.shape)      # Initialize force array
        F_drag = np.zeros(time_arr.shape)      # Initialize force array
        F_lift = np.zeros(time_arr.shape)      # Initialize force array
        accum_work = np.zeros(time_arr.shape) # Initialize accumulated work array
        fs_arr = np.zeros(time_arr.shape)
        cl_arr = np.zeros(time_arr.shape)
        cd_arr = np.zeros(time_arr.shape)
        alpha_arr = np.zeros(time_arr.shape)
        V_rel_arr = np.zeros(time_arr.shape)
        power_arr = np.zeros(time_arr.shape)
        
        for n in range(1, num_of_steps):
            
            time_arr[n] =  n * dt  # Time
            
            x[n] = A * np.sin(omega * time_arr[n]) # Position along x-axis
            
            dxdt = A * omega * np.cos(omega * time_arr[n]) # Velocity along x-axis
            
            # Relative velocity
            V_y = V_0 * np.cos(a_0) + dxdt * np.cos(theta)
            
            V_z = V_0 * np.sin(a_0) + dxdt * np.sin(theta)
            
            V_rel = np.sqrt((V_y)**2 + (V_z)**2)
            V_rel_arr[n] = V_rel
            
            # Angle of attack [rad]
            alpha = np.arctan(V_z / V_y)
            
            alpha_arr[n] = alpha
            
            cl = np.interp(np.rad2deg(alpha), self.alpha_tab, self.cl_tab)
            cd = np.interp(np.rad2deg(alpha), self.alpha_tab, self.cd_tab)
            cd_arr[n] = cd
            
            f_stat = np.interp(np.rad2deg(alpha), self.alpha_tab, self.f_stat_tab)
            cl_inv = np.interp(np.rad2deg(alpha), self.alpha_tab, self.cl_inv_tab)
            cl_fs = np.interp(np.rad2deg(alpha), self.alpha_tab, self.cl_fs_tab)
            
            if use_stall:
                tau_stall = 4 * self.c / V_rel
                
                fs_arr[n] = f_stat + (fs_arr[n-1] - f_stat) * np.exp(-dt/tau_stall)
                
                cl = f_stat * cl_inv + (1-fs_arr[n]) * cl_fs

            cl_arr[n] = cl
            
            # Aerodynamic force 
            
            F_x[n]=0.5 * rho * V_rel**2 * self.c*(cl * np.sin(alpha-theta) - cd * np.cos(alpha-theta))
            
            F_lift[n] = 0.5 * rho * V_rel**2 * self.c* cl * np.sin(alpha-theta)
            
            F_drag[n] = -0.5 * rho * V_rel**2 * self.c* cd * np.cos(alpha-theta)
            
            # Accumulated work
            
            accum_work[n] = accum_work[n-1] + F_x[n] * dxdt * dt
            
        
        # Work done by the aerodynamic force 
        work = A * omega*np.trapz(F_x * np.cos(omega * time_arr), time_arr)
        
        self.sim_time = time_arr
        self.sim_accum_work = accum_work
        self.sim_x = x
        self.sim_alpha = alpha_arr
        self.sim_cl = cl_arr
        self.sim_cd = cd_arr
        self.sim_V_rel = V_rel_arr
        self.sim_F_x = F_x
        self.sim_F_drag = F_drag
        self.sim_F_lift = F_lift
        
        self.sim_power = work / (time_arr[-1]-time_arr[0])
        
        return
    
    def plot_alpha(self):
        """
        plt.figure()
        plt.grid()
        plt.title("Angle of attack")
        plt.xlabel("Time [s]")
        plt.ylabel("Angle of attack [deg]")
        plt.plot(self.sim_time, np.rad2deg(self.sim_alpha))
        plt.show()
        """
        
        fig, ax1 = plt.subplots(1,1)
    
        color = 'tab:orange'
        ax1.grid()
        ax1.set_title('AoA and Vrel')
        ax1.set_ylabel('AoA [deg]', color=color)
        ax1.set_xlabel('Time [s]')
        ax1.plot(self.sim_time, np.rad2deg(self.sim_alpha),color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlim(0,12)
        #ax1.set_ylim(-1,1)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
        color = 'tab:blue'
        ax2.set_ylabel('Vrel [m/s]', color=color)  
        ax2.plot(self.sim_time, self.sim_V_rel)
        ax2.set_ylim(9,11)
    
        #Man skal af en eller anden grund have denne her linje med for at de to y-akser bliver aligned
        ax2.set_yticks(np.linspace(ax2.get_yticks()[0],ax2.get_yticks()[-1],len(ax1.get_yticks())))
        ax2.tick_params(axis='y', labelcolor=color)
        
        
        
        return


    def plot_cdcl(self):
        """
        plt.figure()
        plt.grid()
        plt.title("Drag coefficient")
        plt.xlabel("Time [s]")
        plt.ylabel("Drag coefficient")
        plt.plot(self.sim_time, self.sim_cd)
        plt.show()
        
        plt.figure()
        plt.grid()
        plt.title("Lift coefficient")
        plt.xlabel("Time [s]")
        plt.ylabel("Lift coefficient")
        plt.plot(self.sim_time, self.sim_cl)
        plt.show()
        """
        fig, ax1 = plt.subplots(1,1)
    
        color = 'tab:orange'
        ax1.grid()
        ax1.set_title('Cl and Cd')
        ax1.set_ylabel('Cl [-]', color=color)
        ax1.set_xlabel('Time [s]')
        ax1.plot(self.sim_time, self.sim_cl,color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlim(2,12)
        ax1.set_ylim(min(self.sim_cl),max(self.sim_cl))
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
        color = 'tab:blue'
        ax2.set_ylabel('Cd [N/m]', color=color)  
        ax2.plot(self.sim_time, self.sim_cd)
        ax2.set_ylim(min(self.sim_cd)-0.01,max(self.sim_cd)+0.01)
    
        #Man skal af en eller anden grund have denne her linje med for at de to y-akser bliver aligned
        ax2.set_yticks(np.linspace(ax2.get_yticks()[0],ax2.get_yticks()[-1],len(ax1.get_yticks())))
        ax2.tick_params(axis='y', labelcolor=color)

        return
    
    def plot_x(self):
        plt.figure()
        plt.grid()
        plt.title("Position along x-axis")
        plt.xlabel("Time [s]")
        plt.ylabel("Position along x-axis [m]")
        plt.plot(self.sim_time, self.sim_x)
        plt.show()
        return
    
    def plot_V_rel(self):
        plt.figure()
        plt.grid()
        plt.title("Relative velocity")
        plt.xlabel("Time [s]")
        plt.ylabel("Relative velocity [m/s]")
        plt.plot(self.sim_time[1:], self.sim_V_rel[1:])
        plt.show()
        return
    
    def plot_F_x(self):
        """
        plt.figure()
        plt.grid()
        plt.title("Aerodynamic force")
        plt.xlabel("Time [s]")
        plt.ylabel("Aerodynamic force [N/m]")
        plt.plot(self.sim_time, self.sim_F_x)
        plt.show()
        
        plt.figure()
        plt.grid()
        plt.title("Drag force")
        plt.xlabel("Time [s]")
        plt.ylabel("Fx,d [N/m]")
        plt.plot(self.sim_time, self.sim_F_drag)
        plt.show()
        
        plt.figure()
        plt.grid()
        plt.title("Lift force")
        plt.xlabel("Time [s]")
        plt.ylabel("Fx,l [N/m]")
        plt.plot(self.sim_time, self.sim_F_lift)
        plt.show()
        """
        
        fig, ax1 = plt.subplots(1,1)
    
        color = 'tab:orange'
        ax1.grid()
        ax1.set_title('Drag and lift forces')
        ax1.set_ylabel('Fx,l [N/m]', color=color)
        ax1.set_xlabel('Time [s]')
        ax1.plot(self.sim_time, self.sim_F_lift,color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlim(0,12)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
        color = 'tab:blue'
        ax2.set_ylabel('Fx,d [N/m]', color=color)  
        ax2.plot(self.sim_time, self.sim_F_drag)
    
        #Man skal af en eller anden grund have denne her linje med for at de to y-akser bliver aligned
        ax2.set_yticks(np.linspace(ax2.get_yticks()[0],ax2.get_yticks()[-1],len(ax1.get_yticks())))
        ax2.tick_params(axis='y', labelcolor=color)
        return
    
    def plot_work(self):
        plt.figure()
        plt.grid()
        plt.title("Accumulated work")
        plt.xlabel("Time [s]")
        plt.ylabel("Accumulated work [J]")
        plt.plot(self.sim_time, self.sim_accum_work, label="Accumulated work")
        plt.plot(self.sim_time, self.sim_power * self.sim_time,
                 label=f"Power = {self.sim_power:.2f} W")
        plt.xlim(0,12)
        plt.legend()
        plt.show()
        return


##DET ER HER NEDE DU SKAL ÆNDRE VÆRDIERNE!!!

if __name__ == "__main__":
    airfoil = Airfoil()
    airfoil.calc_work(a_0=np.deg2rad(0),
                      theta=np.deg2rad(0),
                      A=0.2,
                      omega=5,
                      V_0=10,
                      num_of_steps=1000,
                      use_stall=True)
    
    airfoil.plot_work()
    airfoil.plot_alpha()
    airfoil.plot_cdcl()
    #airfoil.plot_x()
    #airfoil.plot_V_rel()
    airfoil.plot_F_x()
    

