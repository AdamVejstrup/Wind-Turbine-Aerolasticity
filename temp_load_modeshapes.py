
# %% Importing the libraries
import numpy as np

# %% Defining the parameters and loading the data

# Modes: first flapwise, first edgewise, second flapwise
omega1f, omega1e, omega2f = 3.93, 6.10, 11.28 # rad/s

# Loading the data
# Order of columns:
# r [m]   u1fy [m]   u1fz [m]   u1ey [m]    u1ez [m]    u2fy [m]     u2fz [m]     m [kg/m]
mode_shapes = np.loadtxt('modeshapes.txt', skiprows=2)

# Defining the mode shapes and mass of the airfoils
u1fy, u1fz, u1ey, u1ez, u2fy, u2fz, r_mass = mode_shapes[:, 1:].T






# %%
