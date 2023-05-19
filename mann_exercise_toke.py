import subprocess
from load_turbulence_box import load
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy import signal
from assignment_functions import rpm2rad, x_mask


def write_mann_input(filename, delta_t, n1, n2, n3, ly, lz, hub_height, z_0, v_0):
    """Writes the input file for the Mann turbulence simulation.

    Parameters
    ----------
    filename : str
        Name of the input file to be written.
    delta_t : float
        Time step [s].
    n1 : int
        Number of spatial dimensions.
    n2 : int
        Number of velocity components.
    n3 : int
        Number of grid points in flow direction.
    ly : float
        Physical length [m] in horizontal direction.
    lz : float
        Physical length [m] in vertical direction.
    hub_height : float
        Hub height [m].
    z_0 : float
        Roughness length [m].
    v_0 : float
        Mean wind speed [m/s].
    """

    # Defining the physical length in flow direction
    # based on delta_t, v_0 and n1
    lx = delta_t * v_0 * (n1-1)

    # Defining the delta values (spacing between points)
    delta_y = ly/(n2-1)
    delta_x = lx/(n1-1)
    delta_z = lz/(n3-1)

    # Remove the file extension from the filename
    cropped_filename = filename.split('.INP')[0]

    # Define input parameters to Mann turbulence simulation
    lines = ['3',  # Number of spatial dimensions
             '3',  # Number of velocity components
             str(n1),  # Number of grid points in flow direction
             str(n2),  # Number of grid points in horizontal direction
             str(n3),  # Number of grid points in vertical direction
             str(lx),  # Physical length [m] in flow direction
             str(ly),  # Physical length [m] in horizontal direction
             str(lz),  # Physical length [m] in vertical direction
             'land',  # Defines the spectre (in this case land)
             str(v_0),  # Mean wind speed [m/s]
             str(hub_height),  # Hub height [m]
             str(z_0),  # Roughness length [m]
             '0',  # Spectrum type
             '-5',  # Seed
             # Wind speed fluctuations in flow [m/s]
             f'{cropped_filename}_sim1.bin',
             # Wind speed fluctuations in horizontal [m/s]
             f'{cropped_filename}_sim2.bin',
             f'{cropped_filename}_sim3.bin']  # Wind speed fluctuations [m/s]

    # If the file with filename does not exist, create the file
    # if not os.path.exists(f'turbulence/{filename}.INP'):
    #     open(f'turbulence/{filename}.INP', 'w').close()

    # Write input parameters to file
    with open(f'turbulence/{filename}', 'w+') as f:
        f.write('\n'.join(lines))

    return


def run_mann_simulation(filename):
    """Runs the Mann turbulence simulation.

    Returns
    -------
    turb : nd_array
        Turbulence data with dimensions (n1, n2, n3)
    x_turb : nd_array
        Array with physical length [m] in flow direction for interpolation
    y_turb : nd_array
        Array with physical length [m] in horizontal direction for interpolation
    z_turb : nd_array
        Array with physical length [m] in vertical direction for interpolation
    """

    # Reading in parameters for the turbulent box
    turbulence_parameters = np.genfromtxt(f'turbulence/{filename}')

    # Dimensions must be ints. These are the points in the box.
    n1, n2, n3 = turbulence_parameters[2:5].astype(int)

    # Length of the box in physical dimensions
    lx, ly, lz = turbulence_parameters[5:8]

    # Hub height
    hub_height = turbulence_parameters[10]

    # Run turbulence simulation
    subprocess.run(['turbulence/windsimu.exe',
                    f'turbulence/{filename}'])

    # Remove the file extension from the filename
    cropped_filename = filename.split('.INP')[0]

    # Load turbulence data with the function from Christian
    turb = load(f'{cropped_filename}_sim1.bin', N=(n1, n2, n3))

    # Reshape turbulence data to fit the relevant dimensions
    turb = np.reshape(turb, (n1, n2, n3))

    # Defining the delta values (spacing between points)
    delta_y = ly/(n2-1)
    delta_x = lx/(n1-1)
    delta_z = lz/(n3-1)

    # Changing dimensions from box to physical dimensions
    # x corresponds to z (time)
    # z corresponds to x
    # y corresponds to y
    x_turb = (np.arange(0, n2)*delta_z
              + (hub_height - (n2-1)*delta_z/2))  # Height
    y_turb = np.arange(0, n3)*delta_y - ((n3-1) * delta_y)/2  # Width
    z_turb = np.arange(0, n1)*delta_x  # Depth (Time)

    return turb, x_turb, y_turb, z_turb


def plot_turbulence(delta_t,
                    v_0,
                    turb,
                    x_turb,
                    y_turb,
                    plane_number=0,
                    H=119,
                    R=89.17):

    # Plane number to plot
    plane_time = delta_t * plane_number

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(y_turb, x_turb, turb[plane_number, :, :])
    # Add a colorbar to a plot
    fig.colorbar(cp, label=f'Turbulence [m/s] at t = {plane_time:.1f} s')

    # Plotting the turbine
    lwd = 3
    ax.scatter([0], [H], color='white', s=100)
    ax.plot([0, 0], [H, H + R], color='white', linewidth=lwd)
    ax.plot([0, R * np.sin(2*np.pi/3)], [H, H + R*np.cos(2*np.pi/3)],
            color='white', linewidth=lwd)
    ax.plot([0, R*np.sin(4*np.pi/3)], [H, H + R * np.cos(4*np.pi/3)],
            color='white', linewidth=lwd)
    ax.plot([0, 0], [x_turb[0], H],
            color='white', linewidth=lwd)

    ax.axis('scaled')
    ax.set_title(f'Wind speed = {v_0} m/s + turbulence')
    ax.set_xlabel('y [m]')
    ax.set_ylabel('x [m]')
    plt.show()


def sim_points(omega,
               timerange,
               v_0,
               delta_t,
               turb,
               x_turb,
               y_turb,
               R=89.17,
               H=119,
               point_x=10,
               point_y=2):

    # Array initialization
    # Azimuth angle of the rotating point [rad]
    theta_arr = np.zeros(timerange)
    time_arr = np.zeros(timerange)      # Time array [s]
    x_arr = np.zeros(timerange)         # x position of the rotating point [m]
    y_arr = np.zeros(timerange)         # y position of the rotating point [m]
    # Wind speed at the rotating point [m/s]
    v_arr_rotating = np.zeros(timerange)
    v_arr_fixed = np.zeros(timerange)   # Wind speed at the fixed point [m/s]
    
    for n in range(1, timerange):
        
        # Update the time and angle
        time_arr[n] = n*delta_t

        theta_arr[n] = omega * time_arr[n]

        # Update the position of the rotating point
        x_arr[n] = R * np.sin(theta_arr[n]) + H
        y_arr[n] = R * np.cos(theta_arr[n])

        # Interpolate the turbulence data
        f = interp2d(x_turb, y_turb, turb[n, :, :], kind='linear')

        # Update the wind speed at the rotating point
        v_arr_rotating[n] = f([x_arr[n]], [y_arr[n]]) + v_0

        # Update the wind speed at the fixed point
        v_arr_fixed[n] = f(point_x, point_y) + v_0

    return time_arr, v_arr_rotating, v_arr_fixed


def plot_sim_points(time_arr, v_arr_rotating, v_arr_fixed,
                    x_lim_low=None, x_lim_high=None,
                    y_lim_low=None, y_lim_high=None,
                    use_mask=True):
    """Plotting the wind speed at the rotating and fixed point
    in time.

    Parameters
    ----------
    time_arr : ndarray
        Time [s]
    v_arr_rotating : ndarray
        Wind speed at the rotating point [m/s]
    v_arr_fixed : ndarray
        Wind speed at the fixed point [m/s]
    """

    if use_mask:
        mask = x_mask(time_arr, x_lim_low, x_lim_high)

    plt.figure()
    plt.grid()
    plt.title('Wind speed at rotating and fixed point')
    if use_mask:
        plt.plot(time_arr[mask], v_arr_rotating[mask], label='Rotating point')
        plt.plot(time_arr[mask], v_arr_fixed[mask], label='Fixed point')
    else:
        plt.plot(time_arr, v_arr_rotating, label='Rotating point')
        plt.plot(time_arr, v_arr_fixed, label='Fixed point')
    plt.xlabel('Time [s]')
    plt.ylabel('Wind speed [m/s]')
    plt.xlim(x_lim_low, x_lim_high)
    plt.ylim(y_lim_low, y_lim_high)
    plt.legend()
    plt.show()

    return


def plot_psd_sim_points(time_arr, v_arr_rotating, v_arr_fixed,
                        omega, freq_unit='Hz',
                        x_lim_low=None, x_lim_high=None,
                        y_lim_low=None, y_lim_high=None,
                        use_mask=True, nperseg=1024):

    # Compute the sampling frequency
    fs=1/(time_arr[1]-time_arr[0])

    # Compute and plot the power spectral density.
    # Check out this site for inputs to the Welch
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html 
    
    
    f_rotating, Pxx_den_rotating = signal.welch(v_arr_rotating, fs, nperseg=nperseg)
    f_fixed, Pxx_den_fixed = signal.welch(v_arr_fixed, fs, nperseg=nperseg)
    
    if use_mask:
        mask = x_mask(f_rotating, x_lim_low, x_lim_high)
        
    plt.figure()
    plt.title('Power spectral density of wind speed')
    
    if freq_unit == 'Hz':
        if use_mask:
            plt.plot(f_rotating[mask], Pxx_den_rotating[mask], label='Rotating point')
            plt.plot(f_fixed[mask], Pxx_den_fixed[mask], label='Fixed point')
        else:
            plt.plot(f_rotating, Pxx_den_rotating, label='Rotating point')
            plt.plot(f_fixed, Pxx_den_fixed, label='Fixed point')
        plt.xlabel('Frequency [Hz]')
        
    elif freq_unit == 'per_revolution':
        if use_mask:
            plt.plot(2*np.pi * f_rotating[mask] / omega, Pxx_den_rotating[mask], label='Rotating point')
            plt.plot(2*np.pi * f_fixed[mask] / omega, Pxx_den_fixed[mask], label='Fixed point')
        else:
            plt.plot(2*np.pi * f_rotating / omega, Pxx_den_rotating, label='Rotating point')
            plt.plot(2*np.pi * f_fixed / omega, Pxx_den_fixed, label='Fixed point')
        plt.xlabel('$2 \pi f / \omega$ [-]')
        
    else:
        raise ValueError(f'freq_unit must be either "Hz" or "per_revolution", not {freq_unit}')
        
    plt.yscale('log')
    plt.grid()
    plt.legend()
    
    plt.ylabel('PSD [(m/s)$^2$/Hz]')
    plt.xlim(x_lim_low, x_lim_high)
    plt.ylim(y_lim_low, y_lim_high)
    plt.show()
    
    return

def calc_turb_intensity(v_0, turb):
    """Calculates the turbulence intensity of the turbulence data.
    The turbulence data is 3D with dimensions (n1, n2, n3).
    The standard deviation is calculated in time to achieve
    a 2D array with dimensions (n2, n3), which is then used
    to calculate the turbulence intensity. The printed output
    is the minimum, maximum and mean turbulence intensity in %
    of that plane.

    Parameters
    ----------
    v_0 : float
        Mean wind speed [m/s]
    turb : ndarray
        3D array with turbulence data with dimensions (n1, n2, n3)
    """
    
    
    # Standard deviation in time
    std = np.std(turb, axis=0)

    # Turbulence intensity
    ti = std/v_0 * 100
    
    ti_max = np.max(ti)
    ti_min = np.min(ti)
    ti_mean = np.mean(ti)
    
    print('Turbulence intensity ranges')
    print(f'from {ti_min:.2f}% to {ti_max:.2f}%')
    print(f'with a mean of {ti_mean:.2f}%')
    
    return

    # The code below is only executed if this file is run as a script
    # and not if it is imported as a module
    # In this way, the function above can be used in the
    # assignment script without running the code below.
if __name__ == '__main__':
    # Defining the input parameters
    delta_t = 0.05
    n1 = 4096
    n2 = 32
    n3 = 32
    ly = 180
    lz = 180
    hub_height = 119
    z_0 = 0.05
    v_0 = 12

    # Defining the filename
    filename = 'mann_exercise_toke.INP'

    # Write the input file
    write_mann_input(filename, delta_t, n1, n2, n3,
                     ly, lz, hub_height, z_0, v_0)

    # Load the turbulence data
    turb, x_turb, y_turb, z_turb = run_mann_simulation(filename)

    plot_turbulence(delta_t, v_0, turb, x_turb, y_turb, plane_number=0)

    # Simulate the point rotating with a given angular velocity
    omega = rpm2rad(8)  # rad/s

    time_arr, v_arr_rotating, v_arr_fixed = sim_points(omega,
                                                       n1,  # Timerange equal to the number of points in the box
                                                       v_0,
                                                       delta_t,
                                                       turb,
                                                       x_turb,
                                                       y_turb,
                                                       R=89.17,
                                                       H=119,
                                                       point_x=10,
                                                       point_y=2)

    plot_sim_points(time_arr, v_arr_rotating, v_arr_fixed, x_lim_low=1)

    plot_psd_sim_points(time_arr, v_arr_rotating, v_arr_fixed,
                        omega, freq_unit='per_revolution', x_lim_low=0, x_lim_high=8, y_lim_low=1e-4)
    
    calc_turb_intensity(v_0, turb)