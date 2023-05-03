
import numpy as np
import matplotlib.pyplot as plt


def x_mask(x, start=None, end=None):
    
    """Function that creates a boolean mask/filter for plots etc.
    The mask/filter is based on x-values and some start value and
    end value of x, which is specified by the user. This functions
    was created to remove the transient part of a time series,
    but can also be used for other x-values e.g. frequencies etc.
    Instead of using this funtions, one can also just set the
    xlim of a plt.plot, but then the ylim will be set based on
    y-values corresponding to all x-values and not just the ones,
    that are actually plotted. This makes it difficult to
    find good values for ylim. When using this filter, matplotlib
    will automatically set the ylim only based on the values
    that are actually plottet.
    
    Parameters
    ----------
    start : integer or real, optional
    All values smaller than start is false in the mask
    
    end : integer or real, optional
    All values larger than end is false in the mask
    
    Returns
    -------
    mask : ndarray
    Array filled with true, expect for values smaller than start
    and values larger than end
    """
    
    # If both a start value and an end value is given
    if start is not None and end is not None:
        mask = (x >= start) & (x <= end)
    
    # If only a start value is given
    elif start is None and end is not None:
        mask = x <= end  
    
    # If only an end value is given
    elif start is not None and end is None:
        mask = x >= start
    
    # If neither start value nor end value is given,
    # return an array filled with True
    else:
        mask = np.full(len(x), True)
    
    return mask

# Plotting generator characteristic
def make_gen_char(omega_rated, K):
    
    omega_start = 6*2*np.pi/60 #6rpm til rad/s
    omega_slut = 11*2*np.pi/60 #6rpm til rad/s
    
    omega_range = np.linspace(omega_start, omega_slut, 100)
    
    low_mask = omega_range < omega_rated
    high_mask = omega_range >= omega_rated
    
    M_g = np.zeros(len(omega_range))
    
    M_g[low_mask] = (K * omega_range**2) [low_mask]
    
    M_g[high_mask] = K * omega_rated**2
    
    plt.figure()
    plt.grid()
    plt.title('Generator characteristic')
    plt.plot(omega_range,M_g/10**6, label = 'Generator torque')
    plt.ylabel('$M_{g} \; [MN \cdot m]$')
    plt.xlabel('$\omega$ [rad/s]')
    plt.axvline(omega_rated, label = 'Rated $\omega$ = {:.2f} rad/s'.format(omega_rated), color = 'grey', linestyle = '--')
    plt.xlim(omega_range[0], omega_range[-1])
    # plt.ylim(bottom=0)
    plt.legend()
    plt.show()
    
    return


def make_position_sys1(blade_element, time_arr, y1_arr, x1_arr, mask, r, B, H):
    
    
    # Plot 1: Blade position of the last blade element (system 1)
    
    plt.figure()
    plt.grid()
    plt.title('Blade position of the last blade element (system 1)')
    # For de tre vinger
    for i in range(B):
        # plt.plot(x1_arr[blade_element,i,1:], y1_arr[blade_element,i,1:],linewidth=7-3*i,label='Blade {}'.format(i+1))
        plt.plot(y1_arr[blade_element,i,1:][mask[1:]], x1_arr[blade_element,i,1:][mask[1:]],linewidth=7-3*i,label='Blade {}'.format(i+1))
    # Tilføjer r
    plt.plot([0,r[blade_element]],[H,H],label='r = {:.2f} m'.format(r[blade_element]))
    # Tilføjer H
    plt.plot([0,0],[0,H],label='H = {} m'.format(H),color='black')
    
    # Ticks
    plt.xticks([-r[blade_element],0,r[blade_element]])
    plt.yticks([0,H-r[blade_element],H,H + r[blade_element]])
    
    
    # Symmetriske axer for cirkel i stedet for oval form
    plt.axis('scaled')
    plt.ylabel('x [m]')
    plt.xlabel('y [m]')
    plt.ylim(bottom=0)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.show()
    
    
    # Plot 2: Blade position of the last blade element (system 1)
    
    plt.figure()
    plt.grid()
    plt.title('x-position of the last blade element (system 1)')
    # For de tre vinger
    for i in range(B):
        plt.plot(time_arr[mask], x1_arr[blade_element,i,:][mask],label='Blade {}'.format(i+1))
    
    plt.ylim(bottom=0)
    plt.xlim(time_arr[mask][0], time_arr[mask][-1])
    plt.xlabel('Time [s]')
    plt.ylabel('x [m]')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.show()
    
    return

def rpm2rad(x):
    return x * (2*np.pi/60)

def rad2rpm(x):
    return x * (60/(2*np.pi))

def calc_lam(omega, V_0, R):
    # Calculate tip speed ratio
    return (omega * R) / V_0

def calc_omega(lam, V_0, R):
    # Calculate omega from tip sp
    return lam * V_0 / R
    




