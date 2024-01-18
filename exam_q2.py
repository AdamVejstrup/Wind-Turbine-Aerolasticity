# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:57:45 2023

@author: Toke Schäffer
"""

import numpy as np
from assignment_functions import solve_eig_prob

k1, k2, k3 = 50, 10, 15
m1, m2, m3 = 5, 2, 1

# M = np.array([[m1+m2+m3, 0, 0],
#               [0, m2, 0],
#               [0, 0, m3]])

M = np.array([[m1+m2+m3, m2+m3, m3],
              [m2+m3, m2+m3, m3],
              [m3, m3, m3]])

K = np.array([[k1, 0, 0],
              [0, k2, 0],
              [0, 0, k3]])


eig_omega, mode_shapes_eig = solve_eig_prob(K, M)
    
eig_f = eig_omega / (2*np.pi)