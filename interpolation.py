# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:35:36 2022

@author: Toke Schäffer
"""
import numpy as np

#TEST OF INTERPOLATION ROUTINE. COMPARE TO INTERP1 IN MATLAB

def force_coeffs_10MW(angle_of_attack,thick,aoa, cl_tab, cd_tab, cm_tab): #Creating a function which takes the angle of attack and the section thickness:
    
    
    # angle_of_attack=5
    # thick = tc[0]
    # cl_tab = cl_stat_tab
    # cd_tab = cd_stat_tab
    # cm_tab = cm_stat_tab
    # aoa = aoa_tab

    thick_prof=np.zeros(6)
    # NOTE THAT IN PYTHON THE INTERPOLATION REQUIRES THAT THE VALUES INCREASE IN THE VECTOR!
    thick_prof[0]=24.1;
    thick_prof[1]=30.1;
    thick_prof[2]=36.0;
    thick_prof[3]=48.0;
    thick_prof[4]=60.0;
    thick_prof[5]=100.0;
    
    files=['FFA-W3-241.txt','FFA-W3-301.txt','FFA-W3-360.txt','FFA-W3-480.txt','FFA-W3-600.txt','cylinder.txt']
    
    cl_aoa=np.zeros([1,len(files)])
    cd_aoa=np.zeros([1,len(files)])
    cm_aoa=np.zeros([1,len(files)])
    
    
    #Interpolate to current angle of attack:
    for i in range(np.size(files)):
        cl_aoa[0,i]=np.interp(angle_of_attack,aoa,cl_tab[:,i])
        cd_aoa[0,i]=np.interp(angle_of_attack,aoa,cd_tab[:,i])
        cm_aoa[0,i]=np.interp(angle_of_attack,aoa,cm_tab[:,i])
    
    #Interpolate to current thickness:
    cl=np.interp(thick,thick_prof,cl_aoa[0,:])
    cd=np.interp(thick,thick_prof,cd_aoa[0,:])
    cm=np.interp(thick,thick_prof,cm_aoa[0,:])

    return cl, cd, cm

