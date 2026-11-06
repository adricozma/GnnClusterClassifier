#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:35:08 2025

@author: adriancozma
"""

"""
    This fonction changes the initial location and velocities with a given amount for 
an array of positions. It is important to mention that the initial Location 
is considered to be centered in (0, 0, 0) thus here we just translate the referential 
"""
import numpy as np

def translate(x_init, y_init, z_init, dest_x, dest_y, dest_z):
    N    = len(x_init)
    ones = np.ones(N)
    return x_init + ones*dest_x,  y_init + ones*dest_y, z_init + ones*dest_z,
 
"""
    This function simply gives a cick to the particles with a certain amopunt
"""
def kick(vx_init, vy_init, vz_init, kickx, kicky, kickz):
    N = len(vx_init)
    ones = np.ones(N)
    return vx_init + ones*kickx, vy_init + ones*kicky, vz_init + ones*kickz

filename = "../AdriansPlummerCluster/adrians_plummer.txt"
data_cluster = np.loadtxt(filename, delimiter=' ')

x_init  = data_cluster[:,0]
y_init  = data_cluster[:,1]
z_init  = data_cluster[:,2]
vx_init = data_cluster[:,3]
vy_init = data_cluster[:,4]
vz_init = data_cluster[:,5]
m       = data_cluster[:,6]

print(len(m), " particles loaded from the file: ", filename)
print(len(x_init))  
print(len(vx_init))
print(len(y_init))
print(len(vy_init))
print(len(z_init))
print(len(vz_init))
#translate the cluster to a given location
dest_x = 17.0    #kpc
dest_y = 0.0     #kpc
dest_z = 0.0     #kpc
x_new, y_new, z_new = translate(x_init, y_init, z_init, dest_x, dest_y, dest_z)

#give a kick to the cluster
kickx = 0.0    #km/s
kicky = 150.0  #km/s
kickz = 0.0    #km/s
vx_new, vy_new, vz_new = kick(vx_init, vy_init, vz_init, kickx, kicky, kickz)

# # adding the supermassive particle at the center of the galaxy in first row
x_new = np.insert(x_new, 0, 0.0)
y_new = np.insert(y_new, 0, 0.0)
z_new = np.insert(z_new, 0, 0.0)
vx_new = np.insert(vx_new, 0, 0.0)
vy_new = np.insert(vy_new, 0, 0.0)
vz_new = np.insert(vz_new, 0, 0.0)

print(len(x_new), " particles after adding the SMBH")
print(len(vx_new))
print(len(y_new))
print(len(vy_new))
print(len(z_new))
print(len(vz_new))
m = np.insert(m, 0, 8.89348741e10)   #mass

units_time    = 0.470430312423675E+15 
units_length  = 0.308567758128200E+22 
units_velocity= (units_length*1e-5)/units_time 

#saving the new data
output = np.column_stack([x_new, y_new, z_new, vx_new/units_velocity, vy_new/units_velocity, vz_new/units_velocity, m/1e9])
np.savetxt(filename,output, fmt='%.6e', delimiter=',')
print("saving file : RAMSES_"+filename)
# %%
