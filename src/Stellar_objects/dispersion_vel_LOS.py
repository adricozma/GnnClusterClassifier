#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 15:12:59 2025

@author: adriancozma
"""

import numpy as np

def partPerBin(x, y, z, Rmin, Rmax, nbins):
    """
    Parameters
    ----------
    i : double array
        the position along the i=X,Y,Z axis
    Rmin : double
        the minimum radious (distance from the center of the closest particle)
    Rmax : double
        the maximum radious (distance from the center of the furthest particle)
    nbins : int
        number of bins from which to draw particles in order to compute de velocity dispersion
    Returns
    -------
    r_centers : double array
        returns the array of the positions used to compute n_part positions centered int the middle of the bins
    n_part : TYPE
        Number of particles in each bin 
    
    This will be useful to decide the minimum number of particles per bins and/or decide the amount of bins one can
    use to compute the dispersion later on
    """
    r = (x**2 + y**2 + z**2)**0.5
    r_bins = np.linspace(Rmin, Rmax, nbins)
    r_centers = 0.5*(r_bins[:-1] + r_bins[1:])
    
    n_part= []
    
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1]) 
        
        if len(mask) > 0:
            N=len(r[mask])
            n_part.append(N)
            
        else:
            #to avoid errors
            n_part.append(np.nan)
    
    r_centers = np.array(r_centers)
    n_part = np.array(n_part)
    
    return r_centers, n_part


def dispersionLOS(x, y, z, vx, vy, vz, Rmin, Rmax, nbins, Nbmin):
    """

    Parameters
    ----------
    i  : double array
        the position along the i=X,Y,Z axis
    vi : double array
        velocity along the i=X,Y,Z axis
    Rmin : double
        the minimum radious (distance from the center of the closest particle)
    Rmax : double
        the maximum radious (distance from the center of the furthest particle)
    nbins : int
        number of bins from which to draw particles in order to compute de velocity dispersion
    Returns
    -------
        double array
        r_centers returns the array of the positions used to compute sigma_LOS positions centered int the middle of the bins
        sigma_LOS(R) i.e it return de velocity dispersion in the lign of sight

    """
    # In our case we assume that vz is in the line of sight because we can decide to do so
    r = (x**2 + y**2 + z**2)**0.5
    #r_bins = np.linspace(Rmin, Rmax, nbins)
    r_bins = np.logspace(np.log10(Rmin), np.log10(Rmax), nbins)
    r_centers = 0.5*(r_bins[:-1] + r_bins[1:])
    
    sigma_LOS = []
    r_c       = []
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        vz_bin = vz[mask]
        #all the velocities in the i-th bin 
        if len(vz_bin) > Nbmin:
            N=len(vz_bin)
            s= ( (np.sum(vz_bin**2) - 1/N * (np.sum(vz_bin))**2)/N )**0.5
            sigma_LOS.append(s)
            r_c.append(r_bins[i])
        # else:
        #     #to avoid errors
        #     sigma_LOS.append(np.nan)
    
    r_centers = np.array(r_centers)
    sigma_LOS = np.array(sigma_LOS)
    r_c       = np.array(r_c)
    return r_c, sigma_LOS

#%% for testing purpuses
# filename = '50kplummerCluster_pNbody_v1_NO-RAMSES.txt'
# f  = np.loadtxt(filename, delimiter=" ")
# x  = f[:,0]
# y  = f[:,1]
# z  = f[:,2]
# vx = f[:,3]
# vy = f[:,4]
# vz = f[:,5]
# m  = f[:,6]
# #%%
# import matplotlib.pyplot as plt
# from adri_plotstyle import init_plot_style
# init_plot_style()
# fs = 14

# nbin = [30]
# Nbmin=[200] 

# for N in Nbmin:
#     title = r"$\mathrm{Velocity\ dispersion\ of\ 5\cdot 10^4\ particles\ as\ a\ function\ of}\ r\ \mathrm{and\ the\ nuber\ of\ bins\ with\ " + str(N)+ "\ particles\ minimum\ per\ bin}$"
#     #plt.figure()
#     #plt.title(title)
#     for n in nbin:
#         r, sigma = dispersionLOS(x, y, z, vx, vy, vz, 1e-2, 30, n, N)
        
#         #lab = r"$\mathrm{" + str(N) + "\ part\ minimum}$"
#         lab = r"$\mathrm{" + str(n) + "\ bins}$"
#         plt.plot(r, sigma, label=lab, zorder=2)
#    # plt.ylim((0,1.25))
#     #plt.xscale("log")
#     plt.legend()
#     plt.grid(True, zorder = 1)
#     plt.xlabel(r"$r\ \mathrm{[pc]}$", fontsize = fs)
#     plt.ylabel(r"$\sigma_{LOS}(r)$", fontsize = fs)
#     plt.show()

# #%%
# nbin = [10, 20, 30, 40, 50, 60]

# plt.figure()
# plt.title(r"$5\cdot10^4\ \mathrm{particles\ extracted\ from\ a\ Plummer\ distribution\ as\ a\ function\ of}\ r\ \mathrm{and\ the\ nuber\ of\ bins}$")
# for n in nbin:
#     r_part, n_part = partPerBin(x, y, z, 1e-2, 30, n)
#     lab = r"$\mathrm{" + str(n) + "\ bins}$"
#     plt.scatter(r_part, n_part, label=lab, zorder=2)
#     plt.legend()
#     plt.grid(True, zorder = 1)
#     plt.xlabel(r"$r\ \mathrm{[pc]}$", fontsize = fs)
#     plt.ylabel(r"$n_{part}(r)$", fontsize = fs)
# plt.show()
#%%





