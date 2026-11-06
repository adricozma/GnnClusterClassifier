#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:53:18 2025

@author: adriancozma
"""
import os
import osyris as o
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import glob
import imageio

def Load_Ramses_Osyris(path, output, len_unit = "kpc", vel_unit = "km / s", mass_unit = "M_sun"):
    
    i = output 
    data = o.RamsesDataset(i, path=path).load("part")
    part = data["part"]

    # Convertir les unités
    part["position"] = part["position"].to(len_unit)
    part["velocity"] = part["velocity"].to(vel_unit)
    part["mass"] = part["mass"].to(mass_unit)

    # Extraire les coordonnées
    x = part["position"].x.values
    y = part["position"].y.values
    z = part["position"].z.values

    vx = part["velocity"].x.values
    vy = part["velocity"].y.values
    vz = part["velocity"].z.values
    
    m = part["mass"].values
    
    print(part["velocity"])
    return x, y, z, vx, vy, vz, m

def isolate_cluster(x, y, z, vx, vy, vz, m):
    """
    Isolate the cluster by removing the most massive particle (assumed to be the center)
    and centering the coordinates around the mean position of the remaining particles.
    Returns the modified coordinates and velocities.
    """
    ind_max = np.argmax(m)
    x_los   = np.delete(x, ind_max)
    y_los   = np.delete(y, ind_max)
    z_los   = np.delete(z, ind_max)
    vx_los  = np.delete(vx, ind_max)
    vy_los  = np.delete(vy, ind_max)
    vz_los  = np.delete(vz, ind_max)

    x_los   = x_los - np.mean(x_los)
    y_los   = y_los - np.mean(y_los)
    z_los   = z_los - np.mean(z_los)
    vx_los  = vx_los
    vy_los  = vy_los
    vz_los  = vz_los

    m_c       = np.delete(m, ind_max)
    print("for testing issues")
    print(m[ind_max])
    print("max mass particle removed")
    return x_los, y_los, z_los, vx_los, vy_los, vz_los, m_c


def make_animation(folder_path, output_file, fps=10):

    search_pattern = os.path.join(folder_path, "*.png")
    filenames = glob.glob(search_pattern)
    
    filenames.sort() 

    if not filenames:
        print("Aucune image trouvée !")
        return

    images = []
    print(f"loading from {len(filenames)} images...")
    
    for filename in filenames:
        images.append(imageio.imread(filename))
        
    # 3. Sauvegarde
    print(f"Creation of {output_file} at {fps} FPS...")
    
    if output_file.endswith('.gif'):
        # duration = 1 / fps
        imageio.mimsave(output_file, images, duration=(1/fps), loop=0)
    else:
        imageio.mimsave(output_file, images, fps=fps)
        
    print("Done !")

def make_gif(path, fps=2):
    """
    another try at making a smart function to crate gifs
    """
    import imageio
    import os
    import re
    images = []
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.png')]
    filenames.sort(key=lambda f: int(re.sub(r'\D', '', f)) if re.search(r'\d', f) else 0)
    for filename in filenames:
        images.append(imageio.imread(os.path.join(path, filename)))
    imageio.mimsave(os.path.join(path, 'animation.mov'), images, fps=fps, loop=0)

