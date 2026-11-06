from scipy.spatial import cKDTree
import itertools
from .dispersion_vel_LOS import dispersionLOS
from .visu import *
import numpy as np
import matplotlib.pyplot as plt
from adri_plotstyle import init_plot_style
init_plot_style(use_latex= True, font="helvet", fontsize = 14)

#Making a galaxy class with the functions to load the data and orientate the cluster
def filter_by_mass(x, y, z, vx, vy, vz, m, mass_value, epsilon=1e-1):
    """
    Version optimisée (vectorisée) : plus de boucle for.
    """
    # Crée un masque booléen instantanément sur tout le tableau
    mask = np.abs(m - mass_value) < epsilon
    
    # Applique le masque
    return x[mask], y[mask], z[mask], vx[mask], vy[mask], vz[mask], m[mask]

class Galaxy:
    def __init__(self, path_to_data, output, len_unit="kpc", vel_unit="km / s", mass_unit="M_sun"):
        # On charge
        raw_data = Load_Ramses_Osyris(path_to_data, output, len_unit, vel_unit, mass_unit)
        
        # On déballe et on sécurise (.ravel() ici est vital)
        self.x = np.asarray(raw_data[0]).ravel()
        self.y = np.asarray(raw_data[1]).ravel()
        self.z = np.asarray(raw_data[2]).ravel()
        self.vx = np.asarray(raw_data[3]).ravel()
        self.vy = np.asarray(raw_data[4]).ravel()
        self.vz = np.asarray(raw_data[5]).ravel()
        self.m  = np.asarray(raw_data[6]).ravel()

        self.N = len(self.x)
        self.output = output
        self.m_part_disk  =  0 
        self.m_part_halo  =  0 
        self.m_part_bulge =  0 

    def center_galaxy(self, galaxy_center=[0,0,0]):
        self.x = self.x - galaxy_center[0]
        self.y = self.y - galaxy_center[1]
        self.z = self.z - galaxy_center[2]
        
    def get_different_mass_values(self):
        print("Different mass values in the galaxy:")
        diff = np.unique(self.m)
        print(diff)
        return diff
    
    def set_1part_disk_mass(self, m_part_disk):
        self.m_part_disk = m_part_disk
    def set_1part_halo_mass(self, m_part_halo):
        self.m_part_halo = m_part_halo
    def set_1part_bulge_mass(self, m_part_bulge):
        self.m_part_bulge = m_part_bulge
    
    # --- Utilisation de la fonction filter_by_mass externe ---
    def get_disk(self):
        return filter_by_mass(self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m, self.m_part_disk)
    def get_halo(self):
        return filter_by_mass(self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m, self.m_part_halo)
    def get_bulge(self):
        return filter_by_mass(self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m, self.m_part_bulge)
    
    # --- CORRECTION ICI : Ajout de self ---
    def density_map(self, x, y, z, k=6):
        # On garde les ravel par sécurité absolue
        x = np.asarray(x).ravel() 
        y = np.asarray(y).ravel()
        z = np.asarray(z).ravel()

        positions = np.vstack((x, y, z)).T
        tree = cKDTree(positions)
        
        distances, _ = tree.query(positions, k=k+1)
        r_k = distances[:, -1]
        
        r_k = np.where(r_k <= 0, 1e-10, r_k) 

        volumes = (4/3) * np.pi * r_k**3
        return k / volumes

    def plot_VolumicDensity(self, halo=True, disk=True, bulge=True, projection='xy', limit=True, figsize=(8,8), colormap='viridis', savefig=False, location=''):
        
        xs, ys, zs = [], [], []
        
        # On vérifie si la liste renvoyée n'est pas vide (len(d[0]) > 0)
        label = "$"
        if halo:
            d = self.get_halo()
            if len(d[0]) > 0: xs.append(d[0]); ys.append(d[1]); zs.append(d[2])
            label += "\mathrm{Halo,}\  "
        if disk:
            d = self.get_disk()
            if len(d[0]) > 0: xs.append(d[0]); ys.append(d[1]); zs.append(d[2])
            label += "\mathrm{Disk,}\  "
        if bulge:
            d = self.get_bulge()
            if len(d[0]) > 0: xs.append(d[0]); ys.append(d[1]); zs.append(d[2])
            label += "\mathrm{Bulge}\  "
        if not xs: 
            print("Aucune composante sélectionnée ou vide."); return
        label += "$"
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)

        density = self.density_map(x, y, z)

        proj_map = {
            'xy': (x, y, 'X', 'Y'),
            'xz': (x, z, 'X', 'Z'),
            'yz': (y, z, 'Y', 'Z')
        }
        
        if projection not in proj_map: raise ValueError("Projection invalide")
        u, v, xlabel, ylabel = proj_map[projection]

        plt.figure(figsize=figsize, facecolor='white')
        plt.grid(True, zorder=1, linestyle ='--', alpha=0.5)
        plt.title(f"Volumic Density (Output {self.output})")
        plt.xlabel(f"${xlabel} \mathrm{{\ [kpc]}}$")
        plt.ylabel(f"${ylabel} \mathrm{{\ [kpc]}}$")
        
        import matplotlib.colors as colors
        sc = plt.scatter(u, v, c=density, s=0.5, cmap=colormap, alpha=0.7, norm=colors.LogNorm(), zorder = 2, label = label)
        
        plt.colorbar(sc, label=r"$\rho$ \ [$\mathrm{part/kpc}^3$]")
        plt.axis("equal")
        plt.legend()
        if limit:
            plt.xlim(u.min(), u.max())
            plt.ylim(v.min(), v.max())

        if savefig:
            plt.savefig(f"{location}", dpi=300, bbox_inches='tight')
            
        plt.show()

    def plot_SurfaceDensity(self, halo=True, disk=True, bulge=True, projection='xy', 
                            bins_per_kpc=10, zoom_factor=1, 
                            figsize=(8,8), colormap='plasma', savefig=False, location=''):
        """
        Plots the Surface Density (Sigma) map using 2D binning.
        La résolution est fixée physiquement par 'bins_per_kpc'.
        """
        
        # --- 1. Collecte des données ---
        xs, ys, zs, ms = [], [], [], []
        
        # On récupère coords ET masses
        if halo:
            d = self.get_halo() 
            if len(d[0]) > 0: 
                xs.append(d[0]); ys.append(d[1]); zs.append(d[2]); ms.append(d[6])
        if disk:
            d = self.get_disk()
            if len(d[0]) > 0: 
                xs.append(d[0]); ys.append(d[1]); zs.append(d[2]); ms.append(d[6])
        if bulge:
            d = self.get_bulge()
            if len(d[0]) > 0: 
                xs.append(d[0]); ys.append(d[1]); zs.append(d[2]); ms.append(d[6])

        if not xs: 
            print("Aucune composante sélectionnée."); return

        # Concaténation (suppose que l'__init__ a déjà fait les ravel())
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)
        m = np.concatenate(ms) 

        # --- 2. Gestion de la projection ---
        proj_map = {
            'xy': (x, y, '$X$', '$Y$'),
            'xz': (x, z, '$X$', '$Z$'),
            'yz': (y, z, '$Y$', '$Z$')
        }
        
        if projection not in proj_map: raise ValueError("Projection invalide (xy, xz, yz)")
        u, v, xlabel, ylabel = proj_map[projection]

        # --- 3. Logique BINS_PER_KPC et ZOOM ---
        
        # a. Définir l'étendue physique max visible
        # On cherche le max absolu pour centrer la vue en (0,0)
        max_val = max(np.max(np.abs(u)), np.max(np.abs(v)))
        limit = max_val / zoom_factor
        
        # The visible box goes from -limit to +limit
        bounds = [[-limit, limit], [-limit, limit]]
        
        # b. Calculer le nombre de bins pour respecter bins_per_kpc
        physical_width = 2 * limit # Largeur totale en kpc
        n_bins = int(physical_width * bins_per_kpc)
        
        # Security to avoid 0 bins if we zoo too much or tooo little data
        if n_bins < 50: n_bins = 50 
        
        print(f"Info: Zoom x{zoom_factor} | View: {physical_width:.1f} kpc | Grid: {n_bins}x{n_bins} pixels")

        # --- 4. Calcul de l'Histogramme 2D ---
        # weights=m for the desinty in mass units
        mass_grid, x_edges, y_edges = np.histogram2d(
            u, v, 
            bins=n_bins, 
            range=bounds, 
            weights=m 
        )

        # Calcul de la surface d'un pixel
        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]
        pixel_area = dx * dy

        # Densité = Masse / Surface
        sigma_map = (mass_grid / pixel_area).T 
        
        # Nettoyage pour l'échelle log
        sigma_map = np.where(sigma_map <= 0, 1e-10, sigma_map)

        # --- 5. Affichage ---
        plt.figure(figsize=figsize, facecolor='white', constrained_layout=True)
        ax = plt.gca()
        ax.set_facecolor('black')

        import matplotlib.colors as colors
        epsilon = 1e6
        im = plt.imshow(
            sigma_map,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin='lower',
            cmap=colormap,
            aspect='equal',
            # PowerNorm avec gamma=0.5 est souvent meilleur que LogNorm pour les galaxies 
            # car il sature moins le bulbe tout en montrant le disque.
            #norm=colors.PowerNorm(gamma=0.5, vmin=sigma_map.min()+epsilon, vmax=sigma_map.max())
            norm=colors.LogNorm(vmin=sigma_map.min()+epsilon, vmax=sigma_map.max())
        )
        # cmap = plt.cm.afmhot

        # cmap.set_under('black') # Important !
        # vmin_cbar  = 1e-2
        # import matplotlib.cm as cm
        # from mpl_toolkits.axes_grid1 import make_axes_locatable

        # sm = cm.ScalarMappable(norm=colors.LogNorm(vmin=vmin_cbar, vmax=sigma_map.max()), cmap=cmap)
        # sm.set_array([])

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)

        # Colorbar
        cbar = plt.colorbar(im, pad=0.02)
        cbar.set_label(r"$\mathrm{Surface\ Density}$\ $\Sigma$ $[M_{\odot} / \mathrm{kpc}^2]$")
        
        # cbar = plt.colorbar(sm, cax=cax, extend='min')
        # cbar.set_label(r'$\Sigma$ $[M_\odot / \mathrm{kpc}^2]$')

        plt.xlabel(f"{xlabel} $[\mathrm{{\ kpc}}]$")
        plt.ylabel(f"{ylabel} $[\mathrm{{\ kpc}}]$")
        #plt.title(f"Surface Density (Output {self.output}) - Proj {projection}")
        
        if savefig:
            # auto cration of a folder
            import os
            if location and not os.path.exists(os.path.dirname(location)) and os.path.dirname(location) != '':
                try: os.makedirs(os.path.dirname(location))
                except: pass
            
            filename = f"{location}" if location else f"SurfDensity_{self.output}_{projection}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.show()
    # 1D Profiles 
    def compute_volume_density_profile(self, components=['halo'], n_bins=50, r_min=0.1, r_max=None):
        """
        Calcule le profil de densité volumique rho(r) en coquilles sphériques.
        Utile pour le HALO.
        Retourne: (radius_centers, density)
        """
        xs, ys, zs, ms = [], [], [], []

        # 1. Récupération des particules selon les composants demandés
        if 'halo' in components:
            d = self.get_halo()
            if len(d[0]) > 0:
                xs.append(d[0]); ys.append(d[1]); zs.append(d[2]); ms.append(d[6])
        if 'disk' in components:
            d = self.get_disk()
            if len(d[0]) > 0:
                xs.append(d[0]); ys.append(d[1]); zs.append(d[2]); ms.append(d[6])
        if 'bulge' in components:
            d = self.get_bulge()
            if len(d[0]) > 0:
                xs.append(d[0]); ys.append(d[1]); zs.append(d[2]); ms.append(d[6])

        if not xs:
            print(f"Warning: No particles found for components {components}")
            return np.array([]), np.array([])

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)
        m = np.concatenate(ms)

        # 2. Calcul du rayon sphérique r
        r = np.sqrt(x**2 + y**2 + z**2)

        if r_max is None: r_max = np.max(r)

        # 3. Création des bins logarithmiques
        bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)

        # 4. Histogramme pondéré par la masse
        mass_in_shells, bin_edges = np.histogram(r, bins=bins, weights=m)

        # 5. Calcul du volume des coquilles (V = 4/3 * pi * (r_out^3 - r_in^3))
        vol_shells = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

        # 6. Densité
        rho = mass_in_shells / vol_shells
        
        # Rayon moyen géométrique ou arithmétique pour le plot
        r_centers = (bin_edges[1:] + bin_edges[:-1]) / 2  # Arithmétique
        # r_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1]) # Géométrique (mieux pour log-log pur)

        return r_centers, rho
    
    def compute_surface_density_profile(self, components=['disk', 'bulge'], n_bins=50, R_min=0.1, R_max=None):
        """
        Calcule le profil de densité surfacique Sigma(R) en anneaux cylindriques.
        Utile pour les BARYONS (Disque + Bulbe).
        Retourne: (radius_centers, surface_density)
        """
        xs, ys, zs, ms = [], [], [], []

        # 1. Récupération (Note: on ignore Z pour le rayon, mais on prend toutes les particules en z)
        if 'halo' in components:
            d = self.get_halo()
            if len(d[0]) > 0: xs.append(d[0]); ys.append(d[1]); ms.append(d[6])
        if 'disk' in components:
            d = self.get_disk()
            if len(d[0]) > 0: xs.append(d[0]); ys.append(d[1]); ms.append(d[6])
        if 'bulge' in components:
            d = self.get_bulge()
            if len(d[0]) > 0: xs.append(d[0]); ys.append(d[1]); ms.append(d[6])

        if not xs:
            print(f"Warning: No particles found for components {components}")
            return np.array([]), np.array([])

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        m = np.concatenate(ms)

        # 2. Calcul du rayon CYLINDRIQUE R
        R = np.sqrt(x**2 + y**2)

        if R_max is None: R_max = np.max(R)

        # 3. Bins logarithmiques
        bins = np.logspace(np.log10(R_min), np.log10(R_max), n_bins + 1)

        # 4. Histogramme pondéré
        mass_in_rings, bin_edges = np.histogram(R, bins=bins, weights=m)

        # 5. Calcul de l'aire des anneaux (A = pi * (R_out^2 - R_in^2))
        area_rings = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)

        # 6. Densité Surfacique
        sigma = mass_in_rings / area_rings

        R_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        return R_centers, sigma





#----------------------------------------------------------------------------------------------
# Function to orientate the cluster, necessary in the cluster class
#----------------------------------------------------------------------------------------------
# def orientate_cluster_geometric(x, y, z, galaxy_center=[0,0,0]):
#     """
#     Oriente le système pour que :
#     - L'axe Y pointe EXACTEMENT vers le centre galactique.
#     - L'axe X est perpendiculaire (tangentiel).
#     - L'axe Z est perpendiculaire au plan orbital.
#     """
#     pos = np.vstack((x, y, z)).T
#     center = np.mean(pos, axis=0)
    
#     # 1. Centrer le cluster
#     centered = pos - center
    
#     # 2. Vecteur vers la galaxie (depuis le centre du cluster)
#     vec_to_galaxy = np.array(galaxy_center) - center
    
#     # Normalisation
#     r_mag = np.linalg.norm(vec_to_galaxy)
#     if r_mag == 0: return x, y, z # Sécurité
    
#     # --- Construction de la base locale (Gram-Schmidt simplifié) ---
    
#     # Nouveau vecteur Y : Pointe vers la galaxie
#     new_y = vec_to_galaxy / r_mag
    
#     # Nouveau vecteur Z : On suppose qu'il est inchangé (si orbite plane XY)
#     # Ou on le force à être (0,0,1)
#     new_z = np.array([0, 0, 1])
    
#     # Nouveau vecteur X : Produit vectoriel (Y cross Z)
#     # Cela garantit un repère orthonormé direct
#     new_x = np.cross(new_y, new_z)
    
#     # Création de la matrice de rotation (les colonnes sont les nouveaux vecteurs de base)
#     # R = [new_x, new_y, new_z]
#     rotation_matrix = np.column_stack((new_x, new_y, new_z))
    
#     # 3. Projection (Rotation)
#     # Pour projeter des coordonnées sur une nouvelle base, on fait : pos @ R
#     rotated = centered @ rotation_matrix
    
#     return rotated[:, 0], rotated[:, 1], rotated[:, 2]

# def Orientate_cluster_smart(x, y, z, galaxy_center=[0,0,0]):
#     """
#     Orients the cluster so that:
#         - principal axis with the largest variance becomes +x
#         - second axis becomes +y
#         - third axis becomes +z
#     Also forces a right-handed coordinate system (det > 0)
#     And optionally flips y so the orbit center is at negative y if needed.
#     """

#     pos = np.vstack((x, y, z)).T

#     # 1. Centering
#     center = np.mean(pos, axis=0)
#     centered = pos - center

#     normalized_galaxy_center = np.array(galaxy_center) - center

#     #rotate the xy plane in the direction of the galaxy center
#     angle = np.arctan2(normalized_galaxy_center[1], normalized_galaxy_center[0])
#     cos_angle = np.cos(angle)
#     sin_angle = np.sin(angle)
#     rotation_matrix = np.array([[cos_angle, -sin_angle, 0],
#                                 [sin_angle,  cos_angle, 0],
#                                 [0,          0,         1]])
#     centered = centered @ rotation_matrix.T

#     # Some old methods using the covariance matrix with its eigenvectors

#     # 2. Covariance + eigenvectors
#     cov = np.cov(centered, rowvar=False)
#     eigvals, eigvecs = np.linalg.eigh(cov)

#     # 3. Sort eigenvectors by descending eigenvalue
#     idx = np.argsort(eigvals)[::-1]
#     axes = eigvecs[:, idx]

#     # 4. Ensure a right-handed system (det = +1)
#     if np.linalg.det(axes) < 0:
#         axes[:, -1] *= -1  # flip last axis
#         print("Flipping axis to ensure right-handed system.")
#     # 5. Rotate positions
#     rotated = centered @ axes
#     #rotated = centered
#     return rotated[:, 0], rotated[:, 1], rotated[:, 2]
    
def orientate_cluster_smart(x, y, z, vx, vy, vz, m, galaxy_position=[0,0,0]):
    """
    Orients the cluster so that:
        - principal axis with the largest variance becomes X
        - second axis becomes Y
        - third axis becomes Z
    
    Orientation constraints:
        - Cluster velocity vector (v_cm) points towards negative X (Left).
        - Vector pointing to Galaxy Center points towards negative Y (Down).
        - Coordinate system is right-handed.
    
    Transforms both positions and velocities into this new frame.
    """
    
    # 1. Center of Mass (Position)
    x_cm = np.sum(m*x)/np.sum(m)
    y_cm = np.sum(m*y)/np.sum(m)
    z_cm = np.sum(m*z)/np.sum(m)
    
    # 2. Center of Mass (Velocity)
    vx_cm = np.sum(m*vx)/np.sum(m)
    vy_cm = np.sum(m*vy)/np.sum(m)
    vz_cm = np.sum(m*vz)/np.sum(m)

    # 3. Center positions and velocities
    pos = np.vstack((x - x_cm, y - y_cm, z - z_cm)).T
    vel = np.vstack((vx, vy, vz)).T
    vel = np.vstack((vx - vx_cm, vy - vy_cm, vz - vz_cm)).T

    # 4. PCA on positions to find orientation
    cov = np.cov(pos, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort eigenvectors by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, idx]
    
    # --- Constraint 1: Velocity points Left (Negative X) ---
    # We want v_cm . X < 0
    v_cm_vec = np.array([vx_cm, vy_cm, vz_cm])
    if np.linalg.norm(v_cm_vec) > 1e-9:
        if np.dot(axes[:, 0], v_cm_vec) > 0:
            axes[:, 0] *= -1
            print("Flipping X axis to align with negative velocity (Left).")

    # --- Constraint 2: Galaxy Vector points Down (Negative Y) ---
    # Vector from Cluster to Galaxy
    # vec_to_gal = np.array(galaxy_position) - np.array([x_cm, y_cm, z_cm])
    # if np.linalg.norm(vec_to_gal) > 1e-9:
    #     # We want vec_to_gal . Y < 0
    #     Dot = np.dot(axes[:, 1], vec_to_gal)
    #     print("dot_product: ", Dot )
    #     if Dot < 0:
    #         axes[:, 1] *= -1
    #         print("Flipping Y axis to align with negative galaxy direction (Down).")
        
    #     if Dot > 0:
    #         axes[:, 1] *= -1
    #         print("Flipping Y axis to align with negative galaxy direction (Down).")

    # # --- Constraint 3: Right-handed system ---
    # if np.linalg.det(axes) < 0:
    #     axes[:, 2] *= -1
    #     print("Flipping Z axis to ensure right-handed system.")
        
    # 5. Rotate positions and velocities
    pos_rot = pos @ axes
    vel_rot = vel @ axes
    
    return (pos_rot[:, 0], pos_rot[:, 1], pos_rot[:, 2], 
            vel_rot[:, 0], vel_rot[:, 1], vel_rot[:, 2])


# def cluster_horzontal(x, y, z, galaxy_center=[0,0,0]):
#     """
#     Orients the cluster so that:
#       - the galaxy center is on the positive y axis
#       - the cluster is centered at (0,0,0)
#     """

#     pos = np.vstack((x, y, z)).T

#     # 1. Centering
#     center = np.mean(pos, axis=0)
#     centered = pos - center

#     normalized_galaxy_center = np.array(galaxy_center) - center

#     #rotate the xy plane in the direction of the galaxy center
#     angle = np.arctan2(normalized_galaxy_center[1], normalized_galaxy_center[0])
#     cos_angle = np.cos(np.pi/2-angle)
#     sin_angle = np.sin(np.pi/2-angle)
#     rotation_matrix = np.array([[cos_angle, -sin_angle, 0],
#                                 [sin_angle,  cos_angle, 0],
#                                 [0,          0,         1]])
#     centered = centered @ rotation_matrix.T

#     return centered[:, 0], centered[:, 1], centered[:, 2]
#----------------------------------------------------------------------------------------------
# def orientate_cluster(x, y, z, galaxy_center=[0,0,0]):
#     """
#     Orients the cluster so that:
#       - principal axis with the largest variance becomes +x
#       - second axis becomes +y
#       - third axis becomes +z
#     Also forces a right-handed coordinate system (det > 0)
#     And optionally flips y so the orbit center is at negative y if needed.
#     """

#     centered = cluster_horzontal(x, y, z, galaxy_center)

#     # Some old methods using the covariance matrix with its eigenvectors

#     # 2. Covariance + eigenvectors
#     cov = np.cov(centered, rowvar=False)
#     eigvals, eigvecs = np.linalg.eigh(cov)

#     # 3. Sort eigenvectors by descending eigenvalue
#     idx = np.argsort(eigvals)[::-1]
#     axes = eigvecs[:, idx]

#     # 4. Ensure a right-handed system (det = +1)
#     if np.linalg.det(axes) < 0:
#         axes[:, -1] *= -1  # flip last axis
#         print("Flipping axis to ensure right-handed system.")
#     # 5. Rotate positions
#     rotated = centered @ axes

#     return rotated[:, 0], rotated[:, 1], rotated[:, 2]
#     #return centered[:, 0], centered[:, 1], centered[:, 2]      

def unit_vector_in_bin(vx, vy, vz, m):
    N = len(vx)
    u_v = np.array([0.0, 0.0, 0.0])
    for i in range(N):
        u_v += m[i] * np.array([vx[i], vy[i], vz[i]])
    return u_v/np.sum(m)

def unit_vector_binning(x, y, z, vx, vy, vz, m, bins_per_kpc=100):
    """
    Retourne deux arrays : 
    1. positions : les coordonnées (x, y, z) du centre des bins
    2. directions : les vecteurs unitaires (vx, vy, vz) dans ces bins
    """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    bin_size = 1.0 / bins_per_kpc 

    x_edges = np.arange(x_min, x_max + bin_size, bin_size)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size)
    z_edges = np.arange(z_min, z_max + bin_size, bin_size)

    positions = [] # Pour stocker le centre du bin (X, Y, Z)
    directions = [] # Pour stocker le vecteur vitesse (Vx, Vy, Vz)

    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            for k in range(len(z_edges) - 1):
                mask = (
                    (x >= x_edges[i]) & (x < x_edges[i + 1]) &
                    (y >= y_edges[j]) & (y < y_edges[j + 1]) &
                    (z >= z_edges[k]) & (z < z_edges[k + 1])
                )
                if np.any(mask):
                    # Calcul du vecteur unitaire
                    u_v_bin = unit_vector_in_bin(vx[mask], vy[mask], vz[mask], m[mask])
                    
                    # Calcul du centre du bin
                    bin_center = np.array([
                        0.5 * (x_edges[i] + x_edges[i + 1]),
                        0.5 * (y_edges[j] + y_edges[j + 1]),
                        0.5 * (z_edges[k] + z_edges[k + 1])
                    ])
                    
                    # On stocke les deux séparément !
                    positions.append(bin_center)
                    directions.append(u_v_bin/100)

    return np.array(positions), np.array(directions)

def plot_unit_vectors_projection(positions, directions, projection="xy", figure=None, color_arr="blue"):
    if figure is None:
        figure = plt.figure()
    
    plt.figure(figure.number)
    
    # positions est de forme (N, 3) -> [x, y, z]
    # directions est de forme (N, 3) -> [vx, vy, vz]

    init_plot_style(use_latex= True, font="helvet", fontsize = 14)
    if projection == "xy":
        plt.quiver(
            positions[:, 0], positions[:, 1],   # <--- X, Y (Position de départ)
            directions[:, 0], directions[:, 1], # <--- U, V (Direction de la flèche)
            angles='xy', scale_units='xy', scale=1, color=color_arr, alpha=0.7
        )
        # plot the arrow with the highest norm in red
        max_norm_index = np.argmax(np.linalg.norm(directions[:, :2], axis=1))
        plt.quiver(
            positions[max_norm_index, 0], positions[max_norm_index, 1],
            directions[max_norm_index, 0], directions[max_norm_index, 1],
            angles='xy', scale_units='xy', scale=1, color='red', alpha=1.0
        )
        plt.xlabel('$X\mathrm{[kpc]}$')
        plt.ylabel('$Y\mathrm{[kpc]}$')
        plt.title('$\mathrm{Trend\ magnitude\ and\ direction\ of\ velocities\ on\ XY\ Plane}$')
        
    elif projection == "xz":
        plt.quiver(
            positions[:, 0], positions[:, 2],   # X, Z
            directions[:, 0], directions[:, 2], # Vx, Vz
            angles='xy', scale_units='xy', scale=1, color=color_arr, alpha=0.7
        )
        # plot the arrow with the highest norm in red
        max_norm_index = np.argmax(np.linalg.norm(directions[:, [0,2]], axis=1))
        plt.quiver(
            positions[max_norm_index, 0], positions[max_norm_index, 2],
            directions[max_norm_index, 0], directions[max_norm_index, 2],   
            angles='xy', scale_units='xy', scale=1, color='red', alpha=1.0
        )
        plt.xlabel('$X\mathrm{[kpc]}$')
        plt.ylabel('$Z\mathrm{[kpc]}$')
        plt.title('$\mathrm{Trend\ magnitude\ and\ direction\ of\ velocities\ on\ XZ\ Plane}$')
        
    elif projection == "yz":
        plt.quiver(
            positions[:, 1], positions[:, 2],   # Y, Z
            directions[:, 1], directions[:, 2], # Vy, Vz
            angles='xy', scale_units='xy', scale=1, color=color_arr, alpha=0.7
        )
        plt.xlabel('$Y\mathrm{[kpc]}$')
        plt.ylabel('$Z\mathrm{[kpc]}$')
        plt.title('$\mathrm{Trend\ magnitude\ and\ direction\ of\ velocities\ on\ YZ\ Plane}$')

    plt.grid()
    plt.axis('equal') # Important pour ne pas déformer les flèches
    plt.show()

#----------------------------------------------------------------------------------------------
# Parent class for the cluster

class Cluster:
    def __init__(self, pos, vel, Ramses = False, path_to_data ="", output = 1, len_unit = "kpc", vel_unit = "km / s", mass_unit = "M_sun"):
        if Ramses:
            self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m = Load_Ramses_Osyris(path_to_data, output, len_unit, vel_unit, mass_unit)
        else:
            # Load data for non-RAMSES case
            self.x = pos[:, 0]
            self.y = pos[:, 1]
            self.z = pos[:, 2]
            self.vx = vel[:, 0]
            self.vy = vel[:, 1] 
            self.vz = vel[:, 2]
            self.m = np.ones_like(self.x)
            pass

        self.N = len(self.x)
        self.galaxy_position = [0, 0, 0]
        self.galaxy_cluster_vector = [0, 0, 0]
        self.output = output

    #Here we remove the that has the greatest mass if this mass is greater than a given Mmax
    def remove_ParentGalaxy(self, Mmax=1e6):
        self.galaxy_position = [self.x[np.argmax(self.m)], self.y[np.argmax(self.m)], self.z[np.argmax(self.m)]]
        mask = self.m <= Mmax
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.z = self.z[mask]
        self.vx = self.vx[mask]
        self.vy = self.vy[mask]
        self.vz = self.vz[mask]
        self.m = self.m[mask]
        self.N = len(self.x)

    def reset_coordinates_after_RAMSES(self, box_size):
        """
        Resets the coordinates of the cluster particles after a RAMSES simulation (centers in (0, 0, 0))
        Parameters:
        - box_size: Size of the simulation box (assumed cubic).
        """
        half_box = box_size / 2.0
        self.x = (self.x - half_box)
        self.y = (self.y - half_box)
        self.z = (self.z - half_box)
        self.galaxy_position = [c - half_box for c in self.galaxy_position]
        print(" we have N =", self.N, " particles after resetting coordinates.")

    def get_galaxy_position(self):
        return self.galaxy_position
    
    def get_all_particles(self):
        return self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m
    
    def center_cluster(self):

        x_cm = np.sum(self.x * self.m) / np.sum(self.m)
        y_cm = np.sum(self.y * self.m) / np.sum(self.m)
        z_cm = np.sum(self.z * self.m) / np.sum(self.m)

       
        self.galaxy_cluster_vector = [x_cm - self.galaxy_position[0], y_cm - self.galaxy_position[1], z_cm - self.galaxy_position[2]]
        self.x -= x_cm
        self.y -= y_cm
        self.z -= z_cm
    

    # def rotate_cluster(self):
    #     self.x, self.y, self.z = orientate_cluster(self.x, self.y, self.z, galaxy_center=self.galaxy_cluster_vector)

    # def make_cluster_horizontal(self):
    #     self.x, self.y, self.z = cluster_horzontal(self.x, self.y, self.z, galaxy_center=self.galaxy_cluster_vector)

     # def orientate_cluster_geometric(self):
    #     self.x, self.y, self.z = orientate_cluster_geometric(self.x, self.y, self.z, galaxy_center=self.galaxy_cluster_vector)  

    # The only good function that centers and orientates the cluster in a proper way
    def orientate_cluster_smart(self):
        self.x, self.y, self.z, self.vx, self.vy, self.vz = orientate_cluster_smart(
            self.x, self.y, self.z, 
            self.vx, self.vy, self.vz, 
            self.m, 
            galaxy_position=self.galaxy_position
        )  

    def get_number_of_particles(self):
        return self.N

    def plot_cluster(self, projection = 'yz', figsize=(8,8), showfig = True, savefig=False, location='no_folder_to_save_figs'):
        print("number of particles in the cluster:", self.N)
        if projection == 'xy':
            u = self.x
            v = self.y
            xlegend = "$X \mathrm{\ [kpc]}$"
            ylegend = "$Y \mathrm{\ [kpc]}$"
        elif projection == 'yz':
            u = self.y
            v = self.z
            xlegend = "$Y \mathrm{\ [kpc]}$"
            ylegend = "$Z \mathrm{\ [kpc]}$"
        elif projection == 'xz':
            u = self.x
            v = self.z
            xlegend = "$X \mathrm{\ [kpc]}$"
            ylegend = "$Z \mathrm{\ [kpc]}$"
        plt.figure(figsize=figsize)
        plt.title(f"$\mathrm{{Cluster\ at\ output\ {self.output}}}$")
        plt.xlabel(xlegend)
        plt.ylabel(ylegend)
        plt.xlim((-25, 25))
        plt.ylim((-25, 25))
        plt.grid(True, zorder = 1)
        plt.scatter(u, v, c='steelblue', s=1, alpha=0.7, label=r'$\mathrm{Cluster\ Particles}$', zorder=2)
        #plt.axis('equal') # keep the right proportion
        plt.legend()
        if savefig:
            plt.savefig(f"{location}/Cluster_output_{self.output}_{projection}.png", dpi=300)
        if showfig:
            plt.show()  

#----------------------------------------------------------------------------------------------
# Function for the child class cluster propreties----------------------------------------------
#----------------------------------------------------------------------------------------------

# def get_density_map(x, y, z, r_search):
#     """
#     Computes the density map: density value for every particle, 
#     based on the number of neighbors within r_search.
#     """
    
#     positions = np.vstack((x, y, z)).T
#     tree = cKDTree(positions)
#     volume = (4/3) * np.pi * r_search**3

#     # Calculate density for each particle position
#     densities = np.array([
#         len(tree.query_ball_point(positions[i], r_search)) / volume 
#         for i in range(len(x))
#     ])

#     return densities

# make a smarter of get_density_map using cKDTree
# we look for the 6 nearst neighbours for each particle an compute the volume of the sphere that contains them
# then we compute the density as 6 divided by this volume
def get_density_map(x, y, z, k=6):
    """
    Computes the density map: density value for every particle, 
    based on the distance to the k-th nearest neighbor.
    """
    
    positions = np.vstack((x, y, z)).T
    tree = cKDTree(positions)

    # Find distances to the k-th nearest neighbor for each particle
    distances, _ = tree.query(positions, k=k+1)  # k+1 because the first neighbor is the point itself
    r_k = distances[:, -1]  # Distance to the k-th nearest neighbor

    # Calculate volume of sphere containing k neighbors
    volumes = (4/3) * np.pi * r_k**3

    # Calculate density for each particle position
    densities = k / volumes

    return densities


def compute_density_at_point(tree, position, r_search, volume):
    """Calculates density at a given point using the pre-built KDTree."""
    idx = tree.query_ball_point(position, r_search)
    return len(idx) / volume

def find_precise_density_peak(x, y, z, r_search, max_iterations=50, initial_step=0.001, tolerance=1e-6):
    """
    Finds the most precise density peak in continuous space using a two-phase 
    (coarse particle guess + iterative spatial refinement) approach.
    """
    r_iter = []
    # --- Setup ---
    positions = np.vstack((x, y, z)).T
    tree = cKDTree(positions)
    volume = (4/3) * np.pi * r_search**3

    # --- Phase 1: Initial Coarse Guess (Find densest particle) ---
    initial_densities = np.array([
        compute_density_at_point(tree, p, r_search, volume) for p in positions
    ])
    r_current = positions[np.argmax(initial_densities)]
    
    # --- Phase 2: Iterative Refinement ---
    step = initial_step
    
    for iteration in range(max_iterations):
        best_r_next = r_current
        max_density_in_step = -1.0

        finesse = 6
        # Generate test points in a grid centered on r_current
        offsets_base = np.array(list(itertools.product(np.linspace(-1, 1, finesse), repeat=3)))
        test_points = r_current + step * offsets_base
        
        # Search the test grid
        for r_test in test_points:
            density_at_test = compute_density_at_point(tree, r_test, r_search, volume)

            if density_at_test > max_density_in_step:
                max_density_in_step = density_at_test
                best_r_next = r_test
                    
        # Check for convergence
        r_iter.append(r_current)
        distance_moved = np.linalg.norm(best_r_next - r_current)
        if distance_moved < tolerance:
            break
                
        r_current = best_r_next
        step *= 0.9 # Shrink step size by 10%
        
    return r_current, max_density_in_step, r_iter

def surface_density_binning(x, y, m, x_lim=None, y_lim=None, bins_per_kpc=100, figsize=(8,6), zoom_factor=1):
    """
    Calcule la densité surfacique (Sigma) par binning sur le plan XY, 
    restreinte à une plage définie.

    Paramètres:
    x, y (np.array): Coordonnées projetées des particules.
    m (np.array): Masses des particules (POIDS pour la densité de masse).
    bins (int): Résolution du maillage.
    x_lim (list): Limites [xmin, xmax] pour l'axe X.
    y_lim (list): Limites [ymin, ymax] pour l'axe Y.

    Returns:
    sigma_map (np.array): Carte 2D de la densité surfacique (Masse / Surface).
    x_edges, y_edges (np.array): Bords des bins, correspondant à [x_lim, y_lim].
    """
    # Si les limites ne sont pas fournies, utiliser l'étendue maximale des données
    # if x_lim is None: x_lim = [np.min(x), np.max(x)]
    # if y_lim is None: y_lim = [np.min(y), np.max(y)]
    
    print("figsize:", figsize)    
    # 1. Récupérer les bornes fournies
    xmin_in, xmax_in = sorted(x_lim)
    Dx = xmax_in - xmin_in

    # 3. Centrer x en 0
    x_min = -0.5 * Dx
    x_max =  0.5 * Dx

    # 4. Ratio basé sur figsize
    ratio = figsize[1] / figsize[0]
    Dy = Dx * ratio
    print("Dy:", Dy)
    print("Dx:", Dx)
    print("figsize ratio:", ratio)

    # 5. Construire range final CENTRÉ et STABLE
    h_range = [
    [x_min / zoom_factor, x_max / zoom_factor],
    [-0.5 * Dy / zoom_factor, 0.5 * Dy / zoom_factor]
    ]
    #h_range = [[np.min(x), np.max(x)], [np.min(y)-1, np.max(y)+1]]
    print(">>> h_range =", h_range, "  (len =", len(h_range), ")")
    #we try to define the number of bins. in a clever way so it treats the same way each output
    range_x_values = np.abs(np.max(x) - np.min(x))
    bins = int(bins_per_kpc * range_x_values)
    # 1. Calculer la masse totale dans chaque bin
    mass_per_bin, x_edges, y_edges = np.histogram2d(
        x, 
        y, 
        bins=bins, 
        weights=np.ones(len(m)),  # We count the number of particles per bin
        range=h_range, # Utiliser la plage définie
        density=False 
    )

    # 2. Déterminer la surface d'un seul bin
    # Les bords des bins correspondent maintenant aux limites x_lim, y_lim.
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    bin_area = dx * dy

    # 3. Calculer la Densité Surfacique (Sigma = Masse / Surface)
    sigma_map = mass_per_bin / bin_area

    return sigma_map, x_edges, y_edges

from matplotlib import colors

# def plot_Ssurface_density(sigma_map, x_edges, y_edges, bins_per_kpc=1000, title="$\mathrm{Surface\ Density\ }\Sigma $", 
#                          colormap = 'plasma', savefig=False, location=""):
#     # ... (le début de la fonction reste le même)
#     import matplotlib.colors as colors
#     # Remplacer les valeurs non positives par une valeur très petite (epsilon)
#     sigma_map_clean = np.where(sigma_map <= 0, 1e-10, sigma_map)

#     plt.figure()
#     # using a logarithmic color scale 
#     #Plotting the equation of the circular trajectory in the xy pleane
#     # circle = plt.Circle((22.5, 22.5), 14, color='white', fill=False, linestyle='--', linewidth=2, alpha=0.5)
#     # plt.gca().add_artist(circle)
#     im = plt.imshow(
#         sigma_map_clean.T,
#         extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
#         origin='lower',
#         aspect='equal', # Maintient l'aspect ratio 1:1 pour les pixels
#         cmap= colormap,
#         #norm=colors.LogNorm( vmin=sigma_map_clean.min(), vmax=sigma_map_clean.max())
#         norm=colors.PowerNorm( gamma=0.5, vmin=sigma_map_clean.min(), vmax=sigma_map_clean.max())
#        # norm=colors.PowerNorm( gamma=0.5, vmin=0, vmax=1e5)
#     )
    
#     # plot the maximum value
#     ind_max = np.unravel_index(np.argmax(sigma_map_clean, axis=None), sigma_map_clean.shape)
#     x_max = 0.5 * (x_edges[ind_max[0]] + x_edges[ind_max[0] + 1])
#     y_max = 0.5 * (y_edges[ind_max[1]] + y_edges[ind_max[1] + 1])
#     plt.scatter(x_max, y_max, color='blue', marker='x', s=100, label=r'$\mathrm{Max\ \Sigma}$')

#     # plot the second maximum value
#     sigma_map_copy = sigma_map_clean.copy()
#     sigma_map_copy[ind_max] = -np.inf  # set the max value to -infinity to find the second max
#     ind_max2 = np.unravel_index(np.argmax(sigma_map_copy, axis=None), sigma_map_copy.shape)
#     x_max2 = 0.5 * (x_edges[ind_max2[0]] + x_edges[ind_max2[0] + 1])
#     y_max2 = 0.5 * (y_edges[ind_max2[1]] + y_edges[ind_max2[1] + 1])        
#     plt.scatter(x_max2, y_max2, color='cyan', marker='x', s=100, label=r'$\mathrm{2nd\ Max\ \Sigma}$')  

#     #plt.legend()
#     cbar = plt.colorbar(im, location='bottom')
#     unit=1000/bins_per_kpc
#     cbar.set_label(f'$\Sigma\ \mathrm{{[Part\  \cdot {int(unit)} pc^{{-2}}]}}$')
#     cbar.formatter.set_powerlimits((0, 0))
#     cbar.update_ticks()
    
#     plt.xlim(x_edges[0], x_edges[-1])
#     plt.ylim(y_edges[0], y_edges[-1])
    
#     plt.xlabel(r"$X \mathrm{\ [kpc]}$")
#     plt.ylabel(r"$Y \mathrm{\ [kpc]}$")
#     plt.title(title)
#     if savefig:
#         #plt.savefig(f"{location}surf_orbit_{title.replace(' ', '_')}.png", dpi=300)
#         plt.savefig(f"{location}SurfaceDensity_{title.replace(' ', '_')}.png", dpi=300)   
#     plt.show()

from scipy.ndimage import uniform_filter

def plot_surface_density(sigma_map, x_edges, y_edges, projection="xy", bins_per_kpc=1000, 
                         search_radius_kpc=0.5, 
                         title="$\mathrm{Surface\ Density\ }\Sigma $", 
                         colormap='plasma', figsize=(8,8), savefig=False, location="", Label = True):
    
    import matplotlib.colors as colors
    
    # 1. Initial cleanup, values <= 0 become epsilon
    sigma_map_clean = np.where(sigma_map <= 0, 1e-10, sigma_map)

    plt.figure(facecolor='white', figsize=figsize)
    plt.gca().set_facecolor('black')

    # 2. Display the surface density map
    im = plt.imshow(
        sigma_map_clean.T,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower',
        aspect='equal', 
        cmap=colormap,
        norm=colors.PowerNorm(gamma=0.5, vmin=sigma_map_clean.min(), vmax=sigma_map_clean.max())
    )

    # # =========================================================================
    # # SMART PEAK SEARCH
    # # =========================================================================
    # Some old essays to find surface density peaks 
    # # A. Preparation for smoothing
    # # dx = x_edges[1] - x_edges[0] 
    # # radius_pixels = int(search_radius_kpc / dx)
    # # if radius_pixels < 1: radius_pixels = 1
    # radius_pixels = int(bins_per_kpc * search_radius_kpc)
    # # B. Smoothing
    # smoothed_map = uniform_filter(sigma_map_clean, size=2*radius_pixels)

    # # C. CRUCIAL CORRECTION: RE-MASKING
    # # We look at the original map. If a pixel was "empty" (or almost empty, <= 2*epsilon),
    # # we forbid the smoothed map from finding a max here.
    # # This prevents finding points outside the real matter.
    # invalid_mask = sigma_map_clean <= 2e-10 
    # smoothed_map[invalid_mask] = -np.inf

    # # D. Coordinates
    # x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    # y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # # E. Left / Right Search
    
    # # --- LEFT ZONE (x < 0) ---
    # map_left = smoothed_map.copy()
    # map_left[x_centers >= 0, :] = -np.inf # Exclude right
    
    # # Check if there are any valid pixels left (not -inf)
    # if np.any(map_left > -1e20): 
    #     ind_L = np.unravel_index(np.argmax(map_left), map_left.shape)
    #     x_L = x_centers[ind_L[0]]
    #     y_L = y_centers[ind_L[1]]
    #     val_L = map_left[ind_L] # Smoothed peak value
    #     has_left = True
    #     print(f"Left peak found at ({x_L:.3f}, {y_L:.3f}) with value {val_L:.3e}")
    # else:
    #     has_left = False
    #     val_L = -np.inf

    # # --- RIGHT ZONE (x >= 0) ---
    # map_right = smoothed_map.copy()
    # map_right[x_centers < 0, :] = -np.inf # Exclude left
    
    # if np.any(map_right > -1e20):
    #     ind_R = np.unravel_index(np.argmax(map_right), map_right.shape)
    #     x_R = x_centers[ind_R[0]]
    #     y_R = y_centers[ind_R[1]]
    #     val_R = map_right[ind_R]
    #     has_right = True
    #     print(f"Right peak found at ({x_R:.3f}, {y_R:.3f}) with value {val_R:.3e}")
    # else:
    #     has_right = False
    #     val_R = -np.inf

    # # F. Decision for plotting (Distance + Comparison)
    # points_to_plot = [] 
    
    # total_x_range = x_edges[-1] - x_edges[0]
    # min_dist_threshold = 0.10 * total_x_range # Distance threshold (10%)
    # # If we have candidates on both sides
    # if has_left and has_right:
    #     dist = np.sqrt((x_L - x_R)**2 + (y_L - y_R)**2)
        
    #     # DISTANCE CONDITION: We display both ONLY if they are far apart
    #     if dist > min_dist_threshold:
    #         if val_L >= val_R:
    #             points_to_plot.append((x_L, y_L, 'X', val_L)) # Max 1
    #             points_to_plot.append((x_R, y_R, '*', val_R)) # Max 2
    #         else:
    #             points_to_plot.append((x_R, y_R, 'X', val_R))
    #             points_to_plot.append((x_L, y_L, '*', val_L))
        
    #     # Otherwise, they are too close (it's the same peak cut by the axis), we keep the best one
    #     else:
    #         if val_L >= val_R:
    #             points_to_plot.append((x_L, y_L, 'X', val_L))
    #         else:
    #             points_to_plot.append((x_R, y_R, 'X', val_R))
                
    # # If we have a candidate on only one side
    # elif has_left:
    #     points_to_plot.append((x_L, y_L, 'X', val_L))
    # elif has_right:
    #     points_to_plot.append((x_R, y_R, 'X', val_R))

    # # G. Final display

    # # for px, py, pmarker, pval in points_to_plot:
    # #    plt.scatter(px, py, color=None, marker=pmarker, s=100, linewidth=0.05, 
    # #              zorder=10, label=r'$\mathrm{Max\ \Sigma} = ' + f"{pval:.3e}" + "$")

    # # =========================================================================

    cbar = plt.colorbar(im, location='bottom')
    unit = 1000/bins_per_kpc if bins_per_kpc != 0 else 1
    cbar.set_label(f'$\Sigma\ \mathrm{{[Part\  \cdot {int(unit)} pc^{{-2}}]}}$')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    
    plt.xlim(x_edges[0], x_edges[-1])
    plt.ylim(y_edges[0], y_edges[-1])
    
    if projection == "xy":
        plt.xlabel(r"$X \mathrm{\ [kpc]}$")
        plt.ylabel(r"$Y \mathrm{\ [kpc]}$")
    elif projection == "xz":
        plt.xlabel(r"$X \mathrm{\ [kpc]}$")
        plt.ylabel(r"$Z \mathrm{\ [kpc]}$")
    elif projection == "yz":
        plt.xlabel(r"$Y \mathrm{\ [kpc]}$")
        plt.ylabel(r"$Z \mathrm{\ [kpc]}$")
    else:
        raise ValueError("Invalid projection. Choose from 'xy', 'xz', 'yz'.")
    
    plt.title(title)

    if Label:
        plt.legend()
    if savefig:
        plt.savefig(f"{location}SurfaceDensity_{title.replace(' ', '_')}.png", dpi=300, facecolor='white')   
    if not savefig:
        plt.show()

# Child class of the cluster with properties methods
def Scatter_Plot(u, v, figure, color='lightgrey'):
    plt.figure(figure)
    init_plot_style(use_latex=True, font="helvet", fontsize=14)
    plt.scatter(u, v, color=color, s=5, alpha=0.7)

class ClusterProperties(Cluster):
    
    def __init__(self, cluster):
        # On copie les données pour éviter d’altérer l'objet original
        self.galaxy_position = cluster.galaxy_position.copy()
        self.galaxy_cluster_vector = cluster.galaxy_cluster_vector.copy()
        self.x = cluster.x.copy()
        self.y = cluster.y.copy()
        self.z = cluster.z.copy()
        self.vx = cluster.vx.copy()
        self.vy = cluster.vy.copy()
        self.vz = cluster.vz.copy()
        self.m = cluster.m.copy()
        self.N = cluster.N

        # ranges pour les futurs plots
        self.range_plot_x = [0, 0]
        self.range_plot_y = [0, 0]

    def remove_orbit_velocity(self, v_orbit):

        # R_xy = np.sqrt(self.x**2 + self.y**2)

        # u_tan_x = -self.y / R_xy
        # u_tan_y =  self.x / R_xy
    
        # # v_vec = v_magnitude * direction
        # self.vx -= v_orbit * u_tan_x
        # self.vy -= v_orbit * u_tan_y
        self.vx -= np.mean(self.vx)
        self.vy -= np.mean(self.vy)
        self.vz -= np.mean(self.vz)

    # ---------------------------
    #   PHYSICAL QUANTITIES
    # ---------------------------
    def get_average_velocity(self):
        # Moyenne pondérée par la masse : c'est la vraie vitesse du centre de masse
        v_cx = np.average(self.vx, weights=self.m)
        v_cy = np.average(self.vy, weights=self.m)
        v_cz = np.average(self.vz, weights=self.m)
        return v_cx, v_cy, v_cz

    def get_relative_velocities(self, average_velocity):
        # Cette fonction était déjà mathématiquement correcte !
        vx_rel = self.vx - average_velocity[0]
        vy_rel = self.vy - average_velocity[1]
        vz_rel = self.vz - average_velocity[2]
        return np.array([vx_rel, vy_rel, vz_rel])

    def get_systemic_velocity_median(self):
        # Totalement insensible aux longues queues de marée et au bruit de fond
        vx_sys = np.median(self.vx)
        vy_sys = np.median(self.vy)
        vz_sys = np.median(self.vz)
        return np.array([vx_sys, vy_sys, vz_sys])
    
    def sigma_LOS(self, Rmin, Rmax, nbins, Nbmin):
        r, sigma = dispersionLOS(
            self.x, self.y, self.z,
            self.vx, self.vy, self.vz,
            Rmin, Rmax, nbins, Nbmin
        )
        return r, sigma
    
    def VolmicDensity_per_particle(self, r_search):
        return get_density_map(self.x, self.y, self.z, k=6)
    
    def density_peak(self, r_search, max_iterations=50, initial_step=0.001, tolerance=1e-6):
        return find_precise_density_peak(
            self.x, self.y, self.z,
            r_search/10,
            max_iterations,
            initial_step,
            tolerance
        )
    
    def get_initial_computed_orbit(self, filename = "Cluster_Orbit.txt"):
        data = np.loadtxt(filename,skiprows=1)
        x_orbit = data[:, 0]
        y_orbit = data[:, 1]
        z_orbit = data[:, 2]
        return x_orbit, y_orbit, z_orbit
    # ---------------------------
    #        PLOTS
    # ---------------------------

    def plot_VolumicDensity(self, r_search=1, projection='xy', title="$\\rho$", colormap='viridis', figsize=(8,6), zoom_factor=1, 
                            savefig=False, fname = "None", location="", limit=False,axlim = 100, show_orbit=False, orbit_filename="Cluster_Orbit.txt"):
        """
        Plots the volumic density.
        projection: str, choice between 'xy', 'xz', 'yz'
        """
    
        # --- 1. Computing the density ---
        # Note: La densité reste une propriété 3D, peu importe la projection choisie
        density = self.VolmicDensity_per_particle(r_search)
        density = np.where(density <= 0, 1e-10, density)

        # --- 2. Selection of axes based on projection ---
        projection = projection.lower() # Pour gérer 'XY', 'xy', etc.
        
        if projection == 'xy':
            data_h, data_v = self.x, self.y
            if show_orbit:
                orbit_h, orbit_v, _ = self.get_initial_computed_orbit(orbit_filename)
            xlabel, ylabel = '$X\mathrm{\ [kpc]}$', '$Y\mathrm{\ [kpc]}$'
        elif projection == 'xz':
            data_h, data_v = self.x, self.z
            if show_orbit:
                orbit_h, _, orbit_v = self.get_initial_computed_orbit(orbit_filename)
            xlabel, ylabel = '$X\mathrm{\ [kpc]}$', '$Z\mathrm{\ [kpc]}$'
        elif projection == 'yz':
            data_h, data_v = self.y, self.z
            if show_orbit:
                _, orbit_h, orbit_v = self.get_initial_computed_orbit(orbit_filename)
                print(orbit_h[0])
            xlabel, ylabel = '$Y\mathrm{\ [kpc]}$', '$Z\mathrm{\ [kpc]}$'
        else:
            raise ValueError("Projection must be 'xy', 'xz', or 'yz'")

        # --- 2. Plotting ---
        plt.figure(facecolor='white', figsize=figsize)
        
        try:
            init_plot_style(use_latex=True, font="helvet", fontsize=14)
        except:
            pass

        ax = plt.gca()
        ax.set_facecolor('black')

        plt.title(title)
        plt.grid(True, zorder=1, alpha=0.3, color='white', linestyle='--')

        # Scatter plot principal
        if show_orbit:
            plt.plot(orbit_h, orbit_v, color='grey', linestyle='-', linewidth=1, alpha=0.7, zorder=2, label=r'$\mathrm{Integrated\ Orbit}$')

        sc = plt.scatter(data_h, data_v, 
                 c=density, 
                 s=0.5, 
                 alpha=0.7, 
                 zorder=3,
                 norm=colors.LogNorm(), # Log scale for color
                 cmap=colormap,
                 ) 
        print("mean data h: ", np.mean(data_h))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        cbar = plt.colorbar(sc, location='bottom')
        cbar.set_label(f'$\\rho\ \\mathrm{{[Part\ \\cdot kpc^{{-3}}]}}$')
        plt.clim(1e-1, 1e9)
        lim = max(np.max(np.abs(self.x)), np.max(np.abs(self.y))) * 1.1
        
        if limit:
            plt.gca().set_ylim(-axlim, axlim)
            plt.gca().set_xlim(-axlim, axlim)
            ax.set_aspect('equal', adjustable='box')
            print("We use limits for the axes")
        else:
            plt.gca().set_ylim(ymin=-lim, ymax=lim)
            plt.gca().set_xlim(xmin=-lim, xmax=lim)
            ax.set_aspect('equal', adjustable='box')

        if savefig:
            plt.savefig(f"{location}VolumicDensity_{fname}.png", dpi=300, facecolor='white')
        if not savefig:
           plt.show()

            

    def SurfaceDensity_map(self, x_lim=None, y_lim=None, bins_per_kpc=100, projection='xy', figsize=(8,6), zoom_factor=1):
        if x_lim is None:
            x_lim = self.range_plot_x
        if y_lim is None:
            y_lim = self.range_plot_y

        if projection == 'xy':
            u = self.x
            v = self.y
        elif projection == 'xz':
            u = self.x
            v = self.z
        elif projection == 'yz':
            u = self.y
            v = self.z
        
        sigma_map, x_edges, y_edges = surface_density_binning(
            u, v, self.m,
            x_lim, y_lim,
            bins_per_kpc,
            figsize=figsize,
            zoom_factor=zoom_factor
        )
        return sigma_map, x_edges, y_edges
    

    def plot_SurfaceDensity(self,x_lim=None, y_lim=None, bins_per_kpc=100, projection = 'xy', search_radius_kpc=0.1,
                            title="$\\Sigma$", colormap='plasma',
                            savefig=False, location="", figsize=(8,6), zoom_factor=1, label = True):
        print("figsize:", figsize)
        sigma_map, x_edges, y_edges = self.SurfaceDensity_map(
            x_lim, y_lim,
            bins_per_kpc=bins_per_kpc,
            projection=projection,
            figsize=figsize,
            zoom_factor=zoom_factor
        )

        plot_surface_density(
            sigma_map, x_edges, y_edges, bins_per_kpc=bins_per_kpc, projection=projection, search_radius_kpc=search_radius_kpc,
            title = title, colormap=colormap, figsize=figsize,
            savefig=savefig,
            location=location, Label = label
        )

    def find_angular_extent(self):
        from scipy.optimize import minimize
        import numpy as np

        def circle_residuals(params, x, y):
            xc, yc, r = params
            return np.sum((np.sqrt((x - xc)**2 + (y - yc)**2) - r)**2)

        x = self.x
        y = self.y
        x_m = np.mean(x)
        y_m = np.mean(y)
        r_m = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))

        initial_guess = (x_m, y_m, r_m)
        result = minimize(circle_residuals, initial_guess, args=(x, y))
        xc, yc, r = result.x

        angles = np.arctan2(y - yc, x - xc)
        angles = np.sort(angles)
        
        diffs = np.diff(angles)
        gap_crossing_pi = (angles[0] + 2*np.pi) - angles[-1]
        all_diffs = np.append(diffs, gap_crossing_pi)
        
        max_gap = np.max(all_diffs)
        angular_extent = 2*np.pi - max_gap

        return angular_extent, (xc, yc), r

    def plot_unitV_trend(self, bins=5, proj="xy", color_dots='lightgrey', color_vectors='cyan', savefig =False, location="", output_number=0):
        #rescaling the velocities by removing the initial kick of the cluster 
        # self.vx -= np.mean(self.vx)
        # self.vy -= np.mean(self.vy)
        # self.vz -= np.mean(self.vz)

        #self.remove_orbit_velocity(v_orbit=395)

        fig = plt.figure()
        plt.grid(True, zorder=1, alpha=0.3, color='white', linestyle='--')
        init_plot_style(use_latex=True, font="helvet", fontsize=14)
        ax = plt.gca()
        ax.set_facecolor('black')

        print("number.  ",fig.number)
        if proj == "xy":
            Scatter_Plot(self.x, self.y, fig.number, color=color_dots)
        if proj == "xz":
            Scatter_Plot(self.x, self.z, fig.number, color=color_dots)
        if proj == "yz":
            Scatter_Plot(self.y, self.z, fig.number, color=color_dots)
         # Compute unit vector binned data
        pos, dirs = unit_vector_binning(
                    self.x, self.y, self.z,
                    self.vx, self.vy, self.vz, self.m, bins_per_kpc=bins
                    )   
        # Plot
        plot_unit_vectors_projection(pos, dirs, projection=proj, figure=fig, color_arr=color_vectors)
        if savefig:
            fig.savefig(f"{location}/output_{output_number}UnitV_Trend_{proj}.png", dpi=300)
        plt.show()

class ClusterSeries:
    def __init__(self, path_to_data, orbit_filename=None, boxsize=500):
        """
        Initialise la série. Ne charge PAS les particules ici pour sauver la RAM.
        
        :param path_to_data: Dossier contenant les outputs RAMSES
        :param orbit_filename: Chemin vers le fichier ASCII de l'orbite (t, x, y, z, vx, vy, vz)
        """
        self.path_to_data = path_to_data
        self.orbit_data = None
        self.boxsize = boxsize
        # Chargement de l'orbite théorique/calculée une seule fois au début
        if orbit_filename and os.path.exists(orbit_filename):
            print(f"Loading orbit from {orbit_filename}...")
            # On suppose un format standard: x, y, z, vx, vy, vz
            self.orbit_data = np.loadtxt(orbit_filename, skiprows=1)
        elif orbit_filename:
            print(f"Warning: Orbit file {orbit_filename} not found.")

    def get_cluster_at_step(self, output_number, remove_galaxy=True, Mmax=1e6):
        """
        Charge un cluster spécifique à la demande.
        """
        # 1. On instancie la classe Cluster existante
        cluster = Cluster(self.path_to_data, output_number)
        
        # 2. On nettoie la galaxie
        if remove_galaxy:
            cluster.remove_ParentGalaxy(Mmax=1e3)
        
        # 3. CRUCIAL : Centrage relatif
        # Pour voir l'orbite, on ne veut PAS centrer le cluster sur lui-même.
        # On veut centrer le repère sur la Galaxie, pour voir où est le cluster.
        gal_pos = cluster.get_galaxy_position()
        
        # On décale les particules par rapport à la galaxie
        # cluster.x -= gal_pos[0]
        # cluster.y -= gal_pos[1]
        # cluster.z -= gal_pos[2]
        cluster.x -= self.boxsize / 2
        cluster.y -= self.boxsize / 2
        cluster.z -= self.boxsize / 2
        
        return cluster

    def plot_orbit_sequence(self, outputs_list, projection='xy', 
                            figsize=(10, 10), cmap='inferno', 
                            delta_tout=3.3557,
                            savefig=False, location="sequence.png"):
        """
        Trace la séquence des clusters sur leur orbite avec densité volumique.
        """
        
        # 1. Configuration du Plot
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('black') # Fond noir pour faire ressortir la densité
        
        # Gestion de la projection pour les indices
        # Orbite file indices: 0:t, 1:x, 2:y, 3:z
        proj_idx_orbit = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)} 
        labels = {'xy': ('X', 'Y'), 'xz': ('X', 'Z'), 'yz': ('Y', 'Z')}
        
        if projection not in proj_idx_orbit: raise ValueError("Projection must be xy, xz, or yz")
        
        ix_orb, iy_orb = proj_idx_orbit[projection]
        xlabel, ylabel = labels[projection]
#
        # 2. Tracer l'Orbite théorique (Ligne de fond)
        if self.orbit_data is not None:
            orb_x = self.orbit_data[:, ix_orb]
            orb_y = self.orbit_data[:, iy_orb]
            ax.plot(orb_x, orb_y, color='white', alpha=0.3, lw=1, ls='--', 
                    zorder=1, label=r'$\mathrm{Integrated\ Orbit}$')
            ax.scatter(orb_x[-1], orb_y[-1], marker = "x", color='white', s=50, alpha=1, zorder=1, 
                       label = "$\mathrm{Initial\ position}$" )
        # 3. Boucle sur les outputs demandés
        sc = None 
        print(f"Plotting sequence for outputs: {outputs_list}")
        
        for output in outputs_list:
            # A. Chargement "Lazy" (Economie de RAM)
            try:
                cl = self.get_cluster_at_step(output)
            except Exception as e:
                print(f"Skipping output {output}: {e}")
                continue

            # B. Récupération des données
            x, y, z = cl.x, cl.y, cl.z
            m = cl.m[0] # Assuming uniform mass for simplicity
            # C. Calcul Densité (Utilise ta fonction existante !)
            # k=6 est ta valeur par défaut dans get_density_map
            rho = get_density_map(x, y, z, k=6) * m
            
            # D. Sélection coordonnées pour le plot selon projection
            if projection == 'xy':
                u, v = x, y
            elif projection == 'xz':
                u, v = x, z
            elif projection == 'yz':
                u, v = y, z
            
            # E. Calcul du temps
            time_gyr = compute_time_simulation(output, delta_tout)

            # F. Scatter Plot
            # zorder élevé pour être au dessus de l'orbite
            sc = ax.scatter(u, v, c=rho, 
                            norm=colors.LogNorm(vmin=np.min(rho), vmax= np.max(rho)), # Ajuste vmin/vmax selon tes données
                            cmap=cmap, s=1.5, alpha=0.8, zorder=2)
            
            # G. Annotation du temps (Positionné au centre de masse actuel du cluster)
            text_x, text_y = np.mean(u), np.mean(v)
            ax.text(text_x, text_y + 1.0, f"${time_gyr:.2f} \\, \\mathrm{{Gyr}}$", 
                    color='white', fontsize=9, ha='center', fontweight='bold', zorder=3)

        # 4. Finalisation
        init_plot_style(use_latex=True, font="helvet", fontsize=14)
        ax.set_xlabel(f"${xlabel}$ $\mathrm{{[kpc]}}$", fontsize=14)
        ax.set_ylabel(f"${ylabel}$ $\mathrm{{[kpc]}}$", fontsize=14)
        ax.axis('equal')
        ax.grid(color='white', alpha=0.1, linestyle='--')

        # Colorbar
        if sc:
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(r"$\mathrm{Volumic\ Density}\ \rho \, [\mathrm{M_\odot \cdot kpc^{-3}}]$", fontsize=12)
            # Astuce pour avoir les ticks de la colorbar lisibles sur fond noir/blanc
            cbar.ax.yaxis.set_tick_params(color='black') 
        
        plt.legend(
                    fancybox=False,      # C'est ça qui fait les coins carrés (True = arrondis)
                    framealpha=1,      # Transparence du fond (0 = invisible, 1 = opaque)
                    facecolor='black',   # Couleur de fond de la légende
                    edgecolor='white',   # Couleur de la bordure (facultatif)
                    labelcolor='white',  # Couleur du texte (indispensable sur fond noir)
                    fontsize=10          # Taille du texte
                )
        if savefig:
            # Création dossier si besoin
            folder = os.path.dirname(location)
            if folder and not os.path.exists(folder): os.makedirs(folder)
            
            plt.savefig(location, dpi=300, bbox_inches='tight')
            print(f"Saved to {location}")
        
        plt.show()

class ClusterSequenceTXT:
    def __init__(self, folder_path, file_pattern="stream_{}.txt", orbit_filename=None):
        """
        file_pattern : Par exemple "stream_{}.txt". 
                       Le {} sera remplacé par le string de temps (ex: "0050.0")
        """
        self.folder_path = folder_path
        self.file_pattern = file_pattern
        self.orbit_data = None
        
        # Chargement de l'orbite
        if orbit_filename and os.path.exists(orbit_filename):
            print(f"Loading orbit from {orbit_filename}...")
            # On suppose le format: t, x, y, z, vx, vy, vz
            self.orbit_data = np.loadtxt(orbit_filename)
        else:
            print(f"Warning: Orbit file {orbit_filename} not found or not provided.")

    def load_cluster_data(self, output_id):
        # output_id est la string "0050.0", "0100.0" etc.
        filename = self.file_pattern.format(output_id)
        path = os.path.join(self.folder_path, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        data = np.loadtxt(path)
        # Supposons colonnes: x, y, z sont 0, 1, 2
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        return x, y, z

    def plot_sequence(self, outputs_list, projection='xy', 
                      figsize=(8, 8), cmap='inferno', 
                      add_orbit_offset=False, 
                      savefig=False, location="sequence_txt.png"):
        
        # 1. FIGURE EN BLANC (Le tour de l'image)
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # 2. INTÉRIEUR DU PLOT EN NOIR (Pour le contraste)
        ax.set_facecolor('black')
        
        # CORRECTION ICI: Si col 0 est le Temps, alors X=1, Y=2, Z=3
        proj_idx_orb = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)} 
        labels = {'xy': ('X', 'Y'), 'xz': ('X', 'Z'), 'yz': ('Y', 'Z')}
        
        if projection not in proj_idx_orb: raise ValueError("Proj must be xy, xz, yz")
        
        ix_orb, iy_orb = proj_idx_orb[projection]
        xlabel, ylabel = labels[projection]

        # Tracer l'orbite (en blanc ou gris clair pour qu'elle se voie sur le fond noir)
        if self.orbit_data is not None:
            ox = self.orbit_data[:, ix_orb]
            oy = self.orbit_data[:, iy_orb]
            ax.plot(ox, oy, color='white', alpha=0.3, lw=1, ls='--', 
                    zorder=1, label='$\mathrm{Integrated\ Orbit}$')
            # Marqueur de fin d'orbite (position initiale ou finale selon le sens)
            ax.scatter(ox[-1], oy[-1], marker = "x", color='white', s=50, alpha=1, zorder=1, 
                       label = "$\mathrm{Initial\ position}$" )
        
        sc = None
        print(f"Plotting sequence for {len(outputs_list)} files...")

        for output_str in outputs_list:
            try:
                # A. Chargement (utilise le nom de fichier complet via le pattern)
                x, y, z = self.load_cluster_data(output_str)
                
                # B. Gestion Position (Offset sur l'orbite)
                try: 
                    # On convertit le string "0050.0" en float 50.0 pour chercher dans l'orbite
                    current_time = float(output_str)
                except ValueError: 
                    current_time = 0.0
                
                if add_orbit_offset and self.orbit_data is not None:
                    # On cherche la ligne correspondant au temps dans le fichier orbite
                    times_orbit = self.orbit_data[:, 0]
                    idx_closest = (np.abs(times_orbit - current_time)).argmin()
                    
                    # On récupère x,y,z (cols 1,2,3) à cet instant
                    orbit_pos_at_t = self.orbit_data[idx_closest, 1:4] 
                    
                    x += orbit_pos_at_t[0]
                    y += orbit_pos_at_t[1]
                    z += orbit_pos_at_t[2]

                # C. Densité
                rho = get_density_map(x, y, z, k=6)

                m=21367/14246
                rho = rho * m
                # D. Projection
                if projection == 'xy':   u, v = x, y
                elif projection == 'xz': u, v = x, z
                elif projection == 'yz': u, v = y, z
                
                # E. Plot
                # Note: vmin/vmax dynamiques par cluster ici. 
                # Si tu veux une échelle fixe pour toute la séquence, fixe vmin=1e-2, vmax=10 par ex.
                sc = ax.scatter(u, v, c=rho, 
                                norm=colors.LogNorm(vmin=rho.min(), vmax=rho.max()), 
                                cmap=cmap, s=1, alpha=0.8, zorder=2)
                
                # F. Texte (En blanc car il est SUR le fond noir)
                # On utilise output_str directement pour l'affichage (ex: "0050.0")
                txt_x, txt_y = np.mean(u), np.mean(v)
                ax.text(txt_x, txt_y + 1.5, f"${(float(output_str)/1000):0.2f}$ $\mathrm{{Gyr}}$", color='white', fontsize=9, ha='center', zorder=3)

            except Exception as e:
                print(f"Error output {output_str}: {e}")
                # import traceback
                # traceback.print_exc()
                continue

        # --- STYLE FINAL (Conservé strictement) ---
        
        # Les Labels des axes en NOIR (car ils sont sur le fond blanc)
        ax.set_xlabel(f"${xlabel}$ $\mathrm{{[kpc]}}$", color='black', fontsize=14)
        ax.set_ylabel(f"${ylabel}$ $\mathrm{{[kpc]}}$", color='black', fontsize=14)
        
        # On s'assure que les ticks sont noirs
        ax.tick_params(colors='black', which='both')
        
        # Grille discrète
        ax.axis('equal')
        ax.grid(color='white', alpha=0.1, linestyle=':')

        # Colorbar
        if sc:
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(r"$\mathrm{Volumic\ Density\ }\rho$ $\mathrm{[M_\odot\ \cdot kpc^{-3}]}$", color='black') # Label noir
            cbar.ax.yaxis.set_tick_params(color='black')     # Ticks noirs
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black') # Chiffres noirs

        # Légende
        legend = ax.legend(
            loc='upper right',
            fancybox=False, 
            framealpha=1, 
            facecolor='black',  # Fond de la légende noir
            fontsize = 10,
            edgecolor='white', 
            labelcolor='white'  # Texte blanc pour être lisible sur le fond noir de la légende
        )

        if savefig:
            # On sauvegarde avec facecolor='white'
            plt.savefig(location, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
#----------------------------------------------------------------------------------------------
#Some small functions to help with the vizualisation 

# Small function to compute the time in Gyr from the output number
def compute_time_simulation(output, delta_tout=3.3557, tend=335.57):
    # tot_outputs = tend/delta_tout
    # t_end_Gyr = tend/67.114
    # time_sim = (output-1)/tot_outputs *t_end_Gyr
    time_sim = (output - 1) * delta_tout / 67.114
    return time_sim
#----------------------------------------------------------------------------------------------


# %%
