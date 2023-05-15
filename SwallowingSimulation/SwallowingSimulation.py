from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm
import sph_functions
import sys
import matplotlib
from numba import njit
import cv2
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

directory = 'D:/Things/Programming/IRP/WCSPH/Numba/InterImages/reconstructed_image_'
directory1 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Swallowing/BoundariesFinal/Initial/boundary'
directory2 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Swallowing/BoundariesFinal/Ghost/boundary'
directory_vel1 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Swallowing/BoundariesFinal/Velocity_initial/vel'
directory_vel2 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Swallowing/BoundariesFinal/Velocity_ghost/vel'

# Initialising Simulation Parameters
gamma = 7  # Adiabatic Index
rho0 = 1000  # Reference Density
nu = 10  # Dynamic Viscosity
g = 9.81  # Gravitational acceleration
g_acc = np.array([0.0, -9.81])
dim = 2  # Number of dimensions
c0 = 3971
c0 = 3900
B = (c0**2) * rho0 / gamma  # Tait EoS Coefficient

PlotEvery = 1
# plt.style.use("dark_background")
# Solid Boundary Initialisation
image_path = directory + str(int(800)) + '.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Select the largest contour, assuming there's only one shape in the image
contour = max(contours, key=cv2.contourArea)
contour = contour[:, 0, :]  # Remove extra dimension
contour[:, 1] = -contour[:, 1]
contour = contour.reshape(-1, 1, 2)

pos_solid1 = np.load(directory1 + str(int(1)) + '.npy')
pos_solid2 = np.load(directory2 + str(int(1)) + '.npy')
pos_solid = np.row_stack((pos_solid1, pos_solid2))

# Determine the bounds of the shape
x_min, y_min = np.min(pos_solid1, axis=0)
x_max, y_max = np.max(pos_solid1, axis=0)

# Create grid points within the bounds
spacing = 1.5
grid_points = sph_functions.create_grid_points(pos_solid1, spacing)
inside_points = sph_functions.filter_points_inside_shape(grid_points, contour)
pos_fluid = sph_functions.remove_points_too_close(inside_points, pos_solid1, 1.0)

# Rotate and cut inside
ycut = -480
centroid = sph_functions.find_centroid(pos_solid1)
rotated_inside = sph_functions.rotate(pos_fluid, 32, centroid)
remaining_inside, cut_inside = sph_functions.cut_bottom_part(rotated_inside, -550)
pos_fluid = sph_functions.rotate(remaining_inside, -32, centroid)

vel_fluid = np.zeros_like(pos_fluid)
vel_half = np.zeros_like(pos_fluid)
n_fluid = int(len(pos_fluid))
print(n_fluid)

h = float(1.3 * spacing)  # Smoothing length
m = float((spacing**2) * rho0)
pos = np.row_stack((pos_fluid, pos_solid))
vel = np.zeros_like(pos)

# plt.scatter(pos_fluid[:, 0], pos_fluid[:, 1], s=1)
# plt.scatter(pos_solid[:, 0], pos_solid[:, 1], s=1)
# plt.show()

# Time stepping
Nt = 1599
dt = 0.00017857
dim = 2
CFL = 0.25 * h / c0
print(dt, h, c0, CFL)

rho_fluid = np.ones((n_fluid, 1)) * 1000
rho_half = np.zeros_like(rho_fluid)

zero_array = np.ones(55)
linspacing = np.linspace(1, 1599, 1599)
timesteps = np.hstack((linspacing[0], zero_array, linspacing[1:]))
# for iter in tqdm(range(Nt + 1)):
for iter in tqdm(timesteps):
    pos_solid1 = np.load(directory1 + str(int(iter)) + '.npy')
    pos_solid2 = np.load(directory2 + str(int(iter)) + '.npy')
    pos_solid = np.row_stack((pos_solid1, pos_solid2))
    pos = np.row_stack((pos_fluid, pos_solid))
    n_solid = int(len(pos_solid))
    n_particles = int(len(pos))
    vel_solid1 = np.load(directory_vel1 + str(int(iter)) + '.npy')
    vel_solid2 = np.load(directory_vel2 + str(int(iter)) + '.npy')
    vel_solid = np.row_stack((vel_solid1, vel_solid2))
    vel = np.row_stack((vel_fluid, vel_solid))

    nan_indices = np.argwhere(np.isnan(pos))

    if len(nan_indices) > 0:
        print("NaN values found at the following indices:")
        for index in nan_indices:
            print(f"Row: {index[0]}, Column: {index[1]}")
            print(rho[index], rho[index].shape)
            print(Pressures[index], Pressures[index].shape)
            print(forces[index], forces[index].shape)
            print(pos[index], pos[index.shape])

    # Nearest Neighbours search using KD Trees
    NeighbourID, Distances = neighbors.KDTree(pos).query_radius(pos, 3*h, return_distance=True, sort_results=True)
    # For each position, calculate nearest neighbours within certain radius

    rho = np.zeros((n_particles, 1))
    # Calculating densities, iterating through each particle
    for i in range(n_particles):  # for all fluid and solid particles
        rho[i] = sph_functions.LoopDensity(NeighbourID[i], pos, pos[i], h, dim, rho[i], m)

    # Calculating pressures
    Pressures = sph_functions.TaitEOSHGCorrection(rho, rho0, gamma, B, n_fluid)

    # Force Computation
    forces = np.zeros_like(pos_fluid)
    dy = np.array([spacing, 0.0])
    for i in range(n_fluid):  # Iterating through fluid particles
        forces[i] = sph_functions.LoopForces(NeighbourID[i], pos, pos[i], h, dim, dy, forces[i], m, Pressures,
                                             Pressures[i], rho, rho[i], rho0, c0, vel, vel[i], gamma, g_acc, nu)
    '''
    # Looping for artificial pressure
    for i in range(n_fluid):
        forces[i] = sph_functions.LoopArtificialPressure(NeighbourID[i], pos, pos[i], h, dim, dy, forces[i], m,
                                                         Pressures, Pressures[i], rho, rho[i])
    '''

    # Find the rows containing NaN values
    nan_rows = np.any(np.isnan(forces), axis=1)

    # Create a new array without the NaN rows
    # forces = forces[~nan_rows]

    # X-SPH Correction term
    XSPH = np.zeros((n_fluid, 2))
    for i in range(n_fluid):  # Iterating through each particle
        XSPH[i] = sph_functions.LoopXSPH(NeighbourID[i], pos, pos[i], vel, vel[i], rho, rho[i], h, dim, m, XSPH[i])

    # Leapfrog integration
    vel_fluid, vel_half, vel, pos_fluid, pos = sph_functions.LeapfrogIntegration(iter, vel_half, vel_fluid, vel, forces,
                                                                                 dt, n_fluid, XSPH, pos_fluid, pos)

    # Visualisation
    directory3 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/SwallowingFiles/Quintic_1d5_1d2_alpha5/position/'
    directory4 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/SwallowingFiles/Quintic_1d5_1d2_alpha5/velocity/'
    directory5 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/SwallowingFiles/Quintic_1d5_1d2_alpha5/acceleration/'
    directory6 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/SwallowingFiles/Quintic_1d5_1d2_alpha5/pressure/'
    directory7 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/SwallowingFiles/Quintic_1d5_1d2_alpha5/density/'
    '''
    np.save(directory3 + 'pos' + str(int(iter)) + '.npy', pos)
    np.save(directory4 + 'vel' + str(int(iter)) + '.npy', vel)
    np.save(directory5 + 'accel' + str(int(iter)) + '.npy', forces)
    np.save(directory6 + 'pressure' + str(int(iter)) + '.npy', Pressures)
    np.save(directory7 + 'density' + str(int(iter)) + '.npy', rho)
    '''
    # Visualisation
    plt.scatter(pos[0:n_fluid, 0], pos[0:n_fluid, 1], s=3)
    plt.scatter(pos[n_fluid:n_particles, 0], pos[n_fluid:n_particles, 1], s=3, c='lightcoral')
    # plt.scatter(posnan[:, 0], posnan[:, 1], s=3, c='red')
    # plt.axis('equal')
    plt.xlim(175, 500)
    plt.ylim(-250, -680)
    plt.xlabel('Width (Pixels)')
    plt.ylabel('Height (Pixels)')
    plt.gca().invert_yaxis()
    plt.show()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
