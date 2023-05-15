"""
Sloshing Case (2D)
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm
import sph_functions
import sys
import matplotlib
from numba import jit, njit

matplotlib.use('TkAgg')

np.set_printoptions(threshold=sys.maxsize)

""" Initial Simulation Parameters """
DomainWidth = 1  # Container width
DomainHeight = 0.5  # Container height
WaterWidth = 1  # Water width
WaterHeight = 0.125  # Water height

gamma = 7.0  # Adiabatic index
rho0 = 1000  # Water base density
nu = 1  # Dynamic viscosity
g = 9.81
g_acc = np.array([0, -9.81])
dim = 2
c0 = 10 * np.sqrt(2 * g * WaterHeight)
B = (c0**2) * rho0 / gamma

PlotEvery = 10  # Plot every n iterations
plt.style.use("dark_background")

# Fluid Particle Initialisation
base = np.linspace(0, WaterWidth, 161)
height = np.linspace(0, WaterHeight, 21)
bb, hh = np.meshgrid(base, height)
bb = np.ravel(bb)
hh = np.ravel(hh)
pos_fluid = np.column_stack((bb, hh))
vel_fluid = np.zeros_like(pos_fluid)
vel_half = np.zeros_like(pos_fluid)

dx = pos_fluid[1, 0] - pos_fluid[0, 0]  # Particle spacing
r = dx / 2  # Particle radius
hdx = 1.3
h = float(hdx * dx)
m_V = np.pi * r ** 2
m = float(dx**2 * rho0)

# Solid Boundary Initialisation
pos_solid1 = sph_functions.tank2D(-dx, DomainWidth + dx, 163, -dx, DomainHeight, 82, dx)
pos_solid2 = sph_functions.tank2D(-2 * dx, DomainWidth + 2 * dx, 165, -2 * dx, DomainHeight + dx, 84, dx)
pos_solid3 = sph_functions.tank2D(-3 * dx, DomainWidth + 3 * dx, 167, -3 * dx, DomainHeight + 2 * dx, 86, dx)
pos_solid4 = sph_functions.tank2D(-4 * dx, DomainWidth + 4 * dx, 169, -4 * dx, DomainHeight + 3 * dx, 88, dx)
pos_solid = np.row_stack((pos_solid1, pos_solid2, pos_solid3, pos_solid4))
vel_solid = np.zeros_like(pos_solid)

pos = np.row_stack((pos_fluid, pos_solid))
vel = np.row_stack((vel_fluid, vel_solid))

# Time stepping
Nt = 50000  # Total time steps
dt = 0.25 * h / c0  # Time step period
print(dt)

T = 1.6  # Period of container
t = 0.0

# Initialising velocities for boundaries - Asin(w)
A = 0.03  # Amplitude based on Lugni et al (2006)
w = 2 * np.pi * (t / T)
x = A * np.sin(w)  # Sinusoidal motion of wall particles
prev_x = 0.0

n_fluid = len(pos_fluid)
n_particles = int(len(pos))
n_solid = len(pos_solid)
print(n_fluid, n_particles)


for iter in tqdm(range(Nt)):
    time = dt * iter
    w = 2 * np.pi * (t/T)
    x = A * np.sin(w)
    delta_x = x - prev_x
    prev_x = x

    array_x = np.zeros((n_solid))
    array_x[:] = delta_x
    pos_solid[:, 0] += array_x
    pos[n_fluid: n_particles] = pos_solid
    t += dt

    # Nearest Neighbours search using KD Trees
    NeighbourID, Distances = neighbors.KDTree(pos).query_radius(pos, 2*h, return_distance=True, sort_results=True)
    # For each position, calculate nearest neighbours within certain radius

    rho = np.zeros((n_particles, 1))

    # Calculating densities, iterating through each particle
    for i in range(n_particles):  # for all fluid and solid particles
        rho[i] = sph_functions.LoopDensity(NeighbourID[i], pos, pos[i], h, dim, rho[i], m)

    # Calculating pressures
    Pressures = sph_functions.TaitEOSHGCorrection(rho, rho0, gamma, B, n_fluid)

    # Force Computation
    forces = np.zeros_like(pos_fluid)
    for i in range(n_fluid):  # Iterating through fluid particles
        forces[i] = sph_functions.LoopForces(NeighbourID[i], pos, pos[i], h, dim, dx, forces[i], m, Pressures,
                                             Pressures[i], rho, rho[i], rho0, c0, vel, vel[i], gamma, g_acc, nu)

    # X-SPH Correction term
    XSPH = np.zeros((n_fluid, 2))
    for i in range(n_fluid):  # Iterating through each particle
        XSPH[i] = sph_functions.LoopXSPH(NeighbourID[i], pos, pos[i], vel, vel[i], rho, rho[i], h, dim, m, XSPH[i])

    """ Integration step """
    # Leapfrog integration scheme
    vel_fluid, vel_half, vel, pos_fluid, pos = sph_functions.LeapfrogIntegration(iter, vel_half, vel_fluid, vel, forces,
                                                                                 dt, n_fluid, XSPH, pos_fluid, pos)

    """ Visualisation """
    # directory = 'D:/Things/Programming/IRP/WCSPH/npyfiles/sloshing_cubickoschier_alp1_xsph/'
    directory = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Sloshing/Sloshing_Cubic_alph1_dx0022_xsph/posfiles/'
    directory1 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Sloshing/Sloshing_Cubic_alph1_dx0022_xsph/density/'
    directory2 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Sloshing/Sloshing_Cubic_alph1_dx0022_xsph/pressure/'
    directory3 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Sloshing/Sloshing_Cubic_alph1_dx0022_xsph/acceleration/'
    directory4 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Sloshing/Sloshing_Cubic_alph1_dx0022_xsph/velocity/'

    plt.style.use("dark_background")
    if iter % PlotEvery == 0:

        np.save(directory + 'pos' + str(iter) + '.npy', pos)
        np.save(directory1 + 'dens' + str(iter) + '.npy', rho)
        np.save(directory2 + 'pressure' + str(iter) + '.npy', Pressures)
        np.save(directory3 + 'accel' + str(iter) + '.npy', forces)
        np.save(directory4 + 'vel' + str(iter) + '.npy', vel_fluid)

        plt.scatter(pos[0:n_fluid, 0], pos[0:n_fluid, 1], s=6)
        plt.scatter(pos[n_fluid:n_particles, 0], pos[n_fluid:n_particles, 1], s=6, color = 'lightcoral')
        plt.title(f't = {round(time, 3)}')
        plt.ylim(-0.09, 0.55)
        # plt.xlim(-1, 1.5)
        # plt.xticks([], [])
        # plt.yticks([], [])
        plt.tight_layout()
        plt.draw()
        plt.pause(0.000001)
        plt.clf()
