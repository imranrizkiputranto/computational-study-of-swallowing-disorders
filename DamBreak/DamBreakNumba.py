from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm
import sph_functions
import sys
import matplotlib
from numba import njit
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

""" Defining simulation parameters """
DomainWidth = 4  # Container width
DomainHeight = 4  # Container height
WaterWidth = 1  # Water column width
WaterHeight = 2  # Water column height

gamma = 7.0  # Adiabatic index
rho0 = 1000  # Water base density
nu = 1  # Dynamic viscosity
g = 9.81  # Gravitational acceleration
dim = 2  # Dimension
g_acc = np.array([0, -9.81])
c0 = 10 * np.sqrt(2 * g * WaterHeight)  # Speed of sound - 10 * vref
B = (c0**2) * rho0 / gamma  # Tait EoS Coefficient

PlotEvery = 20  # Plot every n iterations
plt.style.use("dark_background")

# Fluid Particle Initialisation
base = np.linspace(0, WaterWidth, 34)
height = np.linspace(0, WaterHeight, 67)
bb, hh = np.meshgrid(base, height)
bb = np.ravel(bb)
hh = np.ravel(hh)
pos_fluid = np.column_stack((bb, hh))
vel_fluid = np.zeros_like(pos_fluid)
vel_half = np.zeros_like(pos_fluid)

dx = pos_fluid[1, 0] - pos_fluid[0, 0]  # Particle spacing - currently 0.01
print(dx)

# Solid Boundary Initialisation
pos_solid1 = sph_functions.Container2D(-dx, DomainWidth, 135, -dx, DomainHeight, 135, dx)
pos_solid2 = sph_functions.Container2D(-2 * dx, DomainWidth + dx, 137, -2 * dx, DomainHeight, 136, dx)
pos_solid3 = sph_functions.Container2D(-3 * dx, DomainWidth + 2 * dx, 139, -3 * dx, DomainHeight, 137, dx)
pos_solid4 = sph_functions.Container2D(-4 * dx, DomainWidth + 3 * dx, 141, -4 * dx, DomainHeight, 138, dx)
pos_solid = np.row_stack((pos_solid1, pos_solid2, pos_solid3, pos_solid4))

# Stack fluid and solid particle positions
pos = np.row_stack((pos_fluid, pos_solid))
vel = np.zeros_like(pos)

r = dx / 2  # Particle radius
hdx = 1.3  # Smoothing length to particle spacing ratio
h = float(hdx * dx)  # Smoothing length
m_V = np.pi * r ** 2
# m = m_V * rho0
m = float(dx**2 * rho0)  # Particle mass

# Time stepping
dt = float(0.25 * h / c0)
print(dt)
Nt = 50000  # Total number of time steps

n_fluid = len(pos_fluid)
n_particles = int(len(pos))
print(n_particles, n_fluid)

rho = np.ones((n_particles, 1)) * rho0
rho_half = np.zeros_like(rho)

postimeindex = np.zeros((Nt, 3))
a = 0
time = 0

plt.scatter(pos[:, 0], pos[:, 1], s =9)
plt.show()

for iter in tqdm(range(Nt)):
    time = dt * iter
    # Nearest Neighbours search using KD Trees
    NeighbourID, Distances = neighbors.KDTree(pos).query_radius(pos, 2 * h, return_distance=True, sort_results=True)
    # For each position, calculate nearest neighbours within certain radius

    closest_particle_index = sph_functions.ClosesttoBoundary(pos_fluid, DomainWidth)
    postimeindex[a, 0] = pos[closest_particle_index, 0]
    postimeindex[a, 1] = pos[closest_particle_index, 1]
    postimeindex[a, 2] = dt * iter
    print(postimeindex[a])

    # Getting the densities at each particle location
    # rho = np.zeros(n_particles)
    drho = np.zeros_like(rho)

    # Update densities, iterating through each particle
    for i in range(n_particles):  # for all fluid and solid particles
        drho[i] = sph_functions.LoopContinuity(NeighbourID[i], pos, pos[i], h, dim, drho[i], m, vel, vel[i])

    rho_half, rho = sph_functions.UpdateDensity(n_particles, iter, rho_half, rho, drho, dt)

    # Calculating pressures
    Pressures = sph_functions.TaitEOSHGCorrection(rho, rho0, gamma, B, n_fluid)

    forces = np.zeros_like(pos_fluid)
    dy = np.array([dx, 0.0])
    for i in range(n_fluid):  # Iterating through fluid particles
        forces[i] = sph_functions.LoopForces(NeighbourID[i], pos, pos[i], h, dim, dy, forces[i], m, Pressures,
                                             Pressures[i], rho, rho[i], rho0, c0, vel, vel[i], gamma, g_acc, nu)

    '''
    # Looping for artificial pressure
    for i in range(n_fluid):
        forces[i] = sph_functions.LoopArtificialPressure(NeighbourID[i], pos, pos[i], h, dim, dy, forces[i], m,
                                                         Pressures, Pressures[i], rho, rho[i])
    '''

    # X-SPH Correction term
    XSPH = np.zeros((n_fluid, 2))

    for i in range(n_fluid):  # Iterating through each particle
        XSPH[i] = sph_functions.LoopXSPH(NeighbourID[i], pos, pos[i], vel, vel[i], rho, rho[i], h, dim, m, XSPH[i])

    # Leapfrog integration
    vel_fluid, vel_half, vel, pos_fluid, pos = sph_functions.LeapfrogIntegration(iter, vel_half, vel_fluid, vel, forces,
                                                                                 dt, n_fluid, XSPH, pos_fluid, pos)

    # Visualisation
    directory = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/posfiles/'
    directory1 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/surgefront/'
    directory2 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/density/'
    directory3 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/pressure/'
    directory4 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/acceleration/'
    directory5 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/DamBreak/DamBreak_Cubic_alph1_dx03_xsph/velocity/'

    # np.save(directory1 + 'postimeindex' + str(iter) + '.npy', postimeindex[a])

    if iter % PlotEvery == 0:
        # np.save(directory + 'pos' + str(iter) + '.npy', pos)
        # np.save(directory2 + 'density' + str(iter) + '.npy', rho)
        # np.save(directory3 + 'pressure' + str(iter) + '.npy', Pressures)
        # np.save(directory4 + 'accel' + str(iter) + '.npy', forces)
        # np.save(directory5 + 'vel' + str(iter) + '.npy', vel)

        plt.scatter(pos[0:n_fluid, 0], pos[0:n_fluid, 1], s=2)
        plt.scatter(pos[n_fluid:n_particles, 0], pos[n_fluid:n_particles, 1], s=2, color='lightcoral')
        plt.ylim(-0.5, DomainHeight + 0.5)
        plt.xlim(-0.5, DomainWidth + 0.5)
        plt.title(f't = {round(time, 3)}')
        # plt.xticks([], [])
        # plt.yticks([], [])
        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    a += 1


