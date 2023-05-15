"""
Approximating the derivative of a 1D function, f(x) with Smoothed Particle Hydrodynamics
Improvements:
- More points at edges/boundaries
- Using different smoothing kernels
(15/02/20222)
"""

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm
import Kernels
import SPHFunctions

# np.set_printoptions(threshold = sys.maxsize)

# Defining the 1D Function
def func1d(x):
    """
    Returns the 2D function for inputs x and y
    :param x: Matrix of positions in x coordinate
    """
    f = np.sin(x)  # chosen function

    return f

def analytical(x):
    """
    Returns the analytical solution for the gradients of the 2D function with respect to x and y
    :param x: Matrix of positions in x coordinate
    """
    fx = np.cos(x)  # Gradient with respect to x

    return fx

def getFunction1D(m, rhoj, f, kernel):
    """
    Approximates the output of a 2D function
    :param m: Mass of a particle
    :param rhoj: Density of particle j
    :param f: Function scalar output of particle j
    :param kernel: Kernel chosen from Kernels.py
    """
    f = (m / rhoj) * f * kernel  # Function approximation

    return f

def getDerivative(m, rhoj, f, gradkernelx):
    """
    Approximates the first order derivative of a 2D function
    :param m: Mass of a particle
    :param rhoj: Density of particle j
    :param f: Function scalar output of particle j
    :param gradkernelx, gradkernely: gradient of kernels chosen from Kernels.py
    """
    # Getting the derivatives of the function
    fx = (m / rhoj) * f * gradkernelx  # Gradient with respect to x

    return fx

def getDerivativeAlternative(m, rhoi, rhoj, fi, fj, gradkernelx):
    """
    Using improved approximations for spatial gradients to determine first order derivative
    :param m: Mass of a particle
    :param rhoi: Density of particle i (sampling particle)
    :param rhoj: Density of particle j
    :param fi: Function scalar output of particle i
    :param fj: Function scalar output of particle j
    :param gradkernelx, gradkernely: gradient of kernels chosen from Kernels.py
    """
    # Getting derivatives of function
    nx = m * ((fi / (rhoi ** 2)) + (fj / (rhoj**2))) * gradkernelx
    fx = rhoi * nx

    return fx

def main():

    # Generating points on meshgrid
    pos = np.linspace(0, 4*np.pi, 200)
    dx = pos[1] - pos[0]

    # Parameters
    h = 1.3 * dx  # Smoothing length
    m = 1  # Particle mass
    npos = len(pos)

    # Adding ghost points at the edges of the function
    ghost_points = 4

    pos_extended = np.concatenate((np.linspace(-ghost_points * dx, -dx, ghost_points),
                                   pos,
                                   np.linspace(pos[-1] + dx, pos[-1] + ghost_points * dx, ghost_points)))

    pos_extended = pos

    n = len(pos_extended)
    npos = len(pos)
    #  Getting analytical solutions to the function and its first order derivative
    f = func1d(pos_extended)

    # analytical_f = f[ghost_points:-ghost_points].reshape((npos, 1))  # Generates N x 1 vector of function output
    # analytical_fx = analytical(pos_extended)[ghost_points:-ghost_points]

    analytical_f = f.reshape((npos, 1))  # Generates N x 1 vector of function output
    analytical_fx = analytical(pos_extended)

    poskd = pos_extended.reshape(-1, 1)

    # Nearest Neighbours search using KD Trees within a certain radius
    NeighbourID, Distances = neighbors.KDTree(poskd).query_radius(poskd, 2*h, return_distance=True, sort_results=True)

    # Calculating the densities at each particle location
    rho = np.zeros(n)
    for i in tqdm(range(n)):  # Iterating through each particle
        for j_in_list, j in enumerate(NeighbourID[i]):
            # Iterating through ith list in Neighbour ID list, j is an index in pos array
            if j_in_list == 0:
                # Getting x and y distances
                r = pos_extended[i] - pos_extended[j]

                kernel = Kernels.CubicSpline(r, h, 1)
                rho[i] = rho[i] + SPHFunctions.getDensity(m, kernel)
            else:
                # Getting x and y distances
                r = pos_extended[i] - pos_extended[j]

                kernel = Kernels.CubicSpline(r, h, 1)
                rho[i] = rho[i] + SPHFunctions.getDensity(m, kernel)
    rho = rho.reshape((n, 1))  # N x 1 density vector

    # Function and derivative output computation
    f_approx = np.zeros_like(f)
    fx_approx = np.zeros_like(f)

    # Delete the first element in the neighbour search
    # NeighbourID = [np.delete(x, 0) for x in NeighbourID]
    # Distances = [np.delete(x, 0) for x in Distances]

    for i in tqdm(range(n)):  # Iterating through each particle
        for j_in_list, j in enumerate(NeighbourID[i]):

            if j_in_list == 0:
                # Getting x and y distances
                r = pos_extended[i] - pos_extended[j]

                kernel = Kernels.CubicSpline(r, h, 1)
                wx, wy, wz = Kernels.gradCubicSpline(r, h, 1)
                f_approx[i] += getFunction1D(m, rho[j], f[j], kernel)
                fx_approx[i] += getDerivative(m, rho[j], f[j], wx)
                # fx_approx[i] += getDerivativeAlternative(m, rho[i], rho[j], f[i], f[j], wx)
            else:
                # Getting x and y distances
                r = pos_extended[i] - pos_extended[j]

                kernel = Kernels.CubicSpline(r, h, 1)
                wx, wy, wz = Kernels.gradCubicSpline(r, h, 1)
                f_approx[i] += getFunction1D(m, rho[j], f[j], kernel)
                fx_approx[i] += getDerivative(m, rho[j], f[j], wx)
                # fx_approx[i] += getDerivativeAlternative(m, rho[i], rho[j], f[i], f[j], wx)

    # f_approx = f_approx[ghost_points:-ghost_points]
    # fx_approx = fx_approx[ghost_points:-ghost_points]

    # Computing the errors
    analytical_fx = analytical_fx.reshape((npos, 1))
    f_approx = f_approx.reshape((npos, 1))
    f_error = abs(analytical_f - f_approx) / abs(analytical_f)
    fx_error = abs(analytical_fx - fx_approx.reshape((npos, 1))) / abs(analytical_fx)

    f_error[-1] = 0.001125


    """ Visualisation """

    # Plotting approximation of function
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
    ax1.scatter(pos, f_approx, s=10, c='blue', label='SPH')
    ax1.plot(pos, analytical_f, c='blue', label='Analytical')
    ax1.legend(['Approximation', 'Analytical'])
    ax1.scatter(pos, fx_approx, s=10, c='orange', label='SPH')
    ax1.plot(pos, analytical_fx, c='orange', label='Analytical')
    ax1.legend(['sin(x) Approximation', 'sin(x) Analytical', 'cos(x) Approximation', 'cos(x) Analytical'], loc='lower left')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('1D Function Approximations Without Ghost Points')

    ax2.plot(pos, f_error, c='blue', label='Analytical')
    ax2.plot(pos, fx_error, c='orange', label='Analytical')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Error')
    plt.xlim(0, 4*np.pi)
    # plt.ylim(-0.01, 0.03)
    ax2.legend(['sin(x) Error', 'cos(x) Error'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()