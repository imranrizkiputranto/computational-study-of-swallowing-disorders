"""
Approximating the derivative of a 2D function, f(x,y) with Smoothed Particle Hydrodynamics, utilising a Halton sequence
number generator and KD tree for nearest neighbour search.
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

np.set_printoptions(threshold = sys.maxsize)

# Defining the 2D Function
def func2d(x, y):
    """
    Returns the 2D function for inputs x and y
    :param x: Matrix of positions in x coordinate
    :param y: Matrix of positions in y coordinate
    """
    f = np.sin(5 * x) * np.cos(3 * y)  # chosen function

    return f

def analytical(x, y):
    """
    Returns the analytical solution for the gradients of the 2D function with respect to x and y
    :param x: Matrix of positions in x coordinate
    :param y: Matrix of positions in y coordinate
    """
    fx = 5 * np.cos(5 * x) * np.cos(3 * y)  # Gradient with respect to x
    fy = -3 * np.sin(5 * x) * np.sin(3 * y)  # Gradient with respect to y

    return fx, fy

def getFunction2D(m, rhoj, f, kernel):
    """
    Approximates the output of a 2D function
    :param m: Mass of a particle
    :param rhoj: Density of particle j
    :param f: Function scalar output of particle j
    :param kernel: Kernel chosen from Kernels.py
    """
    f = (m / rhoj) * f * kernel  # Function approximation

    return f

def getDerivative(m, rhoj, f, gradkernelx, gradkernely):
    """
    Approximates the first order derivative of a 2D function
    :param m: Mass of a particle
    :param rhoj: Density of particle j
    :param f: Function scalar output of particle j
    :param gradkernelx, gradkernely: gradient of kernels chosen from Kernels.py
    """
    # Getting the derivatives of the function
    fx = (m / rhoj) * f * gradkernelx  # Gradient with respect to x
    fy = (m / rhoj) * f * gradkernely  # Gradient with respect to y
    f = np.column_stack((fx, fy))

    return f

def getDerivativeAlternative(m, rhoi, rhoj, fi, fj, gradkernelx, gradkernely):
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
    ny = m * ((fi / (rhoi ** 2)) + (fj / (rhoj**2))) * gradkernely

    fx = rhoj * nx
    fy = rhoj * ny
    f = np.column_stack((fx, fy))

    return f

def main():

    # Generating points on meshgrid with Halton sequence
    # pos = SPHFunctions.Halton2D(5000, 4)  # 1000 particles from 0 to 4 on x and y-axis
    x = np.linspace(0, 4, 100)
    y = np.linspace(0, 4, 100)
    xx, yy = np.meshgrid(x, y)
    xx = np.ravel(xx)
    yy = np.ravel(yy)
    pos = np.column_stack((xx, yy))
    dx = pos[1, 0] - pos[0, 0]
    print(dx)

    # Parameters
    h = 2 * dx  # Smoothing length
    m = 1  # Particle mass
    npos = len(pos)

    boundary1 = SPHFunctions.tank2D(-dx, 4 + dx, 100, -dx, 4 + dx, 100, dx)
    boundary2 = SPHFunctions.tank2D(-2 * dx, 4 + 2 * dx, 100, -2 * dx, 4 + 2 * dx, 100, dx)
    boundary3 = SPHFunctions.tank2D(-3 * dx, 4 + 3 * dx, 100, -3 * dx, 4 + 3 * dx, 100, dx)
    boundary4 = SPHFunctions.tank2D(-4 * dx, 4 + 4 * dx, 100, -4 * dx, 4 + 4 * dx, 100, dx)
    boundary5 = SPHFunctions.tank2D(-5 * dx, 4 + 5 * dx, 100, -5 * dx, 4 + 5 * dx, 100, dx)
    boundary = np.row_stack((boundary1, boundary2, boundary3))
    pos_extended = np.row_stack((pos, boundary))
    n = len(pos_extended)

    #  Getting analytical solutions to the function and its first order derivative
    f = func2d(pos_extended[:, 0], pos_extended[:, 1])
    analytical_f = f[0: npos].reshape((npos, 1))  # Generates N x 1 vector of function output
    analytical_fx, analytical_fy = analytical(pos_extended[:, 0], pos_extended[:, 1])

    # Nearest Neighbours search using KD Trees within a certain radius
    NeighbourID, Distances = neighbors.KDTree(pos_extended).query_radius(pos_extended, 2*h, return_distance=True,
                                                                         sort_results=True,)
    # Calculating the densities at each particle location
    rho = np.zeros(n)
    for i in tqdm(range(n)):  # Iterating through each particle
        for j_in_list, j in enumerate(NeighbourID[i]):
            # Iterating through ith list in Neighbour ID list, j is an index in pos array

            if j_in_list == 0:
                continue
            else:
                # Getting x and y distances
                r = pos_extended[i] - pos_extended[j]
                kernel = Kernels.CubicSpline(r, h, 2)
                rho[i] += SPHFunctions.getDensity(m, kernel)
    rho = rho.reshape((n, 1))  # N x 1 density vector

    # Function and derivative output computation
    f_approx = np.zeros_like(f)
    fx_approx = np.zeros_like(analytical_fx)
    fy_approx = np.zeros_like(analytical_fy)
    d_approx = np.column_stack((fx_approx, fy_approx))

    # Delete the first element in the neighbour search
    # NeighbourID = [np.delete(x, 0) for x in NeighbourID]
    # Distances = [np.delete(x, 0) for x in Distances]

    for i in tqdm(range(n)):  # Iterating through each particle
        for j_in_list, j in enumerate(NeighbourID[i]):
            # Getting x and y distances
            r = pos_extended[i] - pos_extended[j]

            if j_in_list == 0:
                continue
            else:
                kernel = Kernels.CubicSpline(r, h, 2)
                wx, wy, wz = Kernels.gradCubicSpline(r, h, 2)
                f_approx[i] = f_approx[i] + getFunction2D(m, rho[j], f[j], kernel)
                # d_approx[i] = d_approx[i] + getDerivative(m, rho[j], f[j], wx, wy)
                d_approx[i] = d_approx[i] + getDerivativeAlternative(m, rho[i], rho[j], f[i], f[j], wx, wy)

    # Computing the errors
    f_error = abs(analytical_f - f_approx[0: npos].reshape((npos, 1))) / abs(analytical_f)
    fx_error = abs(analytical_fx - d_approx[:, 0]) / abs(analytical_fx)
    fy_error = abs(analytical_fy - d_approx[:, 1]) / abs(analytical_fy)

    """ Visualisation """

    plt.figure()
    plt.scatter(pos_extended[:, 0], pos_extended[:, 1])

    # Plotting approximation of function
    plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(pos[:, 0], pos[:, 1], f_approx[0: npos], c=f_approx[0:npos], cmap='inferno')
    plt.title('Approximation of f(x, y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_zlim(-1, 1)

    # Plotting approximation of derivative with respect to x
    plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.scatter3D(pos[0: npos, 0], pos[0: npos, 1], d_approx[0: npos, 0], c=d_approx[0:npos, 0], cmap='inferno')
    plt.title('Approximation of derivative with respect to x')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_zlim(-6, 6)

    # Plotting approximation of derivative with respect to y
    plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.scatter3D(pos[0: npos, 0], pos[0: npos, 1], d_approx[0: npos, 1], c=d_approx[0:npos, 1], cmap='inferno')
    plt.title('Approximation of derivative with respect to y')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_zlim(-4, 4)

    # Plotting error for function approximation
    plt.figure()
    ax5 = plt.axes(projection='3d')
    ax5.scatter3D(pos[:, 0], pos[:, 1], f_error[0: npos], s=10, c='lightcoral')
    plt.title('Function approximation error')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')
    ax5.set_zlim(0.0, 0.050)

    # Plotting error for Derivative approximation wrt x
    plt.figure()
    ax6 = plt.axes(projection='3d')
    ax6.scatter3D(pos[:, 0], pos[:, 1], fx_error[0: npos], s=10, c='lightcoral')
    plt.title('Derivative error with respect to x')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlabel('z')
    ax6.set_zlim(0.0, 0.2)

    # Plotting error for Derivative approximation wrt y
    plt.figure()
    ax7 = plt.axes(projection='3d')
    ax7.scatter3D(pos[:, 0], pos[:, 1], fy_error[0: npos], c='lightcoral')
    plt.title('Derivative error with respect to y')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_zlabel('z')
    ax7.set_zlim(0.0, 0.15)

    plt.show()


if __name__ == '__main__':
    main()
