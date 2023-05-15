import numpy as np
from scipy.stats.qmc import Halton, LatinHypercube

"""
def Halton2D(m, ubound):
    '''
    Generates a random numbers using the Halton sequence generator
    :param n: Total number of particles
    :param ubound: Edge boundaries of mesh
    :return: n x 2 matrix of positions of particles [x, y]
    '''

    n = int(np.floor(m/4))

    sampler = Halton(d=2)  # Creating Halton sampler in 2D - generates numbers between 0 and 1
    sample_q1 = sampler.random(n=n) * ubound  # Scaling points in first quadrant by 5 (5 is upper bound)
    sample_q3 = sampler.random(n=n) * -ubound  # Scaling points in third quadrant by 5 (5 is upper bound)

    # Scaling points in second quadrant by 5 (5 is upper bound)
    sample_q2 = sampler.random(n=n)
    sample_x = sample_q2[:, 0] * -ubound
    sample_y = sample_q2[:, 1] * ubound
    sample_q2 = np.column_stack((sample_x, sample_y))

    # Scaling points in fourth quadrant by 5 (5 is upper bound)
    sample_q4 = sampler.random(n=n)
    sample_x = sample_q4[:, 0] * ubound
    sample_y = sample_q4[:, 1] * -ubound
    sample_q4 = np.column_stack((sample_x, sample_y))

    # Stack all four matrices together
    sample = np.row_stack((sample_q1, sample_q2, sample_q3, sample_q4))
    return sample
"""

def Halton2D(m, ubound):
    '''
    Generates a random numbers using the Halton sequence generator
    :param n: Total number of particles
    :param ubound: Edge boundaries of mesh
    :return: n x 2 matrix of positions of particles [x, y]
    '''

    n = int(m)

    sampler = Halton(d=2)  # Creating Halton sampler in 2D - generates numbers between 0 and 1
    sample = sampler.random(n=n) * ubound  # Scaling points in first quadrant by 5 (5 is upper bound)
    return sample


def Halton3D(n, ubound):
    '''
    Generates a random numbers using the Halton sequence generator
    :param n: Total number of particles
    :param ubound: Edge boundaries of mesh
    :return: n x 2 matrix of positions of particles [x, y]
    '''

    n = int(n)

    sampler = Halton(d=3)  # Creating Halton sampler in 3D - generates numbers between 0 and 1
    sample = sampler.random(n=n) * ubound  # Scaling points in first quadrant by n (n is upper bound)

    return sample


def Container2D(l, r, n_b, b, t, n_w, dx):
    '''
    Generates a container based on the dimensions of the container and the number of particles per side
    :param l: X-coordinate of left boundary
    :param r: X-coordinate of right boundary
    :param n_b: number of particles along top and bottom
    :param b: Y-coordinate of base
    :param t: Y-coordinate of top
    :param n_w: number of particles along the left and right boundaries
    '''

    base = np.zeros((n_b, 2))
    left = np.zeros((n_w - 1, 2))
    right = np.zeros_like(left)

    base[:, 0] = np.linspace(l, r, n_b)
    base[:, 1] = b

    left[:, 1] = np.linspace(b + dx, t, n_w-1)
    left[:, 0] = l

    top = np.zeros_like(base)
    top[:, 0] = base[:, 0]
    top[:, 1] = t

    right[:, 1] = left[:, 1]
    right[:, 0] = r

    container = np.row_stack((base, right, left))

    return container


def tank2D(l, r, n_b, b, t, n_w, dx):
    '''
    Generates a container based on the dimensions of the container and the number of particles per side
    :param l: X-coordinate of left boundary
    :param r: X-coordinate of right boundary
    :param n_b: number of particles along top and bottom
    :param b: Y-coordinate of base
    :param t: Y-coordinate of top
    :param n_w: number of particles along the left and right boundaries
    '''

    base = np.zeros((n_b, 2))
    left = np.zeros((n_w - 1, 2))
    right = np.zeros_like(left)

    base[:, 0] = np.linspace(l, r, n_b)
    base[:, 1] = b

    left[:, 1] = np.linspace(b + dx, t, n_w-1)
    left[:, 0] = l

    top = np.zeros_like(base)
    top[:, 0] = base[:, 0]
    top[:, 1] = t

    right[:, 1] = left[:, 1]
    right[:, 0] = r

    container = np.row_stack((base, top, right, left))

    return container


def ClosesttoBoundary(pos, targetx):
    x_differences = np.abs(pos[:, 0] - targetx)
    closest_index = np.argmin(x_differences)
    return closest_index


def getDivergence(m, rho, ui, uj, vi, vj, wx, wy):
    '''
    Getting the divergence of a function
    :param m: mass of a particle
    :param rho: density of a particle
    :param uj: u component of velocity of jth particle
    :param vj: v component of velocity of jth particle
    :param wx: x gradient of kernel
    :param wy: y gradient of kernel
    :return: Divergence of function
    '''

    dx = (m/rho) * (uj - ui) * wx
    dy = (m/rho) * (vi - vj) * wy
    div = dx + dy

    return div


def getDensity(m, kernel):
    '''
    Get density at sampling locations from SPH Particle Distributions for 2D problem
    :param m: Particle mass (constant)
    :param kernel: Chosen kernel from Kernel.py
    :return rho: Density of particle
    '''

    rho = m * kernel
    return rho


def getContinuity(m, vi, vj, wx, wy):

    v = vi - vj
    w = np.hstack((wx, wy))
    d_rho = m * np.dot(v, w)

    return d_rho


def getContinuityAlternative(m, rhoi, rhoj, vi, vj, wx, wy):

    v = vi - vj
    w = np.hstack((wx, wy))
    d_rho = rhoi * (m / rhoj) * np.dot(v, w)

    return d_rho


def TaitEOS(rho, rhob, gamma, B, n_fluid):
    """
    Returns the pressure of a particle based on the Tait EOS
    :param rho: density of that particle
    :param rhob: Initial density of that particle
    :param gamma: adiabatic exponent = 7
    :param B: Tait Coefficient
    :return: P: Pressure
    """

    n = len(rho)
    Pressures = np.zeros_like(rho)

    for i in range(n):
        Pressures[i] = B * ((rho[i] / rhob) ** gamma - 1)

    return Pressures


def TaitEOSHGCorrection(rho, rhob, gamma, B, n_fluid):

    n = len(rho)

    for i in range(n):
        if i < n_fluid:
            continue
        elif i >= n_fluid:
            if rho[i] >= rhob:
                continue
            elif rho[i] < rhob:
                rho[i] = rhob

    Pressures = np.zeros_like(rho)

    for i in range(n):
        Pressures[i] = B * ((rho[i] / rhob) ** gamma - 1)

    return Pressures


def getPressure(k, rhoi, rho0):
    """
    Returns pressure based on equation o
    :param k:
    :param rho:
    :param rho0:
    :return:
    """
    pi = k * (rhoi - rho0)
    return pi


def ArtificialViscosity(m, alpha, beta, c0, h, rhoi, rhoj, rho0, vi, vj, posi, posj, epsilon, gamma, dim, wx, wy):
    v = vi - vj
    r = posi - posj
    rho = (rhoi + rhoj) / 2
    r_norm = np.linalg.norm(r)

    if dim == 1:
        vdotr = v[0] * r[0]
    elif dim == 2:
        vdotr = v[0] * r[0] + v[1] * r[1]
    elif dim == 3:
        vdotr = v[0] * r[0] + v[1] * r[1] + v[2] * r[2]
    else:
        print('invalid dimension')

    ci = c0 * (rhoi / rho0) ** ((gamma - 1) / 2)
    cj = c0 * (rhoj / rho0) ** ((gamma - 1) / 2)
    c = (ci + cj) / 2

    nu = h * vdotr / ((r_norm**2) + epsilon * (h**2))
    visc = -(alpha * c * nu)/rho + (beta * nu**2) / rho

    if vdotr < 0:
        visc_x = -m * visc * wx
        visc_y = -m * visc * wy
    elif vdotr >= 0:
        visc_x = 0
        visc_y = 0

    viscos = np.hstack((visc_x, visc_y))
    return viscos


def getPressureAcc(m, pi, pj, rhoi, rhoj, wx, wy):
    """
    Get accelerations due to pressure at sampling locations from SPH Particle Distributions
    :param m: Particle mass (constant)
    :param pi and pj: Pressures of ith and jth particles
    :param rhoi and rhoj: Densities of ith and jth particles
    :param wx, wy: Gradients of the smoothing kernel
    :param g: Gravitational acceleration
    """

    ax = -m * ((pi / (rhoi**2)) + (pj / (rhoj**2))) * wx
    ay = -m * ((pi / (rhoi**2)) + (pj / (rhoj**2))) * wy

    a = np.hstack((ax, ay))
    return a


def getPressureAcc_WCSPH(m, pi, pj, rhoi, rhoj, wx, wy):
    '''
    Get accelerations due to pressure at sampling locations from SPH Particle positions
    :param m: Particle mass (constant)
    :param pi and pj: Pressures of ith and jth particles
    :param rhoi and rhoj: Densities of ith and jth particles
    :param wx, wy and wz: Gradients of the smoothing kernel
    :param g: Gravitational acceleration
    '''

    ax = -m * ((pi / (rhoi**2)) + (pj / (rhoj**2))) * wx
    ay = -m * ((pi / (rhoi**2)) + (pj / (rhoj**2))) * wy
    # az = -m * ((pi / rhoi**2) + (pj / rhoj**2)) * wz

    a = np.hstack((ax, ay))
    return a


def getViscosityAcc(m, h, rhoi, rhoj, nu, posi, posj, vi, vj, wx, wy):
    """
    Get viscosity acceleration
    :param m: Particle mass
    :param h: Smoothing length
    :param rhoi: Density if ith particle
    :param rhoj: Density of jth particle
    :param nu: Dynamic viscosity
    :param posi: Position of ith particle
    :param posj: Position of jth particle
    :param vi: Velocity of ith particle
    :param vj: Velocity of jth particle
    :param wx: gradient of kernel wrt x
    :param wy: gradient of kernel wrt y
    """

    r = posi - posj
    v = vi - vj
    r_norm = np.linalg.norm(r)
    w = np.hstack((wx, wy))

    norm = (r_norm**2) + (0.01*h*h)
    norm1 = m * (2 * nu) * np.dot(r, w) / (rhoi * rhoj * norm)

    visc = norm1 * v

    return visc


def getPressureForce(r, m, pi, pj, posi, posj, rhoj, gradkernel):
    '''
    Get pressure at sampling locations from SPH Particle Distributions
    :param dx: x distance
    :param dy: y distance
    :param m: Particle mass (constant)
    :param h: Smoothing length
    :param pi and pj: Pressures of ith and jth particles
    '''

    norm = gradkernel * m
    r_norm = np.linalg.norm(r)
    P = norm * (-(posj-posi) / r_norm) * (pj+pi) / (2 * rhoj)

    return P


def getViscosityForce(m, vi, vj, rhoj, nu, gradgradvisc):
    '''
    Get pressure at sampling locations from SPH Particle Distributions
    :param m: Particle mass (constant)
    :param vi, vj: velocities of particles i and j
    :param rhoj: density of particle j
    '''

    visc = gradgradvisc * nu * m * (vj - vi)/rhoj

    return visc


def getPressureForce_New(dx, dy, m, pi, pj, posi, posj, rhoj, wx, wy):
    '''
    Get pressure at sampling locations from SPH Particle Distributions
    :param dx: x distance
    :param dy: y distance
    :param m: Particle mass (constant)
    :param pi and pj: Pressures of ith and jth particles
    '''

    norm_x = wx * m
    norm_y = wy * m

    r = np.sqrt(dx ** 2 + dy ** 2)
    P_x = norm_x * (-(posj[0]-posi[0]) / r) * (pj+pi) / (2 * rhoj)
    P_y = norm_y * (-(posj[1]-posi[1]) / r) * (pj+pi) / (2 * rhoj)
    P = np.hstack((P_x, P_y))

    return P


def ArtificialPressure(m, pi, pj, wr, wdx, wx, wy, n, eps, rhoi, rhoj):

    # eps ~ 0.2
    fij = wr / wdx
    f = fij**n

    if pi < 0:
        Ri = eps * abs(pi) / (rhoi**2)
    elif pi >= 0:
        Ri = 0

    if pj < 0:
        Rj = eps * abs(pj) / (rhoj**2)
    elif pj >= 0:
        Rj = 0

    R = Ri + Rj

    if pi >= 0 and pj >= 0:
        R = 0.01 * ((pi / (rhoi**2)) + (pj / (rhoj**2)))
    else:
        pass

    ti_x = -m * (R*f) * wx
    ti_y = -m * (R*f) * wy

    ti = np.hstack((ti_x, ti_y))

    return ti




