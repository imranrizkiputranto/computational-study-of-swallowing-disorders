import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import sys
from numba import njit, jit
import cv2

""" Kernels """

@njit
def Gaussian(r, h, dim):

    r_norm = np.linalg.norm(r)
    q = r_norm / h

    # Getting normalisation factor
    if dim == 1:
        norm = 1 / (np.sqrt(np.pi) * h)
    elif dim == 2:
        norm = 1 / (np.pi * (h**2))
    elif dim == 3:
        norm = 1 / (np.pi**(3/2) * (h**3))
    else:
        print('invalid dimension')

    # Getting kernel
    if q <= 3:
        w = norm * np.exp(- (q**2))
    elif q > 3:
        w = 0

    return w


@njit
def gradGaussian(r, h, dim):

    r_norm = np.linalg.norm(r)
    q = r_norm / h

    # Getting normalisation factor
    if dim == 1:
        norm = 1 / (np.sqrt(np.pi) * h)
        if q <= 3:
            wx = norm * np.exp(- (q ** 2)) * (-2 / (h**2)) * r[0]
            wy = 0
            wz = 0
        elif q > 3:
            wx = 0
            wy = 0
            wz = 0
    elif dim == 2:
        norm = 1 / (np.pi * (h**2))
        if q <= 3:
            wx = norm * np.exp(- (q ** 2)) * (-2 / (h**2)) * r[0]
            wy = norm * np.exp(- (q ** 2)) * (-2 / (h**2)) * r[1]
            wz = 0
        elif q > 3:
            wx = 0
            wy = 0
            wz = 0
    elif dim == 3:
        norm = 1 / (np.pi**(3/2) * (h**3))
        if q <= 3:
            wx = norm * np.exp(- (q ** 2)) * (-2 / (h**2)) * r[0]
            wy = norm * np.exp(- (q ** 2)) * (-2 / (h**2)) * r[1]
            wz = norm * np.exp(- (q ** 2)) * (-2 / (h**2)) * r[2]
        elif q > 3:
            wx = 0
            wy = 0
            wz = 0
    else:
        print('invalid dimension')

    return wx, wy, wz


@njit
def CubicSplineKoschier(r, h, dim):

    r_norm = np.linalg.norm(r)
    q = r_norm / h

    # Getting normalisation factor
    if dim == 1:
        norm = 4 / (3*h)
    elif dim == 2:
        norm = 40 / (7 * np.pi * (h**2))
    elif dim == 3:
        norm = 8 / (np.pi * (h**3))
    else:
        print('invalid dimension')

    # Getting kernel
    if 0 <= q <= 0.5:
        w = norm * (6 * ((q**3) - (q**2)) + 1)
    elif 0.5 < q <= 1:
        w = norm * 2 * (1-q)**3
    elif q > 1:
        w = 0

    return w

@njit
def gradCubicSplineKoschier(r, h, dim):

    r_norm = np.linalg.norm(r)  # Gets Euclidean distance between particles
    q = r_norm / h  # normalisation

    # Getting normalisation factor
    if dim == 1:
        norm = 4 / (3*h)
    elif dim == 2:
        norm = 40 / (7 * np.pi * (h**2))
    elif dim == 3:
        norm = 8 / (np.pi * (h**3))
    else:
        print('invalid dimension')

    # Getting norm * dw/dq = w_dash
    if 0 <= q <= 0.5:
        w_dash = norm * 6 * q * ((3*q) - 2)
    elif 0.5 < q <= 1:
        w_dash = -6 * norm * (1-q)**2
    elif q > 1:
        w_dash = 0

    if r_norm > 0.000000000001:
        tmp = w_dash / ((r_norm * h) + 0.00*h**2)
    else:
        tmp = 0

    if dim == 1:
        wx = tmp * r[0]
        wy = 0
        wz = 0
    elif dim == 2:
        wx = tmp * r[0]
        wy = tmp * r[1]
        wz = 0
    elif dim == 3:
        wx = tmp * r[0]
        wy = tmp * r[1]
        wz = tmp * r[2]

    return wx, wy, wz


@njit
def CubicSpline(r, h, dim):
    r_norm = np.linalg.norm(r)  # Gets Euclidean distance between particles
    # r_norm = np.sqrt(r[0]**2 + r[1]**2)
    q = r_norm / h  # normalisation

    # Getting normalisation factor
    if dim == 1:
        norm = 2 / (3*h)
    elif dim == 2:
        norm = 10 / (7 * np.pi * (h**2))
    elif dim == 3:
        norm = 1 / (np.pi * (h**3))
    else:
        print('invalid dimension')

    # Getting kernel
    if 0 <= q <= 1:
        w = norm * (1 - (3/2) * (q**2) * (1 - q/2))
    elif 1 < q <= 2:
        w = (norm / 4) * ((2 - q)**3)
    elif q > 2:
        w = 0

    return w

@njit
def gradCubicSpline(r, h, dim):
    r_norm = np.linalg.norm(r)  # Gets Euclidean distance between particles
    q = r_norm / h  # normalisation

    # Getting normalisation factor
    if dim == 1:
        norm = 2 / (3*h)
    elif dim == 2:
        norm = 10 / (7 * np.pi * (h**2))
    elif dim == 3:
        norm = 1 / (np.pi * (h**3))
    else:
        print('invalid dimension')

    # Getting norm * dw/dq = w_dash
    if 0 <= q <= 1:
        w_dash = norm * (-3 * q * (1 - 0.75*q))
    elif 1 < q <= 2:
        w_dash = (norm / 4) * (-3 * ((2 - q)**2))
    elif q > 2:
        w_dash = 0

    if r_norm > 0.000000000001:
        tmp = w_dash / ((r_norm * h) + 0.00 * h ** 2)
    else:
        tmp = 0

    if dim == 1:
        wx = tmp * r[0]
        wy = 0.0
        wz = 0.0
    elif dim == 2:
        wx = tmp * r[0]
        wy = tmp * r[1]
        wz = 0.0
    elif dim == 3:
        wx = tmp * r[0]
        wy = tmp * r[1]
        wz = tmp * r[2]

    return wx, wy, wz

@njit
def QuinticSpline(r, h, dim):

    r_norm = np.linalg.norm(r)  # Gets Euclidean distance between particles
    q = r_norm / h  # normalisation

    # Getting normalisation factor
    if dim == 1:
        norm = 1 / (120*h)
    elif dim == 2:
        norm = 7 / (478 * np.pi * (h**2))
    elif dim == 3:
        norm = 1 / (120 * np.pi * (h**3))
    else:
        print('invalid dimension')

    # Getting kernel
    if 0 <= q <= 1:
        w = norm * (((3-q)**5) - (6 * (2-q)**5) + (15 * (1-q)**5))
    elif 1 < q <= 2:
        w = norm * (((3-q)**5) - (6 * (2-q)**5))
    elif 2 < q <= 3:
        w = norm * ((3-q)**5)
    elif q > 3:
        w = 0

    return w


@njit
def gradQuinticSpline(r, h, dim):

    r_norm = np.linalg.norm(r)  # Gets Euclidean distance between particles
    q = r_norm / h  # normalisation

    # Getting normalisation factor
    if dim == 1:
        norm = 1 / (120*h)
    elif dim == 2:
        norm = 7 / (478 * np.pi * (h**2))
    elif dim == 3:
        norm = 1 / (120 * np.pi * (h**3))
    else:
        print('invalid dimension')

    # Getting norm * dw/dq = w_dash
    if 0 <= q <= 1:
        w_dash = norm * (-5 * (3-q)**4 + 30 * (2-q)**4 - 75 * (1-q)**4)
    elif 1 < q <= 2:
        w_dash = norm * (-5 * (3-q)**4 + 30 * (2-q)**4)
    elif 2 < q <= 3:
        w_dash = norm * (-5 * (3-q)**4)
    elif q > 3:
        w_dash = 0

    if r_norm > 0.000000000001:
        tmp = w_dash / ((r_norm * h) + 0.00 * h ** 2)
    else:
        tmp = 0

    if dim == 1:
        wx = tmp * r[0]
        wy = 0
        wz = 0
    elif dim == 2:
        wx = tmp * r[0]
        wy = tmp * r[1]
        wz = 0
    elif dim == 3:
        wx = tmp * r[0]
        wy = tmp * r[1]
        wz = tmp * r[2]

    return wx, wy, wz


@njit
def Poly6(r, h, dim):
    r_norm = np.linalg.norm(r)

    if dim == 2:
        norm = 4 / (np.pi * (h**8))
    elif dim == 3:
        norm = 315 / (64 * np.pi * (h**9))

    if r_norm < h:
        w = norm * ((h**2) - (r_norm**2))**3
    elif r_norm >= h:
        w = 0

    return w


@njit
def gradPoly6(r, h, dim):
    r_norm = np.linalg.norm(r)

    if dim == 2:
        norm = 4 / (np.pi * (h ** 8))
    elif dim == 3:
        norm = 315 / (64 * np.pi * (h ** 9))

    if r_norm < h:
        tmp = -6 * norm * ((h**2) - (r_norm**2))**2
    elif r_norm >= h:
        tmp = 0

    if dim == 2:
        wx = tmp * r[0]
        wy = tmp * r[1]
        wz = 0
    elif dim == 3:
        wx = tmp * r[0]
        wy = tmp * r[1]
        wz = tmp * r[2]

    return wx, wy, wz


@njit
def WendlandC2(r, h, dim):

    r_norm = np.linalg.norm(r)
    q = r_norm / h

    if dim == 2:
        norm = 7 / (4 * np.pi * (h**2))
    elif dim == 3:
        norm = 21 / (16 * np.pi * (h**3))

    fac1 = 1 - q/2
    fac2 = 2*q + 1

    if 0 <= q <= 2:
        w = norm * (fac1**4) * fac2
    elif q > 2:
        w = 0

    return w


@njit
def gradWendlandC2(r, h, dim):
    r_norm = np.linalg.norm(r)
    q = r_norm / h

    if dim == 2:
        norm = 7 / (4 * np.pi * (h ** 2))
    elif dim == 3:
        norm = 21 / (16 * np.pi * (h ** 3))

    fac = q-2

    if r_norm > 0.000000000001:
        if 0 <= q <= 2:
            wdash = (norm * (5/8) * (fac**3) * q) / (r_norm * h)
        elif q > 2:
            wdash = 0
    else:
        wdash = 0

    if dim == 2:
        wx = wdash * r[0]
        wy = wdash * r[1]
        wz = 0
    elif dim == 3:
        wx = wdash * r[0]
        wy = wdash * r[1]
        wz = wdash * r[2]

    return wx, wy, wz


@njit
def WendlandC4(r, h, dim):

    r_norm = np.linalg.norm(r)
    q = r_norm / h

    if dim == 2:
        norm = 9 / (4 * np.pi * (h**2))
    elif dim == 3:
        norm = 495 / (256 * np.pi * (h**3))

    fac1 = 1 - (q/2)
    fac2 = ((35/12) * q**2) + (3*q) + 1

    if 0 <= q <= 2:
        w = norm * (fac1**6) * fac2
    elif q > 2:
        w = 0

    return w


@njit
def gradWendlandC4(r, h, dim):

    r_norm = np.linalg.norm(r)
    q = r_norm / h

    if dim == 2:
        norm = 9 / (4 * np.pi * (h ** 2))
    elif dim == 3:
        norm = 495 / (256 * np.pi * (h ** 3))

    fac1 = q-2
    fac2 = (5*q + 2)

    fac = (7 * (fac1 ** 5) * q * fac2) / 96

    if r_norm > 0.000000000001:
        if 0 <= q <= 2:
            wdash = (norm * (5/8) * (fac**3) * q) / (r_norm * h)
        elif q > 2:
            wdash = 0
    else:
        wdash = 0

    if dim == 2:
        wx = wdash * r[0]
        wy = wdash * r[1]
        wz = 0
    elif dim == 3:
        wx = wdash * r[0]
        wy = wdash * r[1]
        wz = wdash * r[2]

    return wx, wy, wz


""" ================================ WCSPH Base ======================================= """


def ClosesttoBoundary(pos, targetx):
    x_differences = np.abs(pos[:, 0] - targetx)
    closest_index = np.argmin(x_differences)
    return closest_index


def Container2D(l, r, n_b, b, t, n_w, dx):

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


@njit
def tank2D(l, r, n_b, b, t, n_w, dx):
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

@njit
def getDensity(m, kernel):

    rho = m * kernel
    return rho


@njit
def getContinuity(m, vi, vj, wx, wy):

    v = vi - vj
    w = np.zeros(2)
    w[0] = wx
    w[1] = wy
    d_rho = m * np.dot(v, w)

    return d_rho


@njit
def getContinuityAlternative(m, rhoi, rhoj, vi, vj, wx, wy):

    v = vi - vj
    w = np.hstack((wx, wy))
    d_rho = rhoi * (m / rhoj) * np.dot(v, w)

    return d_rho


@njit
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


@njit
def getPressureAcc_WCSPH(m, pi, pj, rhoi, rhoj, wx, wy):

    ax = -m * ((pi / (rhoi**2)) + (pj / (rhoj**2))) * wx
    ay = -m * ((pi / (rhoi**2)) + (pj / (rhoj**2))) * wy

    a = np.hstack((ax, ay))

    return a


@njit
def ArtificialViscosity(m, alpha, beta, c0, h, rhoi, rhoj, rho0, vi, vj, posi, posj, epsilon, gamma, dim, wx, wy):
    v = vi - vj
    r = posi - posj
    rho_avg = np.hstack((rhoi, rhoj))
    rho = np.mean(rho_avg)
    r_norm = np.linalg.norm(r)

    vdotr = 0
    for i in range(dim):
        vdotr += v[i] * r[i]

    ci = c0 * (rhoi / rho0) ** ((gamma - 1) / 2)
    cj = c0 * (rhoj / rho0) ** ((gamma - 1) / 2)
    c_avg = np.hstack((ci, cj))
    c = np.mean(c_avg)

    nu = h * vdotr / ((r_norm**2) + epsilon * (h**2))
    visc = float(-(alpha * c * nu)/rho + (beta * nu**2) / rho)

    visc_x = 0.0
    visc_y = 0.0

    if vdotr < 0:
        visc_x = -m * visc * wx
        visc_y = -m * visc * wy

    viscos = np.array([visc_x, visc_y])

    return viscos


@njit
def ClearyArtificialViscosity(m, alpha, beta, c0, h, rhoi, rhoj, rho0, vi, vj, posi, posj, eps, gamma, dim, wx, wy):
    v = vi - vj
    r = posi - posj
    rho_avg = np.hstack((rhoi, rhoj))
    rho = np.mean(rho_avg)
    r_norm = np.linalg.norm(r)
    rhoi = float(rhoi[0])
    rhoj = float(rhoj[0])

    vdotr = 0
    for i in range(dim):
        vdotr += v[i] * r[i]

    # nu_i = (1/8) * alpha * h * ci * rhoi
    # nu_j = (1/8) * alpha * h * cj * rhoj

    nu_i = 0.001
    nu_j = 0.001

    visc = -(19.85 * nu_i * nu_j) / (rhoi * rhoj * (nu_i + nu_j)) * (vdotr / (r_norm**2 + (eps * (h**2))))

    visc_x = -m * visc * wx
    visc_y = -m * visc * wy

    viscos = np.array([visc_x, visc_y])

    return viscos

@njit
def getViscosityAcc(m, h, rhoi, rhoj, nu, posi, posj, vi, vj, wx, wy):

    r = posi - posj
    v = vi - vj
    r_norm = np.linalg.norm(r)
    w = np.zeros(2)
    w[0] = wx
    w[1] = wy

    norm = (r_norm**2) + (0.00*h*h)
    norm1 = m * (2 * nu) * np.dot(r, w) / (rhoi * rhoj * norm)

    visc = norm1 * v

    return visc


@njit
def ArtificialPressure(m, pi, pj, wr, wdx, wx, wy, n, eps, rhoi, rhoj):

    # eps ~ 0.2
    fij = wr / wdx
    f = fij**n

    rhoi = float(rhoi[0])
    rhoj = float(rhoj[0])
    pi = float(pi[0])
    pj = float(pj[0])

    if pi < 0:
        Ri = eps * np.abs(pi) / (rhoi**2)
    elif pi >= 0:
        Ri = 0.0

    if pj < 0:
        Rj = eps * np.abs(pj) / (rhoj**2)
    elif pj >= 0:
        Rj = 0.0

    R = Ri + Rj

    if pi >= 0 and pj >= 0:
        R = 0.01 * ((pi / (rhoi**2)) + (pj / (rhoj**2)))
    else:
        pass

    ti_x = -m * (R*f) * wx
    ti_y = -m * (R*f) * wy

    artpress = np.zeros(2)
    artpress[0] = ti_x
    artpress[1] = ti_y

    return artpress


@njit
def LoopDensity(NeighbourIDi, pos, posi, h, dim, rhoi, m):
    for j_in_list, j in enumerate(NeighbourIDi):
        # Iterating through ith list in Neighbour ID list, j is an index in pos array

        if j_in_list == 0:
            # Getting x and y distances
            r = posi - pos[j]  # position vector

            kernel = QuinticSpline(r, h, dim)
            rhoi += getDensity(m, kernel)  # Summation density formula

        else:
            # Getting x and y distances
            r = posi - pos[j]  # position vector

            kernel = QuinticSpline(r, h, dim)
            rhoi += getDensity(m, kernel)  # Summation density formula

    return rhoi


@njit
def LoopContinuity(NeighbourIDi, pos, posi, h, dim, drhoi, m, vel, veli):

    for j_in_list, j in enumerate(NeighbourIDi):
        # Iterating through ith list in Neighbour ID list, j is an index in pos array

        if j_in_list == 0:
            r = posi - pos[j]  # position vector

            wx, wy, wz = gradQuinticSpline(r, h, dim)
            drhoi += getContinuity(m, veli, vel[j], wx, wy)

        else:
            # Getting x and y distances
            r = posi - pos[j]  # position vector

            wx, wy, wz = gradQuinticSpline(r, h, dim)
            drhoi += getContinuity(m, veli, vel[j], wx, wy)

    return drhoi


@njit
def UpdateDensity(n_particles, iter, rho_half, rho, drho, dt):
    for i in range(n_particles):
        if iter == 0:
            rho_half[i] = rho[i] + ((dt/2) * drho[i])
            rho[i] += (drho[i] * dt)
        elif iter != 0:
            rho_half[i] += dt * drho[i]
            rho[i] = rho_half[i] + (drho[i] * dt/2)

    return rho_half, rho


@njit
def LoopForces(NeighbourIDi, pos, posi, h, dim, dx, forcesi, m, p, pi, rho, rhoi, rho0, c0, vel, veli, gamma, g_acc, nu):

    for j_in_list, j in enumerate(NeighbourIDi):
        if j_in_list == 0:
            continue
        else:
            # Getting x and y distances
            r = posi - pos[j]

            wx, wy, wz = gradQuinticSpline(r, h, dim)

            forcesi += getPressureAcc_WCSPH(m, pi, p[j], rhoi, rho[j], wx, wy)
            '''
            forcesi += ClearyArtificialViscosity(m, 0.1, 0.0, c0, h, rhoi, rho[j], rho0, veli, vel[j], posi, pos[j],
                                                 0.01, gamma, dim, wx, wy)
            '''
            '''
            forcesi += getViscosityAcc(m, h, rhoi, rho[j], nu, posi, pos[j], veli, vel[j], wx, wy)

            '''
            forcesi += ArtificialViscosity(m, 0.5, 0.0, c0, h, rhoi, rho[j], rho0, veli, vel[j], posi, pos[j], 0.01,
                                           gamma, dim, wx, wy)

    forcesi += g_acc

    return forcesi


@njit
def LoopArtificialPressure(NeighbourIDi, pos, posi, h, dim, dx, forcesi, m, p, pi, rho, rhoi):
    for j_in_list, j in enumerate(NeighbourIDi):
        if j_in_list == 0:
            continue
        else:
            # Getting x and y distances
            r = posi - pos[j]

            wr = QuinticSpline(r, h, dim)
            wdx = QuinticSpline(dx, h, dim)
            wx, wy, wz = gradQuinticSpline(r, h, dim)

            forcesi += ArtificialPressure(m, pi, p[j], wr, wdx, wx, wy, 4, 0.2, rhoi, rho[j])

    return forcesi


@njit
def LoopXSPH(NeighbourIDi, pos, posi, vel, veli, rho, rhoi, h, dim, m, XSPHi):
    for j_in_list, j in enumerate(NeighbourIDi):
        if j_in_list == 0:
            continue
        else:
            r = posi - pos[j]
            v = veli - vel[j]
            rho_temp = (rhoi + rho[j]) / 2
            kernel = QuinticSpline(r, h, dim)

            XSPHi += m * (v / rho_temp) * kernel

    return XSPHi


@njit
def LeapfrogIntegration(iter, vel_half, vel_fluid, vel, forces, dt, n_fluid, XSPH, pos_fluid, pos):

    if iter == 0:  # For initial time step
        vel_half = vel_fluid + (forces * (dt/2))  # Velocity at half-time step
        vel_fluid += forces * dt  # Velocity at full time step
        vel[0:n_fluid] = vel_fluid

        pos_fluid += ((vel_half * dt) - 0.5*XSPH*dt)
        # pos_fluid += vel_half * dt
        pos[0:n_fluid] = pos_fluid

    elif iter != 0:  # For all other time steps
        vel_half += forces * dt  # Velocity at half-time step
        vel_fluid = vel_half + (forces * dt/2)
        vel[0:n_fluid] = vel_fluid

        pos_fluid += ((vel_half * dt) - 0.5*XSPH*dt)
        # pos_fluid += vel_half * dt
        pos[0:n_fluid] = pos_fluid

    return vel_fluid, vel_half, vel, pos_fluid, pos


def create_grid_points(shape_outline, spacing):
    x_min, y_min = np.min(shape_outline, axis=0)
    x_max, y_max = np.max(shape_outline, axis=0)

    x_values = np.arange(x_min, x_max + spacing, spacing)
    y_values = np.arange(y_min, y_max + spacing, spacing)

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    return grid_points


def filter_points_inside_shape(points, shape_outline):
    inside_points = []
    for point in points:
        if cv2.pointPolygonTest(shape_outline, tuple(point), measureDist=False) >= 0:
            inside_points.append(point)
    return np.array(inside_points)


def remove_points_too_close(points, boundary_points, distance_threshold):
    filtered_points = []
    for point in points:
        min_distance = np.min(np.sqrt(np.sum((boundary_points - point) ** 2, axis=1)))
        if min_distance >= distance_threshold:
            filtered_points.append(point)
    return np.array(filtered_points)


def find_centroid(points):
    return np.mean(points, axis=0)


def rotate(points, angle, center):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    points_rotated = np.dot(points - center, rotation_matrix) + center
    return points_rotated


def cut_bottom_part(points, cut_y):
    remaining_points = points[points[:, 1] >= cut_y]
    cut_points = points[points[:, 1] < cut_y]
    return remaining_points, cut_points