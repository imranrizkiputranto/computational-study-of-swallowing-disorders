"""
Collection of functions for Smoothing Kernels and their respective first and second order derivative in
1D, 2D and 3D
"""
import numpy as np
import matplotlib.pyplot as plt


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


def gradGaussian(r, h, dim):

    r_norm = np.linalg.norm(r)
    q = r_norm / h

    # Getting normalisation factor
    if dim == 1:
        norm = 1 / (np.sqrt(np.pi) * h)
        if q <= 3:
            wx = norm * np.exp(- (q ** 2)) * (-2 / (h**2)) * r
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
        wx = tmp * r
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


def CubicSpline(r, h, dim):
    """
    Defines Cubic Spline Kernel
    :param r: dim x 1 distance vector
    :param h: Smoothing length
    :param dim: dimension
    """

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

    # Getting kernel
    if 0 <= q <= 1:
        w = norm * (1 - (3/2) * (q**2) * (1 - q/2))
    elif 1 < q <= 2:
        w = (norm / 4) * ((2 - q)**3)
    elif q > 2:
        w = 0

    return w


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
        wx = tmp * r
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


def QuinticSpline(r, h, dim):
    """
    Defines Cubic Spline Kernel
    :param r: dim x 1 distance vector
    :param h: Smoothing length
    :param dim: dimension
    """

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


def gradQuinticSpline(r, h, dim):
    """
    Defines Cubic Spline Kernel first derivative
    :param r: dim x 1 distance vector
    :param h: Smoothing length
    :param dim: dimension
    """

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
        wx = tmp * r
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


def Poly6(r, h, dim):
    """
    Poly6 Kernel
    :param r: 1x2 position difference vector between ith and jth particle
    :param h: Smoothing length
    :param dim: dimensions
    """

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


def gradPoly6(r, h, dim):
    """
    Gradient of 6th order polynomial
    :param r: 1x2 position difference vector between ith and jth particle
    :param h: Smoothing length
    :param dim: dimensions
    """

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


def Spiky(r, h):
    '''
    Defines the Spiky Smoothing Kernel (2D) based on x and y coordinates and the
    smoothing length, h
    :param x: Matrix of positions in x coordinates
    :param y: Matrix of positions in y coordinates
    :param h: Smoothing length
    '''

    r_norm = np.linalg.norm(r)

    if r_norm <= h:
        n = (h-r_norm)**3
    elif r_norm > h:
        n = 0

    m = 15 / (np.pi * h**6)
    w = m * n

    return w


def gradSpiky(r, h):
    '''
    Defines the derivative of the Spiky Smoothing Kernel (2D) based on x and y coordinates and the smoothing length, h,
    with respect to x and y
    :param x: Matrix of positions in x coordinates
    :param y: Matrix of positions in y coordinates
    :param h: Smoothing length
    '''

    r_norm = np.linalg.norm(r)

    if r_norm <= h:
        n = (h - r_norm) ** 2
    elif r_norm > h:
        n = 0

    m = - 45 / (np.pi * h ** 6)
    w = m * n

    return w, w


def WVisc(r, h):
    '''
    Defines the Viscosity Smoothing Kernel (2D) based on x and y coordinates and the smoothing length, h
    :param x: Matrix of positions in x coordinates
    :param y: Matrix of positions in y coordinates
    :param h: Smoothing length
    '''

    r_norm = np.linalg.norm(r)

    if r_norm <= h:
        n = -(r_norm / (2 * h**3)) + (r_norm/h)**2 + (h / (2*r_norm)) - 1
    elif r_norm > h:
        n = 0

    m = 15/(2 * np.pi * h**3)
    w = m * n

    return w


def gradgradWVisc(r, h):
    '''
    Defines the second order derivative of the Viscosity Smoothing Kernel (2D) based on x and y coordinates and the
    smoothing length, h, with respect to x and y
    :param x: Matrix of positions in x coordinates
    :param y: Matrix of positions in y coordinates
    :param h: Smoothing length
    '''

    r_norm = np.linalg.norm(r)

    if r_norm <= h:
        n = h - r_norm
    elif r_norm > h:
        n = 0

    m = 45/(np.pi * h**6)
    w = m * n

    return w


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




