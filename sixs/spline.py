"""
Cubic spline interpolation routines.
Converted from Fortran 77 SPLINE.f and SPLINT.f

Original Fortran code uses 1-based indexing; Python uses 0-based.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def spline(x, y, yp1, ypn):
    """
    Calculate cubic spline coefficients.

    Given arrays x[0:n-1] and y[0:n-1] containing a tabulated function,
    and given values yp1 and ypn for the first derivative of the interpolating
    function at points 0 and n-1, this routine returns an array y2[0:n-1]
    that contains the second derivatives of the interpolating function at
    the tabulated points x[i].

    Parameters
    ----------
    x : ndarray
        Array of x values (must be monotonic)
    y : ndarray
        Array of y values
    yp1 : float
        First derivative at x[0]. If >= 1e30, natural spline (zero 2nd deriv)
    ypn : float
        First derivative at x[n-1]. If >= 1e30, natural spline (zero 2nd deriv)

    Returns
    -------
    y2 : ndarray
        Second derivatives at tabulated points
    """
    n = len(x)
    y2 = np.zeros(n, dtype=np.float64)
    u = np.zeros(n, dtype=np.float64)

    # Fortran: if (yp1.gt..99e30) then
    if yp1 > 0.99e30:
        y2[0] = 0.0
        u[0] = 0.0
    else:
        y2[0] = -0.5
        u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1)

    # Fortran: do 11 i=2,n-1
    # Note: Fortran i=2 to n-1 means second to second-to-last in 1-based indexing
    # In Python 0-based, this is i=1 to n-2
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (6.0 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
                       (y[i] - y[i - 1]) / (x[i] - x[i - 1])) /
                (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p

    # Fortran: if (ypn.gt..99e30) then
    if ypn > 0.99e30:
        qn = 0.0
        un = 0.0
    else:
        qn = 0.5
        un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) /
                                                     (x[n - 1] - x[n - 2]))

    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0)

    # Fortran: do 12 k=n-1,1,-1
    # In Fortran 1-based: k goes from n-1 down to 1
    # In Python 0-based: k goes from n-2 down to 0
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]

    return y2


@jit(nopython=True)
def splint(xa, ya, y2a, x):
    """
    Evaluate cubic spline at point x.

    Given the arrays xa[0:n-1] and ya[0:n-1], which tabulate a function,
    and given the array y2a[0:n-1], which is the output from spline above,
    and given a value of x, this routine returns the cubic-spline
    interpolated value y.

    Parameters
    ----------
    xa : ndarray
        Array of x values (must be monotonic)
    ya : ndarray
        Array of y values
    y2a : ndarray
        Second derivatives from spline()
    x : float
        Point at which to evaluate the spline

    Returns
    -------
    y : float
        Interpolated value at x
    """
    n = len(xa)
    klo = 0
    khi = n - 1

    # Binary search (Fortran used GOTO for loop)
    # Fortran: 1 if (khi-klo.gt.1) then ... goto 1
    while khi - klo > 1:
        k = (khi + klo) // 2
        if xa[k] > x:
            khi = k
        else:
            klo = k

    h = xa[khi] - xa[klo]

    # Fortran: if (h.eq.0.) pause 'bad xa input.'
    if h == 0.0:
        raise ValueError('Bad xa input: xa values must be distinct')

    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h

    # Fortran line continuation with *
    y = (a * ya[klo] + b * ya[khi] +
         ((a**3 - a) * y2a[klo] + (b**3 - b) * y2a[khi]) * (h**2) / 6.0)

    return y


def splie2(x2a, ya):
    """
    Calculate 2D cubic spline coefficients.

    Performs spline initialization on a 2D tabulated function. Given an
    m by n table ya[0:m-1, 0:n-1], and given an array x2a[0:n-1],
    this routine constructs one-dimensional natural cubic splines of the
    rows of ya and returns the second derivatives in the array y2a[0:m-1, 0:n-1].

    Parameters
    ----------
    x2a : ndarray
        Array of x2 values (n elements, must be monotonic)
    ya : ndarray
        2D array of function values (m x n)

    Returns
    -------
    y2a : ndarray
        2D array of second derivatives (m x n)

    Notes
    -----
    Converted from Fortran SPLIE2.f. Uses natural spline boundary conditions
    (yp1=ypn=1e30) for all rows.
    """
    m, n = ya.shape

    # Initialize output array
    y2a = np.zeros((m, n), dtype=np.float64)

    # For each row, calculate spline coefficients
    for j in range(m):
        # Extract row j
        ytmp = ya[j, :].copy()

        # Calculate spline coefficients with natural boundary conditions
        y2tmp = spline(x2a, ytmp, 1.0e30, 1.0e30)

        # Store in output array
        y2a[j, :] = y2tmp

    return y2a


def splin2(x1a, x2a, ya, y2a, x1, x2):
    """
    Evaluate 2D cubic spline interpolation.

    Given x1a[0:m-1], x2a[0:n-1], tabulated function values ya[0:m-1, 0:n-1],
    and tabulated second derivatives y2a[0:m-1, 0:n-1] (from splie2), and
    given values x1 and x2, this routine returns an interpolated function value.

    Parameters
    ----------
    x1a : ndarray
        Array of x1 values (m elements, must be monotonic)
    x2a : ndarray
        Array of x2 values (n elements, must be monotonic)
    ya : ndarray
        2D array of function values (m x n)
    y2a : ndarray
        2D array of second derivatives from splie2 (m x n)
    x1 : float
        First coordinate at which to evaluate
    x2 : float
        Second coordinate at which to evaluate

    Returns
    -------
    y : float
        Interpolated function value at (x1, x2)

    Notes
    -----
    Converted from Fortran SPLIN2.f. Performs bicubic spline interpolation:
    1. Interpolates each row at x2 to get m values
    2. Interpolates those m values at x1 to get final result
    """
    m = len(x1a)
    n = len(x2a)

    # Temporary arrays for intermediate interpolation
    yytmp = np.zeros(m, dtype=np.float64)

    # For each row, interpolate at x2
    for j in range(m):
        # Extract row j data and second derivatives
        ytmp = ya[j, :].copy()
        y2tmp = y2a[j, :].copy()

        # Interpolate this row at x2
        yytmp[j] = splint(x2a, ytmp, y2tmp, x2)

    # Now interpolate the resulting 1D array at x1
    # First calculate spline coefficients for yytmp
    y2tmp = spline(x1a, yytmp, 1.0e30, 1.0e30)

    # Finally interpolate at x1
    y = splint(x1a, yytmp, y2tmp, x1)

    return y
