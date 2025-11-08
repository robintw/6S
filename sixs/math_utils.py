"""
Mathematical utilities for 6S radiative transfer code.

This module contains spline interpolation, Gaussian quadrature, and other
numerical methods converted from Fortran 77.

Functions:
    spline, splint: 1D cubic spline interpolation
    splin2, splie2: 2D cubic spline interpolation
    gauss: Gaussian-Legendre quadrature
    csalbr: Cosine albedo calculation with exponential integrals
"""

import numpy as np
from numba import jit


# Import existing 1D spline functions
from .spline import spline, splint
from .gauss import gauss


@jit(nopython=True)
def splie2(x2a, ya):
    """
    Compute 2D spline coefficients.

    Given a 2D array ya[m,n] tabulated at grid points x2a[n],
    compute the 2nd derivatives for bicubic spline interpolation.

    Parameters
    ----------
    x2a : ndarray
        Array of n x-coordinates (2nd dimension)
    ya : ndarray
        2D array of function values, shape (m, n)

    Returns
    -------
    y2a : ndarray
        2D array of 2nd derivatives, shape (m, n)

    Notes
    -----
    Uses natural spline (zero 2nd derivatives at boundaries).
    For each row, calls spline() to get coefficients.
    """
    m, n = ya.shape
    y2a = np.zeros((m, n), dtype=np.float64)
    ytmp = np.zeros(n, dtype=np.float64)

    # Fortran: do 13 j=1,m (j is row index)
    for j in range(m):
        # Extract row
        for k in range(n):
            ytmp[k] = ya[j, k]

        # Compute spline coefficients for this row
        # Fortran: call spline(x2a,ytmp,n,1.e30,1.e30,y2tmp)
        y2tmp = spline(x2a, ytmp, 1.0e30, 1.0e30)

        # Store coefficients
        for k in range(n):
            y2a[j, k] = y2tmp[k]

    return y2a


@jit(nopython=True)
def splin2(x1a, x2a, ya, y2a, x1, x2):
    """
    Evaluate 2D bicubic spline at point (x1, x2).

    Parameters
    ----------
    x1a : ndarray
        Array of m x-coordinates (1st dimension)
    x2a : ndarray
        Array of n x-coordinates (2nd dimension)
    ya : ndarray
        2D array of function values, shape (m, n)
    y2a : ndarray
        2D array of 2nd derivatives from splie2(), shape (m, n)
    x1 : float
        x-coordinate for evaluation
    x2 : float
        y-coordinate for evaluation

    Returns
    -------
    y : float
        Interpolated value at (x1, x2)

    Notes
    -----
    First interpolates each row at x2, then interpolates the results at x1.
    """
    m, n = ya.shape
    yytmp = np.zeros(m, dtype=np.float64)
    ytmp = np.zeros(n, dtype=np.float64)
    y2tmp = np.zeros(n, dtype=np.float64)

    # Fortran: do 12 j=1,m
    # For each row, interpolate at x2
    for j in range(m):
        # Extract row data
        for k in range(n):
            ytmp[k] = ya[j, k]
            y2tmp[k] = y2a[j, k]

        # Interpolate this row at x2
        # Fortran: call splint(x2a,ytmp,y2tmp,n,x2,yytmp(j))
        yytmp[j] = splint(x2a, ytmp, y2tmp, x2)

    # Now interpolate the column of results at x1
    # Fortran: call spline(x1a,yytmp,m,1.e30,1.e30,y2tmp)
    y2tmp_col = spline(x1a, yytmp, 1.0e30, 1.0e30)

    # Fortran: call splint(x1a,yytmp,y2tmp,m,x1,y)
    y = splint(x1a, yytmp, y2tmp_col, x1)

    return y


# Exponential integral functions for cosine albedo
@jit(nopython=True)
def fintexp1(xtau):
    """
    Compute the first exponential integral E1(xtau).

    Uses polynomial approximation for 0 < xtau < 1.
    Accuracy: ~2e-7

    Parameters
    ----------
    xtau : float
        Optical depth

    Returns
    -------
    float
        E1(xtau) = -Ei(-xtau)
    """
    # Polynomial coefficients
    a = np.array([
        -0.57721566,
        0.99999193,
        -0.24991055,
        0.05519968,
        -0.00976004,
        0.00107857
    ], dtype=np.float64)

    xx = a[0]
    xftau = 1.0

    for i in range(1, 6):
        xftau = xftau * xtau
        xx = xx + a[i] * xftau

    # E1(x) = -γ - ln(x) - Σ(-1)^n x^n / (n·n!)
    return xx - np.log(xtau)


@jit(nopython=True)
def fintexp3(xtau):
    """
    Compute the third exponential integral E3(xtau).

    Parameters
    ----------
    xtau : float
        Optical depth

    Returns
    -------
    float
        E3(xtau)
    """
    # E3(x) = [exp(-x)(1-x) + x²·E1(x)] / 2
    xx = (np.exp(-xtau) * (1.0 - xtau) + xtau * xtau * fintexp1(xtau)) / 2.0
    return xx


@jit(nopython=True)
def csalbr(xtau):
    """
    Calculate cosine albedo for conservative scattering.

    This implements the spherical albedo calculation using
    exponential integrals.

    Parameters
    ----------
    xtau : float
        Optical thickness

    Returns
    -------
    xalb : float
        Spherical albedo

    Notes
    -----
    Formula from Chandrasekhar radiative transfer theory.
    Uses exponential integral E3.
    """
    # Fortran: xalb=(3*xtau-fintexp3(xtau)*(4+2*xtau)+2*exp(-xtau))
    # xalb=xalb/(4.+3*xtau)

    numerator = (3.0 * xtau -
                fintexp3(xtau) * (4.0 + 2.0 * xtau) +
                2.0 * np.exp(-xtau))

    xalb = numerator / (4.0 + 3.0 * xtau)

    return xalb


# Re-export for convenience
__all__ = [
    'spline',
    'splint',
    'splie2',
    'splin2',
    'gauss',
    'csalbr',
    'fintexp1',
    'fintexp3',
]
