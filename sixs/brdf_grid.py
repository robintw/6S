"""
BRDF grid interpolation.

Interpolate BRDF data onto Gaussian quadrature grids.
"""

import numpy as np
from numba import jit
from .math_utils import splie2, splin2


@jit(nopython=True)
def brdfgrid(rm, rp, angmu, angphi, brdfdat):
    """
    Interpolate BRDF data onto a grid of Gaussian quadrature points.

    Uses 2D bicubic spline interpolation to evaluate BRDF at specified
    viewing geometry angles.

    Parameters
    ----------
    rm : ndarray
        Cosines of viewing zenith angles (mu values), shape (mu,)
        Typically from Gaussian quadrature
    rp : ndarray
        Relative azimuth angles (phi values), shape (np,)
        Typically from Gaussian quadrature
    angmu : ndarray
        Cosine grid for BRDF data, shape (nmu,)
    angphi : ndarray
        Azimuth angle grid for BRDF data (degrees), shape (nphi,)
    brdfdat : ndarray
        BRDF values on (angmu, angphi) grid, shape (nmu, nphi)

    Returns
    -------
    brdfint : ndarray
        Interpolated BRDF values, shape (mu, np)
        brdfint[i, j] = BRDF at (rm[i], rp[j])

    Notes
    -----
    Uses bicubic spline interpolation for smooth BRDF evaluation.
    Fortran original used adjustable array dimensions with negative indexing.
    Python version uses standard 0-based indexing.
    """
    mu = len(rm)
    np_points = len(rp)
    nmu = len(angmu)
    nphi = len(angphi)

    # Initialize output array
    brdfint = np.zeros((mu, np_points), dtype=np.float64)

    # Compute 2D spline coefficients
    brdftemp = splie2(angphi, brdfdat)

    # Interpolate at each Gaussian quadrature point
    for j in range(np_points):
        for k in range(mu):
            gaussmu = rm[k]
            gaussphi = rp[j]

            # Evaluate 2D spline at (gaussmu, gaussphi)
            y = splin2(angmu, angphi, brdfdat, brdftemp, gaussmu, gaussphi)

            brdfint[k, j] = y

    return brdfint


__all__ = [
    'brdfgrid',
]
