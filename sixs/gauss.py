"""
Gaussian quadrature routine.
Converted from Fortran 77 GAUSS.f

Computes the abscissas and weights for Gauss-Legendre quadrature.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def gauss(x1, x2, n):
    """
    Calculate Gauss-Legendre quadrature abscissas and weights.

    Given the lower and upper limits of integration x1 and x2, this routine
    returns arrays x[0:n-1] and w[0:n-1] of length n, containing the abscissas
    and weights of the Gauss-Legendre n-point quadrature formula.

    Parameters
    ----------
    x1 : float
        Lower limit of integration
    x2 : float
        Upper limit of integration
    n : int
        Number of quadrature points

    Returns
    -------
    x : ndarray
        Abscissas for quadrature
    w : ndarray
        Weights for quadrature

    Notes
    -----
    This uses double precision arithmetic as in the original Fortran code.
    The convergence criterion is eps = 3e-14.
    """
    x = np.zeros(n, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)

    eps = 3.0e-14
    m = (n + 1) // 2

    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)

    # Fortran: do 12 i=1,m
    # Loop over half the points (symmetry)
    for i in range(m):
        # Initial guess for root
        z = np.cos(np.pi * (i + 0.75) / (n + 0.5))

        # Newton-Raphson iteration (Fortran used labeled GOTO: 1 continue)
        while True:
            p1 = 1.0
            p2 = 0.0

            # Legendre polynomial recurrence
            # Fortran: do 11 j=1,n
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j

            # Derivative of Legendre polynomial
            pp = n * (z * p1 - p2) / (z * z - 1.0)

            z1 = z
            z = z1 - p1 / pp

            # Check convergence (Fortran: if(abs(z-z1).gt.eps)go to 1)
            if abs(z - z1) <= eps:
                break

        # Clean up numerical noise
        if abs(z) < eps:
            z = 0.0

        # Use symmetry to get both roots
        # Fortran arrays are 1-based: x(i) and x(n+1-i)
        # Python arrays are 0-based: x[i] and x[n-1-i]
        x[i] = xm - xl * z
        x[n - 1 - i] = xm + xl * z

        w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp)
        w[n - 1 - i] = w[i]

    return x, w
