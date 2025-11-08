"""
Atmospheric layer discretization.

Functions for discretizing atmospheric layers based on optical thickness.
"""

import numpy as np
from numba import jit
from .exceptions import PhysicalError


@jit(nopython=True)
def discre(ta, ha, tr, hr, it, nt, yy, dd, ppp2, ppp1):
    """
    Discretize atmospheric layers based on optical thickness.

    Uses bisection method to find the altitude where optical thickness
    matches the target value.

    Parameters
    ----------
    ta : float
        Aerosol optical thickness
    ha : float
        Aerosol scale height (km)
    tr : float
        Rayleigh optical thickness
    hr : float
        Rayleigh scale height (km)
    it : int
        Current iteration index
    nt : int
        Total number of layers
    yy : float
        Target optical thickness
    dd : float
        Delta value for convergence check
    ppp2 : float
        Upper bound for bisection
    ppp1 : float
        Lower bound for bisection

    Returns
    -------
    zx : float
        Altitude (km) at target optical thickness
    delta : float
        Fraction parameter

    Raises
    ------
    PhysicalError
        If aerosol altitude is too high (>= 7 km)

    Notes
    -----
    Uses exponential atmosphere model:
    τ(z) = τ_a * exp(-z/h_a) + τ_r * exp(-z/h_r)

    Bisection method finds z where τ(z) = target value.
    """
    # Check aerosol altitude constraint
    if ha >= 7.0:
        # In Numba nopython mode, we can't raise custom exceptions
        # Return special sentinel values instead
        return -999.0, -999.0

    # Initialize step size
    if it == 0:
        dt = 1.0e-17
    else:
        dt = 2.0 * (ta + tr - yy) / (nt - it + 1.0)

    # Iterative refinement
    while True:
        dt = dt / 2.0
        ti = yy + dt

        y1 = ppp2
        y3 = ppp1

        # Bisection method to find altitude
        max_iter = 1000
        for _ in range(max_iter):
            y2 = (y1 + y3) * 0.5
            xx = -y2 / ha

            # Calculate optical thickness at this altitude
            if xx < -18.0:
                x2 = tr * np.exp(-y2 / hr)
            else:
                x2 = ta * np.exp(xx) + tr * np.exp(-y2 / hr)

            xd = abs(ti - x2)

            # Check convergence
            if xd < 0.00001:
                break

            # Update bisection bounds
            if ti < x2:
                y3 = y2
            else:
                y1 = y2

        # Found solution
        zx = y2

        # Calculate delta parameter
        delta = 1.0 / (1.0 + ta * hr / tr / ha * np.exp((zx - ppp1) * (1.0/hr - 1.0/ha)))

        # Check convergence criterion
        ecart = 0.0
        if dd != 0.0:
            ecart = abs((dd - delta) / dd)

        # If converged or first iteration, return
        if (ecart <= 0.75) or (it == 0):
            return zx, delta

        # Otherwise, refine and try again
        # (loop will continue with halved dt)


def discre_safe(ta, ha, tr, hr, it, nt, yy, dd, ppp2, ppp1):
    """
    Safe wrapper for discre that handles exceptions.

    Same parameters and returns as discre(), but raises proper exceptions
    instead of returning sentinel values.
    """
    zx, delta = discre(ta, ha, tr, hr, it, nt, yy, dd, ppp2, ppp1)

    if zx == -999.0:
        raise PhysicalError(
            "Aerosol altitude too high (>= 7 km). "
            "Check aerosol measurements or plane altitude."
        )

    return zx, delta


__all__ = [
    'discre',
    'discre_safe',
]
