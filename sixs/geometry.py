"""
Geometric and positional calculation routines.

Satellite geometry and solar position calculations for various platforms.

Functions:
    possol: Solar position calculation (already implemented in possol.py)
    varsol: Solar constant variability
    equivwl: Equivalent wavelength calculation
"""

import numpy as np
from numba import jit


# Import solar position from separate module
from .possol import possol, day_number, pos_fft
from .solar_irradiance import solirr, equivwl


@jit(nopython=True)
def varsol(jday, month):
    """
    Calculate solar constant variability throughout the year.

    Parameters
    ----------
    jday : int
        Day of the month (1-31)
    month : int
        Month (1-12)

    Returns
    -------
    dsol : float
        Multiplicative factor for mean solar constant

    Notes
    -----
    Accounts for Earth's elliptical orbit around the Sun.
    Solar constant varies by ~±3.3% throughout the year.
    """
    # Calculate day number in year
    if month <= 2:
        j = 31 * (month - 1) + jday
    elif month > 8:
        j = 31 * (month - 1) - (month - 2) // 2 - 2 + jday
    else:
        j = 31 * (month - 1) - (month - 1) // 2 - 2 + jday

    # Solar constant variation
    # Based on Earth-Sun distance variation
    pi = 3.14159265
    om = 2.0 * pi * float(j) / 365.0

    # Fortran continues with calculation
    # dsol = 1.0 + 0.01673 * cos(om - 0.0167)
    # Simplified model of Earth's orbital eccentricity
    dsol = 1.0 / (1.0 - 0.01673 * np.cos(om))**2

    return dsol


__all__ = [
    'possol',
    'day_number',
    'pos_fft',
    'varsol',
    'solirr',
    'equivwl',
]
