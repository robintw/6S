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


def posspo(month, jday, tu, xlon, xlat):
    """
    Calculate solar and viewing geometry for SPOT satellite.

    SPOT is a nadir-viewing satellite, so viewing zenith and azimuth are zero.

    Parameters
    ----------
    month : int
        Month (1-12)
    jday : int
        Day of month (1-31)
    tu : float
        Universal time (decimal hours)
    xlon : float
        Scene center longitude (degrees, -180 to 180)
    xlat : float
        Scene center latitude (degrees, -90 to 90)

    Returns
    -------
    asol : float
        Solar zenith angle (degrees)
    phi0 : float
        Solar azimuth angle (degrees)
    avis : float
        Viewing zenith angle (degrees, always 0 for nadir)
    phiv : float
        Viewing azimuth angle (degrees, always 0 for nadir)

    Notes
    -----
    Converted from Fortran POSSPO.f
    SPOT observes in nadir viewing mode.
    """
    # SPOT is nadir viewing
    avis = 0.0
    phiv = 0.0

    # Calculate solar position
    asol, phi0 = possol(month, jday, tu, xlon, xlat)

    return asol, phi0, avis, phiv


def poslan(month, jday, tu, xlon, xlat):
    """
    Calculate solar and viewing geometry for Landsat satellite.

    Landsat is a nadir-viewing satellite, so viewing zenith and azimuth are zero.

    Parameters
    ----------
    month : int
        Month (1-12)
    jday : int
        Day of month (1-31)
    tu : float
        Universal time (decimal hours)
    xlon : float
        Scene center longitude (degrees, -180 to 180)
    xlat : float
        Scene center latitude (degrees, -90 to 90)

    Returns
    -------
    asol : float
        Solar zenith angle (degrees)
    phi0 : float
        Solar azimuth angle (degrees)
    avis : float
        Viewing zenith angle (degrees, always 0 for nadir)
    phiv : float
        Viewing azimuth angle (degrees, always 0 for nadir)

    Notes
    -----
    Converted from Fortran POSLAN.f
    Landsat observes in nadir viewing mode.
    """
    # Landsat is nadir viewing
    avis = 0.0
    phiv = 0.0

    # Calculate solar position
    asol, phi0 = possol(month, jday, tu, xlon, xlat)

    return asol, phi0, avis, phiv


__all__ = [
    'possol',
    'day_number',
    'pos_fft',
    'varsol',
    'solirr',
    'equivwl',
    'posspo',
    'poslan',
]
