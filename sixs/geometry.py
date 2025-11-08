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


def _goes_geometry(month, jday, tu, nc, nl, sat_lon):
    """
    Generic GOES satellite geometry calculation.

    Parameters
    ----------
    month : int
        Month (1-12)
    jday : int
        Day of month (1-31)
    tu : float
        Universal time (decimal hours)
    nc : int
        Column number in satellite image
    nl : int
        Line number in satellite image
    sat_lon : float
        Satellite sub-point longitude (degrees)

    Returns
    -------
    asol : float
        Solar zenith angle (degrees)
    phi0 : float
        Solar azimuth angle (degrees)
    avis : float
        Viewing zenith angle (degrees)
    phiv : float
        Viewing azimuth angle (degrees)
    xlon : float
        Scene longitude (degrees)
    xlat : float
        Scene latitude (degrees)

    Raises
    ------
    ValueError
        If pixel coordinates are outside valid range

    Notes
    -----
    Converts GOES pixel coordinates to lat/lon using satellite projection.
    Based on geostationary satellite geometry.
    """
    # Pixel coordinates relative to center
    yr = float(nl) - 8665.5
    xr = float(nc) - 6498.5

    # Earth and satellite parameters
    alti = 42107.0 - 6378.155  # Satellite altitude (km)
    re = 6378.155              # Earth equatorial radius (km)
    aaa = 1.0 / 297.0          # Earth flattening coefficient
    rp = re / (1.0 + aaa)      # Polar radius

    pi = 3.1415926
    cdr = pi / 180.0  # Degrees to radians
    crd = 180.0 / pi  # Radians to degrees

    # Pixel size in degrees
    deltax = 18.0 / 12997.0
    deltay = 20.0 / 17331.0

    # Convert pixel offsets to angular coordinates
    x = xr * deltax * cdr
    y = yr * deltay * cdr

    rs = re + alti  # Satellite distance from Earth center

    # Trigonometric calculations
    tanx = np.tan(x)
    tany = np.tan(y)
    val1 = 1.0 + (tanx**2)
    val2 = 1.0 + (tany * (1.0 + aaa))**2
    yk = rs / re
    cosx2 = 1.0 / (val1 * val2)

    # Check if pixel is within visible disk
    if (1.0 / cosx2) > ((yk**2) / (yk**2 - 1.0)):
        raise ValueError('No possibility to compute lat. and long. - pixel outside Earth disk')

    # Calculate intersection with Earth
    sn = (rs - (re * np.sqrt((yk**2) - (yk**2 - 1.0) * (1.0 / cosx2)))) / (1.0 / cosx2)
    zt = rs - sn
    xt = -(sn * tanx)
    yt = sn * tany / np.cos(x)

    # Convert to lat/lon
    teta = np.arcsin(yt / rp)
    ylat = np.arctan(np.tan(teta) * rp / re)
    ylon = np.arctan2(xt, zt)

    xlat = ylat * crd
    xlon = ylon * crd - sat_lon

    # Calculate solar position
    asol, phi0 = possol(month, jday, tu, xlon, xlat)

    # Calculate viewing geometry
    ylon = xlon * pi / 180.0 + sat_lon * cdr
    ylat = xlat * pi / 180.0
    gam = np.sqrt(((1.0 / cosx2) - 1.0) * cosx2)
    avis = np.arcsin((1.0 + alti / re) * gam)
    avis = avis * 180.0 / pi
    phiv = np.arctan2(np.tan(ylon), np.sin(ylat)) + pi
    phiv = phiv * 180.0 / pi

    return asol, phi0, avis, phiv, xlon, xlat


def posge(month, jday, tu, nc, nl):
    """
    Calculate geometry for GOES East satellite.

    Converts pixel coordinates to geographic coordinates and calculates
    solar and viewing geometry for GOES East geostationary satellite.

    Parameters
    ----------
    month : int
        Month (1-12)
    jday : int
        Day of month (1-31)
    tu : float
        Universal time (decimal hours)
    nc : int
        Column number in GOES image
    nl : int
        Line number in GOES image

    Returns
    -------
    asol : float
        Solar zenith angle (degrees)
    phi0 : float
        Solar azimuth angle (degrees)
    avis : float
        Viewing zenith angle (degrees)
    phiv : float
        Viewing azimuth angle (degrees)
    xlon : float
        Scene longitude (degrees)
    xlat : float
        Scene latitude (degrees)

    Notes
    -----
    Converted from Fortran POSGE.f
    GOES East is positioned at 75°W longitude.
    """
    return _goes_geometry(month, jday, tu, nc, nl, sat_lon=75.0)


def posgw(month, jday, tu, nc, nl):
    """
    Calculate geometry for GOES West satellite.

    Converts pixel coordinates to geographic coordinates and calculates
    solar and viewing geometry for GOES West geostationary satellite.

    Parameters
    ----------
    month : int
        Month (1-12)
    jday : int
        Day of month (1-31)
    tu : float
        Universal time (decimal hours)
    nc : int
        Column number in GOES image
    nl : int
        Line number in GOES image

    Returns
    -------
    asol : float
        Solar zenith angle (degrees)
    phi0 : float
        Solar azimuth angle (degrees)
    avis : float
        Viewing zenith angle (degrees)
    phiv : float
        Viewing azimuth angle (degrees)
    xlon : float
        Scene longitude (degrees)
    xlat : float
        Scene latitude (degrees)

    Notes
    -----
    Converted from Fortran POSGW.f
    GOES West is positioned at 135°W longitude.
    """
    return _goes_geometry(month, jday, tu, nc, nl, sat_lon=135.0)


__all__ = [
    'possol',
    'day_number',
    'pos_fft',
    'varsol',
    'solirr',
    'equivwl',
    'posspo',
    'poslan',
    'posge',
    'posgw',
]
