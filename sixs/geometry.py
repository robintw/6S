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


def _geostationary_geometry(month, jday, tu, nc, nl, sat_lon,
                            nl_center, nc_center, alti_km, deltax_deg, deltay_deg):
    """
    Generic geostationary satellite geometry calculation.

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
    nl_center : float
        Line number at image center
    nc_center : float
        Column number at image center
    alti_km : float
        Satellite altitude (km)
    deltax_deg : float
        Pixel size in x direction (degrees/pixel)
    deltay_deg : float
        Pixel size in y direction (degrees/pixel)

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
    Converts geostationary satellite pixel coordinates to lat/lon.
    Used for GOES, Meteosat, and other geostationary platforms.
    """
    # Pixel coordinates relative to center
    yr = float(nl) - nl_center
    xr = float(nc) - nc_center

    # Earth and satellite parameters
    re = 6378.155              # Earth equatorial radius (km)
    aaa = 1.0 / 297.0          # Earth flattening coefficient
    rp = re / (1.0 + aaa)      # Polar radius

    pi = 3.1415926
    cdr = pi / 180.0  # Degrees to radians
    crd = 180.0 / pi  # Radians to degrees

    # Convert pixel offsets to angular coordinates
    x = xr * deltax_deg * cdr
    y = yr * deltay_deg * cdr

    rs = re + alti_km  # Satellite distance from Earth center

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
    avis = np.arcsin((1.0 + alti_km / re) * gam)
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
    # GOES East parameters
    nl_center = 8665.5
    nc_center = 6498.5
    alti_km = 42107.0 - 6378.155
    deltax_deg = 18.0 / 12997.0
    deltay_deg = 20.0 / 17331.0
    sat_lon = 75.0

    return _geostationary_geometry(month, jday, tu, nc, nl, sat_lon,
                                    nl_center, nc_center, alti_km, deltax_deg, deltay_deg)


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
    # GOES West parameters (same as East except longitude)
    nl_center = 8665.5
    nc_center = 6498.5
    alti_km = 42107.0 - 6378.155
    deltax_deg = 18.0 / 12997.0
    deltay_deg = 20.0 / 17331.0
    sat_lon = 135.0

    return _geostationary_geometry(month, jday, tu, nc, nl, sat_lon,
                                    nl_center, nc_center, alti_km, deltax_deg, deltay_deg)


def posmto(month, jday, tu, nc, nl):
    """
    Calculate geometry for Meteosat satellite.

    Converts pixel coordinates to geographic coordinates and calculates
    solar and viewing geometry for Meteosat geostationary satellite.

    Parameters
    ----------
    month : int
        Month (1-12)
    jday : int
        Day of month (1-31)
    tu : float
        Universal time (decimal hours)
    nc : int
        Column number in Meteosat image
    nl : int
        Line number in Meteosat image

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
    Converted from Fortran POSMTO.f
    Meteosat is positioned at 0°E longitude (Prime Meridian).
    """
    # Meteosat parameters
    nl_center = 1250.5
    nc_center = 2500.5
    alti_km = 42164.0 - 6378.155
    deltax_deg = 18.0 / 5000.0
    deltay_deg = 18.0 / 2500.0
    sat_lon = 0.0

    return _geostationary_geometry(month, jday, tu, nc, nl, sat_lon,
                                    nl_center, nc_center, alti_km, deltax_deg, deltay_deg)


def posnoa(month, jday, tu, nc, xlonan, hna, campm=1.0):
    """
    Calculate geometry for NOAA polar-orbiting satellite.

    Uses orbital mechanics to calculate geographic coordinates and geometry
    for NOAA sun-synchronous polar-orbiting satellites.

    Parameters
    ----------
    month : int
        Month (1-12)
    jday : int
        Day of month (1-31)
    tu : float
        Universal time (decimal hours)
    nc : int
        Column number in NOAA image (1-2048, with 1024 = nadir)
    xlonan : float
        Longitude of ascending node (degrees)
    hna : float
        Hour at ascending node - equator crossing time (decimal hours)
    campm : float, optional
        Platform multiplier: +1 for AM platform, -1 for PM platform (default: +1)

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
    Converted from Fortran POSNOA.f

    NOAA satellites are in sun-synchronous polar orbits with:
    - Orbital inclination: 98.96°
    - Altitude: 860 km
    - Scan swath: ±55.385° from nadir
    - Image width: 2048 pixels

    The algorithm uses orbital mechanics to compute:
    1. Satellite position from orbital parameters
    2. Earth location from scan geometry
    3. Viewing angles from satellite-to-ground geometry
    """
    # Orbital parameters for NOAA-6
    pi = 3.1415926
    r = 860.0 / 6378.155              # Altitude ratio (dimensionless)
    ai = 98.96 * pi / 180.0           # Orbit inclination (radians)
    an = 360.0 * pi / (6119.0 * 180.0)  # Angular velocity (rad/s)

    # Convert inputs to appropriate units
    ylonan = xlonan * pi / 180.0      # Ascending node longitude (radians)
    t = tu * 3600.0                   # Universal time (seconds)
    hnam = hna * 3600.0               # Ascending node time (seconds)

    # Time since ascending node
    u = t - hnam
    u = campm * u * an                # Orbital angle

    # Scan angle from nadir
    delt = ((nc - (2048.0 + 1.0) / 2.0) * 55.385 / ((2048.0 - 1.0) / 2.0))
    delt = campm * delt * pi / 180.0

    # Viewing zenith angle (satellite perspective)
    avis = np.arcsin((1.0 + r) * np.sin(delt))
    d = avis - delt  # Parallax correction

    # Satellite-to-ground vector components
    y = np.cos(d) * np.cos(ai) * np.sin(u) - np.sin(ai) * np.sin(d)
    z = np.cos(d) * np.sin(ai) * np.sin(u) + np.cos(ai) * np.sin(d)

    # Geographic latitude
    ylat = np.arcsin(z)

    # Geographic longitude calculation
    cosy = np.cos(d) * np.cos(u) / np.cos(ylat)
    siny = y / np.cos(ylat)
    ylon = np.arcsin(siny)

    # Adjust longitude quadrant
    if cosy <= 0.0:
        if siny > 0.0:
            ylon = pi - ylon
        else:
            ylon = -(pi + ylon)

    # Account for Earth rotation and ascending node longitude
    ylo1 = ylon + ylonan - (t - hnam) * 2.0 * pi / 86400.0

    xlat = ylat * 180.0 / pi
    xlon = ylo1 * 180.0 / pi

    # Calculate solar position
    asol, phi0 = possol(month, jday, tu, xlon, xlat)

    # Calculate viewing azimuth angle
    zlat = np.arcsin(np.sin(ai) * np.sin(u))
    zlon = np.arctan2(np.cos(ai) * np.sin(u), np.cos(u))

    if nc != 1024:  # Not at nadir
        xnum = np.sin(zlon - ylon) * np.cos(zlat) / np.sin(np.abs(d))
        xden = (np.sin(zlat) - np.sin(ylat) * np.cos(d)) / np.cos(ylat) / np.sin(np.abs(d))
        phiv = np.arctan2(xnum, xden)
    else:
        phiv = 0.0

    phiv = phiv * 180.0 / pi
    avis = np.abs(avis) * 180.0 / pi

    return asol, phi0, avis, phiv, xlon, xlat


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
    'posmto',
    'posnoa',
]
