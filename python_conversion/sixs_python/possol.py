"""
Solar position calculation routines.
Converted from Fortran 77 POSSOL.f

Calculates solar zenithal and azimuthal angles given date, time, and location.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def day_number(jday, month, ia):
    """
    Calculate the day number in the year.

    Parameters
    ----------
    jday : int
        Day of the month (1-31)
    month : int
        Month (1-12)
    ia : int
        Year (for leap year calculation, not used if 0)

    Returns
    -------
    j : int
        Day number in the year (1-366)
    """
    # Fortran: if (month.le.2) then
    if month <= 2:
        j = 31 * (month - 1) + jday
        return j

    # Fortran: if (month.gt.8) then
    if month > 8:
        j = 31 * (month - 1) - (month - 2) // 2 - 2 + jday
    else:
        j = 31 * (month - 1) - (month - 1) // 2 - 2 + jday

    # Leap year correction
    # Fortran: if(ia.ne.0 .and. mod(ia,4).eq.0) j=j+1
    if ia != 0 and ia % 4 == 0:
        j = j + 1

    return j


@jit(nopython=True)
def pos_fft(j, tu, xlon, xlat):
    """
    Calculate solar position using Fourier series approximation.

    Parameters
    ----------
    j : int
        Day number in the year (1-366)
    tu : float
        Universal time (decimal hours, 0-24)
    xlon : float
        Longitude (degrees, positive East)
    xlat : float
        Latitude (degrees, positive North)

    Returns
    -------
    asol : float
        Solar zenith angle (degrees)
    phi0 : float
        Solar azimuth angle (degrees, measured from North going East)
    """
    pi = 3.14159265
    fac = pi / 180.0

    # Mean solar time (decimal hours)
    # Fortran: tsm=tu+xlon/15.
    tsm = tu + xlon / 15.0

    xla = xlat * fac
    xj = float(j)
    tet = 2.0 * pi * xj / 365.0

    # Time equation (in decimal minutes)
    a1 = 0.000075
    a2 = 0.001868
    a3 = 0.032077
    a4 = 0.014615
    a5 = 0.040849
    et = a1 + a2 * np.cos(tet) - a3 * np.sin(tet) - a4 * np.cos(2.0 * tet) - a5 * np.sin(2.0 * tet)
    et = et * 12.0 * 60.0 / pi

    # True solar time
    tsv = tsm + et / 60.0
    tsv = tsv - 12.0

    # Hour angle
    ah = tsv * 15.0 * fac

    # Solar declination (in radians)
    b1 = 0.006918
    b2 = 0.399912
    b3 = 0.070257
    b4 = 0.006758
    b5 = 0.000907
    b6 = 0.002697
    b7 = 0.001480
    delta = (b1 - b2 * np.cos(tet) + b3 * np.sin(tet) -
             b4 * np.cos(2.0 * tet) + b5 * np.sin(2.0 * tet) -
             b6 * np.cos(3.0 * tet) + b7 * np.sin(3.0 * tet))

    # Elevation and azimuth
    amuzero = np.sin(xla) * np.sin(delta) + np.cos(xla) * np.cos(delta) * np.cos(ah)
    elev = np.arcsin(amuzero)

    az = np.cos(delta) * np.sin(ah) / np.cos(elev)

    # Fortran: if ( (abs(az)-1.000).gt.0.00000) az = sign(1.,az)
    # Clamp az to [-1, 1] to avoid numerical issues in arcsin
    if abs(az) > 1.0:
        az = np.sign(az)

    caz = (-np.cos(xla) * np.sin(delta) +
           np.sin(xla) * np.cos(delta) * np.cos(ah)) / np.cos(elev)

    azim = np.arcsin(az)

    # Adjust azimuth to correct quadrant
    # Fortran: if(caz.le.0.) azim=pi-azim
    if caz <= 0.0:
        azim = pi - azim

    # Fortran: if(caz.gt.0.and.az.le.0) azim=2*pi+azim
    if caz > 0.0 and az <= 0.0:
        azim = 2.0 * pi + azim

    azim = azim + pi
    pi2 = 2.0 * pi

    if azim > pi2:
        azim = azim - pi2

    # Convert elevation to degrees
    elev = elev * 180.0 / pi

    # Convert to solar zenith angle
    asol = 90.0 - elev

    # Convert azimuth to degrees
    phi0 = azim / fac

    return asol, phi0


def possol(month, jday, tu, xlon, xlat):
    """
    Calculate solar position (zenith and azimuth angles).

    Parameters
    ----------
    month : int
        Month (1-12)
    jday : int
        Day of the month (1-31)
    tu : float
        Universal time (decimal hours, 0-24)
    xlon : float
        Longitude (degrees, positive East)
    xlat : float
        Latitude (degrees, positive North)

    Returns
    -------
    asol : float
        Solar zenith angle (degrees)
    phi0 : float
        Solar azimuth angle (degrees from North going East)

    Raises
    ------
    ValueError
        If the sun is below the horizon (asol > 90)

    Notes
    -----
    In the original Fortran code, this called print_error() which used
    a COMMON block for error handling. In Python, we use exceptions.
    """
    ia = 0
    nojour = day_number(jday, month, ia)

    asol, phi0 = pos_fft(nojour, tu, xlon, xlat)

    # Fortran: if(asol.gt.90) call print_error('The sun is not raised')
    if asol > 90.0:
        raise ValueError('The sun is not raised')

    return asol, phi0
