"""
Atmospheric scattering physics modules.

Functions for calculating molecular (Rayleigh) and aerosol scattering
in the atmosphere.

Functions:
    chand: Chandrasekhar function for molecular reflectance
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def chand(xphi, xmuv, xmus, xtau):
    """
    Chandrasekhar function for molecular (Rayleigh) reflectance.

    Calculates the reflection function of a molecular atmosphere using
    the Chandrasekhar H-function approximation.

    Parameters
    ----------
    xphi : float
        Azimuthal difference between sun and observation (degrees, 0-360)
        xphi=0 in backscattering direction
    xmuv : float
        Cosine of the observation (view) zenith angle (0-1)
    xmus : float
        Cosine of the sun zenith angle (0-1)
    xtau : float
        Molecular optical depth (>0)

    Returns
    -------
    xrray : float
        Molecular reflectance (0-1)

    Notes
    -----
    Converted from Fortran CHAND.f

    Uses a polynomial approximation to the Chandrasekhar H-function
    for Rayleigh scattering. Includes depolarization factor (0.0279)
    and accounts for multiple scattering.

    The algorithm computes three Fourier components of the reflectance:
    - Component 1: Isotropic term (m=0)
    - Component 2: First azimuthal harmonic (m=1)
    - Component 3: Second azimuthal harmonic (m=2)

    References
    ----------
    Chandrasekhar, S., Radiative Transfer, Dover Publications, 1960
    """
    # Coefficients for polynomial approximation
    as0 = np.array([
        0.33243832, -6.777104e-02, 0.16285370,
        1.577425e-03, -0.30924818, -1.240906e-02, -0.10324388,
        3.241678e-02, 0.11493334, -3.503695e-02
    ])
    as1 = np.array([0.19666292, -5.439061e-02])
    as2 = np.array([0.14545937, -2.910845e-02])

    pi = 3.1415927
    fac = pi / 180.0

    # Phase angle (180° - azimuthal difference)
    phios = 180.0 - xphi

    # Cosine of phase angle multiples
    xcosf1 = 1.0
    xcosf2 = np.cos(phios * fac)
    xcosf3 = np.cos(2.0 * phios * fac)

    # Depolarization factor for air molecules
    xdep = 0.0279
    xbeta2 = 0.5

    # Depolarization correction
    xfd = xdep / (2.0 - xdep)
    xfd = (1.0 - xfd) / (1.0 + 2.0 * xfd)

    # Phase function components
    xph1 = 1.0 + (3.0 * xmus**2 - 1.0) * (3.0 * xmuv**2 - 1.0) * xfd / 8.0

    xph2 = -xmus * xmuv * np.sqrt(1.0 - xmus**2) * np.sqrt(1.0 - xmuv**2)
    xph2 = xph2 * xfd * xbeta2 * 1.5

    xph3 = (1.0 - xmus**2) * (1.0 - xmuv**2)
    xph3 = xph3 * xfd * xbeta2 * 0.375

    # Primary scattering contribution
    xitm = (1.0 - np.exp(-xtau * (1.0 / xmus + 1.0 / xmuv))) * xmus / (4.0 * (xmus + xmuv))
    xp1 = xph1 * xitm
    xp2 = xph2 * xitm
    xp3 = xph3 * xitm

    # Multiple scattering contribution
    xitm = (1.0 - np.exp(-xtau / xmus)) * (1.0 - np.exp(-xtau / xmuv))
    cfonc1 = xph1 * xitm
    cfonc2 = xph2 * xitm
    cfonc3 = xph3 * xitm

    # Polynomial basis functions
    xlntau = np.log(xtau)
    pl = np.zeros(10)
    pl[0] = 1.0
    pl[1] = xlntau
    pl[2] = xmus + xmuv
    pl[3] = xlntau * pl[2]
    pl[4] = xmus * xmuv
    pl[5] = xlntau * pl[4]
    pl[6] = xmus**2 + xmuv**2
    pl[7] = xlntau * pl[6]
    pl[8] = xmus**2 * xmuv**2
    pl[9] = xlntau * pl[8]

    # Calculate Fourier series coefficients
    fs0 = 0.0
    for i in range(10):
        fs0 += pl[i] * as0[i]

    fs1 = pl[0] * as1[0] + pl[1] * as1[1]
    fs2 = pl[0] * as2[0] + pl[1] * as2[1]

    # Total contribution for each Fourier component
    xitot1 = xp1 + cfonc1 * fs0 * xmus
    xitot2 = xp2 + cfonc2 * fs1 * xmus
    xitot3 = xp3 + cfonc3 * fs2 * xmus

    # Sum Fourier series
    xrray = xitot1 * xcosf1
    xrray += xitot2 * xcosf2 * 2.0
    xrray += xitot3 * xcosf3 * 2.0
    xrray /= xmus

    return xrray


__all__ = [
    'chand',
]
