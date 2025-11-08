"""
Optical depth calculations.

Rayleigh scattering optical depth and related calculations.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def odrayl(wl, z, p, t, delta=0.0279, sigma=0.0):
    """
    Calculate Rayleigh (molecular) optical depth.

    Parameters
    ----------
    wl : float
        Wavelength (micrometers)
    z : ndarray
        Altitude levels (km), 34 elements
    p : ndarray
        Pressure levels (mbar), 34 elements
    t : ndarray
        Temperature levels (K), 34 elements
    delta : float, optional
        Depolarization factor (default: 0.0279)
    sigma : float, optional
        King factor (default: 0.0)

    Returns
    -------
    tray : float
        Rayleigh optical depth

    Notes
    -----
    Uses Edlen 1966 air refraction index formula (Metrologia, 2, 71-80)
    with water vapor pressure pw=0.

    Formula integrates scattering over atmospheric layers:
    - Accounts for pressure and temperature variation with altitude
    - Uses standard molecular number density ns = 2.54743e19 mol/cm³
    """
    pi = 3.1415926
    ak = 1.0 / wl
    awl = wl

    # Air refraction index (Edlen 1966, Metrologia, 2, 71-80)
    # Setting pw=0 (no water vapor correction to refractive index)
    a1 = 130.0 - ak * ak
    a2 = 38.9 - ak * ak
    a3 = 2406030.0 / a1
    a4 = 15997.0 / a2
    an = (8342.13 + a3 + a4) * 1.0e-08
    an = an + 1.0

    # Rayleigh scattering cross-section coefficient
    # 24π³ * (n²-1)² * (6+3δ)/(6-7δ) / (n²+2)² / λ⁴
    a = (24.0 * pi**3) * ((an*an - 1.0)**2) * (6.0 + 3.0 * delta) / (6.0 - 7.0 * delta)
    a = a / ((an*an + 2.0)**2)

    # Integrate over atmospheric layers
    tray = 0.0
    ns = 2.54743e19  # Standard molecular number density (mol/cm³)

    for k in range(33):  # 0 to 32 (33 layers between 34 levels)
        # Average (p/T) over layer
        dppt = (288.15 / 1013.25) * (p[k] / t[k] + p[k+1] / t[k+1]) / 2.0

        # Scattering coefficient (per km)
        sr = (a * dppt / (awl**4) / ns * 1.0e16) * 1.0e05

        # Add layer contribution
        tray = tray + (z[k+1] - z[k]) * sr

    return tray


__all__ = [
    'odrayl',
]
