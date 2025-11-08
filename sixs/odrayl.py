"""
Rayleigh optical depth calculation for 6S.

This module computes the molecular (Rayleigh) optical depth as a function
of wavelength using the air refraction index formula.
"""

import numpy as np


def odrayl(wl: float, z: np.ndarray = None, p: np.ndarray = None,
           t: np.ndarray = None) -> float:
    """
    Compute Rayleigh optical depth at a given wavelength.

    Uses the Edlen (1966) air refraction index formula and integrates
    over the atmospheric profile to compute total Rayleigh scattering
    optical depth.

    Args:
        wl: Wavelength in micrometers
        z: Altitude profile (km), 34 levels. If None, uses US62 standard
        p: Pressure profile (mb), 34 levels
        t: Temperature profile (K), 34 levels

    Returns:
        Rayleigh optical depth (dimensionless)

    Reference:
        Edlen, B. (1966). The refractive index of air. Metrologia, 2(2), 71-80.
    """
    # Use US62 standard atmosphere if not provided
    if z is None or p is None or t is None:
        from sixs.standard_atmospheres import us62
        z, p, t, _, _ = us62()

    # Constants
    pi = np.pi
    delta = 0.0279  # Depolarization factor
    sigma = 0.056032  # King factor (6+3δ)/(6-7δ)
    ns = 2.54743e19  # Standard number density at STP (molecules/cm³)

    # Wavenumber (inverse wavelength)
    ak = 1.0 / wl
    awl = wl

    # Air refraction index (Edlen 1966, with pw=0 for dry air)
    a1 = 130.0 - ak * ak
    a2 = 38.9 - ak * ak
    a3 = 2406030.0 / a1
    a4 = 15997.0 / a2
    an = (8342.13 + a3 + a4) * 1.0e-08
    an = an + 1.0

    # Rayleigh scattering cross-section
    # σ = (24π³(n²-1)²(6+3δ))/(λ⁴N²(n²+2)²(6-7δ))
    a = (24.0 * pi**3) * ((an * an - 1.0)**2) * (6.0 + 3.0 * delta) / \
        (6.0 - 7.0 * delta) / ((an * an + 2.0)**2)

    # Integrate over atmospheric layers
    tray = 0.0
    for k in range(33):
        # Average density factor for layer
        dppt = (288.15 / 1013.25) * (p[k] / t[k] + p[k + 1] / t[k + 1]) / 2.0

        # Scattering coefficient (km⁻¹)
        sr = (a * dppt / (awl**4) / ns * 1.0e+16) * 1.0e+05

        # Optical depth contribution from layer
        tray += (z[k + 1] - z[k]) * sr

    return tray
