"""
Aerosol optical depth at 550nm from visibility for 6S.

This module computes aerosol optical thickness at 550nm wavelength
given meteorological visibility using empirical vertical profiles.
"""

import numpy as np


def oda550(iaer: int, v: float, z: np.ndarray = None) -> float:
    """
    Compute aerosol optical thickness at 550nm from visibility.

    Uses empirical vertical distributions of aerosol density for two
    extreme visibility conditions (5km and 23km) and interpolates
    for intermediate values.

    Args:
        iaer: Aerosol type (0 = no aerosols)
        v: Meteorological visibility (km)
        z: Altitude levels (km), 34 values. If None, uses US62 standard

    Returns:
        Aerosol optical thickness at 550nm

    Note:
        The vertical profiles are based on empirical models with particle
        number densities (particles/cm³) for visibility extremes of 5km
        and 23km. The formula interpolates between these extremes.
    """
    # Constants
    sigma = 0.056032  # Scattering cross-section factor

    # Vertical distribution of aerosol density for v=23km (particles/cm³)
    an23 = np.array([
        2.828e+03, 1.244e+03, 5.371e+02, 2.256e+02, 1.192e+02,
        8.987e+01, 6.337e+01, 5.890e+01, 6.069e+01, 5.818e+01, 5.675e+01,
        5.317e+01, 5.585e+01, 5.156e+01, 5.048e+01, 4.744e+01, 4.511e+01,
        4.458e+01, 4.314e+01, 3.634e+01, 2.667e+01, 1.933e+01, 1.455e+01,
        1.113e+01, 8.826e+00, 7.429e+00, 2.238e+00, 5.890e-01, 1.550e-01,
        4.082e-02, 1.078e-02, 5.550e-05, 1.969e-08, 0.000e+00
    ])

    # Vertical distribution of aerosol density for v=5km (particles/cm³)
    an5 = np.array([
        1.378e+04, 5.030e+03, 1.844e+03, 6.731e+02, 2.453e+02,
        8.987e+01, 6.337e+01, 5.890e+01, 6.069e+01, 5.818e+01, 5.675e+01,
        5.317e+01, 5.585e+01, 5.156e+01, 5.048e+01, 4.744e+01, 4.511e+01,
        4.458e+01, 4.314e+01, 3.634e+01, 2.667e+01, 1.933e+01, 1.455e+01,
        1.113e+01, 8.826e+00, 7.429e+00, 2.238e+00, 5.890e-01, 1.550e-01,
        4.082e-02, 1.078e-02, 5.550e-05, 1.969e-08, 0.000e+00
    ])

    # Use US62 standard atmosphere if not provided
    if z is None:
        from sixs.standard_atmospheres import us62
        z, _, _, _, _ = us62()

    # Initialize optical thickness
    taer55 = 0.0

    # Check for no aerosols or invalid visibility
    if abs(v) <= 0.0 or iaer == 0:
        return taer55

    # Integrate over vertical layers
    for k in range(32):
        # Layer thickness
        dz = z[k + 1] - z[k]

        # Aerosol densities at layer boundaries
        bn5 = an5[k]
        bn51 = an5[k + 1]
        bn23 = an23[k]
        bn231 = an23[k + 1]

        # Linear interpolation coefficients
        # bnz(v) = az/v - bz
        az = (115.0 / 18.0) * (bn5 - bn23)
        az1 = (115.0 / 18.0) * (bn51 - bn231)
        bz = (5.0 * bn5 / 18.0) - (23.0 * bn23 / 18.0)
        bz1 = (5.0 * bn51 / 18.0) - (23.0 * bn231 / 18.0)

        # Interpolated density at current visibility
        bnz = az / v - bz
        bnz1 = az1 / v - bz1

        # Avoid log of negative or zero values
        if bnz > 0 and bnz1 > 0:
            # Geometric mean of densities, times layer thickness
            ev = dz * np.exp((np.log(bnz) + np.log(bnz1)) * 0.5)
            # Add contribution to optical thickness
            taer55 += ev * sigma * 1.0e-03

    return taer55
