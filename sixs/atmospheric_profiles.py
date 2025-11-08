"""
Standard atmospheric profile models.

Contains 6 standard atmospheric profiles from McClatchey et al.:
- Tropical
- Midlatitude Summer
- Midlatitude Winter
- Subarctic Summer
- Subarctic Winter
- US Standard 1962

Each profile contains 34 vertical levels with:
- z: Altitude (km)
- p: Pressure (mbar)
- t: Temperature (K)
- wh: Water vapor density (g/cm³)
- wo: Ozone concentration (cm-atm)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class AtmosphericProfile:
    """
    Atmospheric profile with vertical structure.

    Attributes
    ----------
    name : str
        Profile name
    z : ndarray
        Altitude (km), 34 levels
    p : ndarray
        Pressure (mbar), 34 levels
    t : ndarray
        Temperature (K), 34 levels
    wh : ndarray
        Water vapor density (g/cm³), 34 levels
    wo : ndarray
        Ozone concentration (cm-atm), 34 levels
    """
    name: str
    z: np.ndarray
    p: np.ndarray
    t: np.ndarray
    wh: np.ndarray
    wo: np.ndarray


# Altitude levels (common to all profiles)
_Z_LEVELS = np.array([
    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
    10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
    20., 21., 22., 23., 24., 25., 30., 35., 40., 45.,
    50., 70., 100., 99999.
], dtype=np.float64)


def tropical():
    """
    Tropical atmosphere profile (McClatchey).

    Returns
    -------
    AtmosphericProfile
        Tropical atmospheric profile
    """
    p = np.array([
        1.013e+03, 9.040e+02, 8.050e+02, 7.150e+02, 6.330e+02, 5.590e+02,
        4.920e+02, 4.320e+02, 3.780e+02, 3.290e+02, 2.860e+02, 2.470e+02,
        2.130e+02, 1.820e+02, 1.560e+02, 1.320e+02, 1.110e+02, 9.370e+01,
        7.890e+01, 6.660e+01, 5.650e+01, 4.800e+01, 4.090e+01, 3.500e+01,
        3.000e+01, 2.570e+01, 1.220e+01, 6.000e+00, 3.050e+00, 1.590e+00,
        8.540e-01, 5.790e-02, 3.000e-04, 0.000e+00
    ], dtype=np.float64)

    t = np.array([
        3.000e+02, 2.940e+02, 2.880e+02, 2.840e+02, 2.770e+02, 2.700e+02,
        2.640e+02, 2.570e+02, 2.500e+02, 2.440e+02, 2.370e+02, 2.300e+02,
        2.240e+02, 2.170e+02, 2.100e+02, 2.040e+02, 1.970e+02, 1.950e+02,
        1.990e+02, 2.030e+02, 2.070e+02, 2.110e+02, 2.150e+02, 2.170e+02,
        2.190e+02, 2.210e+02, 2.320e+02, 2.430e+02, 2.540e+02, 2.650e+02,
        2.700e+02, 2.190e+02, 2.100e+02, 2.100e+02
    ], dtype=np.float64)

    wh = np.array([
        1.900e+01, 1.300e+01, 9.300e+00, 4.700e+00, 2.200e+00, 1.500e+00,
        8.500e-01, 4.700e-01, 2.500e-01, 1.200e-01, 5.000e-02, 1.700e-02,
        6.000e-03, 1.800e-03, 1.000e-03, 7.600e-04, 6.400e-04, 5.600e-04,
        5.000e-04, 4.900e-04, 4.500e-04, 5.100e-04, 5.100e-04, 5.400e-04,
        6.000e-04, 6.700e-04, 3.600e-04, 1.100e-04, 4.300e-05, 1.900e-05,
        6.300e-06, 1.400e-07, 1.000e-09, 0.000e+00
    ], dtype=np.float64)

    wo = np.array([
        5.600e-05, 5.600e-05, 5.400e-05, 5.100e-05, 4.700e-05, 4.500e-05,
        4.300e-05, 4.100e-05, 3.900e-05, 3.900e-05, 3.900e-05, 4.100e-05,
        4.300e-05, 4.500e-05, 4.500e-05, 4.700e-05, 4.700e-05, 6.900e-05,
        9.000e-05, 1.400e-04, 1.900e-04, 2.400e-04, 2.800e-04, 3.200e-04,
        3.400e-04, 3.400e-04, 2.400e-04, 9.200e-05, 4.100e-05, 1.300e-05,
        4.300e-06, 8.600e-08, 4.300e-11, 0.000e+00
    ], dtype=np.float64)

    return AtmosphericProfile("Tropical", _Z_LEVELS.copy(), p, t, wh, wo)


def midlatitude_summer():
    """
    Midlatitude summer atmosphere profile (McClatchey).

    Returns
    -------
    AtmosphericProfile
        Midlatitude summer profile
    """
    p = np.array([
        1.013e+03, 9.020e+02, 8.020e+02, 7.100e+02, 6.280e+02, 5.540e+02,
        4.870e+02, 4.260e+02, 3.720e+02, 3.240e+02, 2.810e+02, 2.430e+02,
        2.090e+02, 1.790e+02, 1.530e+02, 1.300e+02, 1.110e+02, 9.500e+01,
        8.120e+01, 6.950e+01, 5.950e+01, 5.100e+01, 4.370e+01, 3.760e+01,
        3.220e+01, 2.770e+01, 1.320e+01, 6.520e+00, 3.330e+00, 1.760e+00,
        9.510e-01, 6.710e-02, 3.000e-04, 0.000e+00
    ], dtype=np.float64)

    t = np.array([
        2.940e+02, 2.900e+02, 2.850e+02, 2.790e+02, 2.730e+02, 2.670e+02,
        2.610e+02, 2.550e+02, 2.480e+02, 2.420e+02, 2.350e+02, 2.290e+02,
        2.220e+02, 2.160e+02, 2.160e+02, 2.160e+02, 2.160e+02, 2.160e+02,
        2.160e+02, 2.170e+02, 2.180e+02, 2.190e+02, 2.200e+02, 2.220e+02,
        2.230e+02, 2.240e+02, 2.340e+02, 2.450e+02, 2.580e+02, 2.700e+02,
        2.760e+02, 2.180e+02, 2.100e+02, 2.100e+02
    ], dtype=np.float64)

    wh = np.array([
        1.400e+01, 9.300e+00, 5.900e+00, 3.300e+00, 1.900e+00, 1.000e+00,
        6.100e-01, 3.700e-01, 2.100e-01, 1.200e-01, 6.400e-02, 2.200e-02,
        6.000e-03, 1.800e-03, 1.000e-03, 7.600e-04, 6.400e-04, 5.600e-04,
        5.000e-04, 4.900e-04, 4.500e-04, 5.100e-04, 5.100e-04, 5.400e-04,
        6.000e-04, 6.700e-04, 3.600e-04, 1.100e-04, 4.300e-05, 1.900e-05,
        6.300e-06, 1.400e-07, 1.000e-09, 0.000e+00
    ], dtype=np.float64)

    wo = np.array([
        6.000e-05, 6.000e-05, 6.000e-05, 6.200e-05, 6.400e-05, 6.600e-05,
        6.900e-05, 7.500e-05, 7.900e-05, 8.600e-05, 9.000e-05, 1.100e-04,
        1.200e-04, 1.500e-04, 1.800e-04, 1.900e-04, 2.100e-04, 2.400e-04,
        2.800e-04, 3.200e-04, 3.400e-04, 3.900e-04, 4.100e-04, 4.300e-04,
        4.500e-04, 4.300e-04, 3.600e-04, 1.100e-04, 4.300e-05, 1.900e-05,
        6.300e-06, 1.400e-07, 1.000e-09, 0.000e+00
    ], dtype=np.float64)

    return AtmosphericProfile("Midlatitude Summer", _Z_LEVELS.copy(), p, t, wh, wo)


def midlatitude_winter():
    """
    Midlatitude winter atmosphere profile (McClatchey).

    Returns
    -------
    AtmosphericProfile
        Midlatitude winter profile
    """
    p = np.array([
        1.018e+03, 8.973e+02, 7.897e+02, 6.938e+02, 6.081e+02, 5.313e+02,
        4.627e+02, 4.016e+02, 3.473e+02, 2.992e+02, 2.568e+02, 2.199e+02,
        1.882e+02, 1.610e+02, 1.378e+02, 1.178e+02, 1.007e+02, 8.610e+01,
        7.350e+01, 6.280e+01, 5.370e+01, 4.580e+01, 3.910e+01, 3.340e+01,
        2.860e+01, 2.430e+01, 1.110e+01, 5.180e+00, 2.530e+00, 1.290e+00,
        6.820e-01, 4.670e-02, 3.000e-04, 0.000e+00
    ], dtype=np.float64)

    t = np.array([
        2.721e+02, 2.688e+02, 2.654e+02, 2.589e+02, 2.523e+02, 2.457e+02,
        2.391e+02, 2.325e+02, 2.259e+02, 2.193e+02, 2.173e+02, 2.173e+02,
        2.173e+02, 2.173e+02, 2.173e+02, 2.173e+02, 2.163e+02, 2.153e+02,
        2.143e+02, 2.133e+02, 2.193e+02, 2.253e+02, 2.273e+02, 2.283e+02,
        2.293e+02, 2.303e+02, 2.333e+02, 2.443e+02, 2.573e+02, 2.653e+02,
        2.703e+02, 2.193e+02, 2.100e+02, 2.100e+02
    ], dtype=np.float64)

    wh = np.array([
        3.500e+00, 2.500e+00, 1.800e+00, 1.200e+00, 6.600e-01, 3.800e-01,
        2.100e-01, 8.500e-02, 3.500e-02, 1.600e-02, 7.500e-03, 6.900e-03,
        6.000e-03, 1.800e-03, 1.000e-03, 7.600e-04, 6.400e-04, 5.600e-04,
        5.000e-04, 4.900e-04, 4.500e-04, 5.100e-04, 5.100e-04, 5.400e-04,
        6.000e-04, 6.700e-04, 3.600e-04, 1.100e-04, 4.300e-05, 1.900e-05,
        6.300e-06, 1.400e-07, 1.000e-09, 0.000e+00
    ], dtype=np.float64)

    wo = np.array([
        6.000e-05, 6.000e-05, 6.000e-05, 6.000e-05, 6.000e-05, 6.000e-05,
        6.000e-05, 6.200e-05, 6.400e-05, 7.100e-05, 7.500e-05, 7.900e-05,
        8.600e-05, 9.000e-05, 1.100e-04, 1.200e-04, 1.500e-04, 1.800e-04,
        1.900e-04, 2.100e-04, 2.400e-04, 2.800e-04, 3.200e-04, 3.400e-04,
        3.900e-04, 4.100e-04, 4.300e-04, 3.600e-04, 1.900e-04, 1.300e-04,
        6.300e-06, 1.400e-07, 1.000e-09, 0.000e+00
    ], dtype=np.float64)

    return AtmosphericProfile("Midlatitude Winter", _Z_LEVELS.copy(), p, t, wh, wo)


def us_standard_1962():
    """
    US Standard Atmosphere 1962 profile (McClatchey).

    Returns
    -------
    AtmosphericProfile
        US Standard 1962 profile
    """
    p = np.array([
        1.013e+03, 8.986e+02, 7.950e+02, 7.012e+02, 6.166e+02, 5.405e+02,
        4.722e+02, 4.111e+02, 3.565e+02, 3.080e+02, 2.650e+02, 2.270e+02,
        1.940e+02, 1.658e+02, 1.417e+02, 1.211e+02, 1.035e+02, 8.850e+01,
        7.565e+01, 6.467e+01, 5.529e+01, 4.729e+01, 4.047e+01, 3.467e+01,
        2.972e+01, 2.549e+01, 1.197e+01, 5.746e+00, 2.871e+00, 1.491e+00,
        7.978e-01, 5.520e-02, 3.008e-04, 0.000e+00
    ], dtype=np.float64)

    t = np.array([
        2.881e+02, 2.816e+02, 2.751e+02, 2.687e+02, 2.622e+02, 2.557e+02,
        2.492e+02, 2.427e+02, 2.362e+02, 2.297e+02, 2.232e+02, 2.168e+02,
        2.166e+02, 2.166e+02, 2.166e+02, 2.166e+02, 2.166e+02, 2.166e+02,
        2.166e+02, 2.166e+02, 2.166e+02, 2.176e+02, 2.186e+02, 2.196e+02,
        2.206e+02, 2.216e+02, 2.265e+02, 2.365e+02, 2.534e+02, 2.642e+02,
        2.706e+02, 2.197e+02, 2.100e+02, 2.100e+02
    ], dtype=np.float64)

    wh = np.array([
        5.900e+00, 4.200e+00, 2.900e+00, 1.800e+00, 1.100e+00, 6.400e-01,
        3.800e-01, 2.100e-01, 8.500e-02, 3.900e-02, 1.900e-02, 9.300e-03,
        4.500e-03, 1.800e-03, 1.000e-03, 7.600e-04, 6.400e-04, 5.600e-04,
        5.000e-04, 4.900e-04, 4.500e-04, 5.100e-04, 5.100e-04, 5.400e-04,
        6.000e-04, 6.700e-04, 3.600e-04, 1.100e-04, 4.300e-05, 1.900e-05,
        6.300e-06, 1.400e-07, 1.000e-09, 0.000e+00
    ], dtype=np.float64)

    wo = np.array([
        6.000e-05, 6.000e-05, 6.000e-05, 6.200e-05, 6.200e-05, 6.400e-05,
        6.600e-05, 6.900e-05, 7.100e-05, 7.500e-05, 7.900e-05, 8.600e-05,
        9.000e-05, 1.100e-04, 1.200e-04, 1.500e-04, 1.800e-04, 1.900e-04,
        2.100e-04, 2.400e-04, 2.800e-04, 3.200e-04, 3.400e-04, 3.900e-04,
        4.100e-04, 4.300e-04, 4.300e-04, 3.600e-04, 1.800e-04, 9.200e-05,
        4.100e-05, 1.300e-05, 4.300e-06, 0.000e+00
    ], dtype=np.float64)

    return AtmosphericProfile("US Standard 1962", _Z_LEVELS.copy(), p, t, wh, wo)


# Profile registry
_PROFILES: Dict[str, callable] = {
    'tropical': tropical,
    'midlatitude_summer': midlatitude_summer,
    'midlatitude_winter': midlatitude_winter,
    'us_standard_1962': us_standard_1962,
}


def get_profile(name: str) -> AtmosphericProfile:
    """
    Get atmospheric profile by name.

    Parameters
    ----------
    name : str
        Profile name: 'tropical', 'midlatitude_summer', 'midlatitude_winter',
        'us_standard_1962'

    Returns
    -------
    AtmosphericProfile
        Requested atmospheric profile

    Raises
    ------
    ValueError
        If profile name not recognized
    """
    name_lower = name.lower()
    if name_lower not in _PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(_PROFILES.keys())}")

    return _PROFILES[name_lower]()


__all__ = [
    'AtmosphericProfile',
    'tropical',
    'midlatitude_summer',
    'midlatitude_winter',
    'us_standard_1962',
    'get_profile',
]
