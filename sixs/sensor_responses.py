"""
Sensor spectral response functions for 6S.

This module provides spectral response functions for various satellite sensors.
Each function populates a 1501-element array with the sensor's spectral response
and sets the wavelength bounds.

Functions:
    meteo: METEOSAT sensor spectral response
"""

import numpy as np
from typing import Tuple


def meteo() -> Tuple[np.ndarray, float, float]:
    """
    METEOSAT satellite spectral response function.

    The METEOSAT visible channel has a spectral band from approximately
    0.35 to 1.11 micrometers. This function provides the relative spectral
    response across this band at 1501 wavelength points.

    Returns:
        Tuple of (s, wlinf, wlsup):
            s: Spectral response array (1501 elements, values 0-1)
            wlinf: Lower wavelength bound (micrometers)
            wlsup: Upper wavelength bound (micrometers)
    """
    # Initialize spectral response array
    s = np.zeros(1501, dtype=np.float32)

    # METEOSAT spectral response data
    # First 40 values are zero
    sr_data = [
        # Index 40-109 (70 values)
        0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.02,
        0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03,
        0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06,
        0.06, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09,
        0.09, 0.10, 0.10, 0.10, 0.11, 0.11, 0.12,
        0.12, 0.12, 0.13, 0.14, 0.14, 0.15, 0.15,
        0.16, 0.16, 0.17, 0.17, 0.18, 0.18, 0.19,
        0.20, 0.20, 0.21, 0.21, 0.22, 0.23, 0.24,
        0.24, 0.25, 0.26, 0.27, 0.28, 0.28, 0.29,
        0.30, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35,
        # Index 110-179 (70 values)
        0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.40,
        0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.48,
        0.49, 0.50, 0.51, 0.52, 0.53, 0.55, 0.56,
        0.57, 0.58, 0.60, 0.61, 0.62, 0.63, 0.64,
        0.65, 0.65, 0.66, 0.67, 0.67, 0.68, 0.69,
        0.69, 0.70, 0.71, 0.71, 0.72, 0.73, 0.73,
        0.74, 0.76, 0.77, 0.78, 0.78, 0.79, 0.80,
        0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
        0.88, 0.89, 0.89, 0.91, 0.92, 0.93, 0.94,
        0.95, 0.96, 0.96, 0.97, 0.98, 0.98, 0.99,
        # Index 180-249 (70 values)
        0.99, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00,
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.99,
        0.99, 0.99, 0.99, 0.98, 0.98, 0.98, 0.98,
        0.98, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97,
        0.97, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96,
        0.96, 0.96, 0.96, 0.96, 0.95, 0.95, 0.95,
        0.94, 0.93, 0.93, 0.92, 0.92, 0.91, 0.90,
        0.89, 0.89, 0.88, 0.88, 0.87, 0.86, 0.86,
        0.85, 0.85, 0.84, 0.84, 0.83, 0.82, 0.82,
        0.81, 0.80, 0.80, 0.79, 0.79, 0.78, 0.77,
        # Index 250-319 (70 values)
        0.77, 0.76, 0.76, 0.75, 0.75, 0.74, 0.74,
        0.74, 0.73, 0.73, 0.72, 0.71, 0.70, 0.68,
        0.67, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60,
        0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53,
        0.52, 0.51, 0.50, 0.49, 0.49, 0.48, 0.47,
        0.46, 0.45, 0.43, 0.42, 0.41, 0.40, 0.39,
        0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.31,
        0.30, 0.29, 0.28, 0.28, 0.27, 0.25, 0.24,
        0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17,
        0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.11,
        # Index 320-344 (25 values)
        0.10, 0.09, 0.08, 0.08, 0.08, 0.07, 0.06,
        0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03,
        0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.00, 0.00, 0.00,
    ]

    # Fill the spectral response array
    # First 40 values are zero (already initialized)
    # Values 40-344 come from sr_data
    s[40:40 + len(sr_data)] = sr_data
    # Remaining values (345-1500) are zero (already initialized)

    # Wavelength bounds
    wlinf = 0.3499999  # Lower bound (micrometers)
    wlsup = 1.11  # Upper bound (micrometers)

    return s, wlinf, wlsup
