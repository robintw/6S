"""
Hyperspectral sensor response functions.

Spectral response functions for hyperspectral blue bands.
"""

import numpy as np
from numba import jit


# Spectral response data for blue bands
# Band 1: MODIS band 3 (vegetation monitoring at 500m / MVI)
_SR_MODIS_BAND3 = np.array([
    *([0.0] * 75),
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0007, 0.0002, 0.0001, 0.0001, 0.0001,
    0.0001, 0.0001,
    *([0.0] * 1400)
], dtype=np.float64)

# Band 2: Enhanced Thematic Mapper (ETM+) band 1
_SR_ETM_BAND1 = np.array([
    *([0.0] * 74),
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.9800, 0.9770, 0.9650, 0.9628,
    0.9950, 0.9895, 0.9900, 0.9791, 0.9830, 0.9691, 0.9600, 0.7768,
    0.2930, 0.0510, 0.0090,
    *([0.0] * 1392)
], dtype=np.float64)

# Wavelength bounds for each band (micrometers)
_WAVELENGTH_BOUNDS = {
    1: (0.4375, 0.500),  # MODIS band 3
    2: (0.435, 0.52),    # ETM+ band 1
}


def hypblue(iwa):
    """
    Get hyperspectral blue band response function.

    Parameters
    ----------
    iwa : int
        Band selector:
        - 1: MODIS band 3 (vegetation monitoring)
        - 2: Enhanced Thematic Mapper band 1

    Returns
    -------
    s : ndarray
        Spectral response array (1501 elements)
    wlinf : float
        Lower wavelength bound (μm)
    wlsup : float
        Upper wavelength bound (μm)

    Notes
    -----
    MODIS band 3: Blue band for vegetation monitoring at 500m resolution
    ETM+ band 1: Blue band of Enhanced Thematic Mapper
    """
    if iwa == 1:
        s = _SR_MODIS_BAND3.copy()
    elif iwa == 2:
        s = _SR_ETM_BAND1.copy()
    else:
        raise ValueError(f"Invalid band selector iwa={iwa}. Must be 1 or 2.")

    wlinf, wlsup = _WAVELENGTH_BOUNDS[iwa]

    return s, wlinf, wlsup


__all__ = [
    'hypblue',
]
