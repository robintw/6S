"""
Atmospheric gas absorption coefficient loader.

This module provides access to absorption coefficient lookup tables for
atmospheric gases. Data is loaded from NPZ files extracted from the original
Fortran DATA statements.

Gases covered:
- Water vapor (H2O): WAVA1-6
- Carbon dioxide (CO2): DICA1-3
- Oxygen (O2): OXYG3-6
- Ozone (O3): OZON1
- Nitrous oxide (N2O): NIOX1-6
- Methane (CH4): METH1-6
- Carbon monoxide (CO): MOCA1-6

Functions:
    get_absorption_coefficients: Get coefficients for a gas and spectral region
"""

import numpy as np
from pathlib import Path
from functools import lru_cache


# Module data directory
_DATA_DIR = Path(__file__).parent / 'data' / 'absorption'


@lru_cache(maxsize=64)
def _load_absorption_data(gas_name, region):
    """
    Load absorption data from NPZ file.

    Parameters
    ----------
    gas_name : str
        Gas name (wava, dica, oxyg, ozon, niox, meth, moca)
    region : int
        Spectral region number (1-6)

    Returns
    -------
    dict
        Dictionary with 'coefficients', 'wavenumber_lower', 'wavenumber_upper'
        coefficients: shape (256, 6)
        wavenumber_lower: shape (256,)
        wavenumber_upper: shape (256,)
    """
    filename = f'{gas_name}{region}_absorption.npz'
    filepath = _DATA_DIR / filename

    if not filepath.exists():
        # Return None if file doesn't exist (not all gases in all regions)
        return None

    data = np.load(filepath)
    return {
        'coefficients': data['coefficients'],
        'wavenumber_lower': data['wavenumber_lower'],
        'wavenumber_upper': data['wavenumber_upper'],
    }


def get_absorption_coefficients(idgaz, id, inu):
    """
    Get absorption coefficients for a gas and spectral index.

    Parameters
    ----------
    idgaz : int
        Gas identifier:
        1 = H2O (water vapor)
        2 = CO2 (carbon dioxide)
        3 = O2 (oxygen)
        4 = O3 (ozone)
        5 = N2O (nitrous oxide)
        6 = CH4 (methane)
        7 = CO (carbon monoxide)
    id : int
        Spectral region (1-6)
    inu : int
        Spectral index within region (1-256)

    Returns
    -------
    ndarray or None
        Absorption coefficients [a0, a1, a2, a3, a4, a5, wl_min, wl_max]
        or None if no absorption in this band
    """
    # Map gas ID to name
    gas_map = {
        1: 'wava',
        2: 'dica',
        3: 'oxyg',
        4: 'ozon',
        5: 'niox',
        6: 'meth',
        7: 'moca',
    }

    if idgaz not in gas_map:
        raise ValueError(f"Invalid gas ID: {idgaz}")

    gas_name = gas_map[idgaz]

    # Load absorption data
    data = _load_absorption_data(gas_name, id)

    if data is None:
        # No absorption data for this gas in this region
        return None

    # Check spectral index bounds
    if inu < 1 or inu > 256:
        raise ValueError(f"Spectral index {inu} out of range [1, 256]")

    # Get coefficients for this spectral interval
    idx = inu - 1  # Convert to 0-indexed
    coeffs = data['coefficients'][idx, :]  # Shape (6,)
    wn_lower = data['wavenumber_lower'][idx]
    wn_upper = data['wavenumber_upper'][idx]

    # Check if this interval has data
    if wn_lower == 0 and wn_upper == 0:
        # No data for this interval
        return None

    # Return 8-element array: [6 coefficients, 2 wavenumber bounds]
    a = np.zeros(8, dtype=np.float64)
    a[:6] = coeffs
    a[6] = wn_lower
    a[7] = wn_upper

    return a


def wava1(a, inu):
    """Water vapor absorption, spectral region 1 (2500-5060 cm^-1)."""
    coeffs = get_absorption_coefficients(1, 1, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def wava2(a, inu):
    """Water vapor absorption, spectral region 2 (5060-7620 cm^-1)."""
    coeffs = get_absorption_coefficients(1, 2, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def wava3(a, inu):
    """Water vapor absorption, spectral region 3 (7620-10180 cm^-1)."""
    coeffs = get_absorption_coefficients(1, 3, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def wava4(a, inu):
    """Water vapor absorption, spectral region 4 (10180-12740 cm^-1)."""
    coeffs = get_absorption_coefficients(1, 4, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def wava5(a, inu):
    """Water vapor absorption, spectral region 5 (12740-15300 cm^-1)."""
    coeffs = get_absorption_coefficients(1, 5, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def wava6(a, inu):
    """Water vapor absorption, spectral region 6 (15300+ cm^-1)."""
    coeffs = get_absorption_coefficients(1, 6, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def dica1(a, inu):
    """CO2 absorption, spectral region 1 (2500-5060 cm^-1)."""
    coeffs = get_absorption_coefficients(2, 1, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def dica2(a, inu):
    """CO2 absorption, spectral region 2 (5060-7620 cm^-1)."""
    coeffs = get_absorption_coefficients(2, 2, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def dica3(a, inu):
    """CO2 absorption, spectral region 3 (7620-10180 cm^-1)."""
    coeffs = get_absorption_coefficients(2, 3, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def oxyg3(a, inu):
    """O2 absorption, spectral region 3 (7620-10180 cm^-1)."""
    coeffs = get_absorption_coefficients(3, 3, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def oxyg4(a, inu):
    """O2 absorption, spectral region 4 (10180-12740 cm^-1)."""
    coeffs = get_absorption_coefficients(3, 4, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def oxyg5(a, inu):
    """O2 absorption, spectral region 5 (12740-15300 cm^-1)."""
    coeffs = get_absorption_coefficients(3, 5, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def oxyg6(a, inu):
    """O2 absorption, spectral region 6 (15300+ cm^-1)."""
    coeffs = get_absorption_coefficients(3, 6, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def ozon1(a, inu):
    """O3 absorption, spectral region 1 (2500-5060 cm^-1)."""
    coeffs = get_absorption_coefficients(4, 1, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def niox1(a, inu):
    """N2O absorption, spectral region 1 (2500-5060 cm^-1)."""
    coeffs = get_absorption_coefficients(5, 1, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def niox2(a, inu):
    """N2O absorption, spectral region 2 (5060-7620 cm^-1)."""
    coeffs = get_absorption_coefficients(5, 2, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def niox3(a, inu):
    """N2O absorption, spectral region 3 (7620-10180 cm^-1)."""
    coeffs = get_absorption_coefficients(5, 3, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def niox4(a, inu):
    """N2O absorption, spectral region 4 (10180-12740 cm^-1)."""
    coeffs = get_absorption_coefficients(5, 4, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def niox5(a, inu):
    """N2O absorption, spectral region 5 (12740-15300 cm^-1)."""
    coeffs = get_absorption_coefficients(5, 5, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def niox6(a, inu):
    """N2O absorption, spectral region 6 (15300+ cm^-1)."""
    coeffs = get_absorption_coefficients(5, 6, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def meth1(a, inu):
    """CH4 absorption, spectral region 1 (2500-5060 cm^-1)."""
    coeffs = get_absorption_coefficients(6, 1, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def meth2(a, inu):
    """CH4 absorption, spectral region 2 (5060-7620 cm^-1)."""
    coeffs = get_absorption_coefficients(6, 2, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def meth3(a, inu):
    """CH4 absorption, spectral region 3 (7620-10180 cm^-1)."""
    coeffs = get_absorption_coefficients(6, 3, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def meth4(a, inu):
    """CH4 absorption, spectral region 4 (10180-12740 cm^-1)."""
    coeffs = get_absorption_coefficients(6, 4, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def meth5(a, inu):
    """CH4 absorption, spectral region 5 (12740-15300 cm^-1)."""
    coeffs = get_absorption_coefficients(6, 5, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def meth6(a, inu):
    """CH4 absorption, spectral region 6 (15300+ cm^-1)."""
    coeffs = get_absorption_coefficients(6, 6, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def moca1(a, inu):
    """CO absorption, spectral region 1 (2500-5060 cm^-1)."""
    coeffs = get_absorption_coefficients(7, 1, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def moca2(a, inu):
    """CO absorption, spectral region 2 (5060-7620 cm^-1)."""
    coeffs = get_absorption_coefficients(7, 2, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def moca3(a, inu):
    """CO absorption, spectral region 3 (7620-10180 cm^-1)."""
    coeffs = get_absorption_coefficients(7, 3, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def moca4(a, inu):
    """CO absorption, spectral region 4 (10180-12740 cm^-1)."""
    coeffs = get_absorption_coefficients(7, 4, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def moca5(a, inu):
    """CO absorption, spectral region 5 (12740-15300 cm^-1)."""
    coeffs = get_absorption_coefficients(7, 5, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


def moca6(a, inu):
    """CO absorption, spectral region 6 (15300+ cm^-1)."""
    coeffs = get_absorption_coefficients(7, 6, inu)
    if coeffs is not None:
        a[:] = coeffs
    return a


__all__ = [
    'get_absorption_coefficients',
    'wava1', 'wava2', 'wava3', 'wava4', 'wava5', 'wava6',
    'dica1', 'dica2', 'dica3',
    'oxyg3', 'oxyg4', 'oxyg5', 'oxyg6',
    'ozon1',
    'niox1', 'niox2', 'niox3', 'niox4', 'niox5', 'niox6',
    'meth1', 'meth2', 'meth3', 'meth4', 'meth5', 'meth6',
    'moca1', 'moca2', 'moca3', 'moca4', 'moca5', 'moca6',
]
