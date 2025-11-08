"""
Surface albedo calculations.

Computes hemispheric albedo for conservative (non-absorbing) Rayleigh
atmosphere using exponential integral functions.

Converted from Fortran CSALBR.f

Functions:
    csalbr: Compute surface albedo for given optical thickness
    fintexp1: First exponential integral E1(x)
    fintexp3: Third exponential integral E3(x)
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def fintexp1(xtau):
    """
    Compute first exponential integral E1(x).

    Uses polynomial approximation for 0 < xtau < 1 with accuracy 2e-7.

    E1(x) = -γ - ln(x) + x - x²/4 + x³/18 - x⁴/96 + x⁵/600 + ...

    Parameters
    ----------
    xtau : float
        Optical thickness

    Returns
    -------
    float
        First exponential integral E1(xtau)

    Notes
    -----
    Converted from Fortran FINTEXP1

    Uses Euler-Mascheroni constant γ ≈ 0.57721566...
    """
    # Polynomial coefficients for E1 approximation
    a = np.array([
        -0.57721566,  # -γ (Euler-Mascheroni constant)
        0.99999193,
        -0.24991055,
        0.05519968,
        -0.00976004,
        0.00107857
    ])

    # Polynomial evaluation
    xx = a[0]
    xftau = 1.0
    for i in range(1, 6):
        xftau *= xtau
        xx += a[i] * xftau

    return xx - np.log(xtau)


@jit(nopython=True)
def fintexp3(xtau):
    """
    Compute third exponential integral E3(x).

    E3(x) = [exp(-x) * (1 - x) + x² * E1(x)] / 2

    Parameters
    ----------
    xtau : float
        Optical thickness

    Returns
    -------
    float
        Third exponential integral E3(xtau)

    Notes
    -----
    Converted from Fortran FINTEXP3
    """
    e1 = fintexp1(xtau)
    return (np.exp(-xtau) * (1.0 - xtau) + xtau * xtau * e1) / 2.0


@jit(nopython=True)
def csalbr(xtau):
    """
    Compute hemispheric albedo for conservative Rayleigh atmosphere.

    For a conservative (non-absorbing) Rayleigh scattering atmosphere,
    computes the hemispheric (spherical) albedo as a function of
    optical thickness.

    The formula is:
    A = [3τ - E3(τ) * (4 + 2τ) + 2 * exp(-τ)] / (4 + 3τ)

    where E3 is the third exponential integral.

    Parameters
    ----------
    xtau : float
        Optical thickness

    Returns
    -------
    float
        Hemispheric albedo (0 ≤ A ≤ 1)

    Notes
    -----
    Converted from Fortran CSALBR

    For a conservative Rayleigh atmosphere, the albedo approaches:
    - A → 0 as τ → 0 (optically thin)
    - A → 1 as τ → ∞ (optically thick)
    """
    e3 = fintexp3(xtau)
    numerator = 3.0 * xtau - e3 * (4.0 + 2.0 * xtau) + 2.0 * np.exp(-xtau)
    denominator = 4.0 + 3.0 * xtau
    return numerator / denominator


__all__ = ['csalbr', 'fintexp1', 'fintexp3']
