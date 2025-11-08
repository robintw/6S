"""
Cosine albedo calculation for atmospheric radiative transfer.

Converted from Fortran 77 CSALBR.f

Provides calculation of cosine-weighted albedo using exponential integral
approximations for atmospheric scattering.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def fintexp1(xtau):
    """
    Approximate exponential integral E1(xtau) = integral from xtau to inf of exp(-t)/t dt.

    Uses polynomial approximation for 0 < xtau < 1.
    Accuracy: 2e-07

    Parameters
    ----------
    xtau : float
        Optical depth value (should be > 0)

    Returns
    -------
    e1 : float
        Approximate value of E1(xtau)

    Notes
    -----
    Converted from Fortran FINTEXP1 function.
    Uses coefficients from Abramowitz & Stegun or similar reference.
    """
    # Coefficients for polynomial approximation
    a0 = -0.57721566
    a1 = 0.99999193
    a2 = -0.24991055
    a3 = 0.05519968
    a4 = -0.00976004
    a5 = 0.00107857

    # Polynomial evaluation using Horner's method (more efficient)
    xx = a0
    xftau = 1.0

    xftau = xftau * xtau
    xx = xx + a1 * xftau

    xftau = xftau * xtau
    xx = xx + a2 * xftau

    xftau = xftau * xtau
    xx = xx + a3 * xftau

    xftau = xftau * xtau
    xx = xx + a4 * xftau

    xftau = xftau * xtau
    xx = xx + a5 * xftau

    # Add logarithmic term
    fintexp1_val = xx - np.log(xtau)

    return fintexp1_val


@jit(nopython=True)
def fintexp3(xtau):
    """
    Approximate third exponential integral E3(xtau).

    Calculated using E1 integral.

    Parameters
    ----------
    xtau : float
        Optical depth value

    Returns
    -------
    e3 : float
        Approximate value of E3(xtau)

    Notes
    -----
    Converted from Fortran FINTEXP3 function.
    Relationship: E3(x) = [exp(-x)(1-x) + x^2 E1(x)] / 2
    """
    exp_neg_xtau = np.exp(-xtau)
    e1_val = fintexp1(xtau)

    xx = (exp_neg_xtau * (1.0 - xtau) + xtau * xtau * e1_val) / 2.0

    return xx


@jit(nopython=True)
def csalbr(xtau):
    """
    Calculate cosine-weighted albedo for isotropic scattering.

    Computes the spherical albedo for a homogeneous atmosphere with
    conservative (single scattering albedo = 1) isotropic scattering.

    Parameters
    ----------
    xtau : float
        Optical thickness of the layer

    Returns
    -------
    xalb : float
        Cosine-weighted albedo

    Notes
    -----
    Converted from Fortran CSALBR subroutine.

    The formula used is:
    xalb = [3*tau - E3(tau)*(4 + 2*tau) + 2*exp(-tau)] / (4 + 3*tau)

    where E3(tau) is the third exponential integral.

    This represents the fraction of incident radiation that is
    reflected by the layer when weighted by the cosine of the
    incidence angle (for isotropic incident radiation).

    References
    ----------
    Based on radiative transfer theory for isotropic scattering.
    See Chandrasekhar (1960), "Radiative Transfer" or similar texts.
    """
    exp_neg_xtau = np.exp(-xtau)
    e3_val = fintexp3(xtau)

    # Calculate albedo
    numerator = 3.0 * xtau - e3_val * (4.0 + 2.0 * xtau) + 2.0 * exp_neg_xtau
    denominator = 4.0 + 3.0 * xtau

    xalb = numerator / denominator

    return xalb


__all__ = [
    'csalbr',
    'fintexp1',
    'fintexp3',
]
