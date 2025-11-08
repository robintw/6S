"""
Polarization utilities for 6S.

This module provides functions for:
- Computing polarization plane direction from Stokes parameters
- Calculating polarized surface reflectance using Nadal-Breon model

Functions:
    dirpopol: Compute plane of polarization from Q and U Stokes parameters
    polnad: Compute polarized surface reflectance (Nadal-Breon model)
"""

import numpy as np
from typing import Tuple


def dirpopol(xq: float, xu: float) -> float:
    """
    Compute the plane of polarization from Stokes parameters.

    Calculates the polarization direction angle from the Q and U Stokes
    parameters of the electromagnetic field.

    Args:
        xq: Stokes Q parameter
        xu: Stokes U parameter

    Returns:
        Polarization direction angle in degrees

    Note:
        The direction of polarization is defined relative to the scattering
        plane. The angle ranges from -90° to +90°, where:
        - 0° indicates polarization parallel to the scattering plane
        - ±90° indicates polarization perpendicular to the scattering plane
        - ±45° indicates diagonal polarization

    Example:
        >>> dirpol = dirpopol(0.1, 0.0)  # Q>0, U≈0
        >>> dirpol  # 0.0 (parallel to scattering plane)
    """
    pi = np.arccos(-1.0)

    # Handle special cases where one parameter is near zero
    if abs(xq) < 0.00001:
        if xu > 0.0:
            return 45.0
        else:
            return -45.0

    if abs(xu) < 0.00001:
        if xq > 0.0:
            return 0.0
        else:
            return 90.0

    # General case: compute arctangent
    if xq > 0.0:
        dirpol = 90.0 / pi * np.arctan(xu / xq)
        return dirpol

    # Q < 0 case: add or subtract 90° depending on sign of U
    if xu > 0.0:
        dirpol = 90.0 + 90.0 / pi * np.arctan(xu / xq)
    else:
        dirpol = -90.0 + 90.0 / pi * np.arctan(xu / xq)

    return dirpol


def polnad(
    xts: float,
    xtv: float,
    phi: float,
    pveg: float
) -> Tuple[float, float]:
    """
    Compute polarized surface reflectance using the Nadal-Breon model.

    This model calculates the polarized components (Q and U Stokes parameters)
    of surface reflectance based on specular reflection from vegetation and
    bare soil. It uses Fresnel equations with a refractive index of 1.5 for
    vegetation.

    Args:
        xts: Solar zenith angle in degrees
        xtv: View zenith angle in degrees
        phi: Relative azimuth angle between sun and view in degrees
        pveg: Vegetation fraction (0-1), used to weight vegetation vs soil

    Returns:
        Tuple of (ropq, ropu):
            ropq: Q Stokes parameter of polarized surface reflectance
            ropu: U Stokes parameter of polarized surface reflectance

    Note:
        The Nadal-Breon model is particularly useful for vegetated surfaces
        where specular reflection contributes to polarization. The model
        accounts for:
        - Fresnel reflection with refractive index N=1.5
        - Different geometries for vegetation and soil
        - Rotation of polarization plane based on viewing geometry

    Reference:
        Nadal, F., & Bréon, F. M. (1999). Parameterization of surface
        polarized reflectance derived from POLDER spaceborne measurements.
        IEEE Transactions on Geoscience and Remote Sensing, 37(3), 1709-1718.

    Example:
        >>> ropq, ropu = polnad(30.0, 0.0, 0.0, 0.8)  # 30° sun, nadir view
        >>> # ropq and ropu are the polarized reflectance components
    """
    # Refractive index of vegetation (assumed constant)
    N = 1.5

    # Convert angles to radians
    pi = np.arccos(0.0) * 2.0
    dtr = pi / 180.0

    # Compute scattering angle
    csca = -np.cos(xts * dtr) * np.cos(xtv * dtr) - \
           np.sin(xts * dtr) * np.sin(xtv * dtr) * np.cos(phi * dtr)
    sca = np.arccos(csca)

    # Incidence angle for specular reflection
    alpha = (pi - sca) / 2.0

    # Refraction angle (Snell's law)
    alphap = np.arcsin(np.sin(alpha) / N)

    # Cosines for Fresnel equations
    mui = np.cos(alpha)  # Incident angle cosine
    mut = np.cos(alphap)  # Transmitted angle cosine

    # Fresnel reflection coefficients (parallel and perpendicular)
    xf1 = (N * mut - mui) / (N * mut + mui)  # Perpendicular
    xf2 = (N * mui - mut) / (N * mui + mut)  # Parallel

    # Polarized reflectance amplitude (Fresnel equation)
    fpalpha = 0.5 * (xf1 * xf1 - xf2 * xf2)

    # Polarized reflectance for vegetation and soil
    # Vegetation: normalized by cosine of sun and view angles
    rpveg = fpalpha / 4.0 / (np.cos(xts * dtr) + np.cos(xtv * dtr))

    # Soil: normalized by product of cosines
    rpsoil = fpalpha / 4.0 / np.cos(xts * dtr) / np.cos(xtv * dtr)

    # Combined surface (weighted average)
    rpsur = rpveg * pveg + rpsoil * (1.0 - pveg)

    # Compute rotation angle for Q and U components
    muv = np.cos(xtv * dtr)
    mus = np.cos(xts * dtr)
    sinv = np.sin(xtv * dtr)

    # Compute ksi (rotation angle in scattering plane)
    if xtv > 0.5:
        if np.sin(phi * dtr) < 0:
            cksi = (muv * csca + mus) / np.sqrt(1.0 - csca * csca) / sinv
        else:
            cksi = -(muv * csca + mus) / np.sqrt(1.0 - csca * csca) / sinv
    else:
        cksi = 0.0

    # Clamp to valid range [-1, 1]
    if cksi > 1.0:
        cksi = 1.0
    if cksi < -1.0:
        cksi = -1.0

    # Compute Q and U Stokes parameters
    # These are rotated by angle ksi from the principal plane
    ropq = rpsur * (2.0 * cksi * cksi - 1.0)
    ropu = -rpsur * 2.0 * cksi * np.sqrt(1.0 - cksi * cksi)

    return ropq, ropu
