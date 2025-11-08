"""
Atmospheric reflectance calculations.

Computes Rayleigh, aerosol, and mixed atmospheric reflectances using
successive orders of scattering. Handles polarized and non-polarized cases.

Converted from Fortran ATMREF.f

Functions:
    atmref: Main atmospheric reflectance calculation
"""

import numpy as np
from sixs.successive_orders import os, ospol


def atmref(iaer, iaer_prof, tamoy, taer, trmoy, pizmoy, piza,
           tamoyp, taerp, trmoyp, palt, phi, xmus, xmuv, phirad,
           nt, mu, np_angles, rm, gb, rp, ipol, xlm1, xlm2, nfi):
    """
    Compute atmospheric reflectances for Rayleigh, aerosol, and mixed.

    Calculates atmospheric reflectance for three cases:
    1. Pure Rayleigh scattering (rorayl)
    2. Pure aerosol scattering (roaero)
    3. Mixed Rayleigh + aerosol (romix)

    Handles both polarized (Stokes I, Q, U) and non-polarized cases.

    Parameters
    ----------
    iaer : int
        Aerosol model type (0=no aerosol)
    iaer_prof : int
        Aerosol profile type
    tamoy : float
        Truncated total optical thickness (aerosol + Rayleigh)
    taer : float
        Total aerosol optical thickness
    trmoy : float
        Total Rayleigh optical thickness
    pizmoy : float
        Truncated single scattering albedo
    piza : float
        Aerosol single scattering albedo
    tamoyp : float
        Truncated optical thickness above observation plane
    taerp : float
        Aerosol optical thickness above plane
    trmoyp : float
        Rayleigh optical thickness above plane
    palt : float
        Observation altitude (km), > 0 for satellite/aircraft
    phi : float
        Relative azimuth angle (degrees)
    xmus : float
        Cosine of solar zenith angle
    xmuv : float
        Cosine of viewing zenith angle
    phirad : float
        Relative azimuth angle (radians)
    nt : int
        Number of atmospheric layers
    mu : int
        Number of Gauss quadrature angles
    np_angles : int
        Number of azimuth angles
    rm : ndarray
        Gauss angle cosines, shape (2*mu+1,)
    gb : ndarray
        Gauss weights, shape (2*mu+1,)
    rp : ndarray
        Azimuth angles (radians), shape (np_angles,)
    ipol : int
        Polarization flag (0=no polarization, 1=polarization only, 2=both)
    xlm1 : ndarray
        Fourier components output, shape (2*mu+1, np_angles)
    xlm2 : ndarray
        Secondary Fourier components, shape (2*mu+1, np_angles)
    nfi : int
        Number of azimuth angles for output (typically 13)

    Returns
    -------
    rorayl : float
        Rayleigh reflectance (Stokes I)
    roaero : float
        Aerosol reflectance (Stokes I)
    romix : float
        Mixed reflectance (Stokes I)
    rqrayl : float
        Rayleigh reflectance (Stokes Q)
    rqaero : float
        Aerosol reflectance (Stokes Q)
    rqmix : float
        Mixed reflectance (Stokes Q)
    rurayl : float
        Rayleigh reflectance (Stokes U)
    ruaero : float
        Aerosol reflectance (Stokes U)
    rumix : float
        Mixed reflectance (Stokes U)
    rorayl_fi : ndarray
        Rayleigh reflectance vs azimuth, shape (nfi,)
    romix_fi : ndarray
        Mixed reflectance vs azimuth, shape (nfi,)
    nfilut : ndarray
        Number of LUT entries per viewing angle, shape (mu,)
    filut : ndarray
        LUT azimuth angles, shape (mu, 41)
    rolut : ndarray
        Mixed reflectance LUT (Stokes I), shape (mu, 41)
    rolutq : ndarray
        Mixed reflectance LUT (Stokes Q), shape (mu, 41)
    rolutu : ndarray
        Mixed reflectance LUT (Stokes U), shape (mu, 41)

    Notes
    -----
    Converted from Fortran ATMREF.f

    Only valid for palt > 0 (satellite/aircraft observation).
    For ground observations, different calculations are used in the main code.
    """
    # Initialize outputs
    rorayl = 0.0
    roaero = 0.0
    romix = 0.0
    rqrayl = 0.0
    rqaero = 0.0
    rqmix = 0.0
    rurayl = 999.0
    ruaero = 999.0
    rumix = 999.0

    rorayl_fi = np.zeros(nfi)
    romix_fi = np.zeros(nfi)

    nfilut = np.zeros(mu, dtype=np.int32)
    filut = np.zeros((mu, 41))
    rolut = np.zeros((mu, 41))
    rolutq = np.zeros((mu, 41))
    rolutu = np.zeros((mu, 41))

    # Temporary arrays for unused LUT returns
    rolutd = np.zeros((mu, 41))

    # Only process if above ground (palt > 0)
    if palt <= 0.0:
        return (rorayl, roaero, romix, rqrayl, rqaero, rqmix,
                rurayl, ruaero, rumix, rorayl_fi, romix_fi,
                nfilut, filut, rolut, rolutq, rolutu)

    # Set viewing geometry at specific angles
    rm_view = rm.copy()
    rm_view[-mu] = -xmuv  # Downward viewing
    rm_view[mu] = xmuv     # Upward (not used for satellite)
    rm_view[0] = -xmus     # Solar direction

    # Initialize Fourier component arrays
    xqm1 = np.zeros_like(xlm1)
    xum1 = np.zeros_like(xlm1)
    xlphim = np.zeros(nfi)

    # ===== RAYLEIGH REFLECTANCE =====
    tamol = 0.0   # No aerosol
    tamolp = 0.0

    xlm1_ray, xqm1_ray, xum1_ray, xlphim_ray, _, _, _, _, nfilut_ray = ospol(
        iaer_prof, tamol, trmoy, piza, tamolp, trmoyp, palt,
        phirad, nt, mu, np_angles, rm_view, gb, rp
    )

    if ipol != 1:  # Not polarization-only mode
        rorayl = xlm1_ray[-mu, 0] / xmus
        romix = rorayl
        for ifi in range(nfi):
            rorayl_fi[ifi] = xlphim_ray[ifi] / xmus
            romix_fi[ifi] = xlphim_ray[ifi] / xmus

    if ipol != 0:  # Polarization mode
        rorayl = xlm1_ray[-mu, 0] / xmus
        rqrayl = xqm1_ray[-mu, 0] / xmus
        rurayl = xum1_ray[-mu, 0] / xmus
        rqmix = rqrayl
        rumix = rurayl
        for ifi in range(nfi):
            rorayl_fi[ifi] = xlphim_ray[ifi] / xmus

    # If no aerosol, we're done
    if iaer == 0:
        romix = rorayl
        rqmix = rqrayl
        rumix = rurayl
        roaero = 0.0
        rqaero = 0.0
        ruaero = 0.0
        return (rorayl, roaero, romix, rqrayl, rqaero, rqmix,
                rurayl, ruaero, rumix, rorayl_fi, romix_fi,
                nfilut, filut, rolut, rolutq, rolutu)

    # ===== AEROSOL REFLECTANCE =====
    tamol = 0.0   # No Rayleigh
    tamolp = 0.0

    if ipol != 1:  # Non-polarized mode
        xlm1_aer, xlphim_aer, _, _, _, _, _ = os(
            iaer_prof, tamoy, tamol, pizmoy, tamoyp, tamolp, palt,
            phirad, nt, mu, np_angles, rm_view, gb, rp
        )
        roaero = xlm1_aer[-mu, 0] / xmus

    if ipol != 0:  # Polarization mode
        xlm1_aer, xqm1_aer, xum1_aer, xlphim_aer, _, _, _, _, _ = ospol(
            iaer_prof, taer, tamol, piza, taerp, tamolp, palt,
            phirad, nt, mu, np_angles, rm_view, gb, rp
        )
        rqaero = xqm1_aer[-mu, 0] / xmus
        ruaero = xum1_aer[-mu, 0] / xmus
        if ipol == 1:
            roaero = xlm1_aer[-mu, 0] / xmus

    # ===== MIXED (RAYLEIGH + AEROSOL) REFLECTANCE =====
    if ipol != 1:  # Non-polarized mode
        xlm1_mix, xlphim_mix, _, _, _, _, _ = os(
            iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
            phirad, nt, mu, np_angles, rm_view, gb, rp
        )
        romix = xlm1_mix[-mu, 0] / xmus
        for ifi in range(nfi):
            romix_fi[ifi] = xlphim_mix[ifi] / xmus

    if ipol != 0:  # Polarization mode
        xlm1_mix, xqm1_mix, xum1_mix, xlphim_mix, rolut_tmp, rolutq_tmp, rolutu_tmp, filut_tmp, nfilut_tmp = ospol(
            iaer_prof, taer, trmoy, piza, taerp, trmoyp, palt,
            phirad, nt, mu, np_angles, rm_view, gb, rp
        )
        rqmix = xqm1_mix[-mu, 0] / xmus
        rumix = xum1_mix[-mu, 0] / xmus

        # Store LUT data
        nfilut[:] = nfilut_tmp
        filut[:, :] = filut_tmp
        rolut[:, :] = rolut_tmp
        rolutq[:, :] = rolutq_tmp
        rolutu[:, :] = rolutu_tmp

        if ipol == 1:
            romix = xlm1_mix[-mu, 0] / xmus
            for ifi in range(nfi):
                romix_fi[ifi] = xlphim_mix[ifi] / xmus

        # Normalize LUT by solar cosine
        for i in range(mu):
            for j in range(41):
                rolut[i, j] /= xmus
                rolutq[i, j] /= xmus
                rolutu[i, j] /= xmus

    return (rorayl, roaero, romix, rqrayl, rqaero, rqmix,
            rurayl, ruaero, rumix, rorayl_fi, romix_fi,
            nfilut, filut, rolut, rolutq, rolutu)


__all__ = ['atmref']
