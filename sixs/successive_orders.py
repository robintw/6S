"""
Successive orders of scattering calculations.

This module implements the ISO (Isotropic/Successive Orders) algorithm
for computing multiple scattering in the atmosphere.

Converted from Fortran ISO.f, KERNEL.f, DISCRE.f, and AEROPROF.f

Functions:
    discre: Discretize atmosphere layers
    kernel: Compute scattering kernels
    aero_prof: Aerosol profile layering
    iso: Successive orders of scattering calculation
"""

import numpy as np
from numba import jit
import warnings

# Constants from paramdef.inc
NT_P = 30        # Default number of layers
MU_P = 25        # Default number of Gauss angles
MU2_P = 48       # Must be (MU_P-1)*2
NP_P = 49
NFI_P = 181
NQUAD_P = 83     # Must be odd
NT_P_MAX = 100   # Maximum layers
NQMAX_P = 1001   # Maximum quadrature points
NQDEF_P = 83


# Module-level variables for COMMON blocks
# These will be set by calling functions
class AtmosphereState:
    """State variables for atmosphere calculations (replaces COMMON blocks)."""
    def __init__(self):
        self.delta = 0.0279  # Depolarization factor for air
        self.sigma = 0.0     # Not used in current implementation
        self.igmax = 20      # Maximum scattering orders
        self.nquad = NQUAD_P  # Number of quadrature points

        # Aerosol phase function parameters (for KERNEL)
        self.pha = np.zeros(NQMAX_P)
        self.qha = np.zeros(NQMAX_P)
        self.uha = np.zeros(NQMAX_P)
        self.alphal = np.zeros(NQMAX_P + 1)
        self.betal = np.zeros(NQMAX_P + 1)
        self.gammal = np.zeros(NQMAX_P + 1)
        self.zetal = np.zeros(NQMAX_P + 1)

        # Aerosol profile data (for AEROPROF)
        self.num_z = 0
        self.alt_z = np.zeros(NT_P_MAX + 1)
        self.taer_z = np.zeros(NT_P_MAX + 1)
        self.taer55_z = np.zeros(NT_P_MAX + 1)


# Global atmosphere state
_atm_state = AtmosphereState()


def set_aerosol_phase_function(betal_values):
    """
    Set aerosol phase function expansion coefficients.

    Parameters
    ----------
    betal_values : array_like
        Legendre expansion coefficients for aerosol phase function
    """
    global _atm_state
    n = min(len(betal_values), NQMAX_P + 1)
    _atm_state.betal[:n] = betal_values[:n]


def set_aerosol_profile(altitudes, optical_depths):
    """
    Set vertical aerosol profile.

    Parameters
    ----------
    altitudes : array_like
        Altitude levels (km) from top to bottom
    optical_depths : array_like
        Aerosol optical depth at each level
    """
    global _atm_state
    _atm_state.num_z = len(altitudes)
    _atm_state.alt_z[:len(altitudes)] = altitudes
    _atm_state.taer_z[:len(optical_depths)] = optical_depths


def discre(ta, ha, tr, hr, it, nt, yy, dd, ppp2, ppp1):
    """
    Discretize atmosphere to find altitude for given optical depth.

    Uses binary search to find the altitude where the cumulative
    optical depth equals a target value, accounting for exponential
    atmosphere structure.

    Parameters
    ----------
    ta : float
        Total aerosol optical depth
    ha : float
        Aerosol scale height (km)
    tr : float
        Total Rayleigh optical depth
    hr : float
        Rayleigh scale height (km, typically 8)
    it : int
        Current layer index
    nt : int
        Total number of layers
    yy : float
        Optical depth at previous layer
    dd : float
        Previous delta (mixing ratio)
    ppp2 : float
        Upper altitude bound for search (km)
    ppp1 : float
        Lower altitude bound for search (km)

    Returns
    -------
    zx : float
        Altitude (km) where optical depth target is reached
    delta : float
        Rayleigh mixing ratio at this altitude

    Notes
    -----
    Converted from Fortran DISCRE.f

    The function finds altitude z where:
        ta*exp(-z/ha) + tr*exp(-z/hr) = target_optical_depth
    """
    if ha >= 7.0:
        warnings.warn("Check aerosol measurements or plane altitude: ha >= 7")
        return ppp2, 1.0

    # Determine target optical depth increment
    if it == 0:
        dt = 1e-17
    else:
        dt = 2.0 * (ta + tr - yy) / (nt - it + 1.0)

    # Binary search with adaptive step size
    while True:
        dt = dt / 2.0
        ti = yy + dt
        y1 = ppp2
        y3 = ppp1

        # Binary search for altitude
        while True:
            y2 = (y1 + y3) * 0.5
            xx = -y2 / ha

            # Compute optical depth at altitude y2
            if xx < -18:
                x2 = tr * np.exp(-y2 / hr)
            else:
                x2 = ta * np.exp(xx) + tr * np.exp(-y2 / hr)

            xd = abs(ti - x2)

            # Convergence check
            if xd < 0.00001:
                break

            # Adjust search bounds
            if ti < x2:
                y3 = y2
            else:
                y1 = y2

        zx = y2

        # Compute mixing ratio delta (fraction of Rayleigh)
        exp_term = (zx - ppp1) * (1.0 / hr - 1.0 / ha)
        if exp_term < -18:
            delta = 1.0
        else:
            delta = 1.0 / (1.0 + ta * hr / tr / ha * np.exp(exp_term))

        # Check if delta change is acceptable
        ecart = 0.0
        if dd != 0:
            ecart = abs((dd - delta) / dd)

        # If delta changed too much, reduce dt and try again
        if ecart > 0.75 and it != 0:
            continue
        else:
            break

    return zx, delta


def kernel(is_val, mu, rm, betal_vals=None):
    """
    Compute scattering kernel and Legendre polynomials.

    Calculates associated Legendre polynomials and scattering kernel
    for radiative transfer calculations.

    Parameters
    ----------
    is_val : int
        Fourier component index (0, 1, 2, ...)
    mu : int
        Number of Gauss angles
    rm : array_like
        Cosines of Gauss angles, indexed from -mu to +mu
        Shape: (2*mu+1,)
    betal_vals : array_like, optional
        Legendre expansion coefficients for aerosol phase function
        If None, uses global state

    Returns
    -------
    xpl : ndarray
        Second-order Legendre polynomials, shape (2*mu+1,)
    psl : ndarray
        Associated Legendre polynomials, shape (NQMAX_P+2, 2*mu+1)
    bp : ndarray
        Scattering kernel, shape (mu+1, 2*mu+1)

    Notes
    -----
    Converted from Fortran KERNEL.f

    Uses recurrence relations to compute associated Legendre polynomials.
    """
    global _atm_state

    if betal_vals is None:
        betal = _atm_state.betal
    else:
        betal = betal_vals

    nquad = _atm_state.nquad
    ip1 = nquad - 3
    rac3 = np.sqrt(3.0)

    # Initialize arrays
    psl = np.zeros((NQMAX_P + 2, 2*mu + 1))
    bp = np.zeros((mu + 1, 2*mu + 1))
    xpl = np.zeros(2*mu + 1)

    # Compute Legendre polynomials based on Fourier component
    if is_val == 0:
        # m=0 case (azimuthally averaged)
        for j in range(mu + 1):
            c = rm[j]
            psl[0, mu - j] = 1.0
            psl[0, mu + j] = 1.0
            psl[1, mu + j] = c
            psl[1, mu - j] = -c
            xdb = (3.0 * c * c - 1.0) * 0.5
            if abs(xdb) < 1e-30:
                xdb = 0.0
            psl[2, mu - j] = xdb
            psl[2, mu + j] = xdb
        psl[1, mu] = rm[0]  # j=0 case

    elif is_val == 1:
        # m=1 case
        for j in range(mu + 1):
            c = rm[j]
            x = 1.0 - c * c
            psl[0, mu + j] = 0.0
            psl[0, mu - j] = 0.0
            psl[1, mu - j] = np.sqrt(x * 0.5)
            psl[1, mu + j] = np.sqrt(x * 0.5)
            psl[2, mu + j] = c * psl[1, mu + j] * rac3
            psl[2, mu - j] = -psl[2, mu + j]
        psl[2, mu] = -psl[2, mu]  # j=0 correction

    else:
        # m >= 2 case
        a = 1.0
        for i in range(1, is_val + 1):
            a = a * np.sqrt((i + is_val) / i) * 0.5
        b = a * np.sqrt(is_val / (is_val + 1.0)) * np.sqrt((is_val - 1.0) / (is_val + 2.0))

        for j in range(mu + 1):
            c = rm[j]
            xx = 1.0 - c * c
            psl[is_val - 1, mu + j] = 0.0
            xdb = a * xx ** (is_val * 0.5)
            if abs(xdb) < 1e-30:
                xdb = 0.0
            psl[is_val, mu - j] = xdb
            psl[is_val, mu + j] = xdb

    # Recurrence relation for higher order polynomials
    k = 2
    ip = ip1
    if is_val > 2:
        k = is_val

    if k < ip:
        ig = -1
        if is_val == 1:
            ig = 1

        for l in range(k, ip):
            lp = l + 1
            lm = l - 1
            a = (2 * l + 1.0) / np.sqrt((l + is_val + 1.0) * (l - is_val + 1.0))
            b = np.sqrt(float((l + is_val) * (l - is_val))) / (2.0 * l + 1.0)

            for j in range(mu + 1):
                c = rm[j]
                xdb = a * (c * psl[l, mu + j] - b * psl[lm, mu + j])
                if abs(xdb) < 1e-30:
                    xdb = 0.0
                psl[lp, mu + j] = xdb
                if j != 0:
                    psl[lp, mu - j] = ig * psl[lp, mu + j]
            ig = -ig

    # Extract second-order polynomials
    for j in range(2*mu + 1):
        xpl[j] = psl[2, j]

    # Compute scattering kernel bp
    ij = ip1
    for j in range(mu + 1):
        for k_idx in range(2*mu + 1):
            sbp = 0.0
            if is_val <= ij:
                for l in range(is_val, ij + 1):
                    bt = betal[l]
                    sbp += psl[l, mu + j] * psl[l, k_idx] * bt
            if abs(sbp) < 1e-30:
                sbp = 0.0
            bp[j, k_idx] = sbp

    return xpl, psl, bp


def kernelpol(is_val, mu, rm, alphal_vals=None, betal_vals=None,
              gammal_vals=None, zetal_vals=None):
    """
    Compute polarized scattering kernels and associated Legendre polynomials.

    Extends kernel() to handle polarized radiative transfer with Stokes parameters.
    Computes P, R, T components of associated Legendre polynomials for polarization
    and generates six scattering kernel matrices.

    Parameters
    ----------
    is_val : int
        Fourier component index (0, 1, 2, ...)
    mu : int
        Number of Gauss angles
    rm : array_like
        Cosines of Gauss angles, shape (2*mu+1,)
    alphal_vals : array_like, optional
        Legendre expansion coefficients for alpha (a1 component)
        If None, uses global state
    betal_vals : array_like, optional
        Legendre expansion coefficients for beta (a2 component)
        If None, uses global state
    gammal_vals : array_like, optional
        Legendre expansion coefficients for gamma (a3 component)
        If None, uses global state
    zetal_vals : array_like, optional
        Legendre expansion coefficients for zeta (a4 component)
        If None, uses global state

    Returns
    -------
    xpl : ndarray
        P-component second-order polynomials, shape (2*mu+1,)
    xrl : ndarray
        R-component second-order polynomials, shape (2*mu+1,)
    xtl : ndarray
        T-component second-order polynomials, shape (2*mu+1,)
    bp : ndarray
        P-component scattering kernel, shape (mu+1, 2*mu+1)
    gr : ndarray
        R-component scattering kernel, shape (mu+1, 2*mu+1)
    gt : ndarray
        T-component scattering kernel, shape (mu+1, 2*mu+1)
    arr : ndarray
        Alpha-R kernel matrix, shape (mu+1, 2*mu+1)
    art : ndarray
        Alpha-T kernel matrix, shape (mu+1, 2*mu+1)
    att : ndarray
        Alpha-T-T kernel matrix, shape (mu+1, 2*mu+1)

    Notes
    -----
    Converted from Fortran KERNELPOL.f

    The three components P, R, T correspond to the generalized spherical functions
    for polarization:
    - P: Intensity (Stokes I) component
    - R: Linear polarization Q component (horizontal/vertical)
    - T: Linear polarization U component (±45°)

    The kernels combine aerosol scattering (via alpha, beta, gamma, zeta coefficients)
    with molecular Rayleigh scattering to produce the full polarized phase matrix.
    """
    global _atm_state

    # Get phase function coefficients
    if alphal_vals is None:
        alphal = _atm_state.alphal
    else:
        alphal = alphal_vals

    if betal_vals is None:
        betal = _atm_state.betal
    else:
        betal = betal_vals

    if gammal_vals is None:
        gammal = _atm_state.gammal
    else:
        gammal = gammal_vals

    if zetal_vals is None:
        zetal = _atm_state.zetal
    else:
        zetal = zetal_vals

    nquad = _atm_state.nquad
    max_l = nquad - 3
    rac3 = np.sqrt(3.0)

    # Initialize arrays
    # psl: P-component Legendre polynomials
    # rsl: R-component Legendre polynomials
    # tsl: T-component Legendre polynomials
    psl = np.zeros((max_l + 2, 2*mu + 1))
    rsl = np.zeros((max_l + 2, 2*mu + 1))
    tsl = np.zeros((max_l + 2, 2*mu + 1))

    # Output arrays
    xpl = np.zeros(2*mu + 1)
    xrl = np.zeros(2*mu + 1)
    xtl = np.zeros(2*mu + 1)
    bp = np.zeros((mu + 1, 2*mu + 1))
    gr = np.zeros((mu + 1, 2*mu + 1))
    gt = np.zeros((mu + 1, 2*mu + 1))
    arr = np.zeros((mu + 1, 2*mu + 1))
    art = np.zeros((mu + 1, 2*mu + 1))
    att = np.zeros((mu + 1, 2*mu + 1))

    # Compute initial Legendre polynomials based on Fourier component
    if is_val == 0:
        # m=0 case (azimuthally averaged)
        for j in range(mu + 1):
            c = rm[mu + j] if j > 0 else rm[mu]
            # P-component
            psl[0, mu + j] = 1.0
            psl[0, mu - j] = 1.0
            psl[1, mu + j] = c
            psl[1, mu - j] = -c
            xdb = (3.0 * c * c - 1.0) * 0.5
            if abs(xdb) < 1e-30:
                xdb = 0.0
            psl[2, mu + j] = xdb
            psl[2, mu - j] = xdb

            # R-component (same as P for m=0)
            rsl[0, mu + j] = 0.0
            rsl[0, mu - j] = 0.0
            rsl[1, mu + j] = 0.0
            rsl[1, mu - j] = 0.0
            rsl[2, mu + j] = 0.0
            rsl[2, mu - j] = 0.0

            # T-component (zero for m=0)
            tsl[0, mu + j] = 0.0
            tsl[0, mu - j] = 0.0
            tsl[1, mu + j] = 0.0
            tsl[1, mu - j] = 0.0
            tsl[2, mu + j] = 0.0
            tsl[2, mu - j] = 0.0

    elif is_val == 1:
        # m=1 case (first Fourier harmonic)
        for j in range(mu + 1):
            c = rm[mu + j] if j > 0 else rm[mu]
            x = 1.0 - c * c

            # P-component
            psl[0, mu + j] = 0.0
            psl[0, mu - j] = 0.0
            psl[1, mu + j] = np.sqrt(x * 0.5)
            psl[1, mu - j] = np.sqrt(x * 0.5)
            psl[2, mu + j] = c * psl[1, mu + j] * rac3
            psl[2, mu - j] = -psl[2, mu + j]

            # R-component
            rsl[0, mu + j] = 0.0
            rsl[0, mu - j] = 0.0
            rsl[1, mu + j] = c * np.sqrt(x * 0.5)
            rsl[1, mu - j] = rsl[1, mu + j]
            xdb = (1.0 + c * c) * np.sqrt(x * 0.5) * rac3 * 0.5
            if abs(xdb) < 1e-30:
                xdb = 0.0
            rsl[2, mu + j] = xdb
            rsl[2, mu - j] = -xdb

            # T-component
            tsl[0, mu + j] = 0.0
            tsl[0, mu - j] = 0.0
            tsl[1, mu + j] = 0.0
            tsl[1, mu - j] = 0.0
            tsl[2, mu + j] = c * rac3 * np.sqrt(x * 0.5)
            tsl[2, mu - j] = tsl[2, mu + j]

    else:
        # m >= 2 case (higher Fourier harmonics)
        a = 1.0
        for i in range(1, is_val + 1):
            a = a * np.sqrt((i + is_val) / i) * 0.5

        for j in range(mu + 1):
            c = rm[mu + j] if j > 0 else rm[mu]
            xx = 1.0 - c * c

            # P-component
            psl[is_val - 1, mu + j] = 0.0
            psl[is_val - 1, mu - j] = 0.0
            xdb = a * xx ** (is_val * 0.5)
            if abs(xdb) < 1e-30:
                xdb = 0.0
            psl[is_val, mu + j] = xdb
            psl[is_val, mu - j] = xdb

            # R-component
            rsl[is_val - 1, mu + j] = 0.0
            rsl[is_val - 1, mu - j] = 0.0
            xdb = a * c * xx ** (is_val * 0.5)
            if abs(xdb) < 1e-30:
                xdb = 0.0
            rsl[is_val, mu + j] = xdb
            rsl[is_val, mu - j] = xdb

            # T-component
            tsl[is_val - 1, mu + j] = 0.0
            tsl[is_val - 1, mu - j] = 0.0
            tsl[is_val, mu + j] = 0.0
            tsl[is_val, mu - j] = 0.0

    # Recurrence relations for higher order polynomials
    k = 2
    if is_val > 2:
        k = is_val

    if k < max_l:
        ig = -1
        if is_val == 1:
            ig = 1

        for l in range(k, max_l):
            lp = l + 1
            lm = l - 1
            a = (2 * l + 1.0) / np.sqrt((l + is_val + 1.0) * (l - is_val + 1.0))
            b = np.sqrt(float((l + is_val) * (l - is_val))) / (2.0 * l + 1.0)
            c_coef = np.sqrt(float((l + is_val + 1) * (l + is_val))) / (2.0 * l + 1.0)

            for j in range(mu + 1):
                c = rm[mu + j] if j > 0 else rm[mu]

                # P-component recurrence
                xdb = a * (c * psl[l, mu + j] - b * psl[lm, mu + j])
                if abs(xdb) < 1e-30:
                    xdb = 0.0
                psl[lp, mu + j] = xdb
                if j > 0:
                    psl[lp, mu - j] = ig * psl[lp, mu + j]

                # R-component recurrence
                xdb = a * (c * rsl[l, mu + j] - b * rsl[lm, mu + j]) - c_coef * psl[l, mu + j]
                if abs(xdb) < 1e-30:
                    xdb = 0.0
                rsl[lp, mu + j] = xdb
                if j > 0:
                    rsl[lp, mu - j] = ig * rsl[lp, mu + j]

                # T-component recurrence
                xdb = a * (c * tsl[l, mu + j] - b * tsl[lm, mu + j])
                if abs(xdb) < 1e-30:
                    xdb = 0.0
                tsl[lp, mu + j] = xdb
                if j > 0:
                    tsl[lp, mu - j] = ig * tsl[lp, mu + j]

            ig = -ig

    # Extract second-order polynomials
    for j in range(2*mu + 1):
        xpl[j] = psl[2, j]
        xrl[j] = rsl[2, j]
        xtl[j] = tsl[2, j]

    # Compute scattering kernels
    for j in range(mu + 1):
        for k_idx in range(2*mu + 1):
            sbp = 0.0
            sgr = 0.0
            sgt = 0.0
            sarr = 0.0
            sart = 0.0
            satt = 0.0

            if is_val <= max_l:
                for l in range(is_val, max_l + 1):
                    # Get phase function coefficients
                    al = alphal[l]
                    bl = betal[l]
                    gl = gammal[l]
                    zl = zetal[l]

                    # Get Legendre polynomials at j and k
                    pj = psl[l, mu + j]
                    pk = psl[l, k_idx]
                    rj = rsl[l, mu + j]
                    rk = rsl[l, k_idx]
                    tj = tsl[l, mu + j]
                    tk = tsl[l, k_idx]

                    # Accumulate kernel components
                    sbp += bl * pj * pk
                    sgr += gl * rj * pk
                    sgt += -zl * tj * pk
                    sarr += al * rj * rk
                    sart += zl * rj * tk
                    satt += al * tj * tk

            # Apply threshold
            if abs(sbp) < 1e-30:
                sbp = 0.0
            if abs(sgr) < 1e-30:
                sgr = 0.0
            if abs(sgt) < 1e-30:
                sgt = 0.0
            if abs(sarr) < 1e-30:
                sarr = 0.0
            if abs(sart) < 1e-30:
                sart = 0.0
            if abs(satt) < 1e-30:
                satt = 0.0

            bp[j, k_idx] = sbp
            gr[j, k_idx] = sgr
            gt[j, k_idx] = sgt
            arr[j, k_idx] = sarr
            art[j, k_idx] = sart
            att[j, k_idx] = satt

    return xpl, xrl, xtl, bp, gr, gt, arr, art, att


def aero_prof(ta, piz, tr, hr, nt, xmus):
    """
    Divide atmosphere into layers based on aerosol profile.

    Creates atmospheric layers with equal combined (aerosol + molecular)
    optical thickness, using a user-specified vertical aerosol profile.

    Parameters
    ----------
    ta : float
        Total aerosol optical depth
    piz : float
        Single scattering albedo of aerosols
    tr : float
        Total Rayleigh optical depth
    hr : float
        Rayleigh scale height (km)
    nt : int
        Number of layers to create
    xmus : float
        Cosine of solar zenith angle

    Returns
    -------
    h : ndarray
        Cumulative optical depth at each layer boundary, shape (nt+1,)
    ch : ndarray
        Transmission factor exp(-h/xmus)/2, shape (nt+1,)
    ydel : ndarray
        Rayleigh fraction in each layer, shape (nt+1,)
    xdel : ndarray
        Aerosol single-scattering albedo fraction in each layer, shape (nt+1,)
    altc : ndarray
        Altitude at each layer boundary (km), shape (nt+1,)

    Notes
    -----
    Converted from Fortran AEROPROF.f

    Uses global aerosol profile data set via set_aerosol_profile().
    """
    global _atm_state

    h = np.zeros(nt + 1)
    ch = np.zeros(nt + 1)
    ydel = np.zeros(nt + 1)
    xdel = np.zeros(nt + 1)
    altc = np.zeros(nt + 1)

    # Get aerosol profile from global state
    num_z = _atm_state.num_z
    alt_z = _atm_state.alt_z.copy()
    taer_z = _atm_state.taer_z.copy()

    # Add layer above 300 km if needed
    if alt_z[0] < 300:
        taer_z[0] = 0.0
        num_z += 1
        for i in range(num_z - 1):
            alt_z[num_z - i] = alt_z[num_z - i - 1]
            taer_z[num_z - i] = taer_z[num_z - i - 1]

    alt_z[0] = 300.0
    ssa_aer = piz

    # Target optical thickness per layer
    dtau_OS = (tr + ta) / nt

    # Initialize
    i = 0
    dz = 0.0001  # altitude step (km)
    h[0] = 0.0
    altc[0] = 300.0
    z_up = alt_z[0]
    ch[0] = 0.5
    ydel[0] = 1.0
    xdel[0] = 0.0
    j = 1
    n = 1
    dtau_aer = 0.0

    # Step down through atmosphere
    while True:
        i += 1
        z = alt_z[0] - dz * i

        # Rayleigh optical depth increment
        dtau_ray = tr * (np.exp(-z / hr) - np.exp(-z_up / hr))

        # Aerosol optical depth increment
        if n < num_z and alt_z[n - 1] != alt_z[n]:
            dtau_aer += taer_z[n] * dz / (alt_z[n - 1] - alt_z[n])
        if z < alt_z[n] and n < num_z - 1:
            n += 1

        # Total optical depth
        dtau = dtau_ray + dtau_aer

        # Check if we've accumulated enough optical depth for a layer
        if dtau >= dtau_OS and j <= nt:
            altc[j] = z
            h[j] = h[j - 1] + dtau
            ch[j] = np.exp(-h[j] / xmus) / 2.0
            xdel[j] = dtau_aer * ssa_aer / dtau  # aerosol SSA fraction
            ydel[j] = dtau_ray / dtau  # molecular fraction
            j += 1
            z_up = z
            dtau_aer = 0.0

        # Stop when we reach ground
        if z <= 0 or j > nt:
            break

    # Set final layer at ground level
    if j <= nt:
        altc[nt] = 0.0
        h[nt] = tr + ta
        ch[nt] = np.exp(-h[nt] / xmus) / 2.0
        if dtau > 0:
            xdel[nt] = dtau_aer * ssa_aer / dtau
            ydel[nt] = dtau_ray / dtau
        else:
            xdel[nt] = 0.0
            ydel[nt] = 1.0

    return h, ch, ydel, xdel, altc


def iso(iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt, nt, mu, rm, gb):
    """
    Successive orders of scattering calculation.

    Computes atmospheric radiative transfer using the successive orders
    of scattering method. Handles molecular (Rayleigh) and aerosol scattering
    with multiple scattering up to convergence.

    Parameters
    ----------
    iaer_prof : int
        Aerosol profile type
        0 = standard exponential atmosphere
        1 = user-defined profile (requires set_aerosol_profile)
    tamoy : float
        Total aerosol optical depth
    trmoy : float
        Total Rayleigh optical depth
    pizmoy : float
        Aerosol single scattering albedo
    tamoyp : float
        Aerosol optical depth above target altitude
    trmoyp : float
        Rayleigh optical depth above target altitude
    palt : float
        Target altitude (meters)
        > 900: above atmosphere
        0-900: within atmosphere
        <= 0: at surface
    nt : int
        Number of atmospheric layers
    mu : int
        Number of Gauss quadrature angles
    rm : array_like
        Cosines of Gauss angles, shape (2*mu+1,)
        Indexed from -mu to +mu in Fortran; 0 to 2*mu in Python
    gb : array_like
        Gauss weights, shape (2*mu+1,)

    Returns
    -------
    xf : ndarray
        Output transmittances and albedo, shape (3,)
        xf[0] = upward transmittance at plane level (Fortran index -1)
        xf[1] = spherical albedo * 0.5 (Fortran index 0)
        xf[2] = downward transmittance (Fortran index 1)

    Notes
    -----
    Converted from Fortran ISO.f

    This is one of the most complex functions in 6S. It:
    1. Discretizes the atmosphere into layers
    2. Computes scattering source functions
    3. Integrates radiation upward and downward
    4. Iterates until successive orders converge
    5. Applies geometric series acceleration for convergence

    The algorithm follows the discrete ordinates method with
    successive orders of scattering.
    """
    global _atm_state

    # Initialize
    snt = nt
    acu = 1e-20
    acu2 = 1e-3
    ta = tamoy
    piz = pizmoy
    tr = trmoy

    xf = np.zeros(3)

    # Compute optical thickness above plane
    trp = trmoy - trmoyp
    tap = tamoy - tamoyp

    # Determine aerosol scale height
    hr = 8.0  # Rayleigh scale height (km)

    if 0.0 < palt <= 900.0:
        # Plane observation within atmosphere
        if tap > 1e-3:
            ha = -palt / np.log(tap / ta)
        else:
            ha = 2.0
        ntp = nt - 1
    else:
        # Above atmosphere or at surface
        ha = 2.0
        ntp = nt

    # Initialize layer arrays
    h = np.zeros(nt + 1)
    xdel = np.zeros(nt + 1)
    ydel = np.zeros(nt + 1)
    altc = np.zeros(nt + 1)
    ch = np.zeros(nt + 1)

    # Compute mixing ratios for each layer
    # Case 1: Pure Rayleigh
    if ta <= acu2 and tr > ta:
        for j in range(ntp + 1):
            h[j] = j * tr / ntp
            ydel[j] = 1.0
            xdel[j] = 0.0

    # Case 2: Pure aerosol
    elif tr <= acu2 and ta > tr:
        for j in range(ntp + 1):
            h[j] = j * ta / ntp
            ydel[j] = 0.0
            xdel[j] = piz

    # Case 3: Mixed Rayleigh and aerosol
    elif tr > acu2 and ta > acu2:
        if iaer_prof == 0:
            # Standard exponential atmosphere
            h[0] = 0.0
            ydel[0] = 1.0
            xdel[0] = 0.0
            altc[0] = 300.0

            for it in range(1, ntp + 1):
                if it == 1:
                    yy = 0.0
                    dd = 0.0
                else:
                    yy = h[it - 1]
                    dd = ydel[it - 1]

                ppp2 = 300.0
                ppp1 = 0.0

                # Find altitude for this layer
                zx, delta = discre(ta, ha, tr, hr, it, ntp, yy, dd, ppp2, ppp1)

                # Compute optical depths at this altitude
                xxx = -zx / ha
                if xxx < -18:
                    ca = 0.0
                else:
                    ca = ta * np.exp(xxx)

                cr = tr * np.exp(-zx / hr)
                h[it] = cr + ca
                altc[it] = zx

                # Compute mixing ratios
                cr_rate = cr / hr
                ca_rate = ca / ha
                if (cr_rate + ca_rate) > 0:
                    ratio = cr_rate / (cr_rate + ca_rate)
                    xdel[it] = (1.0 - ratio) * piz
                    ydel[it] = ratio
                else:
                    xdel[it] = 0.0
                    ydel[it] = 1.0

        elif iaer_prof == 1:
            # User-defined aerosol profile
            h, ch, ydel, xdel, altc = aero_prof(ta, piz, tr, hr, ntp, rm[mu])

    # Update plane layer if necessary
    if ntp == nt - 1:
        taup = tap + trp
        iplane = -1
        for i in range(ntp + 1):
            if taup >= h[i]:
                iplane = i

        # Check if we need to insert a layer at plane altitude
        th = 0.005
        xt1 = abs(h[iplane] - taup)
        xt2 = abs(h[iplane + 1] - taup) if iplane + 1 <= ntp else th + 1

        if xt1 > th and xt2 > th:
            # Insert new layer
            for i in range(nt, iplane, -1):
                xdel[i] = xdel[i - 1]
                ydel[i] = ydel[i - 1]
                h[i] = h[i - 1]
                altc[i] = altc[i - 1]
        else:
            # Use existing layer
            nt = ntp
            if xt2 < xt1:
                iplane = iplane + 1

        # Set plane layer values
        h[iplane] = taup
        if tr > acu2 and ta > acu2:
            ca = ta * np.exp(-palt / ha)
            cr = tr * np.exp(-palt / hr)
            cr_rate = cr / hr
            ca_rate = ca / ha
            if (cr_rate + ca_rate) > 0:
                ratio = cr_rate / (cr_rate + ca_rate)
                xdel[iplane] = (1.0 - ratio) * piz
                ydel[iplane] = ratio
            else:
                xdel[iplane] = 0.0
                ydel[iplane] = 1.0
            altc[iplane] = palt
        elif tr > acu2:
            ydel[iplane] = 1.0
            xdel[iplane] = 0.0
            altc[iplane] = palt
        else:
            ydel[iplane] = 0.0
            xdel[iplane] = piz
            altc[iplane] = palt
    else:
        iplane = 0 if rm[mu] >= 0 else nt

    # Rayleigh phase function parameters
    delta = _atm_state.delta
    aaaa = delta / (2.0 - delta)
    ron = (1.0 - aaaa) / (1.0 + 2.0 * aaaa)
    beta0 = 1.0
    beta2 = 0.5 * ron

    # Compute scattering kernel
    xpl, psl, bp = kernel(0, mu, rm)

    # Initialize intensity arrays
    i1 = np.zeros((nt + 1, 2*mu + 1))
    i2 = np.zeros((nt + 1, 2*mu + 1))
    i3 = np.zeros(2*mu + 1)

    # Primary scattering - upward radiation
    for k in range(mu + 1, 2*mu + 1):  # Positive mu values
        i1[nt, k] = 1.0
        yy = rm[k]
        for i in range(nt - 1, -1, -1):
            i1[i, k] = np.exp(-(ta + tr - h[i]) / yy)

    # Primary scattering - downward radiation (initially zero)
    for k in range(mu):  # Negative mu values
        for i in range(nt + 1):
            i1[i, k] = 0.0

    # Initialize for iteration
    inm1 = np.zeros(2*mu + 1)
    inm2 = np.zeros(2*mu + 1)

    for k in range(2*mu + 1):
        if k < mu:
            index = nt
        else:
            index = 0
        inm1[k] = i1[index, k]
        inm2[k] = i1[index, k]
        i3[k] = i1[index, k]

    tavion = i1[iplane, mu]
    tavion1 = tavion
    tavion2 = tavion

    # Successive orders iteration
    ig = 1
    igmax = _atm_state.igmax

    while ig <= igmax:
        ig += 1

        # Compute source function at each layer
        for k in range(mu + 1, 2*mu + 1):  # Positive mu
            xpk = xpl[k]
            ypk = xpl[2*mu - k]

            for i in range(nt + 1):
                ii1 = 0.0
                ii2 = 0.0
                x = xdel[i]
                y = ydel[i]

                for j in range(mu + 1, 2*mu + 1):  # Positive j
                    xpj = xpl[j]
                    z = gb[j]
                    xi1 = i1[i, j]
                    xi2 = i1[i, 2*mu - j]

                    bpjk = bp[j - mu, k] * x + y * (beta0 + beta2 * xpj * xpk)
                    bpjmk = bp[j - mu, 2*mu - k] * x + y * (beta0 + beta2 * xpj * ypk)

                    ii2 += z * (xi1 * bpjk + xi2 * bpjmk)
                    ii1 += z * (xi1 * bpjmk + xi2 * bpjk)

                i2[i, k] = ii2
                i2[i, 2*mu - k] = ii1

        # Vertical integration - upward
        for k in range(mu + 1, 2*mu + 1):
            i1[nt, k] = 0.0
            zi1 = 0.0
            yy = rm[k]

            for i in range(nt - 1, -1, -1):
                jj = i + 1
                f = h[jj] - h[i]
                if f > 0:
                    a = (i2[jj, k] - i2[i, k]) / f
                else:
                    a = 0.0
                b = i2[i, k] - a * h[i]
                c = np.exp(-f / yy)
                d = 1.0 - c
                xx = h[i] - h[jj] * c
                zi1 = c * zi1 + (d * (b + a * yy) + a * xx) * 0.5
                i1[i, k] = zi1

        # Vertical integration - downward
        for k in range(mu):
            i1[0, k] = 0.0
            zi1 = 0.0
            yy = abs(rm[k])

            for i in range(1, nt + 1):
                jj = i - 1
                f = h[i] - h[jj]
                if f > 0:
                    c = np.exp(-f / yy)
                    d = 1.0 - c
                    a = (i2[i, k] - i2[jj, k]) / f
                else:
                    c = 1.0
                    d = 0.0
                    a = 0.0
                b = i2[i, k] - a * h[i]
                xx = h[i] - h[jj] * c
                zi1 = c * zi1 + (d * (b + a * yy) + a * xx) * 0.5
                i1[i, k] = zi1

        # Get current scattering order
        in_current = np.zeros(2*mu + 1)
        for k in range(2*mu + 1):
            if k < mu:
                index = nt
            else:
                index = 0
            in_current[k] = i1[index, k]

        tavion0 = i1[iplane, mu]

        # Convergence test (geometric series)
        if ig > 2:
            z_max = 0.0

            # Test at plane level
            a1 = tavion2
            d1 = tavion1
            g1 = tavion0
            if a1 >= acu and d1 >= acu and tavion >= acu:
                if abs(1.0 - g1/d1) > 1e-10:
                    y = ((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/tavion)
                    z_max = max(abs(y), z_max)

            # Test all angles
            for l in range(2*mu + 1):
                if l == mu:
                    continue
                a1 = inm2[l]
                d1 = inm1[l]
                g1 = in_current[l]
                if a1 != 0 and d1 != 0 and i3[l] != 0:
                    if abs(1.0 - g1/d1) > 1e-10:
                        y = ((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/i3[l])
                        z_max = max(abs(y), z_max)

            # Check convergence
            if z_max < 0.0001:
                # Apply geometric series acceleration
                for l in range(2*mu + 1):
                    if l == mu:
                        continue
                    d1 = inm1[l]
                    g1 = in_current[l]
                    if d1 != 0 and abs(1.0 - g1/d1) > 1e-10:
                        y1 = 1.0 - g1/d1
                        g1 = g1 / y1
                        i3[l] += g1

                d1 = tavion1
                g1 = tavion0
                if d1 >= acu and abs(1.0 - g1/d1) > 1e-10:
                    y1 = 1.0 - g1/d1
                    g1 = g1 / y1
                    tavion += g1

                break

            # Store previous orders
            for k in range(2*mu + 1):
                inm2[k] = inm1[k]
            tavion2 = tavion1

        # Store current as previous
        for k in range(2*mu + 1):
            inm1[k] = in_current[k]
        tavion1 = tavion0

        # Add to cumulative total
        for l in range(2*mu + 1):
            i3[l] += in_current[l]
        tavion += tavion0

        # Check if current order is small enough
        z_max = 0.0
        for l in range(2*mu + 1):
            if i3[l] != 0:
                y = abs(in_current[l] / i3[l])
                z_max = max(z_max, y)

        if z_max < 0.00001:
            break

    # Compute output
    xf[2] = i3[2*mu]  # Downward at surface (xf(1) in Fortran)
    xf[0] = tavion     # At plane level (xf(-1) in Fortran)

    # Spherical albedo component
    xf[1] = 0.0
    for k in range(mu + 1, 2*mu + 1):
        xf[1] += rm[k] * gb[k] * i3[2*mu - k]

    # Restore nt
    nt = snt

    return xf


def os(iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
       phirad, nt, mu, naz, rm, gb, rp):
    """
    Successive orders of scattering with full angular radiances.

    Computes angle-dependent radiances and look-up tables for atmospheric
    scattering using the successive orders method with Fourier decomposition.

    Parameters
    ----------
    iaer_prof : int
        Aerosol profile flag (0=standard, 1=user-defined)
    tamoy : float
        Total aerosol optical depth
    trmoy : float
        Total Rayleigh optical depth
    pizmoy : float
        Aerosol single scattering albedo
    tamoyp : float
        Aerosol optical depth above observation plane
    trmoyp : float
        Rayleigh optical depth above observation plane
    palt : float
        Observation altitude (km, 0-900)
    phirad : float
        Azimuthal angle (radians)
    nt : int
        Number of atmospheric layers
    mu : int
        Number of Gauss quadrature points
    naz : int
        Number of azimuth angles
    rm : ndarray
        Gauss quadrature angles (-mu:mu)
    gb : ndarray
        Gauss quadrature weights (-mu:mu)
    rp : ndarray
        Azimuth angles for output (naz)

    Returns
    -------
    xl : ndarray
        Radiances at angles (-mu:mu, naz)
    xlphim : ndarray
        Radiances at plane level (nfi)
    rolut : ndarray
        Look-up table radiances (mu, 41)
    filut : ndarray
        Look-up table azimuth angles (mu, 41)
    nfilut : ndarray
        Number of angles per viewing angle (mu)

    Notes
    -----
    Converted from Fortran OS.f (670 lines)

    This function extends ISO by:
    - Computing radiances at multiple viewing/azimuth angles
    - Generating look-up tables for interpolation
    - Full Fourier decomposition in azimuth

    The algorithm:
    1. Discretizes atmosphere into layers
    2. Initializes look-up table angles
    3. Performs Fourier decomposition (0 to iborm):
       - Computes scattering kernels
       - Calculates primary scattering
       - Integrates vertically (up and down)
       - Iterates successive orders until convergence
       - Accumulates Fourier components
    4. Returns angle-dependent radiances
    """
    import numpy as np_module
    from sixs.successive_orders import _atm_state

    # Constants
    hr = 8.0  # Rayleigh scale height (km)
    accu = 1.0e-20
    accu2 = 1.0e-3
    pi = np_module.pi

    snt = nt
    ta = tamoy
    tr = trmoy
    trp = trmoy - trmoyp
    tap = tamoy - tamoyp
    piz = pizmoy

    iplane = 0
    mum1 = mu - 1

    # Compute aerosol scale height
    if palt <= 900.0 and palt > 0.0:
        if tap > 1.0e-03:
            ha = -palt / np.log(tap / ta)
        else:
            ha = 2.0
        ntp = nt - 1
    else:
        ha = 2.0
        ntp = nt

    xmus = -rm[mu]

    # Atmospheric layering
    h = np_module.zeros(nt + 1)
    ch = np_module.zeros(nt + 1)
    ydel = np_module.zeros(nt + 1)
    xdel = np_module.zeros(nt + 1)
    altc = np_module.zeros(nt + 1)

    # Case 1: Pure Rayleigh
    if ta <= accu2 and tr > ta:
        for j in range(ntp + 1):
            h[j] = j * tr / ntp
            ch[j] = np_module.exp(-h[j] / xmus) / 2.0
            ydel[j] = 1.0
            xdel[j] = 0.0
            if j == 0:
                altc[j] = 300.0
            else:
                altc[j] = -np_module.log(h[j] / tr) * hr

    # Case 2: Pure aerosol
    if tr <= accu2 and ta > tr:
        for j in range(ntp + 1):
            h[j] = j * ta / ntp
            ch[j] = np_module.exp(-h[j] / xmus) / 2.0
            ydel[j] = 0.0
            xdel[j] = piz
            if j == 0:
                altc[j] = 300.0
            else:
                altc[j] = -np_module.log(h[j] / ta) * ha

    # Case 3: Mixed Rayleigh-aerosol (standard profile)
    if tr > accu2 and ta > accu2 and iaer_prof == 0:
        ydel[0] = 1.0
        xdel[0] = 0.0
        h[0] = 0.0
        ch[0] = 0.5
        altc[0] = 300.0
        zx = 300.0

        for it in range(ntp + 1):
            if it == 0:
                yy = 0.0
                dd = 0.0
            else:
                yy = h[it - 1]
                dd = ydel[it - 1]

            zx, delta = discre(ta, ha, tr, hr, it, ntp, yy, dd, 300.0, 0.0)

            xx = -zx / ha
            if xx <= -20.0:
                ca = 0.0
            else:
                ca = ta * np_module.exp(xx)

            xx = -zx / hr
            cr = tr * np_module.exp(xx)
            h[it] = cr + ca
            altc[it] = zx
            ch[it] = np_module.exp(-h[it] / xmus) / 2.0
            cr = cr / hr
            ca = ca / ha
            ratio = cr / (cr + ca)
            xdel[it] = (1.0 - ratio) * piz
            ydel[it] = ratio

    # Case 4: Mixed Rayleigh-aerosol (user profile)
    if tr > accu2 and ta > accu2 and iaer_prof == 1:
        h_tmp, ch_tmp, ydel_tmp, xdel_tmp, altc_tmp = aero_prof(
            ta, piz, tr, hr, ntp, xmus
        )
        h[:ntp+1] = h_tmp[:ntp+1]
        ch[:ntp+1] = ch_tmp[:ntp+1]
        ydel[:ntp+1] = ydel_tmp[:ntp+1]
        xdel[:ntp+1] = xdel_tmp[:ntp+1]
        altc[:ntp+1] = altc_tmp[:ntp+1]

    # Update plane layer if necessary
    if ntp == nt - 1:
        taup = tap + trp
        iplane = -1
        for i in range(ntp + 1):
            if taup >= h[i]:
                iplane = i

        th = 0.0005
        xt1 = abs(h[iplane] - taup)
        xt2 = abs(h[iplane + 1] - taup)

        if xt1 > th and xt2 > th:
            # Shift layers
            for i in range(nt, iplane, -1):
                xdel[i] = xdel[i - 1]
                ydel[i] = ydel[i - 1]
                h[i] = h[i - 1]
                altc[i] = altc[i - 1]
                ch[i] = ch[i - 1]
        else:
            nt = ntp
            if xt2 < xt1:
                iplane = iplane + 1

        h[iplane] = taup
        if tr > accu2 and ta > accu2:
            ca = ta * np_module.exp(-palt / ha)
            cr = tr * np_module.exp(-palt / hr)
            h[iplane] = ca + cr
            cr = cr / hr
            ca = ca / ha
            ratio = cr / (cr + ca)
            xdel[iplane] = (1.0 - ratio) * piz
            ydel[iplane] = ratio
            altc[iplane] = palt
            ch[iplane] = np_module.exp(-h[iplane] / xmus) / 2.0

        if tr > accu2 and ta <= accu2:
            ydel[iplane] = 1.0
            xdel[iplane] = 0.0
            altc[iplane] = palt

        if tr <= accu2 and ta > accu2:
            ydel[iplane] = 0.0
            xdel[iplane] = 1.0 * piz
            altc[iplane] = palt

    # Initialize output arrays
    phi = phirad
    nfi = 13  # Number of azimuth angles for plane observation
    xl = np_module.zeros((2*mu + 1, naz))
    xlphim = np_module.zeros(nfi)

    # Look-up table initialization
    max_lut_angles = 41  # Max number of scattering angles
    rolut = np_module.zeros((mu, max_lut_angles))
    filut = np_module.zeros((mu, max_lut_angles))
    nfilut = np_module.zeros(mu, dtype=np_module.int32)

    its = np_module.arccos(xmus) * 180.0 / pi
    for i in range(mu):
        lutmuv = rm[mu + 1 + i]  # Positive mu values
        luttv = np_module.arccos(lutmuv) * 180.0 / pi
        iscama = 180.0 - abs(luttv - its)
        iscami = 180.0 - (luttv + its)
        nbisca = int((iscama - iscami) / 4.0) + 1
        # Clamp to max_lut_angles
        nbisca = min(nbisca, max_lut_angles)
        nfilut[i] = nbisca
        filut[i, 0] = 0.0
        filut[i, nbisca - 1] = 180.0
        scaa = iscama
        for j in range(1, nbisca - 1):
            scaa = scaa - 4.0
            cscaa = np_module.cos(scaa * pi / 180.0)
            cfi = -(cscaa + xmus * lutmuv) / (
                np_module.sqrt(1.0 - xmus**2) * np_module.sqrt(1.0 - lutmuv**2)
            )
            cfi = max(-1.0, min(1.0, cfi))  # Clamp to valid range
            filut[i, j] = np_module.arccos(cfi) * 180.0 / pi

    # Rayleigh phase function parameters
    delta = _atm_state.delta
    aaaa = delta / (2.0 - delta)
    ron = (1.0 - aaaa) / (1.0 + 2.0 * aaaa)
    beta0 = 1.0
    beta2 = 0.5 * ron

    # Fourier decomposition
    i4 = np_module.zeros(2*mu + 1)
    iborm = _atm_state.nquad - 3
    if abs(xmus - 1.0) < 1.0e-06:
        iborm = 0

    # Main Fourier loop
    for is_val in range(iborm + 1):
        ig = 1
        roavion0 = 0.0
        roavion1 = 0.0
        roavion2 = 0.0
        roavion = 0.0
        i3 = np_module.zeros(2*mu + 1)

        # Kernel computations
        xpl, psl, bp = kernel(is_val, mu, rm)

        if is_val > 0:
            beta0_curr = 0.0
        else:
            beta0_curr = beta0

        # Primary scattering source function
        i2 = np_module.zeros((nt + 1, 2*mu + 1))
        for j in range(2*mu + 1):
            if is_val <= 2:
                spl = xpl[mu]  # xpl(0)
                sa1 = beta0_curr + beta2 * xpl[j] * spl
                sa2 = bp[0, j]
            else:
                sa2 = bp[0, j]
                sa1 = 0.0

            for k in range(nt + 1):
                c = ch[k]
                a = ydel[k]
                b = xdel[k]
                i2[k, j] = c * (sa2 * b + sa1 * a)

        # Vertical integration - primary upward radiation
        i1 = np_module.zeros((nt + 1, 2*mu + 1))
        for k in range(mu + 1, 2*mu + 1):  # Positive mu
            i1[nt, k] = 0.0
            zi1 = i1[nt, k]
            yy = rm[k]
            for i in range(nt - 1, -1, -1):
                jj = i + 1
                f = h[jj] - h[i]
                a_coef = (i2[jj, k] - i2[i, k]) / f
                b_coef = i2[i, k] - a_coef * h[i]
                c = np_module.exp(-f / yy)
                d = 1.0 - c
                xx = h[i] - h[jj] * c
                zi1 = c * zi1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                i1[i, k] = zi1

        # Vertical integration - primary downward radiation
        for k in range(mu):  # Negative mu
            i1[0, k] = 0.0
            zi1 = i1[0, k]
            yy = rm[k]
            for i in range(1, nt + 1):
                jj = i - 1
                f = h[i] - h[jj]
                c = np_module.exp(f / yy)
                d = 1.0 - c
                a_coef = (i2[i, k] - i2[jj, k]) / f
                b_coef = i2[i, k] - a_coef * h[i]
                xx = h[i] - h[jj] * c
                zi1 = c * zi1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                i1[i, k] = zi1

        # Initialize for successive orders
        inm1 = np_module.zeros(2*mu + 1)
        inm2 = np_module.zeros(2*mu + 1)
        in_current = np_module.zeros(2*mu + 1)

        for k in range(2*mu + 1):
            if k < mu:
                index = nt
            else:
                index = 0
            inm1[k] = i1[index, k]
            inm2[k] = i1[index, k]
            i3[k] = i1[index, k]

        roavion2 = i1[iplane, 2*mu]
        roavion = i1[iplane, 2*mu]

        # Loop on successive orders
        igmax = _atm_state.igmax
        for ig in range(2, igmax + 1):
            # Multiple scattering source function
            if is_val <= 2:
                for k in range(mu + 1, 2*mu + 1):
                    xpk = xpl[k]
                    ypk = xpl[2*mu - k]
                    for i in range(nt + 1):
                        ii1 = 0.0
                        ii2 = 0.0
                        x = xdel[i]
                        y = ydel[i]
                        for j in range(mu + 1, 2*mu + 1):
                            xpj = xpl[j]
                            z = gb[j]
                            xi1 = i1[i, j]
                            xi2 = i1[i, 2*mu - j]
                            bpjk = bp[j - mu, k] * x + y * (beta0_curr + beta2 * xpj * xpk)
                            bpjmk = bp[j - mu, 2*mu - k] * x + y * (beta0_curr + beta2 * xpj * ypk)
                            xdb = z * (xi1 * bpjk + xi2 * bpjmk)
                            ii2 += xdb
                            xdb = z * (xi1 * bpjmk + xi2 * bpjk)
                            ii1 += xdb
                        if abs(ii2) < 1.0e-30:
                            ii2 = 0.0
                        if abs(ii1) < 1.0e-30:
                            ii1 = 0.0
                        i2[i, k] = ii2
                        i2[i, 2*mu - k] = ii1
            else:
                for k in range(mu + 1, 2*mu + 1):
                    for i in range(nt + 1):
                        ii1 = 0.0
                        ii2 = 0.0
                        x = xdel[i]
                        for j in range(mu + 1, 2*mu + 1):
                            z = gb[j]
                            xi1 = i1[i, j]
                            xi2 = i1[i, 2*mu - j]
                            bpjk = bp[j - mu, k] * x
                            bpjmk = bp[j - mu, 2*mu - k] * x
                            xdb = z * (xi1 * bpjk + xi2 * bpjmk)
                            ii2 += xdb
                            xdb = z * (xi1 * bpjmk + xi2 * bpjk)
                            ii1 += xdb
                        if abs(ii2) < 1.0e-30:
                            ii2 = 0.0
                        if abs(ii1) < 1.0e-30:
                            ii1 = 0.0
                        i2[i, k] = ii2
                        i2[i, 2*mu - k] = ii1

            # Vertical integration - upward
            for k in range(mu + 1, 2*mu + 1):
                i1[nt, k] = 0.0
                zi1 = i1[nt, k]
                yy = rm[k]
                for i in range(nt - 1, -1, -1):
                    jj = i + 1
                    f = h[jj] - h[i]
                    a_coef = (i2[jj, k] - i2[i, k]) / f
                    b_coef = i2[i, k] - a_coef * h[i]
                    c = np_module.exp(-f / yy)
                    d = 1.0 - c
                    xx = h[i] - h[jj] * c
                    zi1 = c * zi1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                    if abs(zi1) <= 1.0e-20:
                        zi1 = 0.0
                    i1[i, k] = zi1

            # Vertical integration - downward
            for k in range(mu):
                i1[0, k] = 0.0
                zi1 = i1[0, k]
                yy = rm[k]
                for i in range(1, nt + 1):
                    jj = i - 1
                    f = h[i] - h[jj]
                    c = np_module.exp(f / yy)
                    d = 1.0 - c
                    a_coef = (i2[i, k] - i2[jj, k]) / f
                    b_coef = i2[i, k] - a_coef * h[i]
                    xx = h[i] - h[jj] * c
                    zi1 = c * zi1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                    if abs(zi1) <= 1.0e-20:
                        zi1 = 0.0
                    i1[i, k] = zi1

            # Extract current order
            for k in range(2*mu + 1):
                if k < mu:
                    index = nt
                else:
                    index = 0
                in_current[k] = i1[index, k]

            roavion0 = i1[iplane, 2*mu]

            # Convergence test (geometric series)
            if ig > 2:
                z = 0.0
                a1 = roavion2
                d1 = roavion1
                g1 = roavion0
                if a1 >= accu and d1 >= accu and roavion >= accu:
                    y = abs(((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/roavion))
                    z = max(z, y)

                for l in range(2*mu + 1):
                    if l == mu:
                        continue
                    a1 = inm2[l]
                    d1 = inm1[l]
                    g1 = in_current[l]
                    if a1 <= accu or d1 <= accu or i3[l] <= accu:
                        continue
                    y = abs(((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/i3[l]))
                    z = max(z, y)

                if z < 0.0001:
                    # Successful convergence - apply geometric series
                    for l in range(2*mu + 1):
                        y1 = 1.0
                        d1 = inm1[l]
                        g1 = in_current[l]
                        if d1 > accu and abs(g1 - d1) > accu:
                            y1 = 1.0 - g1/d1
                            g1 = g1 / y1
                            i3[l] += g1

                    d1 = roavion1
                    g1 = roavion0
                    if d1 >= accu and abs(g1 - d1) >= accu:
                        y1 = 1.0 - g1/d1
                        g1 = g1 / y1
                        roavion += g1

                    break

                # Store n-2 order
                for k in range(2*mu + 1):
                    inm2[k] = inm1[k]
                roavion2 = roavion1

            # Store n-1 order
            for k in range(2*mu + 1):
                inm1[k] = in_current[k]
            roavion1 = roavion0

            # Add to total
            for l in range(2*mu + 1):
                i3[l] += in_current[l]
            roavion += roavion0

            # Check if order is small enough
            z = 0.0
            for l in range(2*mu + 1):
                if abs(i3[l]) >= accu:
                    y = abs(in_current[l] / i3[l])
                    z = max(z, y)

            if z < 0.00001:
                break

        # Sum Fourier components
        delta0s = 1.0 if is_val == 0 else 2.0
        for l in range(2*mu + 1):
            i4[l] += delta0s * i3[l]

        # Accumulate angle-dependent radiances
        for l in range(naz):
            phi_curr = rp[l]
            for m in range(mum1 + 1):
                if m >= mu:
                    xl[m, l] += delta0s * i3[m] * np_module.cos(is_val * (phi_curr + pi))
                else:
                    xl[m, l] += delta0s * i3[m] * np_module.cos(is_val * phi_curr)

        # Look-up table generation
        for m in range(mu):
            for l in range(nfilut[m]):
                phimul = filut[m, l] * pi / 180.0
                rolut[m, l] += delta0s * i3[mu + 1 + m] * np_module.cos(is_val * (phimul + pi))

        # Special values
        if is_val == 0:
            for k in range(mu + 1, 2*mu + 1):
                xl[mu, 0] += rm[k] * gb[k] * i3[2*mu - k]

        xl[2*mu, 0] += delta0s * i3[2*mu] * np_module.cos(is_val * (phirad + pi))

        for ifi in range(nfi):
            phimul = ifi * pi / (nfi - 1)
            xlphim[ifi] += delta0s * roavion * np_module.cos(is_val * (phimul + pi))

        xl[0, 0] += delta0s * roavion * np_module.cos(is_val * (phirad + pi))

        # Check Fourier convergence
        z = 0.0
        for l in range(2*mu + 1):
            if abs(i4[l]) >= accu:
                x = abs(i3[l] / i4[l])
                z = max(z, x)

        if z <= 0.001:
            break

    # Restore nt
    nt = snt

    return xl, xlphim, rolut, filut, nfilut


def ospol(iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
          phirad, nt, mu, naz, rm, gb, rp):
    """
    Successive orders of scattering with full polarization.

    Computes polarized radiative transfer with Stokes parameters (I, Q, U)
    using successive orders of scattering. Extends os() to include polarization.

    Parameters
    ----------
    iaer_prof : int
        Aerosol profile flag (0=standard, 1=user-defined)
    tamoy : float
        Total aerosol optical depth
    trmoy : float
        Total Rayleigh optical depth
    pizmoy : float
        Aerosol single scattering albedo
    tamoyp : float
        Aerosol optical depth above observation plane
    trmoyp : float
        Rayleigh optical depth above observation plane
    palt : float
        Observation altitude (km, 0-900)
    phirad : float
        Azimuthal angle (radians)
    nt : int
        Number of atmospheric layers
    mu : int
        Number of Gauss quadrature points
    naz : int
        Number of azimuth angles
    rm : ndarray
        Gauss quadrature angles (-mu:mu)
    gb : ndarray
        Gauss quadrature weights (-mu:mu)
    rp : ndarray
        Azimuth angles for output (naz)

    Returns
    -------
    xli : ndarray
        Stokes I radiances at angles (-mu:mu, naz)
    xlq : ndarray
        Stokes Q radiances at angles (-mu:mu, naz)
    xlu : ndarray
        Stokes U radiances at angles (-mu:mu, naz)
    xlphim : ndarray
        Intensity radiances at plane level (nfi)
    rolut : ndarray
        Stokes I look-up table radiances (mu, 41)
    rolutq : ndarray
        Stokes Q look-up table radiances (mu, 41)
    rolutu : ndarray
        Stokes U look-up table radiances (mu, 41)
    filut : ndarray
        Look-up table azimuth angles (mu, 41)
    nfilut : ndarray
        Number of angles per viewing angle (mu)

    Notes
    -----
    Converted from Fortran OSPOL.f (983 lines)

    This function extends os() to handle polarized radiative transfer with
    three Stokes parameters (I, Q, U). The fourth parameter (V, circular
    polarization) is not used in atmospheric scattering.

    The algorithm:
    1. Discretizes atmosphere into layers (same as os)
    2. Sets up polarization phase function parameters
    3. Performs Fourier decomposition with polarized kernels
    4. For each Fourier component:
       - Computes polarized scattering kernels (P, R, T components)
       - Calculates source functions for I, Q, U with polarization coupling
       - Integrates vertically (up and down) for all Stokes parameters
       - Iterates successive orders until convergence
       - Accumulates Fourier components with proper phase
    5. Returns angle-dependent Stokes parameters
    """
    import numpy as np_module
    from sixs.successive_orders import _atm_state

    # Constants
    hr = 8.0  # Rayleigh scale height (km)
    acu = 1.0e-20
    acu2 = 1.0e-4
    pi = np_module.pi

    snt = nt
    ta = tamoy
    tr = trmoy
    trp = trmoy - trmoyp
    tap = tamoy - tamoyp
    piz = pizmoy

    iplane = 0
    mum1 = mu - 1

    # Compute aerosol scale height
    if palt <= 900.0 and palt > 0.0:
        if tap > 1.0e-03:
            ha = -palt / np.log(tap / ta)
        else:
            ha = 2.0
        ntp = nt - 1
    else:
        ha = 2.0
        ntp = nt

    xmus = -rm[mu]

    # Atmospheric layering
    h = np_module.zeros(nt + 1)
    ch = np_module.zeros(nt + 1)
    ydel = np_module.zeros(nt + 1)
    xdel = np_module.zeros(nt + 1)
    altc = np_module.zeros(nt + 1)

    # Case 1: Pure Rayleigh
    if ta <= acu2 and tr > ta:
        for j in range(ntp + 1):
            h[j] = j * tr / ntp
            ch[j] = np_module.exp(-h[j] / xmus) / 2.0
            ydel[j] = 1.0
            xdel[j] = 0.0
            if j == 0:
                altc[j] = 300.0
            else:
                altc[j] = -np_module.log(h[j] / tr) * hr

    # Case 2: Pure aerosol
    if tr <= acu2 and ta > tr:
        for j in range(ntp + 1):
            h[j] = j * ta / ntp
            ch[j] = np_module.exp(-h[j] / xmus) / 2.0
            ydel[j] = 0.0
            xdel[j] = piz
            if j == 0:
                altc[j] = 300.0
            else:
                altc[j] = -np_module.log(h[j] / ta) * ha

    # Case 3: Mixed Rayleigh-aerosol (standard profile)
    if tr > acu2 and ta > acu2 and iaer_prof == 0:
        ydel[0] = 1.0
        xdel[0] = 0.0
        h[0] = 0.0
        ch[0] = 0.5
        altc[0] = 300.0
        zx = 300.0

        for it in range(ntp + 1):
            if it == 0:
                yy = 0.0
                dd = 0.0
            else:
                yy = h[it - 1]
                dd = ydel[it - 1]

            zx, delta = discre(ta, ha, tr, hr, it, ntp, yy, dd, 300.0, 0.0)

            xx = -zx / ha
            if xx <= -20.0:
                ca = 0.0
            else:
                ca = ta * np_module.exp(xx)

            xx = -zx / hr
            cr = tr * np_module.exp(xx)
            h[it] = cr + ca
            altc[it] = zx
            ch[it] = np_module.exp(-h[it] / xmus) / 2.0
            cr = cr / hr
            ca = ca / ha
            ratio = cr / (cr + ca)
            xdel[it] = (1.0 - ratio) * piz
            ydel[it] = ratio

    # Case 4: Mixed Rayleigh-aerosol (user profile)
    if tr > acu2 and ta > acu2 and iaer_prof == 1:
        h_tmp, ch_tmp, ydel_tmp, xdel_tmp, altc_tmp = aero_prof(
            ta, piz, tr, hr, ntp, xmus
        )
        h[:ntp+1] = h_tmp[:ntp+1]
        ch[:ntp+1] = ch_tmp[:ntp+1]
        ydel[:ntp+1] = ydel_tmp[:ntp+1]
        xdel[:ntp+1] = xdel_tmp[:ntp+1]
        altc[:ntp+1] = altc_tmp[:ntp+1]

    # Update plane layer if necessary
    if ntp == nt - 1:
        taup = tap + trp
        iplane = -1
        for i in range(ntp + 1):
            if taup >= h[i]:
                iplane = i

        th = 0.0005
        xt1 = abs(h[iplane] - taup)
        xt2 = abs(h[iplane + 1] - taup)

        if xt1 > th and xt2 > th:
            # Shift layers
            for i in range(nt, iplane, -1):
                xdel[i] = xdel[i - 1]
                ydel[i] = ydel[i - 1]
                h[i] = h[i - 1]
                altc[i] = altc[i - 1]
                ch[i] = ch[i - 1]
        else:
            nt = ntp
            if xt2 < xt1:
                iplane = iplane + 1

        h[iplane] = taup
        if tr > acu2 and ta > acu2:
            ca = ta * np_module.exp(-palt / ha)
            cr = tr * np_module.exp(-palt / hr)
            h[iplane] = ca + cr
            cr = cr / hr
            ca = ca / ha
            ratio = cr / (cr + ca)
            xdel[iplane] = (1.0 - ratio) * piz
            ydel[iplane] = ratio
            altc[iplane] = palt
            ch[iplane] = np_module.exp(-h[iplane] / xmus) / 2.0

        if tr > acu2 and ta <= acu2:
            ydel[iplane] = 1.0
            xdel[iplane] = 0.0
            altc[iplane] = palt

        if tr <= acu2 and ta > acu2:
            ydel[iplane] = 0.0
            xdel[iplane] = 1.0 * piz
            altc[iplane] = palt

    # Initialize output arrays
    phi = phirad
    nfi = 13  # Number of azimuth angles for plane observation
    xli = np_module.zeros((2*mu + 1, naz))
    xlq = np_module.zeros((2*mu + 1, naz))
    xlu = np_module.zeros((2*mu + 1, naz))
    xlphim = np_module.zeros(nfi)

    # Look-up table initialization
    max_lut_angles = 41
    rolut = np_module.zeros((mu, max_lut_angles))
    rolutq = np_module.zeros((mu, max_lut_angles))
    rolutu = np_module.zeros((mu, max_lut_angles))
    filut = np_module.zeros((mu, max_lut_angles))
    nfilut = np_module.zeros(mu, dtype=np_module.int32)

    its = np_module.arccos(xmus) * 180.0 / pi
    for i in range(mu):
        lutmuv = rm[mu + 1 + i]  # Positive mu values
        luttv = np_module.arccos(lutmuv) * 180.0 / pi
        iscama = 180.0 - abs(luttv - its)
        iscami = 180.0 - (luttv + its)
        nbisca = int((iscama - iscami) / 4.0) + 1
        nbisca = min(nbisca, max_lut_angles)
        nfilut[i] = nbisca
        filut[i, 0] = 0.0
        filut[i, nbisca - 1] = 180.0
        scaa = iscama
        for j in range(1, nbisca - 1):
            scaa = scaa - 4.0
            cscaa = np_module.cos(scaa * pi / 180.0)
            cfi = -(cscaa + xmus * lutmuv) / (
                np_module.sqrt(1.0 - xmus**2) * np_module.sqrt(1.0 - lutmuv**2)
            )
            cfi = max(-1.0, min(1.0, cfi))
            filut[i, j] = np_module.arccos(cfi) * 180.0 / pi

    # Rayleigh polarization phase function parameters
    delta = _atm_state.delta
    aaaa = delta / (2.0 - delta)
    ron = (1.0 - aaaa) / (1.0 + 2.0 * aaaa)
    beta0 = 1.0
    beta2 = 0.5 * ron
    gamma2 = -ron * np_module.sqrt(1.5)
    alpha2 = 3.0 * ron

    # Fourier decomposition
    i4 = np_module.zeros(2*mu + 1)
    q4 = np_module.zeros(2*mu + 1)
    u4 = np_module.zeros(2*mu + 1)
    iborm = _atm_state.nquad
    if ta <= acu2:
        iborm = 2
    if abs(xmus - 1.0) < 1.0e-06:
        iborm = 0

    # Main Fourier loop
    for is_val in range(iborm + 1):
        ig = 1
        roIavion = np_module.zeros(4)  # indices -1:2 → 0:3
        roQavion = np_module.zeros(4)
        roUavion = np_module.zeros(4)
        i3 = np_module.zeros(2*mu + 1)
        q3 = np_module.zeros(2*mu + 1)
        u3 = np_module.zeros(2*mu + 1)

        # Kernel computations
        xpl, xrl, xtl, bp, gr, gt, arr, art, att = kernelpol(is_val, mu, rm)

        if is_val > 0:
            beta0_curr = 0.0
        else:
            beta0_curr = beta0

        # Primary scattering source function
        i2 = np_module.zeros((nt + 1, 2*mu + 1))
        q2 = np_module.zeros((nt + 1, 2*mu + 1))
        u2 = np_module.zeros((nt + 1, 2*mu + 1))

        for j in range(2*mu + 1):
            if is_val <= 2:
                spl = xpl[mu]  # xpl(0)
                sa1 = beta0_curr + beta2 * xpl[j] * spl
                sa2 = bp[0, j]
                sb1 = gamma2 * xrl[j] * spl
                sb2 = gr[0, j]
                sc1 = gamma2 * xtl[j] * spl
                sc2 = gt[0, j]
            else:
                sa2 = bp[0, j]
                sa1 = 0.0
                sb2 = gr[0, j]
                sb1 = 0.0
                sc2 = gt[0, j]
                sc1 = 0.0

            for k in range(nt + 1):
                c = ch[k]
                a = ydel[k]
                b = xdel[k]
                i2[k, j] = c * (sa2 * b + sa1 * a)
                q2[k, j] = c * (sb2 * b + sb1 * a)
                u2[k, j] = -c * (sc2 * b + sc1 * a)

        # Vertical integration - primary upward radiation
        i1 = np_module.zeros((nt + 1, 2*mu + 1))
        q1 = np_module.zeros((nt + 1, 2*mu + 1))
        u1 = np_module.zeros((nt + 1, 2*mu + 1))

        for k in range(mu + 1, 2*mu + 1):  # Positive mu
            i1[nt, k] = 0.0
            q1[nt, k] = 0.0
            u1[nt, k] = 0.0
            zi1 = i1[nt, k]
            zq1 = q1[nt, k]
            zu1 = u1[nt, k]
            yy = rm[k]
            for i in range(nt - 1, -1, -1):
                jj = i + 1
                f = h[jj] - h[i]
                c = np_module.exp(-f / yy)
                d = 1.0 - c
                xx = h[i] - h[jj] * c

                a_coef = (i2[jj, k] - i2[i, k]) / f
                b_coef = i2[i, k] - a_coef * h[i]
                zi1 = c * zi1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                i1[i, k] = zi1

                a_coef = (q2[jj, k] - q2[i, k]) / f
                b_coef = q2[i, k] - a_coef * h[i]
                zq1 = c * zq1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                q1[i, k] = zq1

                a_coef = (u2[jj, k] - u2[i, k]) / f
                b_coef = u2[i, k] - a_coef * h[i]
                zu1 = c * zu1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                u1[i, k] = zu1

        # Vertical integration - primary downward radiation
        for k in range(mu):  # Negative mu
            i1[0, k] = 0.0
            q1[0, k] = 0.0
            u1[0, k] = 0.0
            zi1 = i1[0, k]
            zq1 = q1[0, k]
            zu1 = u1[0, k]
            yy = rm[k]
            for i in range(1, nt + 1):
                jj = i - 1
                f = h[i] - h[jj]
                c = np_module.exp(f / yy)
                d = 1.0 - c
                xx = h[i] - h[jj] * c

                a_coef = (i2[i, k] - i2[jj, k]) / f
                b_coef = i2[i, k] - a_coef * h[i]
                zi1 = c * zi1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                i1[i, k] = zi1

                a_coef = (q2[i, k] - q2[jj, k]) / f
                b_coef = q2[i, k] - a_coef * h[i]
                zq1 = c * zq1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                q1[i, k] = zq1

                a_coef = (u2[i, k] - u2[jj, k]) / f
                b_coef = u2[i, k] - a_coef * h[i]
                zu1 = c * zu1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                u1[i, k] = zu1

        # Initialize for successive orders
        inm1 = np_module.zeros(2*mu + 1)
        inm2 = np_module.zeros(2*mu + 1)
        qnm1 = np_module.zeros(2*mu + 1)
        qnm2 = np_module.zeros(2*mu + 1)
        unm1 = np_module.zeros(2*mu + 1)
        unm2 = np_module.zeros(2*mu + 1)

        for k in range(2*mu + 1):
            if k < mu:
                index = nt
            else:
                index = 0
            inm1[k] = i1[index, k]
            inm2[k] = i1[index, k]
            i3[k] = i1[index, k]
            qnm1[k] = q1[index, k]
            qnm2[k] = q1[index, k]
            q3[k] = q1[index, k]
            unm1[k] = u1[index, k]
            unm2[k] = u1[index, k]
            u3[k] = u1[index, k]

        roIavion[3] = i1[iplane, 2*mu]  # roIavion(2)
        roIavion[0] = i1[iplane, 2*mu]  # roIavion(-1)
        roQavion[3] = q1[iplane, 2*mu]
        roQavion[0] = q1[iplane, 2*mu]
        roUavion[3] = u1[iplane, 2*mu]
        roUavion[0] = u1[iplane, 2*mu]

        # Loop on successive orders
        igmax = _atm_state.igmax
        for ig in range(2, igmax + 1):
            # Multiple scattering source function
            if is_val <= 2:
                for k in range(mu + 1, 2*mu + 1):
                    xpk = xpl[k]
                    xrk = xrl[k]
                    xtk = xtl[k]
                    ypk = xpl[2*mu - k]
                    yrk = xrl[2*mu - k]
                    ytk = xtl[2*mu - k]
                    for i in range(nt + 1):
                        ii1 = 0.0
                        ii2 = 0.0
                        qq1 = 0.0
                        qq2 = 0.0
                        uu1 = 0.0
                        uu2 = 0.0
                        x = xdel[i]
                        y = ydel[i]
                        for j in range(mu + 1, 2*mu + 1):
                            z = gb[j]
                            xpj = xpl[j]
                            xrj = xrl[j]
                            xtj = xtl[j]
                            ypj = xpl[2*mu - j]
                            yrj = xrl[2*mu - j]
                            ytj = xtl[2*mu - j]
                            xi1 = i1[i, j]
                            xi2 = i1[i, 2*mu - j]
                            xq1 = q1[i, j]
                            xq2 = q1[i, 2*mu - j]
                            xu1 = u1[i, j]
                            xu2 = u1[i, 2*mu - j]

                            bpjk = bp[j - mu, k] * x + y * (beta0_curr + beta2 * xpj * xpk)
                            bpjmk = bp[j - mu, 2*mu - k] * x + y * (beta0_curr + beta2 * xpj * ypk)
                            gtjk = gt[j - mu, k] * x + y * gamma2 * xpj * xtk
                            gtjmk = gt[j - mu, 2*mu - k] * x + y * gamma2 * xpj * ytk
                            gtkj = gt[k - mu, j] * x + y * gamma2 * xpk * xtj
                            gtkmj = gt[k - mu, 2*mu - j] * x + y * gamma2 * xpk * ytj
                            grjk = gr[j - mu, k] * x + y * gamma2 * xpj * xrk
                            grjmk = gr[j - mu, 2*mu - k] * x + y * gamma2 * xpj * yrk
                            grkj = gr[k - mu, j] * x + y * gamma2 * xpk * xrj
                            grkmj = gr[k - mu, 2*mu - j] * x + y * gamma2 * xpk * yrj
                            arrjk = arr[j - mu, k] * x + y * alpha2 * xrj * xrk
                            arrjmk = arr[j - mu, 2*mu - k] * x + y * alpha2 * xrj * yrk
                            artjk = art[j - mu, k] * x + y * alpha2 * xtj * xrk
                            artjmk = art[j - mu, 2*mu - k] * x + y * alpha2 * xtj * yrk
                            artkj = art[k - mu, j] * x + y * alpha2 * xtk * xrj
                            artkmj = art[k - mu, 2*mu - j] * x + y * alpha2 * xtk * yrj
                            attjk = att[j - mu, k] * x + y * alpha2 * xtj * xtk
                            attjmk = att[j - mu, 2*mu - k] * x + y * alpha2 * xtj * ytk

                            xdb = xi1 * bpjk + xi2 * bpjmk + xq1 * grkj + xq2 * grkmj
                            xdb = xdb - xu1 * gtkj - xu2 * gtkmj
                            ii2 += xdb * z
                            xdb = xi1 * bpjmk + xi2 * bpjk + xq1 * grkmj + xq2 * grkj
                            xdb = xdb + xu1 * gtkmj + xu2 * gtkj
                            ii1 += xdb * z
                            xdb = xi1 * grjk + xi2 * grjmk + xq1 * arrjk + xq2 * arrjmk
                            xdb = xdb - xu1 * artjk + xu2 * artjmk
                            qq2 += xdb * z
                            xdb = xi1 * grjmk + xi2 * grjk + xq1 * arrjmk + xq2 * arrjk
                            xdb = xdb - xu1 * artjmk + xu2 * artjk
                            qq1 += xdb * z
                            xdb = xi1 * gtjk - xi2 * gtjmk + xq1 * artkj + xq2 * artkmj
                            xdb = xdb - xu1 * attjk - xu2 * attjmk
                            uu2 -= xdb * z
                            xdb = xi1 * gtjmk - xi2 * gtjk - xq1 * artkmj - xq2 * artkj
                            xdb = xdb - xu1 * attjmk - xu2 * attjk
                            uu1 -= xdb * z

                        if abs(ii2) < 1.0e-30:
                            ii2 = 0.0
                        if abs(ii1) < 1.0e-30:
                            ii1 = 0.0
                        if abs(qq2) < 1.0e-30:
                            qq2 = 0.0
                        if abs(qq1) < 1.0e-30:
                            qq1 = 0.0
                        if abs(uu2) < 1.0e-30:
                            uu2 = 0.0
                        if abs(uu1) < 1.0e-30:
                            uu1 = 0.0
                        i2[i, k] = ii2
                        i2[i, 2*mu - k] = ii1
                        q2[i, k] = qq2
                        q2[i, 2*mu - k] = qq1
                        u2[i, k] = uu2
                        u2[i, 2*mu - k] = uu1
            else:
                for k in range(mu + 1, 2*mu + 1):
                    for i in range(nt + 1):
                        ii1 = 0.0
                        ii2 = 0.0
                        qq1 = 0.0
                        qq2 = 0.0
                        uu1 = 0.0
                        uu2 = 0.0
                        x = xdel[i]
                        for j in range(mu + 1, 2*mu + 1):
                            z = gb[j]
                            xi1 = i1[i, j]
                            xi2 = i1[i, 2*mu - j]
                            xq1 = q1[i, j]
                            xq2 = q1[i, 2*mu - j]
                            xu1 = u1[i, j]
                            xu2 = u1[i, 2*mu - j]

                            bpjk = bp[j - mu, k] * x
                            bpjmk = bp[j - mu, 2*mu - k] * x
                            gtjk = gt[j - mu, k] * x
                            gtjmk = gt[j - mu, 2*mu - k] * x
                            gtkj = gt[k - mu, j] * x
                            gtkmj = gt[k - mu, 2*mu - j] * x
                            grjk = gr[j - mu, k] * x
                            grjmk = gr[j - mu, 2*mu - k] * x
                            grkj = gr[k - mu, j] * x
                            grkmj = gr[k - mu, 2*mu - j] * x
                            arrjk = arr[j - mu, k] * x
                            arrjmk = arr[j - mu, 2*mu - k] * x
                            artjk = art[j - mu, k] * x
                            artjmk = art[j - mu, 2*mu - k] * x
                            artkj = art[k - mu, j] * x
                            artkmj = art[k - mu, 2*mu - j] * x
                            attjk = att[j - mu, k] * x
                            attjmk = att[j - mu, 2*mu - k] * x

                            xdb = xi1 * bpjk + xi2 * bpjmk + xq1 * grkj + xq2 * grkmj
                            xdb = xdb - xu1 * gtkj - xu2 * gtkmj
                            ii2 += xdb * z
                            xdb = xi1 * bpjmk + xi2 * bpjk + xq1 * grkmj + xq2 * grkj
                            xdb = xdb + xu1 * gtkmj + xu2 * gtkj
                            ii1 += xdb * z
                            xdb = xi1 * grjk + xi2 * grjmk + xq1 * arrjk + xq2 * arrjmk
                            xdb = xdb - xu1 * artjk + xu2 * artjmk
                            qq2 += xdb * z
                            xdb = xi1 * grjmk + xi2 * grjk + xq1 * arrjmk + xq2 * arrjk
                            xdb = xdb - xu1 * artjmk + xu2 * artjk
                            qq1 += xdb * z
                            xdb = xi1 * gtjk - xi2 * gtjmk + xq1 * artkj + xq2 * artkmj
                            xdb = xdb - xu1 * attjk - xu2 * attjmk
                            uu2 -= xdb * z
                            xdb = xi1 * gtjmk - xi2 * gtjk - xq1 * artkmj - xq2 * artkj
                            xdb = xdb - xu1 * attjmk - xu2 * attjk
                            uu1 -= xdb * z

                        if abs(ii2) < 1.0e-30:
                            ii2 = 0.0
                        if abs(ii1) < 1.0e-30:
                            ii1 = 0.0
                        if abs(qq2) < 1.0e-30:
                            qq2 = 0.0
                        if abs(qq1) < 1.0e-30:
                            qq1 = 0.0
                        if abs(uu2) < 1.0e-30:
                            uu2 = 0.0
                        if abs(uu1) < 1.0e-30:
                            uu1 = 0.0
                        i2[i, k] = ii2
                        i2[i, 2*mu - k] = ii1
                        q2[i, k] = qq2
                        q2[i, 2*mu - k] = qq1
                        u2[i, k] = uu2
                        u2[i, 2*mu - k] = uu1

            # Vertical integration - upward
            for k in range(mu + 1, 2*mu + 1):
                i1[nt, k] = 0.0
                q1[nt, k] = 0.0
                u1[nt, k] = 0.0
                zi1 = i1[nt, k]
                zq1 = q1[nt, k]
                zu1 = u1[nt, k]
                yy = rm[k]
                for i in range(nt - 1, -1, -1):
                    jj = i + 1
                    f = h[jj] - h[i]
                    c = np_module.exp(-f / yy)
                    d = 1.0 - c
                    xx = h[i] - h[jj] * c

                    a_coef = (i2[jj, k] - i2[i, k]) / f
                    b_coef = i2[i, k] - a_coef * h[i]
                    zi1 = c * zi1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                    if abs(zi1) <= 1.0e-20:
                        zi1 = 0.0
                    i1[i, k] = zi1

                    a_coef = (q2[jj, k] - q2[i, k]) / f
                    b_coef = q2[i, k] - a_coef * h[i]
                    zq1 = c * zq1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                    if abs(zq1) <= 1.0e-20:
                        zq1 = 0.0
                    q1[i, k] = zq1

                    a_coef = (u2[jj, k] - u2[i, k]) / f
                    b_coef = u2[i, k] - a_coef * h[i]
                    zu1 = c * zu1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                    if abs(zu1) <= 1.0e-20:
                        zu1 = 0.0
                    u1[i, k] = zu1

            # Vertical integration - downward
            for k in range(mu):
                i1[0, k] = 0.0
                q1[0, k] = 0.0
                u1[0, k] = 0.0
                zi1 = i1[0, k]
                zq1 = q1[0, k]
                zu1 = u1[0, k]
                yy = rm[k]
                for i in range(1, nt + 1):
                    jj = i - 1
                    f = h[i] - h[jj]
                    c = np_module.exp(f / yy)
                    d = 1.0 - c
                    xx = h[i] - h[jj] * c

                    a_coef = (i2[i, k] - i2[jj, k]) / f
                    b_coef = i2[i, k] - a_coef * h[i]
                    zi1 = c * zi1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                    if abs(zi1) <= 1.0e-20:
                        zi1 = 0.0
                    i1[i, k] = zi1

                    a_coef = (q2[i, k] - q2[jj, k]) / f
                    b_coef = q2[i, k] - a_coef * h[i]
                    zq1 = c * zq1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                    if abs(zq1) <= 1.0e-20:
                        zq1 = 0.0
                    q1[i, k] = zq1

                    a_coef = (u2[i, k] - u2[jj, k]) / f
                    b_coef = u2[i, k] - a_coef * h[i]
                    zu1 = c * zu1 + (d * (b_coef + a_coef * yy) + a_coef * xx) * 0.5
                    if abs(zu1) <= 1.0e-20:
                        zu1 = 0.0
                    u1[i, k] = zu1

            # Extract current order
            in_current = np_module.zeros(2*mu + 1)
            qn_current = np_module.zeros(2*mu + 1)
            un_current = np_module.zeros(2*mu + 1)
            for k in range(2*mu + 1):
                if k < mu:
                    index = nt
                else:
                    index = 0
                in_current[k] = i1[index, k]
                qn_current[k] = q1[index, k]
                un_current[k] = u1[index, k]

            roIavion[1] = i1[iplane, 2*mu]  # roIavion(0)
            roQavion[1] = q1[iplane, 2*mu]
            roUavion[1] = u1[iplane, 2*mu]

            # Convergence test (geometric series)
            if ig > 2:
                z = 0.0
                # Test I component at plane
                a1 = abs(roIavion[3])  # roIavion(2)
                d1 = abs(roIavion[2])  # roIavion(1)
                g1 = abs(roIavion[1])  # roIavion(0)
                r1 = abs(roIavion[0])  # roIavion(-1)
                if a1 >= acu and d1 >= acu and r1 >= acu:
                    a1 = roIavion[3]
                    d1 = roIavion[2]
                    g1 = roIavion[1]
                    r1 = roIavion[0]
                    if abs(1.0 - g1/d1) > 1e-10:
                        y = abs(((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/r1))
                        z = max(z, y)

                # Test Q component at plane
                a1 = abs(roQavion[3])
                d1 = abs(roQavion[2])
                g1 = abs(roQavion[1])
                r1 = abs(roQavion[0])
                if a1 >= acu and d1 >= acu and r1 >= acu:
                    a1 = roQavion[3]
                    d1 = roQavion[2]
                    g1 = roQavion[1]
                    r1 = roQavion[0]
                    if abs(1.0 - g1/d1) > 1e-10:
                        y = abs(((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/r1))
                        z = max(z, y)

                # Test U component at plane
                a1 = abs(roUavion[3])
                d1 = abs(roUavion[2])
                g1 = abs(roUavion[1])
                r1 = abs(roUavion[0])
                if a1 >= acu and d1 >= acu and r1 >= acu:
                    a1 = roUavion[3]
                    d1 = roUavion[2]
                    g1 = roUavion[1]
                    r1 = roUavion[0]
                    if abs(1.0 - g1/d1) > 1e-10:
                        y = abs(((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/r1))
                        z = max(z, y)

                # Test all angles
                for l in range(2*mu + 1):
                    if l == mu:
                        continue
                    # Test I
                    a1 = inm2[l]
                    d1 = inm1[l]
                    g1 = in_current[l]
                    if abs(a1) > acu and abs(d1) > acu and abs(i3[l]) > acu:
                        if abs(1.0 - g1/d1) > 1e-10:
                            y = abs(((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/i3[l]))
                            z = max(z, y)
                    # Test Q
                    a1 = qnm2[l]
                    d1 = qnm1[l]
                    g1 = qn_current[l]
                    if abs(a1) > acu and abs(d1) > acu and abs(q3[l]) > acu:
                        if abs(1.0 - g1/d1) > 1e-10:
                            y = abs(((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/q3[l]))
                            z = max(z, y)
                    # Test U
                    a1 = unm2[l]
                    d1 = unm1[l]
                    g1 = un_current[l]
                    if abs(a1) > acu and abs(d1) > acu and abs(u3[l]) > acu:
                        if abs(1.0 - g1/d1) > 1e-10:
                            y = abs(((g1/d1 - d1/a1) / ((1.0 - g1/d1)**2)) * (g1/u3[l]))
                            z = max(z, y)

                if z < 0.01:
                    # Successful convergence - apply geometric series
                    for l in range(2*mu + 1):
                        y1 = 1.0
                        d1 = inm1[l]
                        if abs(d1) > acu:
                            g1 = in_current[l]
                            if abs(g1 - d1) > acu:
                                y1 = 1.0 - g1/d1
                                g1 = g1 / y1
                                i3[l] += g1
                        y1 = 1.0
                        d1 = qnm1[l]
                        if abs(d1) > acu:
                            g1 = qn_current[l]
                            if abs(g1 - d1) > acu:
                                y1 = 1.0 - g1/d1
                                g1 = g1 / y1
                                q3[l] += g1
                        y1 = 1.0
                        d1 = unm1[l]
                        if abs(d1) > acu:
                            g1 = un_current[l]
                            if abs(g1 - d1) > acu:
                                y1 = 1.0 - g1/d1
                                g1 = g1 / y1
                                u3[l] += g1

                    y1 = 1.0
                    d1 = roIavion[2]
                    if abs(d1) >= acu:
                        g1 = roIavion[1]
                        if abs(g1 - d1) >= acu:
                            y1 = 1.0 - g1/d1
                            g1 = g1 / y1
                        roIavion[0] += g1

                    y1 = 1.0
                    d1 = roQavion[2]
                    if abs(d1) >= acu:
                        g1 = roQavion[1]
                        if abs(g1 - d1) >= acu:
                            y1 = 1.0 - g1/d1
                            g1 = g1 / y1
                        roQavion[0] += g1

                    y1 = 1.0
                    d1 = roUavion[2]
                    if abs(d1) >= acu:
                        g1 = roUavion[1]
                        if abs(g1 - d1) >= acu:
                            y1 = 1.0 - g1/d1
                            g1 = g1 / y1
                        roUavion[0] += g1

                    break

                # Store n-2 order
                for k in range(2*mu + 1):
                    inm2[k] = inm1[k]
                    qnm2[k] = qnm1[k]
                    unm2[k] = unm1[k]
                roIavion[3] = roIavion[2]
                roQavion[3] = roQavion[2]
                roUavion[3] = roUavion[2]

            # Store n-1 order
            for k in range(2*mu + 1):
                inm1[k] = in_current[k]
                qnm1[k] = qn_current[k]
                unm1[k] = un_current[k]
            roIavion[2] = roIavion[1]
            roQavion[2] = roQavion[1]
            roUavion[2] = roUavion[1]

            # Add to total
            for l in range(2*mu + 1):
                i3[l] += in_current[l]
                q3[l] += qn_current[l]
                u3[l] += un_current[l]
            roIavion[0] += roIavion[1]
            roQavion[0] += roQavion[1]
            roUavion[0] += roUavion[1]

            # Check if order is small enough
            z = 0.0
            for l in range(2*mu + 1):
                if abs(i3[l]) >= acu:
                    y = abs(in_current[l] / i3[l])
                    z = max(z, y)
                if abs(q3[l]) >= acu:
                    y = abs(qn_current[l] / q3[l])
                    z = max(z, y)
                if abs(u3[l]) >= acu:
                    y = abs(un_current[l] / u3[l])
                    z = max(z, y)

            if z < 1.0e-8:
                break

        # Sum Fourier components
        delta0s = 1.0 if is_val == 0 else 2.0
        for l in range(2*mu + 1):
            i4[l] += abs(delta0s * i3[l])
            q4[l] += abs(q3[l])
            u4[l] += abs(u3[l])

        # Accumulate angle-dependent radiances
        for l in range(naz):
            phi_curr = rp[l]
            for m in range(mum1 + 1):
                if m >= mu:
                    xli[m, l] += delta0s * i3[m] * np_module.cos(is_val * (phi_curr + pi))
                    xlq[m, l] += delta0s * q3[m] * np_module.cos(is_val * (phi_curr + pi))
                    xlu[m, l] += delta0s * u3[m] * np_module.sin(is_val * (phi_curr + pi))
                else:
                    xli[m, l] += delta0s * i3[m] * np_module.cos(is_val * phi_curr)
                    xlq[m, l] += delta0s * q3[m] * np_module.cos(is_val * phi_curr)
                    xlu[m, l] += delta0s * u3[m] * np_module.sin(is_val * phi_curr)

        # Look-up table generation
        for m in range(mu):
            for l in range(nfilut[m]):
                phimul = filut[m, l] * pi / 180.0
                rolut[m, l] += delta0s * i3[mu + 1 + m] * np_module.cos(is_val * (phimul + pi))
                rolutq[m, l] += delta0s * q3[mu + 1 + m] * np_module.cos(is_val * (phimul + pi))
                rolutu[m, l] += delta0s * u3[mu + 1 + m] * np_module.sin(is_val * (phimul + pi))

        # Special values
        if is_val == 0:
            for k in range(mu + 1, 2*mu):
                xli[mu, 0] += rm[k] * gb[k] * i3[2*mu - k]
                xlq[mu, 0] += rm[k] * gb[k] * q3[2*mu - k]
                xlu[mu, 0] += rm[k] * gb[k] * u3[2*mu - k]

        xli[2*mu, 0] += delta0s * i3[2*mu] * np_module.cos(is_val * (phirad + pi))
        xlq[2*mu, 0] += delta0s * q3[2*mu] * np_module.cos(is_val * (phirad + pi))
        xlu[2*mu, 0] += delta0s * u3[2*mu] * np_module.sin(is_val * (phirad + pi))

        xli[0, 0] += delta0s * roIavion[0] * np_module.cos(is_val * (phirad + pi))
        xlq[0, 0] += delta0s * roQavion[0] * np_module.cos(is_val * (phirad + pi))
        xlu[0, 0] += delta0s * roUavion[0] * np_module.sin(is_val * (phirad + pi))

        for ifi in range(nfi):
            phimul = ifi * pi / (nfi - 1)
            xlphim[ifi] += delta0s * roIavion[0] * np_module.cos(is_val * (phimul + pi))

        # Check Fourier convergence
        z = 0.0
        for l in range(2*mu + 1):
            if abs(i4[l]) >= acu:
                x = abs(delta0s * i3[l] / i4[l])
                z = max(z, x)
            if abs(q4[l]) >= acu:
                x = abs(q3[l] / q4[l])
                z = max(z, x)
            if abs(u4[l]) >= acu:
                x = abs(u3[l] / u4[l])
                z = max(z, x)

        if z <= 0.0005:
            break

    # Restore nt
    nt = snt

    return xli, xlq, xlu, xlphim, rolut, rolutq, rolutu, filut, nfilut


__all__ = [
    'discre',
    'kernel',
    'kernelpol',
    'aero_prof',
    'iso',
    'os',
    'ospol',
    'set_aerosol_phase_function',
    'set_aerosol_profile',
    'AtmosphereState',
]
