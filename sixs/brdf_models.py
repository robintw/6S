"""
BRDF (Bidirectional Reflectance Distribution Function) models.

Surface reflectance models for various land and water surfaces.
Includes simple parametric models (Minnaert, Walthall, Roujean) and
water reflectance lookup tables.
"""

import numpy as np
from numba import jit
from .gauss import gauss


# ==============================================================================
# Simple Albedo Models
# ==============================================================================

@jit(nopython=True)
def minnalbe(par1, par2):
    """
    Minnaert albedo model.

    Parameters
    ----------
    par1 : float
        Minnaert parameter k
    par2 : float
        Brightness parameter

    Returns
    -------
    brdfalb : float
        Spherical albedo
    """
    brdfalb = 2.0 * par2 / (par1 + 1.0)
    return brdfalb


@jit(nopython=True)
def modisalbe(p1, p2, p3):
    """
    MODIS albedo model.

    Simple linear combination model for MODIS band albedo.

    Parameters
    ----------
    p1 : float
        Constant term
    p2 : float
        Coefficient for first parameter
    p3 : float
        Coefficient for second parameter

    Returns
    -------
    brdfalb : float
        Spherical albedo

    Notes
    -----
    Formula: p1 + p2*0.189184 - p3*1.377622
    """
    brdfalb = p1 + p2 * 0.189184 - p3 * 1.377622
    return brdfalb


# ==============================================================================
# Simple BRDF Models
# ==============================================================================

@jit(nopython=True)
def minnbrdf(par1, par2, rm, rp):
    """
    Minnaert BRDF model.

    Classical Minnaert empirical model for surface reflectance.

    Parameters
    ----------
    par1 : float
        Minnaert k parameter (typically 0.5-1.5)
    par2 : float
        Surface albedo
    rm : ndarray
        Cosines of viewing zenith angles, shape (mu,)
        rm[0] is the solar zenith angle cosine
    rp : ndarray
        Relative azimuth angles, shape (np,)

    Returns
    -------
    brdfint : ndarray
        BRDF values, shape (mu, np)

    Notes
    -----
    Minnaert model: ρ(μ_s, μ_v) = A * (μ_s * μ_v)^(k-1)
    where μ_s is cosine of solar zenith, μ_v is cosine of viewing zenith
    """
    mu = len(rm)
    np_points = len(rp)

    xmu = rm[0]  # Solar zenith cosine

    brdfint = np.zeros((mu, np_points), dtype=np.float64)

    for k in range(np_points):
        for j in range(mu):
            view = rm[j]
            brdfint[j, k] = 0.5 * par2 * (par1 + 1.0) * ((xmu * view) ** (par1 - 1.0))

    return brdfint


@jit(nopython=True)
def waltbrdf(a, ap, b, c, rm, rp):
    """
    Walthall BRDF model.

    Semi-empirical model from Walthall et al. (Applied Optics, 1985).

    Parameters
    ----------
    a : float
        Coefficient for θ_s² * θ_v² term
    ap : float
        Coefficient for (θ_s² + θ_v²) term
    b : float
        Coefficient for θ_s * θ_v * cos(φ) term
    c : float
        Constant term
    rm : ndarray
        Cosines of viewing zenith angles, shape (mu+1,)
        rm[0] is solar zenith angle, rm[-1] is azimuth offset
    rp : ndarray
        Relative azimuth angles, shape (np,)

    Returns
    -------
    brdfint : ndarray
        BRDF values, shape (mu, np)

    Notes
    -----
    Model: ρ = a*θ_s²*θ_v² + a'*(θ_s² + θ_v²) + b*θ_s*θ_v*cos(φ) + c

    Modified slightly to match reciprocity principle.
    Reference: Applied Optics, Vol 24, No 3, pp 383-387 (1985)
    """
    mu = len(rm) - 1  # Account for rm[-1] being azimuth offset
    np_points = len(rp)

    xmu = rm[0]
    ts = np.arccos(xmu)

    brdfint = np.zeros((mu, np_points), dtype=np.float64)

    for k in range(np_points):
        for j in range(mu):
            view = rm[j]
            tv = np.arccos(view)

            if j == mu - 1:
                fi = rm[-1]
            else:
                fi = rp[k] + rm[-1]

            phi = fi

            brdfint[j, k] = (a * (ts * ts * tv * tv) +
                            ap * (ts * ts + tv * tv) +
                            b * ts * tv * np.cos(phi) + c)

    return brdfint


@jit(nopython=True)
def roujbrdf(k0, k1, k2, rm, rp):
    """
    Roujean BRDF model.

    Semi-empirical kernel-driven model from Roujean et al.

    Parameters
    ----------
    k0 : float
        Isotropic scattering parameter
    k1 : float
        Geometric-optical kernel weight
    k2 : float
        Volume scattering kernel weight
    rm : ndarray
        Cosines of viewing zenith angles, shape (mu+1,)
        rm[0] is solar zenith, rm[-1] is azimuth offset
    rp : ndarray
        Relative azimuth angles, shape (np,)

    Returns
    -------
    brdfint : ndarray
        BRDF values, shape (mu, np)

    Notes
    -----
    Model: ρ = k0 + k1*f1 + k2*f2
    where f1 is geometric kernel and f2 is volume scattering kernel.

    Reference: JGR 1992, Vol. 97, No D18, Pages 20,445-20,468
    """
    mu = len(rm) - 1
    np_points = len(rp)
    pi = np.pi

    xmus = rm[0]

    brdfint = np.zeros((mu, np_points), dtype=np.float64)

    for k in range(np_points):
        for j in range(mu):
            xmuv = rm[j]

            if j == mu - 1:
                fi = rm[-1]
            else:
                fi = rp[k] + rm[-1]

            fr = np.arccos(np.cos(fi))
            tts = np.tan(np.arccos(xmus))
            ttv = np.tan(np.arccos(xmuv))

            # Phase angle
            cpsi = xmus * xmuv + np.sin(np.arccos(xmus)) * np.sin(np.arccos(xmuv)) * np.cos(fi)

            if cpsi < 1.0:
                psi = np.arccos(cpsi)
            else:
                psi = 0.0

            # Volume scattering kernel (f2)
            f2 = (4.0 / (3.0 * pi * (xmus + xmuv)) *
                  ((pi / 2.0 - psi) * cpsi + np.sin(psi)) - 1.0 / 3.0)

            # Geometric kernel (f1)
            ft = tts * tts + ttv * ttv - 2.0 * tts * ttv * np.cos(fr)
            f1 = (0.5 * ((pi - fr) * np.cos(fr) + np.sin(fr)) * tts * ttv -
                  tts - ttv - np.sqrt(ft)) / pi

            brdfint[j, k] = k0 + k1 * f1 + k2 * f2

    return brdfint


# ==============================================================================
# Albedo Integration Functions (using Gaussian quadrature)
# ==============================================================================

def waltalbe(a, ap, b, c):
    """
    Compute spherical albedo for Walthall BRDF model.

    Integrates the Walthall BRDF over all viewing and illumination angles
    using Gaussian quadrature.

    Parameters
    ----------
    a, ap, b, c : float
        Walthall model parameters

    Returns
    -------
    brdfalb : float
        Spherical albedo
    """
    nta = 24
    nfa = 48
    pi = np.pi

    teta1 = 0.0
    teta2 = pi / 2.0
    ta, wta = gauss(teta1, teta2, nta)

    phi1 = 0.0
    phi2 = 2.0 * pi
    fa, wfa = gauss(phi1, phi2, nfa)

    brdfalb = 0.0
    summ = 0.0

    for k in range(nfa):
        for j in range(nta):
            for l in range(nta):
                si2 = np.sin(ta[j])
                si1 = np.sin(ta[l])
                mu2 = np.cos(ta[j])
                mu1 = np.cos(ta[l])
                ts = ta[j]
                tv = ta[l]
                phi = fa[k]

                pond = mu1 * mu2 * si1 * si2 * wfa[k] * wta[j] * wta[l]
                brdfint = (a * ts * ts * tv * tv +
                          ap * (ts * ts + tv * tv) +
                          b * ts * tv * np.cos(phi) + c)

                brdfalb = brdfalb + brdfint * pond
                summ = summ + pond

    brdfalb = brdfalb / summ
    return brdfalb


def roujalbe(k0, k1, k2):
    """
    Compute spherical albedo for Roujean BRDF model.

    Integrates the Roujean BRDF over all viewing and illumination angles
    using Gaussian quadrature.

    Parameters
    ----------
    k0, k1, k2 : float
        Roujean model parameters

    Returns
    -------
    brdfalb : float
        Spherical albedo
    """
    nta = 24
    nfa = 48
    pi = np.pi

    teta1 = 0.0
    teta2 = pi / 2.0
    ta, wta = gauss(teta1, teta2, nta)

    phi1 = 0.0
    phi2 = 2.0 * pi
    fa, wfa = gauss(phi1, phi2, nfa)

    brdfalb = 0.0
    summ = 0.0

    for k in range(nfa):
        for j in range(nta):
            for l in range(nta):
                si2 = np.sin(ta[j])
                si1 = np.sin(ta[l])
                mu2 = np.cos(ta[j])
                mu1 = np.cos(ta[l])
                ts = ta[j]
                tv = ta[l]
                fr = np.arccos(np.cos(fa[k]))

                pond = mu1 * mu2 * si1 * si2 * wfa[k] * wta[j] * wta[l]

                tts = np.tan(ts)
                ttv = np.tan(tv)
                xmus = np.cos(ts)
                xmuv = np.cos(tv)

                cpsi = xmus * xmuv + np.sin(ts) * np.sin(tv) * np.cos(fr)

                if cpsi < 1.0:
                    psi = np.arccos(cpsi)
                else:
                    psi = 0.0

                f2 = (4.0 / (3.0 * pi * (xmus + xmuv)) *
                      ((pi / 2.0 - psi) * cpsi + np.sin(psi)) - 1.0 / 3.0)

                ft = tts * tts + ttv * ttv - 2.0 * tts * ttv * np.cos(fr)
                f1 = (0.5 * ((pi - fr) * np.cos(fr) + np.sin(fr)) * tts * ttv -
                      tts - ttv - np.sqrt(ft)) / pi

                brdfalb = brdfalb + (k0 + k1 * f1 + k2 * f2) * pond
                summ = summ + pond

    brdfalb = brdfalb / summ
    return brdfalb


# ==============================================================================
# Water Reflectance Lookup Tables
# ==============================================================================

# Clear water reflectance (0.25-1.5 μm, 1501 points at 0.0025 μm spacing)
# Values given between 0.5 and 1.0 microns, outside this interval set to 0
_CLEARW_REFLECTANCE = np.array([
    *([0.0] * 58),
    0.00000, 0.02050, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100,
    0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100,
    0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100,
    0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100,
    0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100,
    0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100, 0.04100,
    0.04100, 0.04150, 0.04200, 0.04250, 0.04300, 0.04350, 0.04400,
    0.04400, 0.04400, 0.04500, 0.04600, 0.04650, 0.04700, 0.04800,
    0.04900, 0.04950, 0.05000, 0.05100, 0.05200, 0.05300, 0.05400,
    0.05450, 0.05500, 0.05550, 0.05600, 0.05750, 0.05900, 0.05950,
    0.06000, 0.06050, 0.06100, 0.06100, 0.06100, 0.06000, 0.05900,
    # Continuation from sr(136,1501)
    0.05800, 0.05700, 0.05550, 0.05400, 0.05350, 0.05300, 0.05200,
    0.05100, 0.05050, 0.05000, 0.04950, 0.04900, 0.04800, 0.04700,
    0.04650, 0.04600, 0.04600, 0.04600, 0.04550, 0.04500, 0.04450,
    0.04400, 0.04350, 0.04300, 0.04300, 0.04300, 0.04200, 0.04100,
    0.04050, 0.04000, 0.03900, 0.03800, 0.03750, 0.03700, 0.03700,
    0.03700, 0.03650, 0.03600, 0.03450, 0.03300, 0.03250, 0.03200,
    0.03150, 0.03100, 0.03000, 0.02900, 0.02800, 0.02700, 0.02550,
    0.02400, 0.02350, 0.02300, 0.02200, 0.02100, 0.01950, 0.01800,
    0.01650, 0.01500, 0.01350, 0.01200, 0.01050, 0.00900, 0.00850,
    0.00800, 0.00700, 0.00600, 0.00500, 0.00400, 0.00300, 0.00200,
    0.00150, 0.00100, 0.00050, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000,
    *([0.0] * 1280)
], dtype=np.float64)


# Lake water reflectance (0.25-1.5 μm, 1501 points)
# Values given between 0.35 and 1.0 microns
_LAKEW_REFLECTANCE = np.array([
    *([0.0] * 40),
    0.00000, 0.02250, 0.04500, 0.04600, 0.04700, 0.04750, 0.04800,
    0.04900, 0.05000, 0.05050, 0.05100, 0.05100, 0.05100, 0.05200,
    0.05300, 0.05300, 0.05300, 0.05400, 0.05500, 0.05600, 0.05700,
    0.05750, 0.05800, 0.05850, 0.05900, 0.05950, 0.06000, 0.06000,
    0.06000, 0.06100, 0.06200, 0.06300, 0.06400, 0.06450, 0.06500,
    0.06600, 0.06700, 0.06750, 0.06800, 0.06900, 0.07000, 0.07000,
    0.07000, 0.06950, 0.06900, 0.07000, 0.07100, 0.07100, 0.07100,
    0.07150, 0.07200, 0.07300, 0.07400, 0.07400, 0.07400, 0.07450,
    0.07500, 0.07550, 0.07600, 0.07600, 0.07600, 0.07650, 0.07700,
    0.07700, 0.07700, 0.07700, 0.07700, 0.07700, 0.07700, 0.07800,
    # sr(111,180)
    0.07900, 0.07950, 0.08000, 0.08000, 0.08000, 0.08050, 0.08100,
    0.08100, 0.08100, 0.08150, 0.08200, 0.08200, 0.08200, 0.08200,
    0.08200, 0.08250, 0.08300, 0.08250, 0.08200, 0.08250, 0.08300,
    0.08250, 0.08200, 0.08200, 0.08200, 0.08200, 0.08200, 0.08200,
    0.08200, 0.08200, 0.08200, 0.08100, 0.08000, 0.07950, 0.07900,
    0.07900, 0.07900, 0.07800, 0.07700, 0.07600, 0.07500, 0.07450,
    0.07400, 0.07300, 0.07200, 0.07150, 0.07100, 0.07050, 0.07000,
    0.06900, 0.06800, 0.06700, 0.06600, 0.06550, 0.06500, 0.06450,
    0.06400, 0.06350, 0.06300, 0.06200, 0.06100, 0.06050, 0.06000,
    0.05950, 0.05900, 0.05800, 0.05700, 0.05650, 0.05600, 0.05500,
    # sr(181,1501)
    0.05400, 0.05350, 0.05300, 0.05200, 0.05100, 0.05050, 0.05000,
    0.04900, 0.04800, 0.04750, 0.04700, 0.04650, 0.04600, 0.04550,
    0.04500, 0.04450, 0.04400, 0.04300, 0.04200, 0.04150, 0.04100,
    0.04050, 0.04000, 0.03950, 0.03900, 0.03800, 0.03700, 0.03650,
    0.03600, 0.03550, 0.03500, 0.03450, 0.03400, 0.03300, 0.03200,
    0.03200, 0.03200, 0.03150, 0.03100, 0.03050, 0.03000, 0.03000,
    0.03000, 0.02900, 0.02800, 0.02750, 0.02700, 0.02700, 0.02700,
    0.02600, 0.02500, 0.02450, 0.02400, 0.02400, 0.02400, 0.02350,
    0.02300, 0.02250, 0.02200, 0.02200, 0.02200, 0.02100, 0.02000,
    0.02000, 0.02000, 0.02000, 0.02000, 0.01950, 0.01900, 0.01900,
    0.01900, 0.01900, 0.01900, 0.01900, 0.01900, 0.01900, 0.01900,
    0.01900, 0.01900, 0.01850, 0.01800, 0.01700, 0.01600, 0.01600,
    0.01600, 0.01550, 0.01500, 0.01450, 0.01400, 0.01300, 0.01200,
    0.01200, 0.01200, 0.01200, 0.01200, 0.01150, 0.01100, 0.01100,
    0.01100, 0.01000, 0.00900, 0.00850, 0.00800, 0.00800, 0.00800,
    0.00700, 0.00600, 0.00550, 0.00500, 0.00500, 0.00500, 0.00400,
    0.00300, 0.00250, 0.00200, 0.00100, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000,
    *([0.0] * 1200)
], dtype=np.float64)


def clearw():
    """
    Get clear water reflectance spectrum.

    Returns
    -------
    r : ndarray
        Reflectance values, 1501 points covering 0.25-1.5 μm

    Notes
    -----
    Values are given between 0.5 and 1.0 microns.
    Outside this interval the values are set to 0.
    """
    return _CLEARW_REFLECTANCE.copy()


def lakew():
    """
    Get lake water reflectance spectrum.

    Returns
    -------
    r : ndarray
        Reflectance values, 1501 points covering 0.25-1.5 μm

    Notes
    -----
    Values are given between 0.35 and 1.0 microns.
    Outside this interval the values are set to 0.
    """
    return _LAKEW_REFLECTANCE.copy()


__all__ = [
    # Albedo models
    'minnalbe',
    'modisalbe',
    'waltalbe',
    'roujalbe',
    # BRDF models
    'minnbrdf',
    'waltbrdf',
    'roujbrdf',
    # Water reflectance
    'clearw',
    'lakew',
]
