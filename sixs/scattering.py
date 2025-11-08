"""
Atmospheric scattering physics modules.

Functions for calculating molecular (Rayleigh) and aerosol scattering
in the atmosphere.

Functions:
    chand: Chandrasekhar function for molecular reflectance
    scatra: Scattering transmittance calculation
    odrayl: Rayleigh optical depth calculation
"""

import numpy as np
from numba import jit

from sixs.cosine_albedo import csalbr
from sixs.successive_orders import iso as iso_full


@jit(nopython=True)
def chand(xphi, xmuv, xmus, xtau):
    """
    Chandrasekhar function for molecular (Rayleigh) reflectance.

    Calculates the reflection function of a molecular atmosphere using
    the Chandrasekhar H-function approximation.

    Parameters
    ----------
    xphi : float
        Azimuthal difference between sun and observation (degrees, 0-360)
        xphi=0 in backscattering direction
    xmuv : float
        Cosine of the observation (view) zenith angle (0-1)
    xmus : float
        Cosine of the sun zenith angle (0-1)
    xtau : float
        Molecular optical depth (>0)

    Returns
    -------
    xrray : float
        Molecular reflectance (0-1)

    Notes
    -----
    Converted from Fortran CHAND.f

    Uses a polynomial approximation to the Chandrasekhar H-function
    for Rayleigh scattering. Includes depolarization factor (0.0279)
    and accounts for multiple scattering.

    The algorithm computes three Fourier components of the reflectance:
    - Component 1: Isotropic term (m=0)
    - Component 2: First azimuthal harmonic (m=1)
    - Component 3: Second azimuthal harmonic (m=2)

    References
    ----------
    Chandrasekhar, S., Radiative Transfer, Dover Publications, 1960
    """
    # Coefficients for polynomial approximation
    as0 = np.array([
        0.33243832, -6.777104e-02, 0.16285370,
        1.577425e-03, -0.30924818, -1.240906e-02, -0.10324388,
        3.241678e-02, 0.11493334, -3.503695e-02
    ])
    as1 = np.array([0.19666292, -5.439061e-02])
    as2 = np.array([0.14545937, -2.910845e-02])

    pi = 3.1415927
    fac = pi / 180.0

    # Phase angle (180° - azimuthal difference)
    phios = 180.0 - xphi

    # Cosine of phase angle multiples
    xcosf1 = 1.0
    xcosf2 = np.cos(phios * fac)
    xcosf3 = np.cos(2.0 * phios * fac)

    # Depolarization factor for air molecules
    xdep = 0.0279
    xbeta2 = 0.5

    # Depolarization correction
    xfd = xdep / (2.0 - xdep)
    xfd = (1.0 - xfd) / (1.0 + 2.0 * xfd)

    # Phase function components
    xph1 = 1.0 + (3.0 * xmus**2 - 1.0) * (3.0 * xmuv**2 - 1.0) * xfd / 8.0

    xph2 = -xmus * xmuv * np.sqrt(1.0 - xmus**2) * np.sqrt(1.0 - xmuv**2)
    xph2 = xph2 * xfd * xbeta2 * 1.5

    xph3 = (1.0 - xmus**2) * (1.0 - xmuv**2)
    xph3 = xph3 * xfd * xbeta2 * 0.375

    # Primary scattering contribution
    xitm = (1.0 - np.exp(-xtau * (1.0 / xmus + 1.0 / xmuv))) * xmus / (4.0 * (xmus + xmuv))
    xp1 = xph1 * xitm
    xp2 = xph2 * xitm
    xp3 = xph3 * xitm

    # Multiple scattering contribution
    xitm = (1.0 - np.exp(-xtau / xmus)) * (1.0 - np.exp(-xtau / xmuv))
    cfonc1 = xph1 * xitm
    cfonc2 = xph2 * xitm
    cfonc3 = xph3 * xitm

    # Polynomial basis functions
    xlntau = np.log(xtau)
    pl = np.zeros(10)
    pl[0] = 1.0
    pl[1] = xlntau
    pl[2] = xmus + xmuv
    pl[3] = xlntau * pl[2]
    pl[4] = xmus * xmuv
    pl[5] = xlntau * pl[4]
    pl[6] = xmus**2 + xmuv**2
    pl[7] = xlntau * pl[6]
    pl[8] = xmus**2 * xmuv**2
    pl[9] = xlntau * pl[8]

    # Calculate Fourier series coefficients
    fs0 = 0.0
    for i in range(10):
        fs0 += pl[i] * as0[i]

    fs1 = pl[0] * as1[0] + pl[1] * as1[1]
    fs2 = pl[0] * as2[0] + pl[1] * as2[1]

    # Total contribution for each Fourier component
    xitot1 = xp1 + cfonc1 * fs0 * xmus
    xitot2 = xp2 + cfonc2 * fs1 * xmus
    xitot3 = xp3 + cfonc3 * fs2 * xmus

    # Sum Fourier series
    xrray = xitot1 * xcosf1
    xrray += xitot2 * xcosf2 * 2.0
    xrray += xitot3 * xcosf3 * 2.0
    xrray /= xmus

    return xrray


def iso(iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt, nt, mu,
        rm_neg_mu, rm_0, rm_pos_mu, gb, xf_result):
    """
    Wrapper for ISO successive orders of scattering calculation.

    Adapts the SCATRA calling convention to the full ISO function.

    Parameters
    ----------
    iaer_prof : int
        Aerosol profile identifier
    tamoy : float
        Mean aerosol optical depth
    trmoy : float
        Mean molecular (Rayleigh) optical depth
    pizmoy : float
        Mean single scattering albedo
    tamoyp : float
        Aerosol optical depth above target altitude
    trmoyp : float
        Molecular optical depth above target altitude
    palt : float
        Target altitude (meters)
    nt : int
        Number of atmospheric layers
    mu : int
        Number of Gauss quadrature angles
    rm_neg_mu : float
        Cosine value at -mu index
    rm_0 : float
        Cosine value at 0 index (solar zenith)
    rm_pos_mu : float
        Cosine value at +mu index
    gb : array_like
        Gauss weights
    xf_result : array_like
        Output array for transmittances (modified in place)
        xf_result[0] = upward/downward transmittance (Fortran index -1)
        xf_result[1] = spherical albedo * 0.5 (Fortran index 0)
        xf_result[2] = downward/upward transmittance (Fortran index 1)

    Notes
    -----
    This wrapper converts the SCATRA calling convention (individual angle values)
    to the full ISO calling convention (full rm and gb arrays).
    """
    # Initialize rm with Gauss quadrature angles on [0, 1]
    # This mimics what the Fortran code expects
    x_gauss, w_gauss = np.polynomial.legendre.leggauss(mu)
    # Map from [-1, 1] to [0, 1]
    rm_gauss = (x_gauss + 1.0) / 2.0

    # Build full rm array: negative angles, zero, positive angles
    rm_full = np.zeros(2 * mu + 1)
    rm_full[0:mu] = -rm_gauss[::-1]  # Negative values (reversed order)
    rm_full[mu] = 0.0
    rm_full[mu+1:2*mu+1] = rm_gauss  # Positive values

    # Override specific values as SCATRA does
    rm_full[0] = rm_neg_mu     # rm(-mu) -> index 0
    rm_full[mu] = rm_0          # rm(0) -> index mu
    rm_full[2*mu] = rm_pos_mu  # rm(mu) -> index 2*mu

    # Call the full ISO function
    xf = iso_full(iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp,
                   palt, nt, mu, rm_full, gb)

    # Copy results to output array
    xf_result[0] = xf[0]
    xf_result[1] = xf[1]
    xf_result[2] = xf[2]


def scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu, rm, gb,
           xmus, xmuv):
    """
    Calculate scattering transmittances for different atmospheric components.

    Computes direct and diffuse transmittances for upward and downward paths,
    and spherical albedo for Rayleigh scattering, aerosol scattering, and
    their combination.

    Parameters
    ----------
    iaer_prof : int
        Aerosol profile identifier
    taer : float
        Total aerosol optical depth
    taerp : float
        Aerosol optical depth above target altitude
    tray : float
        Total molecular (Rayleigh) optical depth
    trayp : float
        Molecular optical depth above target altitude
    piza : float
        Mean single scattering albedo
    palt : float
        Target altitude (meters), >900 means above atmosphere
    nt : int
        Number of atmospheric layers
    mu : int
        Number of Gauss quadrature angles
    rm : array_like
        Array of cosine values (size 2*mu+1)
        Fortran indexing: rm(-mu:mu)
        Python: rm[0] corresponds to -mu, rm[mu] to 0, rm[2*mu] to +mu
    gb : array_like
        Gauss weights (size 2*mu+1)
    xmus : float
        Cosine of solar zenith angle
    xmuv : float
        Cosine of view zenith angle

    Returns
    -------
    dict
        Dictionary containing:
        - 'total': dict with keys 'ddir', 'ddif', 'udir', 'udif', 'sphalb'
        - 'rayleigh': dict with keys 'ddir', 'ddif', 'udir', 'udif', 'sphalb'
        - 'aerosol': dict with keys 'ddir', 'ddif', 'udir', 'udif', 'sphalb'

        Where:
        - ddir: downward direct transmittance
        - ddif: downward diffuse transmittance
        - udir: upward direct transmittance
        - udif: upward diffuse transmittance
        - sphalb: spherical albedo

    Notes
    -----
    Converted from Fortran SCATRA.f

    Processes three cases:
    1. Rayleigh (molecular) scattering only
    2. Aerosol scattering only
    3. Combined Rayleigh + aerosol

    Uses exponential transmission for direct beams and calls ISO subroutine
    for diffuse transmittances (currently stubbed).
    """
    # Initialize output values
    result = {
        'total': {'ddir': 1.0, 'ddif': 0.0, 'udir': 1.0, 'udif': 0.0, 'sphalb': 0.0},
        'rayleigh': {'ddir': 1.0, 'ddif': 0.0, 'udir': 1.0, 'udif': 0.0, 'sphalb': 0.0},
        'aerosol': {'ddir': 1.0, 'ddif': 0.0, 'udir': 1.0, 'udif': 0.0, 'sphalb': 0.0}
    }

    # Working variables
    xtrans = np.zeros(3)  # Array for ISO output: [downward, sphalb, upward]

    # Loop over three cases: Rayleigh, aerosol, combined
    for it in range(1, 4):  # it=1,2,3 in Fortran
        # it=1: Rayleigh only
        # it=2: Aerosol only (skip if no aerosol)
        # it=3: Rayleigh + aerosol

        if it == 2 and taer <= 0.0:
            continue

        # --- Case 1: Rayleigh scattering only ---
        if it == 1:
            if palt > 900:
                # Above atmosphere - simple exponential formulas
                udiftt = (2.0/3.0 + xmuv) + (2.0/3.0 - xmuv) * np.exp(-tray/xmuv)
                udiftt = udiftt / ((4.0/3.0) + tray) - np.exp(-tray/xmuv)

                ddiftt = (2.0/3.0 + xmus) + (2.0/3.0 - xmus) * np.exp(-tray/xmus)
                ddiftt = ddiftt / ((4.0/3.0) + tray) - np.exp(-tray/xmus)

                ddirtt = np.exp(-tray/xmus)
                udirtt = np.exp(-tray/xmuv)

                sphalbt = csalbr(tray)

            elif palt < 900:
                # Within atmosphere
                tamol = 0.0
                tamolp = 0.0

                # Upward transmittance
                iso(iaer_prof, tamol, tray, piza, tamolp, trayp,
                    palt, nt, mu, -xmuv, xmus, xmuv, gb, xtrans)

                udiftt = xtrans[0] - np.exp(-trayp/xmuv)
                udirtt = np.exp(-trayp/xmuv)

                # Downward transmittance
                ddiftt = (2.0/3.0 + xmus) + (2.0/3.0 - xmus) * np.exp(-tray/xmus)
                ddiftt = ddiftt / ((4.0/3.0) + tray) - np.exp(-tray/xmus)
                ddirtt = np.exp(-tray/xmus)

                udirtt = np.exp(-tray/xmuv)
                sphalbt = csalbr(tray)

            if palt <= 0.0:
                # At or below surface
                udiftt = 0.0
                udirtt = 1.0

            # Store Rayleigh results
            result['rayleigh']['ddir'] = ddirtt
            result['rayleigh']['ddif'] = ddiftt
            result['rayleigh']['udir'] = udirtt
            result['rayleigh']['udif'] = udiftt
            result['rayleigh']['sphalb'] = sphalbt

        # --- Case 2: Aerosol scattering only ---
        elif it == 2:
            tamol = 0.0
            tamolp = 0.0

            # Upward transmittance
            iso(iaer_prof, taer, tamol, piza, taerp, tamolp,
                palt, nt, mu, -xmuv, xmus, xmuv, gb, xtrans)

            udiftt = xtrans[0] - np.exp(-taerp/xmuv)
            udirtt = np.exp(-taerp/xmuv)

            # Downward transmittance
            iso(iaer_prof, taer, tamol, piza, taerp, tamolp,
                999.0, nt, mu, -xmus, xmus, xmus, gb, xtrans)

            ddirtt = np.exp(-taer/xmus)
            ddiftt = xtrans[2] - np.exp(-taer/xmus)
            sphalbt = xtrans[1] * 2.0

            if palt <= 0.0:
                udiftt = 0.0
                udirtt = 1.0

            # Store aerosol results
            result['aerosol']['ddir'] = ddirtt
            result['aerosol']['ddif'] = ddiftt
            result['aerosol']['udir'] = udirtt
            result['aerosol']['udif'] = udiftt
            result['aerosol']['sphalb'] = sphalbt

        # --- Case 3: Combined Rayleigh + aerosol ---
        elif it == 3:
            # Upward transmittance
            iso(iaer_prof, taer, tray, piza, taerp, trayp,
                palt, nt, mu, -xmuv, xmus, xmuv, gb, xtrans)

            udirtt = np.exp(-(taerp + trayp)/xmuv)
            udiftt = xtrans[0] - np.exp(-(taerp + trayp)/xmuv)

            # Downward transmittance
            iso(iaer_prof, taer, tray, piza, taerp, trayp,
                999.0, nt, mu, -xmus, xmus, xmus, gb, xtrans)

            ddiftt = xtrans[2] - np.exp(-(taer + tray)/xmus)
            ddirtt = np.exp(-(taer + tray)/xmus)
            sphalbt = xtrans[1] * 2.0

            if palt <= 0.0:
                udiftt = 0.0
                udirtt = 1.0

            # Store total results
            result['total']['ddir'] = ddirtt
            result['total']['ddif'] = ddiftt
            result['total']['udir'] = udirtt
            result['total']['udif'] = udiftt
            result['total']['sphalb'] = sphalbt

    return result


def odrayl(wl, z, p, t):
    """
    Calculate Rayleigh (molecular) optical depth as a function of wavelength.

    Uses the Edlen (1966) formula for air refraction index and integrates
    over atmospheric layers to compute total Rayleigh optical depth.

    Parameters
    ----------
    wl : float
        Wavelength in micrometers (μm)
    z : array_like
        Altitude levels in km, shape (34,)
    p : array_like
        Pressure levels in mb, shape (34,)
    t : array_like
        Temperature levels in K, shape (34,)

    Returns
    -------
    tray : float
        Total Rayleigh optical depth (dimensionless)

    Notes
    -----
    Converted from Fortran ODRAYL.f

    The calculation uses:
    - Air refraction index from Edlen 1966 (Metrologia, 2, 71-80)
    - Depolarization factor delta (from global state, typically 0.0279)
    - Integration over 33 atmospheric layers

    References
    ----------
    Edlen, B. (1966). The refractive index of air. Metrologia, 2(2), 71.

    Examples
    --------
    >>> # Standard atmosphere profile
    >>> z = np.array([...])  # 34 altitude levels
    >>> p = np.array([...])  # 34 pressure levels
    >>> t = np.array([...])  # 34 temperature levels
    >>> wl = 0.55  # 550 nm in microns
    >>> tray = odrayl(wl, z, p, t)
    """
    from sixs.successive_orders import _atm_state

    # Constants
    pi = 3.1415926
    ak = 1.0 / wl
    awl = wl

    # Get depolarization factor from global state
    delta = _atm_state.delta  # typically 0.0279

    # Air refraction index (Edlen 1966, Metrologia 2, 71-80)
    # Setting partial water vapor pressure pw=0
    a1 = 130.0 - ak * ak
    a2 = 38.9 - ak * ak
    a3 = 2406030.0 / a1
    a4 = 15997.0 / a2
    an = (8342.13 + a3 + a4) * 1.0e-08
    an = an + 1.0

    # Rayleigh scattering cross section
    a = (24.0 * pi**3) * ((an * an - 1.0)**2) * (6.0 + 3.0 * delta) / (6.0 - 7.0 * delta)
    a = a / ((an * an + 2.0)**2)

    # Integrate over atmospheric layers
    tray = 0.0
    ns = 2.54743e+19  # Number density at standard conditions

    for k in range(33):  # 33 layers (0-32)
        # Average pressure-temperature ratio for layer
        dppt = (288.15 / 1013.25) * (p[k] / t[k] + p[k+1] / t[k+1]) / 2.0

        # Rayleigh scattering coefficient
        sr = (a * dppt / (awl**4) / ns * 1.0e+16) * 1.0e+05

        # Add layer contribution (layer thickness * scattering coefficient)
        tray += (z[k+1] - z[k]) * sr

    return tray


def trunca(pha, qha, uha, alphal, betal, gammal, zetal, nquad, ipol):
    """
    Compute Legendre expansion coefficients from phase function.

    This implements the delta-M truncation method for forward-peaked
    phase functions. Computes Legendre polynomial expansion coefficients
    from the discrete phase function values.

    Converted from Fortran TRUNCA.f

    Parameters
    ----------
    pha : ndarray
        Phase function values at quadrature angles, shape (nquad,)
    qha : ndarray or None
        Q Stokes parameter phase function, shape (nquad,) or None
    uha : ndarray or None
        U Stokes parameter phase function, shape (nquad,) or None
    alphal : ndarray
        Output: alpha Legendre coefficients, shape (nquad,)
        Modified in place
    betal : ndarray
        Output: beta Legendre coefficients, shape (nquad,)
        Modified in place
    gammal : ndarray
        Output: gamma Legendre coefficients, shape (nquad,)
        Modified in place
    zetal : ndarray
        Output: zeta Legendre coefficients, shape (nquad,)
        Modified in place
    nquad : int
        Number of quadrature points
    ipol : int
        Polarization flag (0=no polarization, 1=polarization)

    Returns
    -------
    coeff : float
        Truncation coefficient (currently returns 0.0)

    Notes
    -----
    The function modifies the input arrays alphal, betal, gammal, zetal
    in place with the computed Legendre expansion coefficients.

    The coefficients are normalized by betal(0) at the end.
    """
    # Initialize arrays
    nbmu = nquad
    nbmu_2 = (nbmu - 3) // 2

    # Allocate working arrays
    cgaus_S = np.zeros(nquad)
    pdgs_S = np.zeros(nquad)
    pl = np.zeros(nquad + 1)
    pol = np.zeros(nquad + 1)
    deltal = np.zeros(nquad)

    # Calculate Gauss quadrature points
    from sixs.gauss import gauss
    cosang, weight = gauss(-1.0, 1.0, nbmu - 3)

    # Set up quadrature points and weights
    cgaus_S[0] = -1.0
    pdgs_S[0] = 0.0

    for j in range(nbmu_2):
        cgaus_S[j + 1] = cosang[j]
        pdgs_S[j + 1] = weight[j]

    cgaus_S[nbmu_2 + 1] = 0.0
    pdgs_S[nbmu_2 + 1] = 0.0

    for j in range(nbmu_2, nbmu - 3):
        cgaus_S[j + 2] = cosang[j]
        pdgs_S[j + 2] = weight[j]

    cgaus_S[nbmu - 1] = 1.0
    pdgs_S[nbmu - 1] = 0.0

    # Initialize Legendre coefficients
    for k in range(nbmu - 2):
        alphal[k] = 0.0
        betal[k] = 0.0
        gammal[k] = 0.0
        deltal[k] = 0.0
        zetal[k] = 0.0

    # Compute beta Legendre coefficients from phase function
    for j in range(nbmu):
        x = pha[j] * pdgs_S[j]
        rm = cgaus_S[j]
        pl[0] = 1.0

        for k in range(nbmu - 2):
            # Recurrence relation for Legendre polynomials
            if k == 0:
                pl[1] = rm
            else:
                pl[k + 1] = ((2 * k + 1) * rm * pl[k] - k * pl[k - 1]) / (k + 1)

            betal[k] += x * pl[k]

    # Normalize beta coefficients
    for k in range(nbmu - 2):
        betal[k] = (2 * k + 1) * 0.5 * betal[k]

        # Set negative coefficients to zero
        if betal[k] < 0:
            for j in range(k, nbmu - 2):
                betal[j] = 0.0
            break

    # Handle polarization case
    if ipol != 0 and qha is not None and uha is not None:
        for j in range(nbmu):
            x = qha[j] * pdgs_S[j]
            xx = uha[j] * pdgs_S[j]
            rm = cgaus_S[j]

            # Initialize polarization polynomials
            pol[0] = 0.0
            pol[1] = 0.0
            pol[2] = 3.0 * (1.0 - rm**2) / 2.0 / np.sqrt(6.0)

            pl[0] = 1.0

            for k in range(2, nbmu - 2):
                # Recurrence for polarization polynomials
                d = (2.0 * k + 1.0) / np.sqrt((k + 3) * (k - 1))
                e = np.sqrt((k + 2) * (k - 2)) / (2.0 * k + 1.0)
                pol[k + 1] = d * (rm * pol[k] - e * pol[k - 1])
                gammal[k] += x * pol[k]

            for k in range(nbmu - 2):
                if k == 0:
                    pl[1] = rm
                else:
                    pl[k + 1] = ((2.0 * k + 1.0) * rm * pl[k] - k * pl[k - 1]) / (k + 1.0)
                deltal[k] += xx * pl[k]

        # Normalize polarization coefficients
        for k in range(nbmu - 2):
            deltal[k] = deltal[k] * (2.0 * k + 1.0) / 2.0
            gammal[k] = gammal[k] * (2.0 * k + 1.0) / 2.0

        # Compute alpha and zeta coefficients
        for i in range(2, nbmu - 2):
            co1 = 4.0 * (2.0 * i + 1.0) / (i * (i - 1) * (i + 1) * (i + 2))
            co2 = i * (i - 1) / ((i + 1) * (i + 2))
            co3 = co2 * deltal[i]
            co2 = co2 * betal[i]

            nn = i // 2
            mm = (i - 1) // 2

            som1 = 0.0
            som2 = 0.0
            som3 = 0.0
            som4 = 0.0

            for j in range(1, nn + 1):
                c2 = (i - 1) * (i - 1) - 3.0 * (2 * j - 1) * (i - j)
                som1 += c2 * betal[i - 2 * j]
                som2 += c2 * deltal[i - 2 * j]

            for j in range(mm + 1):
                c2 = (i - 1) * (i - 1) - 3.0 * j * (2 * i - 2 * j - 1)
                som3 += c2 * betal[i - 2 * j - 1]
                som4 += c2 * deltal[i - 2 * j - 1]

            zetal[i] = co3 - co1 * (som2 - som3)
            alphal[i] = co2 - co1 * (som1 - som4)

        # Normalize all coefficients by betal[0]
        z1p = betal[0]
        if z1p != 0:
            for k in range(nbmu - 2):
                alphal[k] /= z1p
                betal[k] /= z1p
                gammal[k] /= z1p
                zetal[k] /= z1p

    coeff = 0.0
    return coeff


__all__ = [
    'chand',
    'scatra',
    'iso',
    'odrayl',
    'trunca',
]
