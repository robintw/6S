"""
Atmospheric scattering physics modules.

Functions for calculating molecular (Rayleigh) and aerosol scattering
in the atmosphere.

Functions:
    chand: Chandrasekhar function for molecular reflectance
    scatra: Scattering transmittance calculation
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


__all__ = [
    'chand',
    'scatra',
    'iso',
]
