"""
Aerosol optical properties module.

This module sets up aerosol optical properties including extinction,
scattering, asymmetry parameters, and phase functions. It handles multiple
aerosol components and mixing.

Converted from Fortran AEROSO.f

Functions:
    aeroso: Main aerosol properties setup
"""

import numpy as np
from sixs.gauss import gauss


# Standard wavelengths for aerosol calculations (microns)
WLDISC = np.array([
    0.350, 0.400, 0.412, 0.443, 0.470, 0.488, 0.515, 0.550,
    0.590, 0.633, 0.670, 0.694, 0.760, 0.860, 1.240, 1.536,
    1.650, 1.950, 2.250, 3.750
])


def aeroso(iaer, co, xmud, wldis, nquad, ipol=0, aer_file=None):
    """
    Compute aerosol optical properties.

    Sets up aerosol extinction, scattering coefficients, asymmetry parameter,
    and phase functions for various aerosol models. Handles multi-component
    mixing.

    Parameters
    ----------
    iaer : int
        Aerosol model type:
        0 = No aerosols
        1-4 = Standard 4-component models (dust, water, oceanic, soot)
        5 = Background desert model (BDM)
        6 = Biomass burning model (BBM)
        7 = Stratospheric model (STM)
        8-11 = User-defined size distributions (requires MIE)
        12 = User-defined from file
    co : array_like
        Component mixing ratios, shape (4,)
    xmud : float
        Cosine of scattering angle for phase function interpolation
    wldis : array_like
        Wavelengths in microns, shape (20,)
    nquad : int
        Number of quadrature points for phase function
    ipol : int, optional
        Polarization flag (0=no polarization, 1=with polarization)
    aer_file : str, optional
        File path for user-defined aerosol model (iaer=12)

    Returns
    -------
    ext : ndarray
        Extinction coefficient, shape (20,)
    ome : ndarray
        Single scattering albedo, shape (20,)
    gasym : ndarray
        Asymmetry parameter, shape (20,)
    phase : ndarray
        Phase function at scattering angle xmud, shape (20,)
    qhase : ndarray
        Q Stokes parameter at scattering angle, shape (20,)
    uhase : ndarray
        U Stokes parameter at scattering angle, shape (20,)
    phasel : ndarray
        Phase function at all angles, shape (20, nquad)
    qhasel : ndarray
        Q Stokes at all angles, shape (20, nquad)
    uhasel : ndarray
        U Stokes at all angles, shape (20, nquad)

    Notes
    -----
    Converted from Fortran AEROSO.f

    For iaer=8-11, this function interfaces with the MIE module to compute
    aerosol properties from size distributions.

    For iaer=1-7, pre-computed lookup tables are used (these would need to
    be loaded from data files or converted separately).

    For iaer=12, aerosol properties are read from an external file.
    """
    pi = np.pi

    # Initialize outputs
    ext = np.zeros(20)
    sca = np.zeros(20)
    ome = np.zeros(20)
    gasym = np.zeros(20)
    phase = np.zeros(20)
    qhase = np.zeros(20)
    uhase = np.zeros(20)
    phasel = np.zeros((20, nquad))
    qhasel = np.zeros((20, nquad))
    uhasel = np.zeros((20, nquad))

    # Set up Gauss quadrature points
    cgaus_S = np.zeros(nquad)
    pdgs_S = np.zeros(nquad)
    nbmu = nquad
    nbmu_2 = (nbmu - 3) // 2

    cosang, weight = gauss(-1.0, 1.0, nbmu - 3)

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

    # Handle user-defined aerosol from file (iaer=12)
    if iaer == 12:
        if aer_file is None:
            raise ValueError("aer_file required for iaer=12")

        ext, sca, ome, gasym, phasel, qhasel, uhasel = _read_aerosol_file(
            aer_file, ipol
        )

        # Interpolate phase function to xmud
        j1, j2, coef = _find_angle_indices(xmud, cgaus_S, nbmu)
        for l in range(20):
            phase[l] = phasel[l, j1] + coef * (phasel[l, j1] - phasel[l, j2])
            if ipol != 0:
                qhase[l] = qhasel[l, j1] + coef * (qhasel[l, j1] - qhasel[l, j2])
                uhase[l] = uhasel[l, j1] + coef * (uhasel[l, j1] - uhasel[l, j2])

        return ext, ome, gasym, phase, qhase, uhase, phasel, qhasel, uhasel

    # No aerosol case
    if iaer == 0:
        # Set a minimal extinction at 0.443 microns for normalization
        ext[3] = 1.0  # l=4 in Fortran (0.443 microns)
        return ext, ome, gasym, phase, qhase, uhase, phasel, qhasel, uhasel

    # Find angle indices for phase function interpolation
    j1, j2, coef = _find_angle_indices(xmud, cgaus_S, nbmu)

    # Handle special aerosol models (iaer 5-11)
    if iaer >= 5 and iaer <= 11:
        if iaer == 5:
            # Background desert model
            raise NotImplementedError("BDM model requires lookup table conversion")
        elif iaer == 6:
            # Biomass burning model
            raise NotImplementedError("BBM model requires lookup table conversion")
        elif iaer == 7:
            # Stratospheric model
            raise NotImplementedError("STM model requires lookup table conversion")
        elif iaer >= 8 and iaer <= 11:
            # User-defined from size distribution - call MIE
            from sixs.mie import mie

            # These parameters would need to be passed in or set up
            # For now, raise an error indicating MIE integration needed
            raise NotImplementedError(
                "MIE integration requires size distribution parameters. "
                "Call mie() directly or extend this function with size distribution setup."
            )

        icp = 1  # Single component for special models
        cij = np.array([1.0, 0.0, 0.0, 0.0])

    else:
        # Standard 4-component models (iaer 1-4)
        # These would load from DUST, WATE, OCEA, SOOT lookup tables
        raise NotImplementedError(
            "Standard 4-component models require lookup table conversion. "
            "Component models: DUST, WATE, OCEA, SOOT (each ~70KB of data)"
        )

    return ext, ome, gasym, phase, qhase, uhase, phasel, qhasel, uhasel


def _find_angle_indices(xmud, cgaus_S, nbmu):
    """
    Find bracketing angle indices for interpolation.

    Parameters
    ----------
    xmud : float
        Target angle cosine
    cgaus_S : ndarray
        Gauss quadrature angle cosines
    nbmu : int
        Number of angles

    Returns
    -------
    j1 : int
        Lower bracket index
    j2 : int
        Upper bracket index
    coef : float
        Interpolation coefficient
    """
    for k in range(nbmu - 1):
        if xmud >= cgaus_S[k] and xmud < cgaus_S[k + 1]:
            j1 = k
            j2 = j1 + 1
            coef = -(xmud - cgaus_S[j1]) / (cgaus_S[j2] - cgaus_S[j1])
            return j1, j2, coef

    # Default to last interval if not found
    j1 = nbmu - 2
    j2 = nbmu - 1
    coef = 0.0
    return j1, j2, coef


def _read_aerosol_file(filename, ipol):
    """
    Read user-defined aerosol model from file.

    Parameters
    ----------
    filename : str
        Path to aerosol data file
    ipol : int
        Polarization flag

    Returns
    -------
    ext : ndarray
        Extinction, shape (20,)
    sca : ndarray
        Scattering, shape (20,)
    ome : ndarray
        Single scattering albedo, shape (20,)
    gasym : ndarray
        Asymmetry parameter, shape (20,)
    phasel : ndarray
        Phase function, shape (20, nbmu)
    qhasel : ndarray
        Q Stokes parameter, shape (20, nbmu)
    uhasel : ndarray
        U Stokes parameter, shape (20, nbmu)
    """
    with open(filename, 'r') as f:
        # Read number of angles
        nbmu = int(f.readline().strip())

        # Skip header line
        f.readline()

        # Read optical properties (20 wavelengths)
        ext = np.zeros(20)
        sca = np.zeros(20)
        ome = np.zeros(20)
        gasym = np.zeros(20)

        for l in range(20):
            line = f.readline().split()
            ext[l] = float(line[0])
            sca[l] = float(line[1])
            ome[l] = float(line[2])
            gasym[l] = float(line[3])

        # Skip 3 lines
        for _ in range(3):
            f.readline()

        # Read phase functions
        phasel = np.zeros((20, nbmu))
        for k in range(nbmu):
            line = f.readline().split()
            for l in range(20):
                phasel[l, k] = float(line[l + 1])  # Skip angle in first column

        qhasel = np.zeros((20, nbmu))
        uhasel = np.zeros((20, nbmu))

        if ipol != 0:
            # Read Q Stokes parameter
            for k in range(nbmu):
                line = f.readline().split()
                for l in range(20):
                    qhasel[l, k] = float(line[l + 1])

            # Read U Stokes parameter
            for k in range(nbmu):
                line = f.readline().split()
                for l in range(20):
                    uhasel[l, k] = float(line[l + 1])

    return ext, sca, ome, gasym, phasel, qhasel, uhasel


def mix_aerosol_components(ex, sc, asy, ph, qh, uh, cij, icp, nquad):
    """
    Mix multiple aerosol components.

    Computes weighted average of optical properties from multiple
    aerosol components based on volume mixing ratios.

    Parameters
    ----------
    ex : ndarray
        Extinction for each component, shape (4, 20)
    sc : ndarray
        Scattering for each component, shape (4, 20)
    asy : ndarray
        Asymmetry parameter for each component, shape (4, 20)
    ph : ndarray
        Phase function for each component, shape (4, 20, nquad)
    qh : ndarray
        Q Stokes for each component, shape (4, 20, nquad)
    uh : ndarray
        U Stokes for each component, shape (4, 20, nquad)
    cij : ndarray
        Volume mixing ratios, shape (4,)
    icp : int
        Number of components
    nquad : int
        Number of quadrature angles

    Returns
    -------
    ext : ndarray
        Mixed extinction, shape (20,)
    ome : ndarray
        Mixed single scattering albedo, shape (20,)
    gasym : ndarray
        Mixed asymmetry parameter, shape (20,)
    phasel : ndarray
        Mixed phase function, shape (20, nquad)
    qhasel : ndarray
        Mixed Q Stokes, shape (20, nquad)
    uhasel : ndarray
        Mixed U Stokes, shape (20, nquad)
    """
    ext_mix = np.zeros(20)
    sca_mix = np.zeros(20)
    gasym_mix = np.zeros(20)
    phasel_mix = np.zeros((20, nquad))
    qhasel_mix = np.zeros((20, nquad))
    uhasel_mix = np.zeros((20, nquad))

    for l in range(20):
        for j in range(icp):
            ext_mix[l] += ex[j, l] * cij[j]
            sca_mix[l] += sc[j, l] * cij[j]
            gasym_mix[l] += sc[j, l] * cij[j] * asy[j, l]

            for k in range(nquad):
                phasel_mix[l, k] += sc[j, l] * cij[j] * ph[j, l, k]
                qhasel_mix[l, k] += sc[j, l] * cij[j] * qh[j, l, k]
                uhasel_mix[l, k] += sc[j, l] * cij[j] * uh[j, l, k]

        # Normalize
        ome_mix_l = sca_mix[l] / ext_mix[l]
        gasym_mix[l] = gasym_mix[l] / sca_mix[l]

        for k in range(nquad):
            phasel_mix[l, k] = phasel_mix[l, k] / sca_mix[l]
            qhasel_mix[l, k] = qhasel_mix[l, k] / sca_mix[l]
            uhasel_mix[l, k] = uhasel_mix[l, k] / sca_mix[l]

    ome_mix = sca_mix / ext_mix

    return ext_mix, ome_mix, gasym_mix, phasel_mix, qhasel_mix, uhasel_mix


__all__ = ['aeroso', 'mix_aerosol_components', 'WLDISC']
