"""
Discrete ordinates computation for atmospheric scattering.

Main computation loop that:
1. Loops over 20 discrete wavelengths
2. Computes atmospheric reflectances (Ray human, aerosol, mixed)
3. Computes scattering transmittances (direct and diffuse)

Converted from Fortran DISCOM.f

Functions:
    discom: Main discrete ordinates computation
"""

import numpy as np
from sixs.odrayl import odrayl
from sixs.scattering import trunca, scatra
from sixs.atmospheric_reflectance import atmref


def discom(idatmp, iaer, iaer_prof, xmus, xmuv, phi, taer55, taer55p,
           palt, phirad, nt, mu, np_angles, rm, gb, rp, ftray, ipol, xlm1, xlm2,
           nfi, ext, ome, gasym, phase, qhase, uhase, wldis, wlinf, wlsup,
           alphal, betal, gammal, zetal, phasel, qhasel, uhasel, nquad,
           alt_z, taer_z, taer55_z, num_z):
    """
    Compute atmospheric optical properties at discrete wavelengths.

    Loops over 20 standard wavelengths and computes:
    - Rayleigh optical depth
    - Atmospheric reflectances (Rayleigh, aerosol, mixed)
    - Scattering transmittances (downward and upward, direct and diffuse)
    - Spherical albedos

    Stores results in arrays indexed by wavelength for later interpolation.

    Parameters
    ----------
    idatmp : int
        Observation level (0=ground, 4=TOA, other=plane)
    iaer : int
        Aerosol model (0=no aerosol)
    iaer_prof : int
        Aerosol profile type
    xmus : float
        Cosine of solar zenith angle
    xmuv : float
        Cosine of viewing zenith angle
    phi : float
        Relative azimuth angle (degrees)
    taer55 : float
        Aerosol optical depth at 550 nm
    taer55p : float
        Aerosol optical depth above plane at 550 nm
    palt : float
        Observation altitude (km)
    phirad : float
        Relative azimuth angle (radians)
    nt : int
        Number of atmospheric layers
    mu : int
        Number of viewing angles for quadrature
    np_angles : int
        Number of azimuth angles
    rm : ndarray
        Gauss angle cosines, shape (2*mu+1,)
    gb : ndarray
        Gauss weights, shape (2*mu+1,)
    rp : ndarray
        Azimuth angles (radians), shape (np_angles,)
    ftray : float
        Fraction of Rayleigh above plane
    ipol : int
        Polarization flag (0=no, 1=yes, 2=both)
    xlm1 : ndarray
        Fourier components workspace, shape (2*mu+1, np_angles)
    xlm2 : ndarray
        Secondary Fourier workspace, shape (2*mu+1, np_angles)
    nfi : int
        Number of azimuth output angles
    ext : ndarray
        Aerosol extinction at 20 wavelengths, shape (20,)
    ome : ndarray
        Aerosol single scattering albedo, shape (20,)
    gasym : ndarray
        Aerosol asymmetry parameter, shape (20,)
    phase : ndarray
        Aerosol phase function, shape (20,)
    qhase : ndarray
        Q Stokes phase function, shape (20,)
    uhase : ndarray
        U Stokes phase function, shape (20,)
    wldis : ndarray
        Discrete wavelengths (microns), shape (20,)
    wlinf : float
        Lower wavelength bound for computation
    wlsup : float
        Upper wavelength bound for computation
    alphal : ndarray
        Legendre coefficients for phase function expansion
    betal : ndarray
        Legendre coefficients
    gammal : ndarray
        Legendre coefficients
    zetal : ndarray
        Legendre coefficients
    phasel : ndarray
        Phase function at all angles, shape (20, nquad)
    qhasel : ndarray
        Q Stokes at all angles, shape (20, nquad)
    uhasel : ndarray
        U Stokes at all angles, shape (20, nquad)
    nquad : int
        Number of quadrature angles
    alt_z : ndarray
        Altitude profile (km), shape (num_z+1,)
    taer_z : ndarray
        Aerosol optical depth profile, shape (num_z+1,)
    taer55_z : ndarray
        Aerosol optical depth at 550 nm profile, shape (num_z+1,)
    num_z : int
        Number of aerosol profile layers

    Returns
    -------
    roatm : ndarray
        Atmospheric reflectances [Rayleigh, mixed, aerosol], shape (3, 20)
    rqatm : ndarray
        Q Stokes atmospheric reflectances, shape (3, 20)
    ruatm : ndarray
        U Stokes atmospheric reflectances, shape (3, 20)
    dtdir : ndarray
        Downward direct transmittance, shape (3, 20)
    dtdif : ndarray
        Downward diffuse transmittance, shape (3, 20)
    utdir : ndarray
        Upward direct transmittance, shape (3, 20)
    utdif : ndarray
        Upward diffuse transmittance, shape (3, 20)
    sphal : ndarray
        Spherical albedo, shape (3, 20)
    trayl : ndarray
        Rayleigh optical depth, shape (20,)
    traypl : ndarray
        Rayleigh optical depth above plane, shape (20,)
    roatm_fi : ndarray
        Reflectance vs azimuth, shape (3, 20, nfi)
    roluts : ndarray
        Reflectance look-up table, shape (20, mu, 41)
    rolutsq : ndarray
        Q Stokes LUT, shape (20, mu, 41)
    rolutsu : ndarray
        U Stokes LUT, shape (20, mu, 41)
    nfilut : ndarray
        Number of LUT entries per angle, shape (mu,)
    filut : ndarray
        LUT azimuth angles, shape (mu, 41)

    Notes
    -----
    Converted from Fortran DISCOM.f

    This is the main computation loop that prepares all atmospheric
    optical properties at discrete wavelengths for later spectral
    interpolation by interp().
    """
    # Initialize output arrays
    roatm = np.zeros((3, 20))
    rqatm = np.zeros((3, 20))
    ruatm = np.zeros((3, 20))
    dtdir = np.zeros((3, 20))
    dtdif = np.zeros((3, 20))
    utdir = np.zeros((3, 20))
    utdif = np.zeros((3, 20))
    sphal = np.zeros((3, 20))
    trayl = np.zeros(20)
    traypl = np.zeros(20)
    roatm_fi = np.zeros((3, 20, nfi))

    roluts = np.zeros((20, mu, 41))
    rolutsq = np.zeros((20, mu, 41))
    rolutsu = np.zeros((20, mu, 41))
    nfilut = np.zeros(mu, dtype=np.int32)
    filut = np.zeros((mu, 41))

    # Loop over 20 discrete wavelengths
    for l in range(20):
        wl = wldis[l]

        # Skip if outside requested wavelength range
        if wlsup < wldis[0] and l <= 1:
            continue
        if wlinf > wldis[19] and l >= 18:
            continue
        if l < 19 and wldis[l] < wlinf and wldis[l + 1] < wlinf:
            continue
        if l > 0 and wldis[l] > wlsup and wldis[l - 1] > wlsup:
            continue

        # Compute Rayleigh optical depth
        tray = odrayl(wl)

        # Handle plane observation
        if idatmp == 0 or idatmp == 4:
            trayp = tray if idatmp == 4 else 0.0
        else:
            trayp = tray * ftray

        trayl[l] = tray
        traypl[l] = trayp

        # Compute aerosol optical properties at this wavelength
        taer = taer55 * ext[l]
        taerp = taer55p * ext[l]
        piza = ome[l]

        # Update aerosol profile for this wavelength
        if iaer_prof > 0:
            for i in range(num_z):
                taer_z[i] = taer55_z[i] * ext[l]

        # Set up phase function for truncation
        if iaer != 0:
            # Copy phase function for this wavelength
            pha = phasel[l, :nquad]
            if ipol != 0:
                qha = qhasel[l, :nquad]
                uha = uhasel[l, :nquad]

            # Call truncation to get Legendre expansion
            coeff = trunca(pha, qha if ipol != 0 else None,
                          uha if ipol != 0 else None,
                          alphal, betal, gammal, zetal, nquad, ipol)
        else:
            coeff = 0.0

        # Compute truncated optical properties
        tamoy = taer * (1.0 - piza * coeff)
        tamoyp = taerp * (1.0 - piza * coeff)
        pizmoy = piza * (1.0 - coeff) / (1.0 - piza * coeff)

        # Compute atmospheric reflectances
        (rorayl, roaero, romix, rqrayl, rqaero, rqmix,
         rurayl, ruaero, rumix, rorayl_fi_l, romix_fi_l,
         nfilut_l, filut_l, rolut_l, rolutq_l, rolutu_l) = atmref(
            iaer, iaer_prof, tamoy, taer, tray, pizmoy, piza,
            tamoyp, taerp, trayp, palt, phi, xmus, xmuv, phirad,
            nt, mu, np_angles, rm, gb, rp, ipol, xlm1, xlm2, nfi
        )

        # Store reflectances
        roatm[0, l] = rorayl
        roatm[1, l] = romix
        roatm[2, l] = roaero

        rqatm[0, l] = rqrayl
        rqatm[1, l] = rqmix
        rqatm[2, l] = rqaero

        ruatm[0, l] = rurayl
        ruatm[1, l] = rumix
        ruatm[2, l] = ruaero

        # Store azimuthal reflectances
        for ifi in range(nfi):
            roatm_fi[0, l, ifi] = rorayl_fi_l[ifi]
            roatm_fi[1, l, ifi] = romix_fi_l[ifi]

        # Store look-up tables
        nfilut[:] = nfilut_l
        filut[:, :] = filut_l
        for i in range(mu):
            for j in range(41):
                roluts[l, i, j] = rolut_l[i, j]
                rolutsq[l, i, j] = rolutq_l[i, j]
                rolutsu[l, i, j] = rolutu_l[i, j]

        # Compute scattering transmittances
        result = scatra(
            iaer_prof, tamoy, tamoyp, tray, trayp, pizmoy,
            palt, nt, mu, rm, gb, xmus, xmuv
        )

        # Extract transmittances from result dictionary
        # result['rayleigh'], result['total'], result['aerosol']
        ddirtr = result['rayleigh']['ddir']
        ddiftr = result['rayleigh']['ddif']
        udirtr = result['rayleigh']['udir']
        udiftr = result['rayleigh']['udif']
        sphalbr = result['rayleigh']['sphalb']

        ddirtt = result['total']['ddir']
        ddiftt = result['total']['ddif']
        udirtt = result['total']['udir']
        udiftt = result['total']['udif']
        sphalbt = result['total']['sphalb']

        ddirta = result['aerosol']['ddir']
        ddifta = result['aerosol']['ddif']
        udirta = result['aerosol']['udir']
        udifta = result['aerosol']['udif']
        sphalba = result['aerosol']['sphalb']

        # Store transmittances
        dtdir[0, l] = ddirtr
        dtdif[0, l] = ddiftr
        dtdir[1, l] = ddirtt
        dtdif[1, l] = ddiftt
        dtdir[2, l] = ddirta
        dtdif[2, l] = ddifta

        utdir[0, l] = udirtr
        utdif[0, l] = udiftr
        utdir[1, l] = udirtt
        utdif[1, l] = udiftt
        utdir[2, l] = udirta
        utdif[2, l] = udifta

        sphal[0, l] = sphalbr
        sphal[1, l] = sphalbt
        sphal[2, l] = sphalba

    return (roatm, rqatm, ruatm, dtdir, dtdif, utdir, utdif, sphal,
            trayl, traypl, roatm_fi, roluts, rolutsq, rolutsu,
            nfilut, filut)


__all__ = ['discom']
