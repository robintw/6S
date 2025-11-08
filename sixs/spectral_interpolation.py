"""
Spectral interpolation for atmospheric optical properties.

Interpolates all atmospheric scattering and absorption quantities from
discrete wavelength values to a specific target wavelength using log-log
or linear interpolation as appropriate.

Converted from Fortran INTERP.f

Functions:
    interp: Spectral interpolation of atmospheric properties
"""

import numpy as np


def interp(iaer, idatmp, wl, taer55, taer55p, xmud,
           roatm, rqatm, ruatm, ext, ome, gasym, phase, qhase, uhase,
           dtdir, dtdif, utdir, utdif, sphal, wldis, trayl, traypl,
           delta, sigma, roatm_fi, roluts, rolutsq, rolutsu, nfilut, nfi, mu):
    """
    Interpolate atmospheric optical properties to target wavelength.

    Performs spectral interpolation of all atmospheric quantities from
    discrete wavelength values (20 standard wavelengths in wldis) to the
    target wavelength wl. Uses log-log interpolation for positive quantities
    and linear interpolation for quantities that can change sign or be near zero.

    Parameters
    ----------
    iaer : int
        Aerosol model (0=no aerosol)
    idatmp : int
        Observation level (0=ground, other=above ground)
    wl : float
        Target wavelength (microns)
    taer55 : float
        Aerosol optical depth at 550 nm
    taer55p : float
        Aerosol optical depth above plane at 550 nm
    xmud : float
        Cosine of scattering angle
    roatm : ndarray
        Atmospheric reflectances, shape (3, 20)
    rqatm : ndarray
        Q Stokes atmospheric reflectances, shape (3, 20)
    ruatm : ndarray
        U Stokes atmospheric reflectances, shape (3, 20)
    ext : ndarray
        Aerosol extinction, shape (20,)
    ome : ndarray
        Aerosol single scattering albedo, shape (20,)
    gasym : ndarray
        Aerosol asymmetry parameter, shape (20,)
    phase : ndarray
        Aerosol phase function at xmud, shape (20,)
    qhase : ndarray
        Q Stokes phase function, shape (20,)
    uhase : ndarray
        U Stokes phase function, shape (20,)
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
    wldis : ndarray
        Discrete wavelengths (microns), shape (20,)
    trayl : ndarray
        Rayleigh optical depth, shape (20,)
    traypl : ndarray
        Rayleigh optical depth above plane, shape (20,)
    delta : float
        Depolarization factor for Rayleigh scattering
    sigma : float
        Depolarization parameter
    roatm_fi : ndarray
        Reflectance vs azimuth, shape (3, 20, nfi)
    roluts : ndarray
        Look-up table for reflectance, shape (20, mu, 41)
    rolutsq : ndarray
        Look-up table for Q Stokes, shape (20, mu, 41)
    rolutsu : ndarray
        Look-up table for U Stokes, shape (20, mu, 41)
    nfilut : ndarray
        Number of LUT entries per viewing angle, shape (mu,)
    nfi : int
        Number of azimuth angles
    mu : int
        Number of viewing angles

    Returns
    -------
    dict
        Dictionary with interpolated values:
        - romix, rorayl, roaero: Mixed, Rayleigh, aerosol reflectance (I)
        - rqmix, rqrayl, rqaero: Q Stokes reflectances
        - rumix, rurayl, ruaero: U Stokes reflectances
        - phaa, phar: Aerosol and Rayleigh phase functions
        - qhaa, qhar: Q Stokes phase functions
        - uhaa, uhar: U Stokes phase functions
        - tsca: Aerosol scattering optical depth
        - tray, trayp: Rayleigh optical depth (total, above plane)
        - taer, taerp: Aerosol optical depth (total, above plane)
        - dtott, utott: Total transmittance (down, up)
        - dtotr, utotr: Rayleigh transmittance (down, up)
        - dtota, utota: Aerosol transmittance (down, up)
        - astot, asray, asaer: Spherical albedo (total, Rayleigh, aerosol)
        - romix_fi, rorayl_fi: Reflectance vs azimuth, shape (nfi,)
        - rolut, rolutq, rolutu: Interpolated LUTs, shape (mu, 41)

    Notes
    -----
    Converted from Fortran INTERP.f

    Uses log-log interpolation: y = beta * (wl ** alpha)
    where alpha = log(y2/y1) / log(wl2/wl1)
          beta = y1 / (wl1 ** alpha)

    For quantities that can be negative or near zero, uses linear interpolation.
    """
    # Find bracketing wavelengths
    linf = 0
    for ll in range(19):
        if wl > wldis[ll] and wl <= wldis[ll + 1]:
            linf = ll
            break
    if wl > wldis[19]:
        linf = 18

    lsup = linf + 1
    wlinf = wldis[linf]
    coef = np.log(wldis[lsup] / wldis[linf])
    coefl = (wl - wldis[linf]) / (wldis[lsup] - wldis[linf])

    # Depolarization factors for Rayleigh
    depolar1 = 2.0 * (1.0 - delta) / (2.0 + delta)
    depolar2 = 3.0 * delta / (2.0 + delta)

    # Initialize outputs
    result = {
        'romix': 0.0, 'rorayl': 0.0, 'roaero': 0.0,
        'rqmix': 0.0, 'rqrayl': 0.0, 'rqaero': 0.0,
        'rumix': 0.0, 'rurayl': 0.0, 'ruaero': 0.0,
        'phaa': 0.0, 'phar': 0.0,
        'qhaa': 0.0, 'qhar': 0.0,
        'uhaa': 0.0, 'uhar': 0.0,
        'tsca': 0.0, 'tray': 0.0, 'trayp': 0.0,
        'taer': 0.0, 'taerp': 0.0,
        'dtott': 0.0, 'utott': 0.0,
        'dtotr': 0.0, 'utotr': 0.0,
        'dtota': 1.0, 'utota': 1.0,
        'astot': 0.0, 'asray': 0.0, 'asaer': 0.0,
        'romix_fi': np.zeros(nfi),
        'rorayl_fi': np.zeros(nfi),
        'rolut': np.zeros((mu, 41)),
        'rolutq': np.zeros((mu, 41)),
        'rolutu': np.zeros((mu, 41)),
    }

    # === PHASE FUNCTIONS (Stokes I) ===
    if iaer != 0:
        alpha = np.log(phase[lsup] / phase[linf]) / coef
        beta = phase[linf] / (wlinf ** alpha)
        result['phaa'] = beta * (wl ** alpha)

    result['phar'] = depolar1 * 0.75 * (1.0 + xmud * xmud) + depolar2

    # === REFLECTANCES (Stokes I) ===
    if idatmp != 0:
        # Rayleigh reflectance
        alpha = np.log(roatm[0, lsup] / roatm[0, linf]) / coef
        beta = roatm[0, linf] / (wlinf ** alpha)
        result['rorayl'] = beta * (wl ** alpha)

        for ifi in range(nfi):
            alpha = np.log(roatm_fi[0, lsup, ifi] / roatm_fi[0, linf, ifi]) / coef
            beta = roatm_fi[0, linf, ifi] / (wlinf ** alpha)
            result['rorayl_fi'][ifi] = beta * (wl ** alpha)

        # Mixed reflectance
        alpha = np.log(roatm[1, lsup] / roatm[1, linf]) / coef
        beta = roatm[1, linf] / (wlinf ** alpha)
        result['romix'] = beta * (wl ** alpha)

        for ifi in range(nfi):
            alpha = np.log(roatm_fi[1, lsup, ifi] / roatm_fi[1, linf, ifi]) / coef
            beta = roatm_fi[1, linf, ifi] / (wlinf ** alpha)
            result['romix_fi'][ifi] = beta * (wl ** alpha)

        # Aerosol reflectance
        if iaer != 0:
            alpha = np.log(roatm[2, lsup] / roatm[2, linf]) / coef
            beta = roatm[2, linf] / (wlinf ** alpha)
            result['roaero'] = beta * (wl ** alpha)

        # Look-up table interpolation
        for i in range(mu):
            for j in range(nfilut[i]):
                # Stokes I
                alpha = np.log(roluts[lsup, i, j] / roluts[linf, i, j]) / coef
                beta = roluts[linf, i, j] / (wlinf ** alpha)
                result['rolut'][i, j] = beta * (wl ** alpha)

                # Stokes Q (with linear fallback)
                if rolutsq[lsup, i, j] > 0.001 and rolutsq[linf, i, j] > 0.001:
                    alpha = np.log(rolutsq[lsup, i, j] / rolutsq[linf, i, j]) / coef
                    beta = rolutsq[linf, i, j] / (wlinf ** alpha)
                    result['rolutq'][i, j] = beta * (wl ** alpha)
                else:
                    result['rolutq'][i, j] = (rolutsq[linf, i, j] +
                                              (rolutsq[lsup, i, j] - rolutsq[linf, i, j]) * coefl)

                # Stokes U (with linear fallback)
                if rolutsu[lsup, i, j] > 0.001 and rolutsu[linf, i, j] > 0.001:
                    alpha = np.log(rolutsu[lsup, i, j] / rolutsu[linf, i, j]) / coef
                    beta = rolutsu[linf, i, j] / (wlinf ** alpha)
                    result['rolutu'][i, j] = beta * (wl ** alpha)
                else:
                    result['rolutu'][i, j] = (rolutsu[linf, i, j] +
                                              (rolutsu[lsup, i, j] - rolutsu[linf, i, j]) * coefl)

    # === STOKES Q PARAMETERS ===
    if iaer != 0:
        if qhase[lsup] > 0.001 and qhase[linf] > 0.001:
            alpha = np.log(qhase[lsup] / qhase[linf]) / coef
            beta = qhase[linf] / (wlinf ** alpha)
            result['qhaa'] = beta * (wl ** alpha)
        else:
            result['qhaa'] = qhase[linf] + (qhase[lsup] - qhase[linf]) * coefl

    result['qhar'] = depolar1 * 0.75 * (xmud * xmud - 1.0)

    if idatmp != 0:
        # Rayleigh Q
        test1, test2 = abs(rqatm[0, linf]), abs(rqatm[0, lsup])
        test3 = rqatm[0, lsup] * rqatm[0, linf]
        if test1 < 0.001 or test2 < 0.001 or test3 < 0.0:
            result['rqrayl'] = rqatm[0, linf] + (rqatm[0, lsup] - rqatm[0, linf]) * coefl
        else:
            alpha = np.log(rqatm[0, lsup] / rqatm[0, linf]) / coef
            beta = rqatm[0, linf] / (wlinf ** alpha)
            result['rqrayl'] = beta * (wl ** alpha)

        # Mixed Q
        test1, test2 = abs(rqatm[1, linf]), abs(rqatm[1, lsup])
        test3 = rqatm[1, lsup] * rqatm[1, linf]
        if test1 < 0.001 or test2 < 0.001 or test3 < 0.0:
            result['rqmix'] = rqatm[1, linf] + (rqatm[1, lsup] - rqatm[1, linf]) * coefl
        else:
            alpha = np.log(rqatm[1, lsup] / rqatm[1, linf]) / coef
            beta = rqatm[1, linf] / (wlinf ** alpha)
            result['rqmix'] = beta * (wl ** alpha)

        # Aerosol Q
        if iaer != 0:
            test1, test2 = abs(rqatm[2, linf]), abs(rqatm[2, lsup])
            test3 = rqatm[2, lsup] * rqatm[2, linf]
            if test1 < 0.001 or test2 < 0.001 or test3 < 0.0:
                result['rqaero'] = rqatm[2, linf] + (rqatm[2, lsup] - rqatm[2, linf]) * coefl
            else:
                alpha = np.log(rqatm[2, lsup] / rqatm[2, linf]) / coef
                beta = rqatm[2, linf] / (wlinf ** alpha)
                result['rqaero'] = beta * (wl ** alpha)

    # === STOKES U PARAMETERS ===
    if iaer != 0:
        if uhase[lsup] > 0.001 and uhase[linf] > 0.001:
            alpha = np.log(uhase[lsup] / uhase[linf]) / coef
            beta = uhase[linf] / (wlinf ** alpha)
            result['uhaa'] = beta * (wl ** alpha)
        else:
            result['uhaa'] = uhase[linf] + (uhase[lsup] - uhase[linf]) * coefl

    result['uhar'] = depolar1 * 1.5 * xmud

    if idatmp != 0:
        # Rayleigh U
        test1, test2 = abs(ruatm[0, linf]), abs(ruatm[0, lsup])
        test3 = ruatm[0, lsup] * ruatm[0, linf]
        if test1 < 0.001 or test2 < 0.001 or test3 < 0.0:
            result['rurayl'] = ruatm[0, linf] + (ruatm[0, lsup] - ruatm[0, linf]) * coefl
        else:
            alpha = np.log(ruatm[0, lsup] / ruatm[0, linf]) / coef
            beta = ruatm[0, linf] / (wlinf ** alpha)
            result['rurayl'] = beta * (wl ** alpha)

        # Mixed U
        test1, test2 = abs(ruatm[1, linf]), abs(ruatm[1, lsup])
        test3 = ruatm[1, lsup] * ruatm[1, linf]
        if test1 < 0.001 or test2 < 0.001 or test3 < 0.0:
            result['rumix'] = ruatm[1, linf] + (ruatm[1, lsup] - ruatm[1, linf]) * coefl
        else:
            alpha = np.log(ruatm[1, lsup] / ruatm[1, linf]) / coef
            beta = ruatm[1, linf] / (wlinf ** alpha)
            result['rumix'] = beta * (wl ** alpha)

        # Aerosol U
        if iaer != 0:
            test1, test2 = abs(ruatm[2, linf]), abs(ruatm[2, lsup])
            test3 = ruatm[2, lsup] * ruatm[2, linf]
            if test1 < 0.001 or test2 < 0.001 or test3 < 0.0:
                result['ruaero'] = ruatm[2, linf] + (ruatm[2, lsup] - ruatm[2, linf]) * coefl
            else:
                alpha = np.log(ruatm[2, lsup] / ruatm[2, linf]) / coef
                beta = ruatm[2, linf] / (wlinf ** alpha)
                result['ruaero'] = beta * (wl ** alpha)

    # === OPTICAL DEPTHS ===
    # Rayleigh
    alpha = np.log(trayl[lsup] / trayl[linf]) / coef
    beta = trayl[linf] / (wlinf ** alpha)
    result['tray'] = beta * (wl ** alpha)

    if idatmp != 0:
        alpha = np.log(traypl[lsup] / traypl[linf]) / coef
        beta = traypl[linf] / (wlinf ** alpha)
        result['trayp'] = beta * (wl ** alpha)

    # Aerosol
    if iaer != 0:
        alpha = np.log(ext[lsup] * ome[lsup] / (ext[linf] * ome[linf])) / coef
        beta = ext[linf] * ome[linf] / (wlinf ** alpha)
        result['tsca'] = taer55 * beta * (wl ** alpha) / ext[7]  # Normalize by 0.55 µm

        alpha = np.log(ext[lsup] / ext[linf]) / coef
        beta = ext[linf] / (wlinf ** alpha)
        result['taerp'] = taer55p * beta * (wl ** alpha) / ext[7]
        result['taer'] = taer55 * beta * (wl ** alpha) / ext[7]

    # === TRANSMITTANCES ===
    # Rayleigh downward
    drinf = dtdif[0, linf] + dtdir[0, linf]
    drsup = dtdif[0, lsup] + dtdir[0, lsup]
    alpha = np.log(drsup / drinf) / coef
    beta = drinf / (wlinf ** alpha)
    result['dtotr'] = beta * (wl ** alpha)

    # Total downward
    dtinf = dtdif[1, linf] + dtdir[1, linf]
    dtsup = dtdif[1, lsup] + dtdir[1, lsup]
    alpha = np.log((dtsup * drinf) / (dtinf * drsup)) / coef
    beta = (dtinf / drinf) / (wlinf ** alpha)
    dtotc = beta * (wl ** alpha)

    # Aerosol downward
    if iaer != 0:
        dainf = dtdif[2, linf] + dtdir[2, linf]
        dasup = dtdif[2, lsup] + dtdir[2, lsup]
        alpha = np.log(dasup / dainf) / coef
        beta = dainf / (wlinf ** alpha)
        result['dtota'] = beta * (wl ** alpha)

    result['dtott'] = dtotc * result['dtotr']

    # Rayleigh upward
    urinf = utdif[0, linf] + utdir[0, linf]
    ursup = utdif[0, lsup] + utdir[0, lsup]
    alpha = np.log(ursup / urinf) / coef
    beta = urinf / (wlinf ** alpha)
    result['utotr'] = beta * (wl ** alpha)

    # Total upward
    utinf = utdif[1, linf] + utdir[1, linf]
    utsup = utdif[1, lsup] + utdir[1, lsup]
    alpha = np.log((utsup * urinf) / (utinf * ursup)) / coef
    beta = (utinf / urinf) / (wlinf ** alpha)
    utotc = beta * (wl ** alpha)

    # Aerosol upward
    if iaer != 0:
        uainf = utdif[2, linf] + utdir[2, linf]
        uasup = utdif[2, lsup] + utdir[2, lsup]
        alpha = np.log(uasup / uainf) / coef
        beta = uainf / (wlinf ** alpha)
        result['utota'] = beta * (wl ** alpha)

    result['utott'] = utotc * result['utotr']

    # === SPHERICAL ALBEDOS ===
    # Rayleigh
    arinf, arsup = sphal[0, linf], sphal[0, lsup]
    alpha = np.log(arsup / arinf) / coef
    beta = arinf / (wlinf ** alpha)
    result['asray'] = beta * (wl ** alpha)

    # Total
    atinf, atsup = sphal[1, linf], sphal[1, lsup]
    alpha = np.log(atsup / atinf) / coef
    beta = atinf / (wlinf ** alpha)
    result['astot'] = beta * (wl ** alpha)

    # Aerosol
    if iaer != 0:
        aainf, aasup = sphal[2, linf], sphal[2, lsup]
        alpha = np.log(aasup / aainf) / coef
        beta = aainf / (wlinf ** alpha)
        result['asaer'] = beta * (wl ** alpha)

    return result


__all__ = ['interp']
