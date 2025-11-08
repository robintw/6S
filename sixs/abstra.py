"""
Atmospheric gaseous absorption module.

This module computes transmittance for atmospheric gases including water vapor,
ozone, carbon dioxide, oxygen, nitrous oxide, methane, and carbon monoxide.
Uses wavelength-dependent absorption coefficients from spectral band models.

Converted from Fortran ABSTRA.f

Functions:
    abstra: Main atmospheric absorption calculation
"""

import numpy as np


# Spectral region boundaries (cm^-1)
IVLI = np.array([2500, 5060, 7620, 10180, 12740, 15300])

# Ozone absorption coefficients (102 values)
CO3 = np.array([
    4.50e-03, 8.00e-03, 1.07e-02, 1.10e-02, 1.27e-02, 1.71e-02,
    2.00e-02, 2.45e-02, 3.07e-02, 3.84e-02, 4.78e-02, 5.67e-02,
    6.54e-02, 7.62e-02, 9.15e-02, 1.00e-01, 1.09e-01, 1.20e-01,
    1.28e-01, 1.12e-01, 1.11e-01, 1.16e-01, 1.19e-01, 1.13e-01,
    1.03e-01, 9.24e-02, 8.28e-02, 7.57e-02, 7.07e-02, 6.58e-02,
    5.56e-02, 4.77e-02, 4.06e-02, 3.87e-02, 3.82e-02, 2.94e-02,
    2.09e-02, 1.80e-02, 1.91e-02, 1.66e-02, 1.17e-02, 7.70e-03,
    6.10e-03, 8.50e-03, 6.10e-03, 3.70e-03, 3.20e-03, 3.10e-03,
    2.55e-03, 1.98e-03, 1.40e-03, 8.25e-04, 2.50e-04, 0.,
    0., 0., 5.65e-04, 2.04e-03, 7.35e-03, 2.03e-02,
    4.98e-02, 1.18e-01, 2.46e-01, 5.18e-01, 1.02e+00, 1.95e+00,
    3.79e+00, 6.65e+00, 1.24e+01, 2.20e+01, 3.67e+01, 5.95e+01,
    8.50e+01, 1.26e+02, 1.68e+02, 2.06e+02, 2.42e+02, 2.71e+02,
    2.91e+02, 3.02e+02, 3.03e+02, 2.94e+02, 2.77e+02, 2.54e+02,
    2.26e+02, 1.96e+02, 1.68e+02, 1.44e+02, 1.17e+02, 9.75e+01,
    7.65e+01, 6.04e+01, 4.62e+01, 3.46e+01, 2.52e+01, 2.00e+01,
    1.57e+01, 1.20e+01, 1.00e+01, 8.80e+00, 8.30e+00, 8.60e+00
])

# Water vapor continuum absorption coefficients (15 values)
CCH2O = np.array([
    0.00, 0.19, 0.15, 0.12, 0.10, 0.09, 0.10, 0.12,
    0.15, 0.17, 0.20, 0.24, 0.28, 0.33, 0.00
])


def abstra(idatm, wl, xmus, xmuv, uw, uo3, uwus, uo3us,
           idatmp, uwpl, uo3pl, uwusp, uo3usp,
           z, p, t, wh, wo, zpl, ppl, tpl, whpl, wopl):
    """
    Calculate atmospheric gaseous absorption transmittances.

    Computes transmittance for downward, upward, and total paths for
    seven atmospheric gases: water vapor, CO2, O2, O3, N2O, CH4, CO.

    Parameters
    ----------
    idatm : int
        Atmospheric model identifier:
        0 = No atmosphere
        1-7 = Standard atmosphere models
        8 = User-defined with scaling
    wl : float
        Wavelength in microns
    xmus : float
        Cosine of solar zenith angle
    xmuv : float
        Cosine of viewing zenith angle
    uw : float
        Total water vapor content (g/cm²)
    uo3 : float
        Total ozone content (atm-cm)
    uwus : float
        Standard water vapor content
    uo3us : float
        Standard ozone content
    idatmp : int
        Plane atmospheric model (0, 4, 8, or 1-7)
    uwpl : float
        Water vapor content above plane
    uo3pl : float
        Ozone content above plane
    uwusp : float
        Standard water vapor above plane
    uo3usp : float
        Standard ozone above plane
    z : ndarray
        Altitude levels (km), shape (34,)
    p : ndarray
        Pressure levels (mb), shape (34,)
    t : ndarray
        Temperature levels (K), shape (34,)
    wh : ndarray
        Water vapor density (g/m³), shape (34,)
    wo : ndarray
        Ozone density (g/m³), shape (34,)
    zpl : ndarray
        Altitude levels for plane, shape (34,)
    ppl : ndarray
        Pressure levels for plane, shape (34,)
    tpl : ndarray
        Temperature levels for plane, shape (34,)
    whpl : ndarray
        Water vapor density for plane, shape (34,)
    wopl : ndarray
        Ozone density for plane, shape (34,)

    Returns
    -------
    dict
        Dictionary containing transmittances:
        - dtwava, utwava, ttwava: Water vapor (down, up, total)
        - dtozon, utozon, ttozon: Ozone
        - dtdica, utdica, ttdica: CO2
        - dtoxyg, utoxyg, ttoxyg: O2
        - dtniox, utniox, ttniox: N2O
        - dtmeth, utmeth, ttmeth: CH4
        - dtmoca, utmoca, ttmoca: CO

    Notes
    -----
    Converted from Fortran ABSTRA.f

    Uses spectral band models (WAVA, DICA, OXYG, OZON, NIOX, METH, MOCA)
    for wavelength-dependent absorption. Implements Goody band model for
    some gases and lookup tables for others (e.g., ozone).
    """
    accu = 1e-10

    # Initialize all transmittances to 1.0 (no absorption)
    result = {
        'dtwava': 1.0, 'utwava': 1.0, 'ttwava': 1.0,
        'dtcont': 1.0, 'utcont': 1.0, 'ttcont': 1.0,
        'dtozon': 1.0, 'utozon': 1.0, 'ttozon': 1.0,
        'dtdica': 1.0, 'utdica': 1.0, 'ttdica': 1.0,
        'dtoxyg': 1.0, 'utoxyg': 1.0, 'ttoxyg': 1.0,
        'dtniox': 1.0, 'utniox': 1.0, 'ttniox': 1.0,
        'dtmeth': 1.0, 'utmeth': 1.0, 'ttmeth': 1.0,
        'dtmoca': 1.0, 'utmoca': 1.0, 'ttmoca': 1.0,
    }

    # No atmosphere case
    if idatm == 0:
        return result

    # Check for valid zenith angles
    if xmus == 0.0 or xmuv == 0.0:
        raise ValueError("Error on zenithal angle (near 90 deg)")

    # Physical constants
    p0 = 1013.25  # Standard pressure (mb)
    g = 98.1  # Gravity (m/s²)
    t0 = 250.0  # Reference temperature (K)

    # Volumic mass ratios (kg/m³ at STP)
    air = 0.028964 / 0.0224
    roco2 = 0.044 / 0.0224
    rmo2 = 0.032 / 0.0224
    rmo3 = 0.048 / 0.0224
    rmn2o = 0.044 / 0.0224
    rmch4 = 0.016 / 0.0224
    rmco = 0.028 / 0.0224

    # Standard values
    uwus_std = 1.424
    uo3us_std = 0.344

    # Gas scaling ratios
    rat = np.ones(10)
    if idatm == 8:
        rat[0] = uw / uwus_std  # H2O
        rat[1] = 1.0  # CO2
        rat[2] = 1.0  # O2
        rat[3] = uo3 / uo3us_std  # O3
        rat[4] = 1.0  # N2O
        rat[5] = 1.0  # CH4
        rat[6] = 1.0  # CO
        rat[7] = uw / uwus_std  # continuum
        rat[8] = uw / uwus_std
        rat[9] = uw / uwus_std

    # Plane scaling ratios
    ratpl = np.ones(10)
    if idatmp == 8:
        ratpl[0] = uwpl / uwusp
        ratpl[1] = 1.0
        ratpl[2] = 1.0
        ratpl[3] = uo3pl / uo3usp
        ratpl[4] = 1.0
        ratpl[5] = 1.0
        ratpl[6] = 1.0
        ratpl[7] = uwpl / uwusp
        ratpl[8] = uwpl / uwusp
        ratpl[9] = uwpl / uwusp

    # Wavenumber (cm^-1)
    v = 1.0e4 / wl
    iv = int(v / 5.0) * 5

    # Determine spectral region
    id = ((iv - 2500) // 10) // 256 + 1
    if id > 6:
        return result

    inu = (iv - IVLI[id - 1]) // 10 + 1 if id <= 6 else 0

    # Storage for transmittances by gas
    tnu = np.ones((10, 3))

    # Loop over 7 gases
    for idgaz in range(1, 8):
        # Get absorption coefficients for this gas and spectral region
        a = _get_absorption_coefficients(idgaz, id, inu, v, iv)

        if a is None:
            continue

        # Calculate mixing ratios for each layer
        rm = np.zeros(34)
        r2 = np.zeros(34)
        r3 = np.zeros(34)
        tp = np.zeros(34)

        for k in range(33):
            roair = air * 273.16 * p[k] / (1013.25 * t[k])
            tp[k] = (t[k] + t[k + 1]) / 2.0
            te = tp[k] - t0
            te2 = te * te
            phi = np.exp(a[2] * te + a[3] * te2)
            psi = np.exp(a[4] * te + a[5] * te2)

            # Gas-specific mixing ratios
            if idgaz == 1:  # H2O
                rm[k] = wh[k] / (roair * 1000.0)
            elif idgaz == 2:  # CO2
                rm[k] = 3.3e-04 * roco2 / air
            elif idgaz == 3:  # O2
                rm[k] = 0.20947 * rmo2 / air
            elif idgaz == 4:  # O3
                rm[k] = wo[k] / (roair * 1000.0)
            elif idgaz == 5:  # N2O
                rm[k] = 310.0e-09 * rmn2o / air
            elif idgaz == 6:  # CH4
                rm[k] = 1.72e-06 * rmch4 / air
            elif idgaz == 7:  # CO
                rm[k] = 1.0e-09 * rmco / air

            r2[k] = rm[k] * phi
            r3[k] = rm[k] * psi

        # Pressure-weighted integration
        uu = 0.0
        u = 0.0
        up = 0.0

        for k in range(1, 33):
            ds = (p[k - 1] - p[k]) / p[0]
            ds2 = (p[k - 1]**2 - p[k]**2) / (2.0 * p[0] * p0)
            uu += ((rm[k] + rm[k - 1]) / 2.0) * ds * rat[idgaz - 1]
            u += ((r2[k] + r2[k - 1]) / 2.0) * ds * rat[idgaz - 1]
            up += ((r3[k] + r3[k - 1]) / 2.0) * ds2 * rat[idgaz - 1]

        uu = uu * p[0] * 100.0 / g
        u = u * p[0] * 100.0 / g
        up = up * p[0] * 100.0 / g

        # Convert to appropriate units
        if idgaz == 4:  # O3
            uu = 1000.0 * uu / rmo3
        elif idgaz == 2:  # CO2
            uu = 1000.0 * uu / roco2
        elif idgaz == 5:  # N2O
            uu = 1000.0 * uu / rmn2o
        elif idgaz == 6:  # CH4
            uu = 1000.0 * uu / rmch4
        elif idgaz == 7:  # CO
            uu = 1000.0 * uu / rmco

        # Plane atmosphere calculations
        if idatmp == 0 or idatmp == 4:
            uupl = uu
            upl = u
            uppl = up
        else:
            rmpl = np.zeros(34)
            r2pl = np.zeros(34)
            r3pl = np.zeros(34)

            for k in range(33):
                roair = air * 273.16 * ppl[k] / (1013.25 * tpl[k])
                tp[k] = (tpl[k] + tpl[k + 1]) / 2.0
                te = tp[k] - t0
                te2 = te * te
                phi = np.exp(a[2] * te + a[3] * te2)
                psi = np.exp(a[4] * te + a[5] * te2)

                if idgaz == 1:
                    rmpl[k] = whpl[k] / (roair * 1000.0)
                elif idgaz == 2:
                    rmpl[k] = 3.3e-04 * roco2 / air
                elif idgaz == 3:
                    rmpl[k] = 0.20947 * rmo2 / air
                elif idgaz == 4:
                    rmpl[k] = wopl[k] / (roair * 1000.0)
                elif idgaz == 5:
                    rmpl[k] = 310.0e-09 * rmn2o / air
                elif idgaz == 6:
                    rmpl[k] = 1.72e-06 * rmch4 / air
                elif idgaz == 7:
                    rmpl[k] = 1.0e-09 * rmco / air

                r2pl[k] = rmpl[k] * phi
                r3pl[k] = rmpl[k] * psi

            uupl = 0.0
            upl = 0.0
            uppl = 0.0

            for k in range(1, 33):
                ds = (ppl[k - 1] - ppl[k]) / ppl[0]
                ds2 = (ppl[k - 1]**2 - ppl[k]**2) / (2.0 * ppl[0] * p0)
                uupl += ((rmpl[k] + rmpl[k - 1]) / 2.0) * ds * ratpl[idgaz - 1]
                upl += ((r2pl[k] + r2pl[k - 1]) / 2.0) * ds * ratpl[idgaz - 1]
                uppl += ((r3pl[k] + r3pl[k - 1]) / 2.0) * ds2 * ratpl[idgaz - 1]

            uupl = uupl * ppl[0] * 100.0 / g
            upl = upl * ppl[0] * 100.0 / g
            uppl = uppl * ppl[0] * 100.0 / g

            if idgaz == 4:
                uupl = 1000.0 * uupl / rmo3
            elif idgaz == 2:
                uupl = 1000.0 * uupl / roco2
            elif idgaz == 5:
                uupl = 1000.0 * uupl / rmn2o
            elif idgaz == 6:
                uupl = 1000.0 * uupl / rmch4
            elif idgaz == 7:
                uupl = 1000.0 * uupl / rmco

        # Path amounts
        uud = uu / xmus  # Downward
        uuu = uupl / xmuv  # Upward
        uut = uu / xmus + uupl / xmuv  # Total

        # Calculate transmittances
        if idgaz == 1 or idgaz == 4:
            # Special handling for H2O and O3
            if idgaz == 1:
                # Water vapor continuum
                if 2350 <= iv <= 3000:
                    xi = (v - 2350.0) / 50.0 + 1.0
                    nh = int(xi + 1.001)
                    xh = xi - float(nh)
                    ah2o = CCH2O[nh - 1] + xh * (CCH2O[nh - 1] - CCH2O[nh - 2])
                    result['dtcont'] = np.exp(-ah2o * uud)
                    result['utcont'] = np.exp(-ah2o * uuu)
                    result['ttcont'] = np.exp(-ah2o * uut)

            if idgaz == 4:
                # Ozone lookup table
                if iv >= 13000:
                    if iv <= 23400:
                        xi = (v - 13000.0) / 200.0 + 1.0
                    elif iv >= 27500:
                        xi = (v - 27500.0) / 500.0 + 57.0
                    else:
                        tnu[idgaz - 1, 0] = 1.0
                        tnu[idgaz - 1, 1] = 1.0
                        tnu[idgaz - 1, 2] = 1.0
                        continue

                    n = int(xi + 1.001)
                    xd = xi - float(n)
                    ako3 = CO3[n - 1] + xd * (CO3[n - 1] - CO3[n - 2])

                    test1 = min(ako3 * uud, 86.0)
                    test2 = min(ako3 * uuu, 86.0)
                    test3 = min(ako3 * uut, 86.0)

                    tnu[idgaz - 1, 0] = np.exp(-test1)
                    tnu[idgaz - 1, 1] = np.exp(-test2)
                    tnu[idgaz - 1, 2] = np.exp(-test3)
                    continue

        # Standard gases - use Goody model
        if (idgaz == 2 and iv > 9620) or (idgaz == 3 and iv > 15920) or \
           (idgaz == 4 and iv > 3020):
            tnu[idgaz - 1, 0] = 1.0
            tnu[idgaz - 1, 1] = 1.0
            tnu[idgaz - 1, 2] = 1.0
            continue

        # Downward path
        ud = u / xmus
        upd = up / xmus
        udt = ud if not (ud == 0 and upd == 0) else 1.0
        atest = a[1] if not (a[1] == 0 and a[0] == 0) else 1.0
        updt = upd if not (ud == 0 and upd == 0) else 1.0

        tn = a[1] * upd / (2 * udt)
        tt = 1 + 4 * (a[0] / atest) * ((ud * ud) / updt)
        y = -tn * (np.sqrt(tt) - 1)

        if idgaz == 1:
            y = -a[0] * ud / np.sqrt(1 + (a[0] / atest) * (ud * ud / updt))

        tnu[idgaz - 1, 0] = np.exp(y)

        # Upward path
        udp = upl / xmuv
        updp = uppl / xmuv
        udtp = udp if not (udp == 0 and updp == 0) else 1.0
        updtp = updp if not (udp == 0 and updp == 0) else 1.0

        tn = a[1] * updp / (2 * udtp)
        tt = 1 + 4 * (a[0] / atest) * ((udp * udp) / updtp)
        y = -tn * (np.sqrt(tt) - 1)

        if idgaz == 1:
            y = -a[0] * udp / np.sqrt(1 + (a[0] / atest) * (udp * udp / updtp))

        tnu[idgaz - 1, 1] = np.exp(y)

        # Total path
        ut = u / xmus + upl / xmuv
        upt = up / xmus + uppl / xmuv
        utt = ut if not (ut == 0 and upt == 0) else 1.0
        uptt = upt if not (ut == 0 and upt == 0) else 1.0

        tn = a[1] * upt / (2 * utt)
        tt = 1 + 4 * (a[0] / atest) * ((ut * ut) / uptt)
        y = -tn * (np.sqrt(tt) - 1)

        if idgaz == 1:
            y = -a[0] * ut / np.sqrt(1 + (a[0] / atest) * (ut * ut / uptt))

        tnu[idgaz - 1, 2] = np.exp(y)

    # Apply accuracy threshold and store results
    ptest = tnu[0, 0] * result['dtcont']
    result['dtwava'] = ptest if ptest > accu else 0.0

    ptest = tnu[0, 1] * result['utcont']
    result['utwava'] = ptest if ptest > accu else 0.0

    ptest = tnu[0, 2] * result['ttcont']
    result['ttwava'] = ptest if ptest > accu else 0.0

    result['dtdica'] = tnu[1, 0]
    result['utdica'] = tnu[1, 1]
    result['ttdica'] = tnu[1, 2]

    result['dtoxyg'] = tnu[2, 0]
    result['utoxyg'] = tnu[2, 1]
    result['ttoxyg'] = tnu[2, 2]

    result['dtozon'] = tnu[3, 0]
    result['utozon'] = tnu[3, 1]
    result['ttozon'] = tnu[3, 2]

    result['dtniox'] = tnu[4, 0]
    result['utniox'] = tnu[4, 1]
    result['ttniox'] = tnu[4, 2]

    result['dtmeth'] = tnu[5, 0]
    result['utmeth'] = tnu[5, 1]
    result['ttmeth'] = tnu[5, 2]

    result['dtmoca'] = tnu[6, 0]
    result['utmoca'] = tnu[6, 1]
    result['ttmoca'] = tnu[6, 2]

    # For ground-level observation (idatmp == 0), upward transmittance = 1
    if idatmp == 0:
        result['ttwava'] = result['dtwava']
        result['utwava'] = 1.0
        result['ttdica'] = result['dtdica']
        result['utdica'] = 1.0
        result['ttoxyg'] = result['dtoxyg']
        result['utoxyg'] = 1.0
        result['ttozon'] = result['dtozon']
        result['utozon'] = 1.0
        result['ttniox'] = result['dtniox']
        result['utniox'] = 1.0
        result['ttmeth'] = result['dtmeth']
        result['utmeth'] = 1.0
        result['ttmoca'] = result['dtmoca']
        result['utmoca'] = 1.0

    return result


def _get_absorption_coefficients(idgaz, id, inu, v, iv):
    """
    Get absorption coefficients for a gas and spectral region.

    Parameters
    ----------
    idgaz : int
        Gas identifier (1=H2O, 2=CO2, 3=O2, 4=O3, 5=N2O, 6=CH4, 7=CO)
    id : int
        Spectral region (1-6)
    inu : int
        Index within spectral region
    v : float
        Wavenumber (cm^-1)
    iv : int
        Wavenumber rounded to nearest 5 cm^-1

    Returns
    -------
    ndarray or None
        Absorption coefficients [a0, a1, a2, a3, a4, a5, a6, a7]
        Returns None if no absorption in this band
    """
    # Import the appropriate WAVA module
    # These will be implemented separately as part of Wave 3.4

    # Placeholder: Return zeros for now until WAVA modules are converted
    # Each spectral region (id) has different gas coverage:
    # id=1 (2500-5060): H2O, CO2, O3, N2O, CH4, CO
    # id=2 (5060-7620): H2O, CO2, N2O, CH4, CO
    # id=3 (7620-10180): H2O, CO2, O2, N2O, CH4, CO
    # id=4 (10180-12740): H2O, O2, N2O, CH4, CO
    # id=5 (12740-15300): H2O, O2, N2O, CH4, CO
    # id=6 (15300+): H2O, O2, N2O, CH4, CO

    # Gases not present in certain bands
    if id == 1:
        if idgaz == 3:  # O2 not in band 1
            return None
    elif id == 2:
        if idgaz in [3, 4]:  # O2, O3 not in band 2
            return None
    elif id in [3, 4, 5, 6]:
        if idgaz == 4:  # O3 not in bands 3-6
            return None
        if idgaz == 2 and id >= 4:  # CO2 not in bands 4-6
            return None

    # For now, return zeros until WAVA modules are implemented
    # This will be replaced with calls to wava1-6, dica1-3, oxyg3-6, etc.
    return np.zeros(8)


__all__ = ['abstra', 'CO3', 'CCH2O', 'IVLI']
