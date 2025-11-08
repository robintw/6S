"""
Atmospheric profile utilities for 6S.

This module provides functions for:
- Vertical layering of aerosol profiles
- Plane altitude adjustments to atmospheric profiles
- Ground target altitude adjustments

Functions:
    aero_prof: Divide atmosphere into equal optical thickness layers
    presplane: Adjust atmospheric profile for plane altitude
    pressure: Adjust atmospheric profile for ground target altitude
"""

import numpy as np
from typing import Tuple


def aero_prof(
    ta: float,
    piz: float,
    tr: float,
    hr: float,
    nt: int,
    xmus: float,
    alt_z: np.ndarray,
    taer_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide atmosphere into layers with equal total optical thickness.

    Creates a vertical discretization where each layer has the same combined
    (molecular + aerosol) optical thickness for use in radiative transfer.

    Args:
        ta: Total aerosol optical thickness
        piz: Aerosol single scattering albedo
        tr: Total Rayleigh optical thickness
        hr: Rayleigh scale height (km)
        nt: Number of layers to create
        xmus: Cosine of solar zenith angle
        alt_z: Altitude boundaries for aerosol profile (km)
        taer_z: Aerosol optical thickness at each altitude boundary

    Returns:
        Tuple of (h, ch, ydel, xdel, altc):
            h: Cumulative optical thickness at layer boundaries
            ch: Chapman function values (exp(-h/xmus)/2)
            ydel: Molecular fraction in each layer
            xdel: Aerosol fraction in each layer
            altc: Altitude at center of each layer (km)
    """
    # Ensure arrays are mutable copies
    alt_z = np.array(alt_z, dtype=np.float64)
    taer_z = np.array(taer_z, dtype=np.float64)
    num_z = len(alt_z) - 1  # Number of aerosol layers

    # If maximum aerosol height < 300 km, add layer above with zero aerosol
    if alt_z[0] < 300:
        taer_z = np.concatenate([[0.0], taer_z])
        alt_z = np.concatenate([[alt_z[0]], alt_z])
        num_z += 1

    # Set top of atmosphere to 300 km
    alt_z[0] = 300.0
    ssa_aer = piz

    # Target optical thickness per layer
    dtau_OS = (tr + ta) / nt

    # Initialize arrays
    h = np.zeros(nt + 1)
    ch = np.zeros(nt + 1)
    ydel = np.zeros(nt + 1)
    xdel = np.zeros(nt + 1)
    altc = np.zeros(nt + 1)

    # Initial values
    h[0] = 0.0
    altc[0] = 300.0
    ch[0] = 0.5
    ydel[0] = 1.0
    xdel[0] = 0.0

    # Layer construction
    i = 0
    dz = 0.0001  # 0.1 m step size
    z_up = alt_z[0]
    j = 1
    n = 1
    dtau_aer = 0.0

    while True:
        i += 1
        z = alt_z[0] - dz * i

        # Rayleigh optical thickness increment
        dtau_ray = tr * (np.exp(-z / hr) - np.exp(-z_up / hr))

        # Aerosol optical thickness increment
        if n < len(alt_z):
            dtau_aer += taer_z[n] * dz / (alt_z[n - 1] - alt_z[n])
            if z < alt_z[n]:
                n += 1

        # Total optical thickness
        dtau = dtau_ray + dtau_aer

        # Check if layer is complete
        if dtau >= dtau_OS:
            altc[j] = z
            h[j] = h[j - 1] + dtau
            ch[j] = np.exp(-h[j] / xmus) / 2
            xdel[j] = dtau_aer * ssa_aer / dtau  # Aerosol portion
            ydel[j] = dtau_ray / dtau  # Molecular portion
            j += 1
            z_up = z
            dtau_aer = 0.0

            if j > nt:
                break

        if z <= 0:
            break

    # Set final layer to ground level
    altc[nt] = 0
    h[nt] = tr + ta
    ch[nt] = np.exp(-h[nt] / xmus) / 2
    if dtau > 0:
        xdel[nt] = dtau_aer * ssa_aer / dtau
        ydel[nt] = dtau_ray / dtau

    return h, ch, ydel, xdel, altc


def presplane(
    uw: float,
    uo3: float,
    xpp: float,
    z: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    wh: np.ndarray,
    wo: np.ndarray,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Adjust atmospheric profile for plane altitude observation.

    Creates a modified atmospheric profile with the plane altitude as a new
    level, and computes updated water vapor and ozone content.

    Args:
        uw: Initial water vapor content (not used, will be recalculated)
        uo3: Initial ozone content (not used, will be recalculated)
        xpp: Plane altitude above ground (km)
        z: Altitude levels (km)
        p: Pressure levels (mb)
        t: Temperature levels (K)
        wh: Water vapor concentration (g/m³)
        wo: Ozone concentration (g/m³)

    Returns:
        Tuple of (uw_new, uo3_new, ftray, zpl, ppl, tpl, whpl, wopl):
            uw_new: Updated water vapor content (g/cm²)
            uo3_new: Updated ozone content (cm-atm)
            ftray: Rayleigh optical thickness correction factor
            zpl: Modified altitude profile
            ppl: Modified pressure profile
            tpl: Modified temperature profile
            whpl: Modified water vapor profile
            wopl: Modified ozone profile
    """
    # Make copies of input arrays
    z = np.array(z, dtype=np.float64)
    p = np.array(p, dtype=np.float64)
    t = np.array(t, dtype=np.float64)
    wh = np.array(wh, dtype=np.float64)
    wo = np.array(wo, dtype=np.float64)

    # Adjust altitude to absolute altitude
    xpp = xpp + z[0]
    if xpp >= 100.0:
        xpp = 1000.0

    # Find bounding levels for interpolation
    i = 0
    while z[i] <= xpp:
        i += 1
    isup = i
    iinf = i - 1

    # Log-linear interpolation for pressure
    xa = (z[isup] - z[iinf]) / np.log(p[isup] / p[iinf])
    xb = z[isup] - xa * np.log(p[isup])
    ps = np.exp((xpp - xb) / xa)

    # Linear interpolation for temperature, water vapor, ozone
    xalt = xpp
    xtemp = (t[isup] - t[iinf]) / (z[isup] - z[iinf])
    xtemp = xtemp * (xalt - z[iinf]) + t[iinf]

    xwo = (wo[isup] - wo[iinf]) / (z[isup] - z[iinf])
    xwo = xwo * (xalt - z[iinf]) + wo[iinf]

    xwh = (wh[isup] - wh[iinf]) / (z[isup] - z[iinf])
    xwh = xwh * (xalt - z[iinf]) + wh[iinf]

    # Create modified profile
    zpl = np.zeros(34)
    ppl = np.zeros(34)
    tpl = np.zeros(34)
    whpl = np.zeros(34)
    wopl = np.zeros(34)

    # Copy levels below plane
    for i in range(iinf + 1):
        zpl[i] = z[i]
        ppl[i] = p[i]
        tpl[i] = t[i]
        whpl[i] = wh[i]
        wopl[i] = wo[i]

    # Insert plane level
    zpl[iinf + 1] = xalt
    ppl[iinf + 1] = ps
    tpl[iinf + 1] = xtemp
    whpl[iinf + 1] = xwh
    wopl[iinf + 1] = xwo

    # Fill remaining levels with plane values
    for i in range(iinf + 2, 34):
        zpl[i] = zpl[iinf + 1]
        ppl[i] = ppl[iinf + 1]
        tpl[i] = tpl[iinf + 1]
        whpl[i] = whpl[iinf + 1]
        wopl[i] = wopl[iinf + 1]

    # Compute modified H2O and O3 integrated content
    uw_new = 0.0
    uo3_new = 0.0
    g = 98.1  # Gravitational acceleration (m/s²)
    air = 0.028964 / 0.0224  # Air molar mass / molar volume
    ro3 = 0.048 / 0.0224  # Ozone molar mass / molar volume

    # Compute Rayleigh optical thickness correction factor
    rt = 0.0
    rp = 0.0
    rmwh = np.zeros(34)
    rmo3 = np.zeros(34)

    for k in range(33):
        roair = air * 273.16 * ppl[k] / (1013.25 * tpl[k])
        rmwh[k] = wh[k] / (roair * 1000.0)
        rmo3[k] = wo[k] / (roair * 1000.0)
        rt += (p[k + 1] / t[k + 1] + p[k] / t[k]) * (z[k + 1] - z[k])
        rp += (ppl[k + 1] / tpl[k + 1] + ppl[k] / tpl[k]) * (zpl[k + 1] - zpl[k])

    ftray = rp / rt

    # Integrate water vapor and ozone
    for k in range(1, 33):
        ds = (ppl[k - 1] - ppl[k]) / ppl[0]
        uw_new += ((rmwh[k] + rmwh[k - 1]) / 2.0) * ds
        uo3_new += ((rmo3[k] + rmo3[k - 1]) / 2.0) * ds

    uw_new = uw_new * ppl[0] * 100.0 / g
    uo3_new = uo3_new * ppl[0] * 100.0 / g
    uo3_new = 1000.0 * uo3_new / ro3

    return uw_new, uo3_new, ftray, zpl, ppl, tpl, whpl, wopl


def pressure(
    uw: float,
    uo3: float,
    xps: float,
    z: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    wh: np.ndarray,
    wo: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Adjust atmospheric profile for ground target altitude.

    Modifies the atmospheric profile to place the target at a specified
    altitude (typically negative for below sea level or positive for elevated
    targets), and computes updated water vapor and ozone content.

    Args:
        uw: Initial water vapor content (not used, will be recalculated)
        uo3: Initial ozone content (not used, will be recalculated)
        xps: Target altitude (negative value) (km)
        z: Altitude levels (km)
        p: Pressure levels (mb)
        t: Temperature levels (K)
        wh: Water vapor concentration (g/m³)
        wo: Ozone concentration (g/m³)

    Returns:
        Tuple of (uw_new, uo3_new, z_mod, p_mod, t_mod, wh_mod, wo_mod):
            uw_new: Updated water vapor content (g/cm²)
            uo3_new: Updated ozone content (cm-atm)
            z_mod: Modified altitude profile
            p_mod: Modified pressure profile
            t_mod: Modified temperature profile
            wh_mod: Modified water vapor profile
            wo_mod: Modified ozone profile
    """
    # Make copies of input arrays
    z = np.array(z, dtype=np.float64)
    p = np.array(p, dtype=np.float64)
    t = np.array(t, dtype=np.float64)
    wh = np.array(wh, dtype=np.float64)
    wo = np.array(wo, dtype=np.float64)

    # Convert to positive altitude
    xps = -xps
    if xps >= 100.0:
        xps = 99.99

    # Find bounding levels for interpolation
    i = 0
    while z[i] <= xps:
        i += 1
    isup = i
    iinf = i - 1

    # Log-linear interpolation for pressure
    xa = (z[isup] - z[iinf]) / np.log(p[isup] / p[iinf])
    xb = z[isup] - xa * np.log(p[isup])
    ps = np.exp((xps - xb) / xa)

    # Linear interpolation for temperature, water vapor, ozone
    xalt = xps
    xtemp = (t[isup] - t[iinf]) / (z[isup] - z[iinf])
    xtemp = xtemp * (xalt - z[iinf]) + t[iinf]

    xwo = (wo[isup] - wo[iinf]) / (z[isup] - z[iinf])
    xwo = xwo * (xalt - z[iinf]) + wo[iinf]

    xwh = (wh[isup] - wh[iinf]) / (z[isup] - z[iinf])
    xwh = xwh * (xalt - z[iinf]) + wh[iinf]

    # Update atmospheric profile
    # First level: target altitude
    z[0] = xalt
    p[0] = ps
    t[0] = xtemp
    wh[0] = xwh
    wo[0] = xwo

    # Shift remaining levels
    for i in range(1, 33 - iinf + 2):
        if i + iinf - 1 < len(z):
            z[i] = z[i + iinf - 1]
            p[i] = p[i + iinf - 1]
            t[i] = t[i + iinf - 1]
            wh[i] = wh[i + iinf - 1]
            wo[i] = wo[i + iinf - 1]

    # Fill in remaining levels with linear interpolation
    l = 33 - iinf + 1
    if l < 34:
        for i in range(l + 1, 34):
            frac = (i - l) / (34 - l)
            z[i] = (z[33] - z[l]) * frac + z[l]
            p[i] = (p[33] - p[l]) * frac + p[l]
            t[i] = (t[33] - t[l]) * frac + t[l]
            wh[i] = (wh[33] - wh[l]) * frac + wh[l]
            wo[i] = (wo[33] - wo[l]) * frac + wo[l]

    # Compute modified H2O and O3 integrated content
    uw_new = 0.0
    uo3_new = 0.0
    g = 98.1  # Gravitational acceleration (m/s²)
    air = 0.028964 / 0.0224  # Air molar mass / molar volume
    ro3 = 0.048 / 0.0224  # Ozone molar mass / molar volume

    rmwh = np.zeros(34)
    rmo3 = np.zeros(34)

    for k in range(33):
        roair = air * 273.16 * p[k] / (1013.25 * t[k])
        rmwh[k] = wh[k] / (roair * 1000.0)
        rmo3[k] = wo[k] / (roair * 1000.0)

    # Integrate water vapor and ozone
    for k in range(1, 33):
        ds = (p[k - 1] - p[k]) / p[0]
        uw_new += ((rmwh[k] + rmwh[k - 1]) / 2.0) * ds
        uo3_new += ((rmo3[k] + rmo3[k - 1]) / 2.0) * ds

    uw_new = uw_new * p[0] * 100.0 / g
    uo3_new = uo3_new * p[0] * 100.0 / g
    uo3_new = 1000.0 * uo3_new / ro3

    return uw_new, uo3_new, z, p, t, wh, wo
