"""
Mie scattering calculations for aerosols.

This module implements Mie theory for computing scattering and extinction
efficiency factors, phase functions, and polarization for spherical aerosol particles.

Converted from Fortran MIE.f

Functions:
    mie: Main function for aerosol Mie scattering
    exscphase: Mie scattering efficiency and phase function calculation
"""

import numpy as np
from numba import jit


# Maximum series expansion terms (nser in Fortran)
NSER_MAX = 10000000


def mie(iaer, wldis, rmax, rmin, icp, rn, ri, x1, x2, x3, cij,
        irsunph, rsunph, nrsunph, nquad, ipol=0):
    """
    Compute Mie scattering for aerosol size distributions.

    Integrates Mie theory over particle size distributions to compute
    wavelength-dependent extinction, scattering, asymmetry parameter,
    and phase functions for aerosol mixtures.

    Parameters
    ----------
    iaer : int
        Aerosol model type (8, 9, 10, or 11):
        8 = Log-normal distribution
        9 = Modified Gamma distribution
        10 = Junge power-law distribution
        11 = Sun photometer size distribution
    wldis : array_like
        Wavelengths in microns, shape (20,)
    rmax : float
        Maximum particle radius (microns)
    rmin : float
        Minimum particle radius (microns)
    icp : int
        Number of particle components (1-4)
    rn : ndarray
        Real refractive index, shape (20, 4) [wavelength, component]
    ri : ndarray
        Imaginary refractive index, shape (20, 4)
    x1 : array_like
        Size distribution parameter 1, shape (4,)
    x2 : array_like
        Size distribution parameter 2, shape (4,)
    x3 : array_like
        Size distribution parameter 3, shape (4,)
    cij : array_like
        Volume mixing ratios for components, shape (4,)
    irsunph : int
        Number of sun photometer size bins (for iaer=11)
    rsunph : array_like
        Sun photometer radii, shape (50,)
    nrsunph : array_like
        Sun photometer size distribution dV/dlog(r), shape (50,)
    nquad : int
        Number of quadrature angles for phase function
    ipol : int, optional
        Polarization flag (0=no polarization, 1=compute polarization)

    Returns
    -------
    ex : ndarray
        Extinction coefficient, shape (4, 20) [component, wavelength]
    sc : ndarray
        Scattering coefficient, shape (4, 20)
    asy : ndarray
        Asymmetry parameter, shape (4, 20)
    ph : ndarray
        Phase function (intensity), shape (20, nquad)
    qh : ndarray
        Q Stokes parameter phase function, shape (20, nquad)
    uh : ndarray
        U Stokes parameter phase function, shape (20, nquad)

    Notes
    -----
    Converted from Fortran MIE.f

    The size distribution functions:
    - iaer=8: Log-normal: n(r) ∝ exp(-(log(r/r0)/σ)²)
    - iaer=9: Modified Gamma: n(r) ∝ r^α * exp(-β*r^γ)
    - iaer=10: Junge power-law: n(r) ∝ r^(-α)
    - iaer=11: Sun photometer: tabulated dV/dlog(r)
    """
    from sixs.gauss import gauss

    pi = np.pi
    rlogpas = 0.011  # Logarithmic step in radius
    nbmu = nquad

    # Initialize outputs
    ex = np.zeros((4, 20))
    sc = np.zeros((4, 20))
    asy = np.zeros((4, 20))
    ph = np.zeros((20, nquad))
    qh = np.zeros((20, nquad))
    uh = np.zeros((20, nquad))

    # Internal arrays
    np_particles = np.zeros(4)  # Relative number of particles
    ext = np.zeros((20, 4))
    sca = np.zeros((20, 4))
    p1 = np.zeros((20, 4, nquad))
    q1 = np.zeros((20, 4, nquad))
    u1 = np.zeros((20, 4, nquad))
    p11 = np.zeros(nquad)
    q11 = np.zeros(nquad)
    u11 = np.zeros(nquad)
    vi = np.zeros(4)  # Volume for each component

    # Set up Gauss quadrature points for phase function
    cgaus_S = np.zeros(nquad)
    pdgs_S = np.zeros(nquad)
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

    # Loop over particle components
    for i in range(icp):
        r = rmin
        dr = r * (10**rlogpas - 1.0)

        # Loop over particle radii
        while r < rmax:
            # Compute size distribution nr (dn/dr or dV/dlogr)
            # Different formulas based on aerosol type
            if iaer == 8:
                # Log-normal distribution
                nr = np.log10(r / x1[i]) ** 2
                nr = nr / (np.log10(x2[i]) ** 2)
                nr = np.exp(-nr / 2.0)
                nr = nr / np.sqrt(2.0 * pi) / np.log(10.0) / r / np.log10(x2[i])

            elif iaer == 9:
                # Modified Gamma distribution
                r0 = 1.0
                arg = -x2[i] * ((r / r0) ** x3[i])
                if arg > -300.0:
                    nr = ((r / r0) ** x1[i]) * np.exp(arg)
                else:
                    nr = 0.0

            elif iaer == 10:
                # Junge power-law
                r0 = 0.1
                nr = r0 ** (-x1[i])
                if r > r0:
                    nr = r ** (-x1[i])

            elif iaer == 11:
                # Sun photometer distribution
                nr = 0.0
                for j in range(1, irsunph):
                    if r - rsunph[j] < 0.000001:
                        nr = (r - rsunph[j - 1]) / (rsunph[j] - rsunph[j - 1])
                        nr = nrsunph[j - 1] + nr * (nrsunph[j] - nrsunph[j - 1])
                        break
                # Convert dV/dlogr to dn/dr
                nr = nr * 3.0 / (pi * r * r * r * r * 4.0)

            else:
                raise ValueError(f"Invalid aerosol type iaer={iaer}, must be 8-11")

            xndpr2 = nr * dr * pi * (r ** 2)
            np_particles[i] += nr * dr

            # Loop over wavelengths
            for l in range(20):
                alpha = 2.0 * pi * r / wldis[l]
                # Call Mie scattering calculation
                Qext, Qsca, p11_tmp, q11_tmp, u11_tmp = exscphase(
                    alpha, rn[l, i], ri[l, i], ipol, cgaus_S, pdgs_S
                )

                ext[l, i] += xndpr2 * Qext
                sca[l, i] += xndpr2 * Qsca

                # Phase function for each type of particle
                for k in range(nbmu):
                    p1[l, i, k] += 4.0 * p11_tmp[k] * xndpr2
                    if ipol != 0:
                        q1[l, i, k] += 4.0 * q11_tmp[k] * xndpr2
                        u1[l, i, k] += 4.0 * u11_tmp[k] * xndpr2

            vi[i] += r * r * r * nr * dr

            # Increment radius
            r = r + dr
            dr = r * (10**rlogpas - 1.0)

    # Mix different types of particles
    # Normalize volume fractions
    sigm = 0.0
    for i in range(icp):
        vi[i] = 4 * pi * vi[i] / 3
        sigm += cij[i] / vi[i]

    # Recalculate cij coefficients
    for j in range(icp):
        cij[j] = (cij[j] / vi[j] / sigm)

    # Compute mixed extinction and scattering
    for l in range(20):
        for i in range(icp):
            ex[0, l] += cij[i] * ext[l, i]
            sc[0, l] += cij[i] * sca[l, i]

    # Compute mixed phase functions and asymmetry parameter
    for l in range(20):
        asy_n = 0.0
        asy_d = 0.0
        for k in range(nbmu):
            ph[l, k] = 0.0
            for i in range(icp):
                ph[l, k] += cij[i] * p1[l, i, k]
            ph[l, k] = ph[l, k] / sc[0, l]

            if ipol != 0:
                qh[l, k] = 0.0
                uh[l, k] = 0.0
                for i in range(icp):
                    qh[l, k] += cij[i] * q1[l, i, k]
                    uh[l, k] += cij[i] * u1[l, i, k]
                qh[l, k] = qh[l, k] / sc[0, l]
                uh[l, k] = uh[l, k] / sc[0, l]

            asy_n += cgaus_S[k] * ph[l, k] * pdgs_S[k]
            asy_d += ph[l, k] * pdgs_S[k]

        asy[0, l] = asy_n / asy_d

    # Normalize by particle number
    for i in range(icp):
        for l in range(20):
            ex[i, l] = ext[l, i] / np_particles[i]
            sc[i, l] = sca[l, i] / np_particles[i]

    return ex, sc, asy, ph, qh, uh


@jit(nopython=True)
def exscphase(X, nr, ni, ipol, cgaus_S, pdgs_S):
    """
    Compute Mie scattering efficiency and phase functions.

    Uses Mie theory to calculate scattering and extinction efficiency factors
    (Qsca and Qext) and phase functions including polarization.

    Parameters
    ----------
    X : float
        Size parameter (2πr/λ)
    nr : float
        Real part of refractive index
    ni : float
        Imaginary part of refractive index
    ipol : int
        Polarization flag (0=no, 1=yes)
    cgaus_S : ndarray
        Gauss quadrature angles (cosines), shape (nquad,)
    pdgs_S : ndarray
        Gauss quadrature weights, shape (nquad,)

    Returns
    -------
    Qext : float
        Extinction efficiency
    Qsca : float
        Scattering efficiency
    p11 : ndarray
        Intensity phase function, shape (nquad,)
    q11 : ndarray
        Q Stokes parameter phase function, shape (nquad,)
    u11 : ndarray
        U Stokes parameter phase function, shape (nquad,)

    Notes
    -----
    Converted from Fortran EXSCPHASE

    Implements Mie scattering using:
    - Bessel function recurrence relations
    - Riccati-Bessel functions
    - Mie coefficients An and Bn
    - Legendre polynomial recurrence
    """
    nbmu = len(cgaus_S)

    # Compute complex refractive index reciprocal
    Ren = nr / (nr * nr + ni * ni)
    Imn = ni / (nr * nr + ni * ni)

    # Size parameter for complex refractive index
    Y = X * np.sqrt(nr * nr + ni * ni)

    # Determine maximum order of computation (mu)
    # From Corbato, J. Assoc. Computing Machinery, 1959
    N = int(0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * Y * Y))) + 1
    if N == 1:
        N = 2

    # Compute upper limit for series
    Np = N
    Up = 2.0 * Y / (2.0 * Np + 1.0)
    mu1 = int(Np + 30.0 * (0.10 + 0.35 * Up * (2 - Up * Up) / 2.0 / (1 - Up)))

    Np = int(Y - 0.5 + np.sqrt(30.0 * 0.35 * Y))
    mu2 = 1000000
    if Np > N:
        Up = 2.0 * Y / (2.0 * Np + 1.0)
        mu2 = int(Np + 30.0 * (0.10 + 0.35 * Up * (2 - Up * Up) / 2.0 / (1 - Up)))

    mu = min(mu1, mu2)

    if mu >= NSER_MAX:
        raise ValueError(f"Error: nser too small, mu={mu}")
    if mu <= 0:
        raise ValueError(f"Error: mu too small, mu={mu}")

    # Allocate arrays
    xj = np.zeros(mu + 2)
    xy = np.zeros(mu + 2)
    Rn = np.zeros(mu + 1)
    RDnX = np.zeros(mu + 1)
    RDnY = np.zeros(mu + 1)
    IDnY = np.zeros(mu + 1)
    RGnX = np.zeros(mu + 1)
    IGnX = np.zeros(mu + 1)
    RAn = np.zeros(mu + 1)
    IAn = np.zeros(mu + 1)
    RBn = np.zeros(mu + 1)
    IBn = np.zeros(mu + 1)
    PIn = np.zeros(mu + 2)
    TAUn = np.zeros(mu + 2)
    p11 = np.zeros(nbmu)
    q11 = np.zeros(nbmu)
    u11 = np.zeros(nbmu)

    # Downward recursion for Bessel function ratio Rn
    Rn[mu] = 0.0
    k = mu + 1
    while True:
        k = k - 1
        xj[k] = 0.0
        Rn[k - 1] = X / (2.0 * k + 1.0 - X * Rn[k])

        if k == 2:
            mub = mu
            xj[mub + 1] = 0.0
            xj[mub] = 1.0
            break

        if Rn[k - 1] > 1.0:
            mub = k - 1
            xj[mub + 1] = Rn[mub]
            xj[mub] = 1.0
            break

    # Upward recursion for Bessel function j
    for k in range(mub, 0, -1):
        xj[k - 1] = (2.0 * k + 1.0) * xj[k] / X - xj[k + 1]

    coxj = (xj[0] - X * xj[1]) * np.cos(X) + X * xj[0] * np.sin(X)

    # Downward recursion for Dn(alpha) and Dn(alpha*m)
    RDnY[mu] = 0.0
    IDnY[mu] = 0.0
    RDnX[mu] = 0.0

    for k in range(mu, 0, -1):
        RDnX[k - 1] = k / X - 1.0 / (RDnX[k] + k / X)
        XnumRDnY = RDnY[k] + Ren * k / X
        XnumIDnY = IDnY[k] + Imn * k / X
        XdenDnY = XnumRDnY * XnumRDnY + XnumIDnY * XnumIDnY
        RDnY[k - 1] = k * Ren / X - XnumRDnY / XdenDnY
        IDnY[k - 1] = k * Imn / X + XnumIDnY / XdenDnY

    # Initialize upward recursions
    xy[0] = -np.cos(X) / X  # xy(-1) in Fortran
    xy[1] = np.sin(X) / X   # xy(0) in Fortran
    RGnX[0] = 0.0
    IGnX[0] = -1.0
    Qsca = 0.0
    Qext = 0.0

    # Main loop for Mie coefficients
    for k in range(1, mu + 1):
        if k <= mub:
            xj[k] = xj[k] / coxj
        else:
            xj[k] = Rn[k - 1] * xj[k - 1]

        # Bessel function y
        xy[k + 1] = (2.0 * k - 1.0) * xy[k] / X - xy[k - 1]
        xJonH = xj[k] / (xj[k] * xj[k] + xy[k + 1] * xy[k + 1])

        # Gn(alpha)
        XdenGNX = (RGnX[k - 1] - k / X) ** 2 + IGnX[k - 1] * IGnX[k - 1]
        RGnX[k] = (k / X - RGnX[k - 1]) / XdenGNX - k / X
        IGnX[k] = IGnX[k - 1] / XdenGNX

        # An(alpha) and Bn(alpha) coefficients
        Xnum1An = RDnY[k] - nr * RDnX[k]
        Xnum2An = IDnY[k] + ni * RDnX[k]
        Xden1An = RDnY[k] - nr * RGnX[k] - ni * IGnX[k]
        Xden2An = IDnY[k] + ni * RGnX[k] - nr * IGnX[k]
        XdenAn = Xden1An * Xden1An + Xden2An * Xden2An
        RAnb = (Xnum1An * Xden1An + Xnum2An * Xden2An) / XdenAn
        IAnb = (-Xnum1An * Xden2An + Xnum2An * Xden1An) / XdenAn
        RAn[k] = xJonH * (xj[k] * RAnb - xy[k + 1] * IAnb)
        IAn[k] = xJonH * (xy[k + 1] * RAnb + xj[k] * IAnb)

        Xnum1Bn = nr * RDnY[k] + ni * IDnY[k] - RDnX[k]
        Xnum2Bn = nr * IDnY[k] - ni * RDnY[k]
        Xden1Bn = nr * RDnY[k] + ni * IDnY[k] - RGnX[k]
        Xden2Bn = nr * IDnY[k] - ni * RDnY[k] - IGnX[k]
        XdenBn = Xden1Bn * Xden1Bn + Xden2Bn * Xden2Bn
        RBnb = (Xnum1Bn * Xden1Bn + Xnum2Bn * Xden2Bn) / XdenBn
        IBnb = (-Xnum1Bn * Xden2Bn + Xnum2Bn * Xden1Bn) / XdenBn
        RBn[k] = xJonH * (xj[k] * RBnb - xy[k + 1] * IBnb)
        IBn[k] = xJonH * (xy[k + 1] * RBnb + xj[k] * IBnb)

        # Convergence test (Deirmendjian et al., 1961)
        test = (RAn[k]**2 + IAn[k]**2 + RBn[k]**2 + IBn[k]**2) / k
        if test < 1.0e-14:
            mu = k
            if mu <= 0:
                raise ValueError("Warning: mu < 0")
            break

        # Scattering and extinction efficiency
        xpond = 2.0 / X / X * (2.0 * k + 1)
        Qsca += xpond * (RAn[k]**2 + IAn[k]**2 + RBn[k]**2 + IBn[k]**2)
        Qext += xpond * (RAn[k] + RBn[k])

    # Compute amplitude functions S1 and S2
    for j in range(nbmu):
        xmud = cgaus_S[j]
        RS1 = 0.0
        RS2 = 0.0
        IS1 = 0.0
        IS2 = 0.0
        PIn[0] = 0.0
        PIn[1] = 1.0
        TAUn[1] = xmud

        for k in range(1, mu + 1):
            co_n = (2.0 * k + 1.0) / k / (k + 1.0)
            RS1 += co_n * (RAn[k] * PIn[k] + RBn[k] * TAUn[k])
            RS2 += co_n * (RAn[k] * TAUn[k] + RBn[k] * PIn[k])
            IS1 += co_n * (IAn[k] * PIn[k] + IBn[k] * TAUn[k])
            IS2 += co_n * (IAn[k] * TAUn[k] + IBn[k] * PIn[k])
            PIn[k + 1] = ((2.0 * k + 1) * xmud * PIn[k] - (k + 1.0) * PIn[k - 1]) / k
            TAUn[k + 1] = (k + 1.0) * xmud * PIn[k + 1] - (k + 2.0) * PIn[k]

        # Scattering intensity efficiency
        p11[j] = (RS1 * RS1 + IS1 * IS1 + RS2 * RS2 + IS2 * IS2) / X / X / 2.0
        if ipol != 0:
            q11[j] = (RS2 * RS2 + IS2 * IS2 - RS1 * RS1 - IS1 * IS1) / X / X / 2.0
            u11[j] = (2.0 * RS2 * RS1 + 2.0 * IS2 * IS1) / X / X / 2.0

    return Qext, Qsca, p11, q11, u11


__all__ = ['mie', 'exscphase']
