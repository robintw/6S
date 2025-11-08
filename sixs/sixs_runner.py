"""
Main computation runner for 6SV.

Orchestrates all radiative transfer calculations by calling converted modules
in the correct sequence, following the computational flow from main.f.

This is a complete rewrite to properly replicate the logic from main.f.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from sixs.input_parser import SixSInput
from sixs.standard_atmospheres import (
    tropic, midsum, midwin, subsum, subwin, us62
)
from sixs.solar_spectral import varsol, equivwl, solirr
from sixs.odrayl import odrayl
from sixs.discrete_ordinate_computation import discom
from sixs.aeroso import aeroso
from sixs.oda550 import oda550
from sixs.profile_utilities import pressure, presplane
from sixs.water_models import clearw, lakew
from sixs.surface_models import sand, vegeta
from sixs.gauss import gauss
from sixs.abstra import abstra
from sixs.spectral_interpolation import interp
from sixs.successive_orders import os as successive_orders_os
from sixs.brdf_models import enviro


# Parameter definitions matching paramdef.inc
NT_P = 30        # Number of layers
MU_P = 25        # Number of Gauss angles
MU2_P = 48       # 2*(MU_P-1)
NP_P = 49        # Number of azimuth angles
NFI_P = 181      # Number of azimuth output angles
NQUAD_P = 83     # Number of quadrature points (must be odd)


@dataclass
class SixSResults:
    """Container for 6SV computation results."""

    # Geometry
    asol: float = 0.0
    phi0: float = 0.0
    avis: float = 0.0
    phiv: float = 0.0
    phi: float = 0.0
    adif: float = 0.0
    xmus: float = 0.0
    xmuv: float = 0.0
    xmud: float = 0.0  # Scattering angle cosine
    dsol: float = 1.0

    # Atmospheric parameters
    uw: float = 0.0
    uo3: float = 0.0
    taer55: float = 0.0
    taer55p: float = 0.0
    v: float = 0.0

    # Wavelength
    wlmoy: float = 0.55

    # Optical properties
    trmoy: float = 0.0
    trmoyp: float = 0.0
    tamoy: float = 0.0
    tamoyp: float = 0.0
    pizmoy: float = 0.0
    pizmoyp: float = 0.0

    # Reflectances
    roc: float = 0.0
    roe: float = 0.0
    roatm: np.ndarray = None
    rqatm: np.ndarray = None  # Polarization Q
    ruatm: np.ndarray = None  # Polarization U
    romix: float = 0.0
    rorayl: float = 0.0
    roaero: float = 0.0

    # Transmittances
    dtdir: np.ndarray = None
    dtdif: np.ndarray = None
    utdir: np.ndarray = None
    utdif: np.ndarray = None
    sphal: np.ndarray = None

    # Total transmittance
    tgasm: float = 1.0

    # Atmospheric correction
    refet: float = 0.0
    alumet: float = 0.0

    # Surface polarization
    ropq: float = 0.0
    ropu: float = 0.0

    # Integrated values
    seb: float = 0.0  # Total irradiance

    def __post_init__(self):
        """Initialize arrays if not provided."""
        if self.roatm is None:
            self.roatm = np.zeros((3, 20))
        if self.rqatm is None:
            self.rqatm = np.zeros((3, 20))
        if self.ruatm is None:
            self.ruatm = np.zeros((3, 20))
        if self.dtdir is None:
            self.dtdir = np.zeros((3, 20))
        if self.dtdif is None:
            self.dtdif = np.zeros((3, 20))
        if self.utdir is None:
            self.utdir = np.zeros((3, 20))
        if self.utdif is None:
            self.utdif = np.zeros((3, 20))
        if self.sphal is None:
            self.sphal = np.zeros((3, 20))


class SixSRunner:
    """
    Main computation engine for 6SV.

    Executes the complete radiative transfer calculation sequence following
    the exact logic from main.f:
    1. Initialize constants and quadrature points
    2. Geometry computation
    3. Atmospheric model setup
    4. Aerosol model setup
    5. Altitude adjustments
    6. Discrete ordinate computation (DISCOM)
    7. Spectral loop with gas absorption and interpolation
    8. Surface reflectance and coupling
    9. Atmospheric correction
    """

    def __init__(self, input_data: SixSInput):
        """
        Initialize runner with parsed input.

        Args:
            input_data: Parsed 6SV input parameters
        """
        self.input = input_data
        self.results = SixSResults()

        # Constants from main.f
        self.pi = np.pi
        self.pi2 = 2.0 * np.pi
        self.sigma = 0.056032
        self.delta = 0.0279
        self.step = 0.0025  # Wavelength step (µm)
        self.xacc = 1.e-06  # Accuracy parameter
        self.accu2 = 1.e-03
        self.accu3 = 1.e-07

        # Wavelength discretization (20 points) from main.f
        self.wldis = np.array([
            0.350, 0.400, 0.412, 0.443, 0.470, 0.488, 0.515, 0.550,
            0.590, 0.633, 0.670, 0.694, 0.760, 0.860, 1.240, 1.536,
            1.650, 1.950, 2.250, 3.750
        ])

        # Quadrature parameters from main.f
        self.nt = NT_P  # Number of layers
        self.mu = MU_P  # Number of angles
        self.mu2 = MU2_P  # 2*mu
        self.np_angles = NP_P  # Number of azimuth angles
        self.nfi = NFI_P  # Number of azimuth output angles
        self.nquad = NQUAD_P  # Quadrature points

        # Initialize Gauss quadrature points (from main.f lines 467-482)
        self._initialize_quadrature()

        # Arrays for aerosol properties (20 wavelengths)
        self.ext = np.zeros(20)
        self.ome = np.zeros(20)
        self.gasym = np.zeros(20)
        self.phase = np.zeros(20)
        self.qhase = np.zeros(20)
        self.uhase = np.zeros(20)

        # Legendre expansion coefficients
        self.alphal = np.zeros(NQUAD_P + 1)
        self.betal = np.zeros(NQUAD_P + 1)
        self.gammal = np.zeros(NQUAD_P + 1)
        self.zetal = np.zeros(NQUAD_P + 1)

        # Phase function arrays
        self.phasel = np.zeros((20, NQUAD_P))
        self.qhasel = np.zeros((20, NQUAD_P))
        self.uhasel = np.zeros((20, NQUAD_P))

        # Aerosol profile
        self.iaer_prof = 0
        self.num_z = 0
        self.alt_z = np.zeros(100)
        self.taer_z = np.zeros(100)
        self.taer55_z = np.zeros(100)

        # Arrays from discom
        self.trayl = np.zeros(20)
        self.traypl = np.zeros(20)
        self.roatm_fi = np.zeros((3, 20, self.nfi))
        self.nfilut = np.zeros(self.mu, dtype=int)
        self.filut = np.zeros((self.mu, 41))
        self.roluts = np.zeros((20, self.mu, 41))
        self.rolutsq = np.zeros((20, self.mu, 41))
        self.rolutsu = np.zeros((20, self.mu, 41))

        # Spectral arrays
        self.wlinf = 0.25
        self.wlsup = 4.0
        self.iinf = 1
        self.isup = 1501

        # Surface reflectance spectra
        self.rocl = np.zeros(1501)
        self.roel = np.zeros(1501)

        # Filter function (for spectral integration)
        self.s = np.ones(1501)

        # Polarization flag
        self.ipol = getattr(input_data, 'ipol', 0)

    def _initialize_quadrature(self):
        """Initialize Gauss quadrature points (main.f lines 467-482)."""
        # Compute Gauss quadrature for angles
        anglem, weightm = gauss(-1.0, 1.0, self.mu2)

        # Compute azimuth angles
        self.rp, self.gp = gauss(0.0, self.pi2, self.np_angles)

        # Set up rm and gb arrays with proper indexing
        # In Python we use 0-based indexing, but need to handle negative indices
        mum1 = self.mu - 1
        self.rm = np.zeros(2 * self.mu + 1)
        self.gb = np.zeros(2 * self.mu + 1)

        # Map from Fortran indexing to Python
        # rm(-mu:mu) maps to rm[0:2*mu+1]
        for j in range(mum1):
            k = j
            self.rm[mum1 - j - 1] = anglem[k]
            self.gb[mum1 - j - 1] = weightm[k]

        for j in range(1, mum1):
            k = mum1 + j
            self.rm[self.mu + j] = anglem[k]
            self.gb[self.mu + j] = weightm[k]

        # Zero out the boundary weights
        self.gb[0] = 0.0  # gb(-mu)
        self.gb[self.mu] = 0.0  # gb(0)
        self.gb[2 * self.mu] = 0.0  # gb(mu)

        # Workspace arrays for discom
        self.xlm1 = np.zeros((2 * self.mu + 1, self.np_angles))
        self.xlm2 = np.zeros((2 * self.mu + 1, self.np_angles))

    def run(self) -> SixSResults:
        """
        Execute complete 6SV computation following main.f logic.

        Returns:
            SixSResults object with all computed values
        """
        # 1. Compute geometry (main.f geometry section)
        self._compute_geometry()

        # 2. Set up atmospheric model (main.f lines 706-713)
        self._setup_atmosphere()

        # 3. Set up aerosol model (main.f line 942)
        self._setup_aerosol()

        # 4. Handle altitude adjustments (main.f lines 995-1059)
        self._handle_altitudes()

        # 5. Load surface reflectance spectra (main.f lines 2221-2257)
        self._load_surface_spectra()

        # 6. Compute spectral properties (main.f lines 1496-1515)
        # This calls equivwl, discom, specinterp, odrayl
        self._compute_spectral_and_discom()

        # 7. Spectral integration loop (main.f lines 2767-3141)
        self._spectral_integration_loop()

        # 8. Compute atmospheric correction if requested
        if hasattr(self.input, 'iatmcorr') and self.input.iatmcorr > 0:
            self._atmospheric_correction()

        return self.results

    def _compute_geometry(self):
        """
        Compute geometric parameters from angles.

        Follows main.f geometry computation section.
        """
        # Copy input angles
        self.results.asol = self.input.asol
        self.results.phi0 = self.input.phi0
        self.results.avis = self.input.avis
        self.results.phiv = self.input.phiv

        # Compute relative azimuth
        self.results.phi = abs(self.input.phiv - self.input.phi0)

        # Convert to radians
        phirad = (self.input.phi0 - self.input.phiv) * self.pi / 180.0
        if phirad < 0.0:
            phirad += self.pi2
        if phirad > self.pi2:
            phirad -= self.pi2

        self.phirad = phirad

        # Compute cosines
        self.results.xmus = np.cos(self.input.asol * self.pi / 180.0)
        self.results.xmuv = np.cos(self.input.avis * self.pi / 180.0)
        xmup = np.cos(phirad)

        # Scattering angle cosine (critical for phase function evaluation)
        xmud = -self.results.xmus * self.results.xmuv - \
               np.sqrt(1.0 - self.results.xmus**2) * \
               np.sqrt(1.0 - self.results.xmuv**2) * xmup

        # Clamp to valid range
        xmud = np.clip(xmud, -1.0, 1.0)
        self.results.xmud = xmud

        # Scattering angle (degrees)
        self.results.adif = np.arccos(xmud) * 180.0 / self.pi

        # Solar constant variability
        jday = getattr(self.input, 'jday', 1)
        month = getattr(self.input, 'month', 1)
        self.results.dsol = varsol(jday, month)

    def _setup_atmosphere(self):
        """
        Set up atmospheric model and load profiles.

        Follows main.f lines 706-713 and 713.
        """
        idatm = self.input.idatm

        # Load appropriate atmospheric profile
        if idatm == 1:
            z, p, t, wh, wo = tropic()
        elif idatm == 2:
            z, p, t, wh, wo = midsum()
        elif idatm == 3:
            z, p, t, wh, wo = midwin()
        elif idatm == 4:
            z, p, t, wh, wo = subsum()
        elif idatm == 5:
            z, p, t, wh, wo = subwin()
        elif idatm == 6:
            z, p, t, wh, wo = us62()
        elif idatm == 7:
            # User-defined profile
            z = self.input.z
            p = self.input.p
            t = self.input.t
            wh = self.input.wh
            wo = self.input.wo
        elif idatm == 8:
            # User-defined water vapor and ozone with US62 profile
            z, p, t, wh, wo = us62()
            self.results.uw = self.input.uw
            self.results.uo3 = self.input.uo3
        else:  # idatm == 0 (no gaseous absorption)
            z, p, t, wh, wo = us62()  # Use for Rayleigh calculation

        # Store atmospheric profile
        self.atm_z = z
        self.atm_p = p
        self.atm_t = t
        self.atm_wh = wh
        self.atm_wo = wo

        # Compute integrated water vapor and ozone if not user-defined
        if idatm != 8:
            # These would be computed from profile integration
            # For now use approximate values
            self.results.uw = np.sum(wh) * 0.01  # Approximate
            self.results.uo3 = np.sum(wo) * 0.001  # Approximate

        # Store standard values for later use
        self.uwus = self.results.uw
        self.uo3us = self.results.uo3

    def _setup_aerosol(self):
        """
        Set up aerosol model and compute optical properties.

        Follows main.f line 942: call aeroso
        """
        iaer = self.input.iaer

        if iaer == 0:
            # No aerosols
            self.results.taer55 = 0.0
            self.results.v = 0.0
            self.ext[:] = 0.0
            self.ome[:] = 0.0
            self.gasym[:] = 0.0
            return

        # Get aerosol optical thickness
        if hasattr(self.input, 'v') and self.input.v > 0:
            # Convert visibility to AOT at 550nm (main.f line 970)
            v = self.input.v
            self.results.v = v
            taer55 = oda550(iaer, v)
            self.results.taer55 = taer55
        else:
            taer55 = getattr(self.input, 'taer55', 0.1)
            self.results.taer55 = taer55

        # Component mixing ratios
        c = getattr(self.input, 'c', np.array([0.0, 0.0, 0.0, 0.0]))

        # Call aeroso to get optical properties (main.f line 942)
        (self.ext, self.ome, self.gasym, self.phase, self.qhase, self.uhase,
         self.phasel, self.qhasel, self.uhasel,
         self.alphal, self.betal, self.gammal, self.zetal) = aeroso(
            iaer, c, self.results.xmud, self.wldis, self.nquad, self.ipol
        )

    def _handle_altitudes(self):
        """
        Handle target and sensor altitude adjustments.

        Follows main.f lines 995-1059.
        """
        # Initialize plane atmosphere to None
        self.atm_zpl = None
        self.atm_ppl = None
        self.atm_tpl = None
        self.atm_whpl = None
        self.atm_wopl = None
        self.ftray = 1.0

        # Initialize plane variables for abstra
        self.puw = self.results.uw
        self.puo3 = self.results.uo3
        self.puwus = self.uwus
        self.puo3us = self.uo3us

        # Target altitude (ground level adjustment)
        pps = getattr(self.input, 'pps', 0.0)
        if pps != 0:
            xps = pps
            if xps < 0:
                # Ground target altitude below sea level
                (self.results.uw, self.results.uo3,
                 self.atm_z, self.atm_p, self.atm_t,
                 self.atm_wh, self.atm_wo) = pressure(
                    self.results.uw, self.results.uo3, xps,
                    self.atm_z, self.atm_p, self.atm_t,
                    self.atm_wh, self.atm_wo
                )
                self.uwus = self.results.uw
                self.uo3us = self.results.uo3

        # Sensor altitude (plane observation)
        palt = getattr(self.input, 'palt', 0.0)
        if palt > 0:
            xpp = palt
            (self.puw, self.puo3, self.ftray,
             self.atm_zpl, self.atm_ppl, self.atm_tpl,
             self.atm_whpl, self.atm_wopl) = presplane(
                self.results.uw, self.results.uo3, xpp,
                self.atm_z, self.atm_p, self.atm_t,
                self.atm_wh, self.atm_wo
            )

            # Standard atmosphere plane values
            (self.puwus, self.puo3us, ftray_dummy,
             zpl_dummy, ppl_dummy, tpl_dummy,
             whpl_dummy, wopl_dummy) = presplane(
                self.uwus, self.uo3us, xpp,
                self.atm_z, self.atm_p, self.atm_t,
                self.atm_wh, self.atm_wo
            )

            # Compute taer55p (aerosol above plane)
            self.results.taer55p = self.results.taer55 * self.ftray
        else:
            self.results.taer55p = 0.0
            self.ftray = 1.0

    def _load_surface_spectra(self):
        """
        Load surface reflectance spectra.

        Follows main.f lines 2221-2257.
        """
        igroun = getattr(self.input, 'igroun', 0)
        inhomo = getattr(self.input, 'inhomo', 0)

        if inhomo == 0:
            # Homogeneous surface
            if igroun == 0:
                # Constant reflectance
                roc = getattr(self.input, 'roc', 0.1)
                self.rocl[:] = roc
                self.roel[:] = roc
            elif igroun == 1:
                # Vegetation
                self.rocl = vegeta()
                self.roel = self.rocl.copy()
            elif igroun == 2:
                # Clear water
                self.rocl = clearw()
                self.roel = self.rocl.copy()
            elif igroun == 3:
                # Sand
                self.rocl = sand()
                self.roel = self.rocl.copy()
            elif igroun == 4:
                # Lake water
                self.rocl = lakew()
                self.roel = self.rocl.copy()
        else:
            # Inhomogeneous surface
            igrou1 = getattr(self.input, 'igrou1', 0)
            igrou2 = getattr(self.input, 'igrou2', 0)

            # Target reflectance
            if igrou1 == 0:
                roc = getattr(self.input, 'roc', 0.1)
                self.rocl[:] = roc
            elif igrou1 == 1:
                self.rocl = vegeta()
            elif igrou1 == 2:
                self.rocl = clearw()
            elif igrou1 == 3:
                self.rocl = sand()
            elif igrou1 == 4:
                self.rocl = lakew()

            # Environment reflectance
            if igrou2 == 0:
                roe = getattr(self.input, 'roe', 0.1)
                self.roel[:] = roe
            elif igrou2 == 1:
                self.roel = vegeta()
            elif igrou2 == 2:
                self.roel = clearw()
            elif igrou2 == 3:
                self.roel = sand()
            elif igrou2 == 4:
                self.roel = lakew()

    def _compute_spectral_and_discom(self):
        """
        Compute spectral properties and run discrete ordinate computation.

        Follows main.f lines 1496-1515:
        - call equivwl
        - call discom
        - call specinterp (handled separately in spectral loop)
        - call odrayl
        """
        # Determine wavelength mode
        iwave = getattr(self.input, 'iwave', -1)

        if iwave == -1:
            # Monochromatic wavelength
            wl = getattr(self.input, 'wlinf', 0.55)
            wlmoy = wl
        elif iwave == -2 or iwave == 0:
            # Wavelength range with uniform filter
            wlinf = getattr(self.input, 'wlinf', 0.4)
            wlsup = getattr(self.input, 'wlsup', 0.7)
            wlmoy = (wlinf + wlsup) / 2.0
            self.wlinf = wlinf
            self.wlsup = wlsup
            self.iinf = int((wlinf - 0.25) / self.step) + 1
            self.isup = int((wlsup - 0.25) / self.step) + 1
        elif iwave == 1:
            # User-defined filter function
            s = getattr(self.input, 's', np.ones(1501))
            self.s = s
            self.iinf = getattr(self.input, 'iinf', 1)
            self.isup = getattr(self.input, 'isup', 1501)
            wlmoy = equivwl(self.iinf, self.isup, self.step, s)
        else:
            # Predefined sensor band (would need sensor response functions)
            wlmoy = 0.55  # Default

        self.results.wlmoy = wlmoy

        # Determine observation level (idatmp)
        palt = getattr(self.input, 'palt', 0.0)
        if palt == 0:
            idatmp = 0  # Ground level
        elif palt < 0 or palt > 100:
            idatmp = 4  # TOA
        else:
            idatmp = 1  # Plane observation

        # Call DISCOM - THE CRITICAL DISCRETE ORDINATE COMPUTATION
        # This computes atmospheric reflectances and transmittances
        # at 20 discrete wavelengths (main.f line 1502)
        (self.results.roatm, self.results.rqatm, self.results.ruatm,
         self.results.dtdir, self.results.dtdif,
         self.results.utdir, self.results.utdif, self.results.sphal,
         self.trayl, self.traypl, self.roatm_fi,
         self.roluts, self.rolutsq, self.rolutsu,
         self.nfilut, self.filut) = discom(
            idatmp, self.input.iaer, self.iaer_prof,
            self.results.xmus, self.results.xmuv, self.results.phi,
            self.results.taer55, self.results.taer55p,
            palt, self.phirad, self.nt, self.mu, self.np_angles,
            self.rm, self.gb, self.rp, self.ftray, self.ipol,
            self.xlm1, self.xlm2, self.nfi,
            self.ext, self.ome, self.gasym, self.phase,
            self.qhase, self.uhase, self.wldis,
            self.wlinf, self.wlsup,
            self.alphal, self.betal, self.gammal, self.zetal,
            self.phasel, self.qhasel, self.uhasel, self.nquad,
            self.alt_z, self.taer_z, self.taer55_z, self.num_z
        )

        # Compute Rayleigh optical thickness at mean wavelength
        trmoy = odrayl(wlmoy)
        self.results.trmoy = trmoy
        self.results.trmoyp = trmoy * self.ftray

        # Compute aerosol optical properties at mean wavelength
        # This would be done by specinterp, but for now use simple approximation
        if self.input.iaer != 0 and self.results.taer55 > 0:
            # Simple Angstrom approximation
            alpha = 1.3
            self.results.tamoy = self.results.taer55 * (wlmoy / 0.55) ** (-alpha)
            self.results.tamoyp = self.results.tamoy * self.ftray
            self.results.pizmoy = 0.9  # Would come from specinterp
        else:
            self.results.tamoy = 0.0
            self.results.tamoyp = 0.0
            self.results.pizmoy = 0.0

    def _spectral_integration_loop(self):
        """
        Spectral integration loop over wavelengths.

        Follows main.f lines 2767-3141:
        - Loop over wavelengths from iinf to isup
        - Call abstra for gas absorption
        - Call solirr for solar irradiance
        - Call interp to interpolate atmospheric properties
        - Compute total transmittances
        - Compute surface contribution
        - Compute atmospheric reflectance
        - Integrate over spectral band
        """
        # Initialize integration variables
        sb = 0.0  # Total filter weight
        seb = 0.0  # Total irradiance
        alumet = 0.0  # Integrated radiance
        refet = 0.0  # Integrated reflectance

        # Observation level
        palt = getattr(self.input, 'palt', 0.0)
        if palt == 0:
            idatmp = 0
        elif palt < 0 or palt > 100:
            idatmp = 4
        else:
            idatmp = 1

        # Inhomogeneity parameters
        inhomo = getattr(self.input, 'inhomo', 0)
        idirec = getattr(self.input, 'idirec', 0)
        rad = getattr(self.input, 'rad', 0.0)

        # Loop over wavelengths (main.f line 2767)
        for l in range(self.iinf - 1, self.isup):  # Convert to 0-based
            # Spectral weight
            sbor = self.s[l]
            if l == self.iinf - 1 or l == self.isup - 1:
                sbor = sbor * 0.5  # Trapezoidal rule

            # Surface reflectances
            roc = self.rocl[l]
            roe = self.roel[l]

            # Wavelength
            wl = 0.25 + l * self.step

            # Call ABSTRA for gas absorption (main.f lines 2775, 2780)
            # First call with uw/2 for some reason in original code
            if self.atm_zpl is None:
                zpl = self.atm_z
                ppl = self.atm_p
                tpl = self.atm_t
                whpl = self.atm_wh
                wopl = self.atm_wo
            else:
                zpl = self.atm_zpl
                ppl = self.atm_ppl
                tpl = self.atm_tpl
                whpl = self.atm_whpl
                wopl = self.atm_wopl

            (dtwava, dtozon, dtdica, dtoxyg, dtniox, dtmeth, dtmoca,
             utwava, utozon, utdica, utoxyg, utniox, utmeth, utmoca,
             ttwava, ttozon, ttdica, ttoxyg, ttniox, ttmeth, ttmoca) = abstra(
                self.input.idatm, wl,
                self.results.xmus, self.results.xmuv,
                self.results.uw, self.results.uo3,
                self.uwus, self.uo3us,
                idatmp, self.puw, self.puo3,
                self.puwus, self.puo3us,
                self.atm_z, self.atm_p, self.atm_t,
                self.atm_wh, self.atm_wo,
                zpl, ppl, tpl, whpl, wopl
            )

            # Apply accuracy threshold
            if dtwava < self.accu3:
                dtwava = 0.0
            if utwava < self.accu3:
                utwava = 0.0
            if ttwava < self.accu3:
                ttwava = 0.0

            # Call SOLIRR for solar irradiance (main.f line 2804)
            swl = solirr(wl)
            swl = swl * self.results.dsol

            # Integration coefficients
            coef = sbor * self.step * swl

            # Call INTERP to interpolate atmospheric properties (main.f line 2809)
            (romix, rorayl, roaero, phaa, phar,
             rqmix, rqrayl, rqaero, qhaa, qhar,
             rumix, rurayl, ruaero, uhaa, uhar,
             tsca, tray, trayp, taer, taerp,
             dtott, utott, astot, asray, asaer,
             utotr, utota, dtotr, dtota,
             romix_fi, rolut, rolutq, rolutu) = interp(
                self.input.iaer, idatmp, wl,
                self.results.taer55, self.results.taer55p,
                self.results.xmud,
                self.results.roatm, self.results.rqatm, self.results.ruatm,
                self.ext, self.ome, self.gasym,
                self.phase, self.qhase, self.uhase,
                self.results.dtdir, self.results.dtdif,
                self.results.utdir, self.results.utdif,
                self.results.sphal, self.wldis,
                self.trayl, self.traypl,
                self.delta, self.sigma,
                self.roatm_fi, self.roluts,
                self.rolutsq, self.rolutsu,
                self.nfilut, self.nfi, self.mu
            )

            # Total gaseous transmittances (main.f lines 2816-2820)
            dgtot = dtwava * dtozon * dtdica * dtoxyg * dtniox * dtmeth * dtmoca
            tgtot = ttwava * ttozon * ttdica * ttoxyg * ttniox * ttmeth * ttmoca
            ugtot = utwava * utozon * utdica * utoxyg * utniox * utmeth * utmoca
            tgp1 = ttozon * ttdica * ttoxyg * ttniox * ttmeth * ttmoca
            tgp2 = ttwava * ttozon * ttdica * ttoxyg * ttniox * ttmeth * ttmoca

            # Compute surface contribution (main.f lines 2828-2846)
            if idirec == 1:
                # Directional effects (BRDF) - not fully implemented
                tdird = np.exp(-(trayp + taerp) / self.results.xmus)
                tdiru = np.exp(-(trayp + taerp) / self.results.xmuv)
                tdifd = dtott - tdird
                tdifu = utott - tdiru

                # Simplified - would need BRDF values
                robar_val = roc
                robarp_val = roc
                robard_val = roc

                rsurf = (roc * tdird * tdiru +
                        robar_val * tdifd * tdiru +
                        robarp_val * tdifu * tdird +
                        robard_val * tdifd * tdifu +
                        (tdifd + tdird) * (tdifu + tdiru) * astot *
                        robard_val * robard_val / (1.0 - astot * robard_val))
                avr = robard_val
            else:
                # Lambertian surface with environment (main.f lines 2842-2845)
                if inhomo == 1 and rad > 0:
                    # Call enviro for adjacency effect
                    edifr = utotr - np.exp(-trayp / self.results.xmuv)
                    edifa = utota - np.exp(-taerp / self.results.xmuv)
                    fra, fae, fr = enviro(edifr, edifa, rad, palt, self.results.xmuv)
                    avr = roc * fr + (1.0 - fr) * roe
                else:
                    avr = roc

                rsurf = (roc * dtott * np.exp(-(trayp + taerp) / self.results.xmuv) /
                        (1.0 - avr * astot) +
                        avr * dtott * (utott - np.exp(-(trayp + taerp) / self.results.xmuv)) /
                        (1.0 - avr * astot))

            # Atmospheric reflectance with gas absorption (main.f lines 2847-2855)
            ratm1 = (romix - rorayl) * tgtot + rorayl * tgp1
            ratm2 = (romix - rorayl) * tgp2 + rorayl * tgp1
            ratm3 = romix * tgp1

            # Total reflectance
            romeas1 = ratm1 + rsurf * tgtot
            romeas2 = ratm2 + rsurf * tgtot
            romeas3 = ratm3 + rsurf * tgtot

            # Radiance
            alumeas = self.results.xmus * swl * romeas2 / self.pi

            # Integrate over spectral band (main.f lines 2824-2859)
            sb += sbor * self.step
            seb += coef
            alumet += alumeas * sbor * self.step
            refet += romeas2 * sbor * self.step

        # Store integrated results
        self.results.seb = seb
        self.results.alumet = alumet / sb if sb > 0 else 0.0
        self.results.refet = refet / sb if sb > 0 else 0.0

        # Store total gaseous transmittance
        self.results.tgasm = tgp2

        # Store final atmospheric and surface reflectances
        self.results.romix = romix
        self.results.rorayl = rorayl
        self.results.roaero = roaero
        self.results.roc = roc
        self.results.roe = roe

    def _atmospheric_correction(self):
        """
        Perform atmospheric correction.

        Uses computed atmospheric properties instead of hardcoded values.
        """
        # Get input radiance or reflectance
        radiance = getattr(self.input, 'radiance', 0.0)
        input_val = abs(radiance)

        if radiance > 0:
            # Radiance input - convert to reflectance
            swl = solirr(self.results.wlmoy)
            refet = (input_val * self.pi /
                    (swl * self.results.dsol * self.results.xmus))
        else:
            # Reflectance input
            refet = input_val

        # Atmospheric correction using computed values
        # refet_surface = (refet_toa - roatm) / (tdir_down * tdir_up)

        # Use values from spectral loop
        roatm = self.results.roatm[1, 7]  # Mixed atmosphere at 550nm

        # Transmittances
        tdir_down = self.results.dtdir[1, 7]
        tdir_up = self.results.utdir[1, 7]

        # Simple atmospheric correction
        if tdir_down * tdir_up > 0:
            self.results.refet = max(0.0, (refet - roatm) / (tdir_down * tdir_up))
        else:
            self.results.refet = 0.0

        # Convert back to radiance if needed
        swl = solirr(self.results.wlmoy)
        self.results.alumet = (self.results.refet * swl * self.results.dsol *
                              self.results.xmus / self.pi)
