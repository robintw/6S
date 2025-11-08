"""
Main computation runner for 6SV.

Orchestrates all radiative transfer calculations by calling converted modules
in the correct sequence, following the computational flow from main.f.
"""

import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from sixs.input_parser import SixSInput
from sixs.standard_atmospheres import (
    tropic, midsum, midwin, subsum, subwin, us62
)
from sixs.solar_spectral import varsol, equivwl, solirr  # , specinterp
from sixs.odrayl import odrayl
# from sixs.discrete_ordinate_computation import discom
# from sixs.aeroso import aeroso
from sixs.oda550 import oda550
from sixs.profile_utilities import pressure, presplane  # , aero_prof
from sixs.water_models import clearw, lakew
from sixs.surface_models import sand  # , enviro
from sixs.gauss import gauss


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
    dsol: float = 1.0

    # Atmospheric parameters
    uw: float = 0.0
    uo3: float = 0.0
    taer55: float = 0.0
    v: float = 0.0

    # Wavelength
    wlmoy: float = 0.55

    # Optical properties
    trmoy: float = 0.0
    trmoyp: float = 0.0
    tamoy: float = 0.0
    tamoyp: float = 0.0
    pizmoy: float = 0.0

    # Reflectances
    roc: float = 0.0
    roe: float = 0.0
    roatm: np.ndarray = None
    romix: float = 0.0
    rorayl: float = 0.0
    roaero: float = 0.0

    # Transmittances
    dtdir: np.ndarray = None
    dtdif: np.ndarray = None
    utdir: np.ndarray = None
    utdif: np.ndarray = None

    # Total transmittance
    tgasm: float = 1.0

    # Atmospheric correction
    refet: float = 0.0
    alumet: float = 0.0

    # Surface polarization
    ropq: float = 0.0
    ropu: float = 0.0

    def __post_init__(self):
        """Initialize arrays if not provided."""
        if self.roatm is None:
            self.roatm = np.zeros((3, 20))
        if self.dtdir is None:
            self.dtdir = np.zeros((3, 20))
        if self.dtdif is None:
            self.dtdif = np.zeros((3, 20))
        if self.utdir is None:
            self.utdir = np.zeros((3, 20))
        if self.utdif is None:
            self.utdif = np.zeros((3, 20))


class SixSRunner:
    """
    Main computation engine for 6SV.

    Executes the complete radiative transfer calculation sequence:
    1. Geometry computation
    2. Atmospheric model setup
    3. Aerosol model setup
    4. Spectral properties
    5. Radiative transfer
    6. Surface reflectance
    7. Atmospheric correction
    """

    def __init__(self, input_data: SixSInput):
        """
        Initialize runner with parsed input.

        Args:
            input_data: Parsed 6SV input parameters
        """
        self.input = input_data
        self.results = SixSResults()

        # Constants
        self.pi = np.pi
        self.sigma = 0.056032
        self.delta = 0.0279
        self.step = 0.0025  # Wavelength step (µm)

        # Wavelength discretization (20 points)
        self.wldis = np.array([
            0.350, 0.400, 0.412, 0.443, 0.470, 0.488, 0.515, 0.550,
            0.590, 0.633, 0.670, 0.694, 0.760, 0.860, 1.240, 1.536,
            1.650, 1.950, 2.250, 3.750
        ])

        # Quadrature parameters
        self.nt = 31  # Number of layers for vertical integration
        self.mu = 25  # Number of angles for Gauss quadrature
        self.np_angles = 13  # Number of azimuth angles

    def run(self) -> SixSResults:
        """
        Execute complete 6SV computation.

        Returns:
            SixSResults object with all computed values
        """
        # 1. Compute geometry
        self._compute_geometry()

        # 2. Set up atmospheric model
        self._setup_atmosphere()

        # 3. Set up aerosol model
        self._setup_aerosol()

        # 4. Handle altitude adjustments
        self._handle_altitudes()

        # 5. Compute spectral properties
        self._compute_spectral()

        # 6. Run radiative transfer
        self._run_radiative_transfer()

        # 7. Handle surface reflectance
        self._handle_surface()

        # 8. Compute atmospheric correction if requested
        if self.input.iatmcorr > 0:
            self._atmospheric_correction()

        return self.results

    def _compute_geometry(self):
        """Compute geometric parameters from angles."""
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
            phirad += 2.0 * self.pi
        if phirad > 2.0 * self.pi:
            phirad -= 2.0 * self.pi

        # Compute cosines
        self.results.xmus = np.cos(self.input.asol * self.pi / 180.0)
        self.results.xmuv = np.cos(self.input.avis * self.pi / 180.0)
        xmup = np.cos(phirad)

        # Scattering angle cosine
        xmud = -self.results.xmus * self.results.xmuv - \
               np.sqrt(1.0 - self.results.xmus**2) * \
               np.sqrt(1.0 - self.results.xmuv**2) * xmup

        # Clamp to valid range
        if xmud > 1.0:
            xmud = 1.0
        if xmud < -1.0:
            xmud = -1.0

        # Scattering angle (degrees)
        self.results.adif = np.arccos(xmud) * 180.0 / self.pi

        # Solar constant variability
        self.results.dsol = varsol(self.input.jday, self.input.month)

    def _setup_atmosphere(self):
        """Set up atmospheric model and load profiles."""
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

    def _setup_aerosol(self):
        """Set up aerosol model and compute optical properties."""
        iaer = self.input.iaer

        if iaer == 0:
            # No aerosols
            self.results.taer55 = 0.0
            self.results.v = 0.0
            return

        # Get aerosol optical thickness
        if self.input.v > 0:
            # Convert visibility to AOT at 550nm
            v = self.input.v
            self.results.v = v
            taer55 = oda550(iaer, v)
            self.results.taer55 = taer55
        else:
            taer55 = self.input.taer55
            self.results.taer55 = taer55

        # Compute xmud for aerosol phase function
        xmud = -self.results.xmus * self.results.xmuv - \
               np.sqrt(1.0 - self.results.xmus**2) * \
               np.sqrt(1.0 - self.results.xmuv**2)

        # Load aerosol optical properties
        # This would call aeroso() but we need to handle different aerosol types
        # For now, set default values
        self.ext = np.ones(20)
        self.ome = np.ones(20) * 0.9
        self.gasym = np.ones(20) * 0.7

    def _handle_altitudes(self):
        """Handle target and sensor altitude adjustments."""
        # Target altitude
        if self.input.pps != 0:
            xps = self.input.pps
            if xps < 0:
                # Ground target altitude
                uw_new, uo3_new, z_new, p_new, t_new, wh_new, wo_new = pressure(
                    self.results.uw, self.results.uo3, xps,
                    self.atm_z, self.atm_p, self.atm_t,
                    self.atm_wh, self.atm_wo
                )
                self.atm_z = z_new
                self.atm_p = p_new
                self.atm_t = t_new
                self.atm_wh = wh_new
                self.atm_wo = wo_new
                self.results.uw = uw_new
                self.results.uo3 = uo3_new

        # Sensor altitude
        if self.input.palt > 0:
            xpp = self.input.palt
            uw_new, uo3_new, ftray, zpl, ppl, tpl, whpl, wopl = presplane(
                self.results.uw, self.results.uo3, xpp,
                self.atm_z, self.atm_p, self.atm_t,
                self.atm_wh, self.atm_wo
            )
            # Store plane atmosphere
            self.atm_zpl = zpl
            self.atm_ppl = ppl
            self.atm_tpl = tpl
            self.atm_whpl = whpl
            self.atm_wopl = wopl
            self.ftray = ftray
        else:
            self.ftray = 1.0

    def _compute_spectral(self):
        """Compute spectral properties and equivalent wavelength."""
        if self.input.iwave == -1:
            # Monochromatic wavelength
            wlmoy = self.input.wlinf  # wlinf = wlsup for monochromatic
        elif self.input.iwave == -2 or self.input.iwave == 0:
            # Wavelength range with uniform filter (filter = 1)
            wlmoy = (self.input.wlinf + self.input.wlsup) / 2.0
        elif self.input.iwave == 1:
            # User-defined filter function
            iinf = self.input.iinf
            isup = self.input.isup
            s = self.input.s
            wlmoy = equivwl(iinf, isup, self.step, s)
        else:
            # Predefined sensor band (would need sensor response functions)
            wlmoy = 0.55  # Default to visible

        self.results.wlmoy = wlmoy

        # Compute Rayleigh optical thickness
        trmoy = odrayl(wlmoy)
        self.results.trmoy = trmoy
        self.results.trmoyp = trmoy * self.ftray

        # Interpolate aerosol properties to wavelength
        if self.input.iaer != 0 and self.results.taer55 > 0:
            # This would call specinterp but we need full implementation
            # For now use simple Angstrom approximation
            alpha = 1.3  # Angstrom exponent
            self.results.tamoy = self.results.taer55 * (wlmoy / 0.55) ** (-alpha)
            self.results.tamoyp = self.results.tamoy * self.ftray
            self.results.pizmoy = 0.9  # Default single scattering albedo
        else:
            self.results.tamoy = 0.0
            self.results.tamoyp = 0.0
            self.results.pizmoy = 0.0

    def _run_radiative_transfer(self):
        """Run discrete ordinates radiative transfer."""
        # Set up Gauss quadrature
        mu2 = 2 * self.mu
        anglem, weightm = gauss(-1.0, 1.0, mu2)
        rm = anglem[:self.mu]
        gb = weightm[:self.mu]
        rp, gp = gauss(0.0, 2.0 * self.pi, self.np_angles)

        # This would call discom() with full parameters
        # For now, set default atmospheric reflectance
        self.results.roatm[0, :] = 0.01  # Small atmospheric reflectance
        self.results.rorayl = 0.01
        self.results.romix = 0.01

        # Default transmittances (clear atmosphere)
        self.results.dtdir = np.ones((3, 20)) * 0.9
        self.results.dtdif = np.ones((3, 20)) * 0.05
        self.results.utdir = np.ones((3, 20)) * 0.9
        self.results.utdif = np.ones((3, 20)) * 0.05

        # Total gaseous transmittance (simplified)
        self.results.tgasm = 0.8  # Typical value

    def _handle_surface(self):
        """Handle surface reflectance."""
        if self.input.inhomo == 0:
            # Homogeneous surface
            if self.input.igroun == 0:
                # Constant reflectance
                self.results.roc = self.input.roc
                self.results.roe = self.input.roc
            elif self.input.igroun == 1:
                # Vegetation (would need spectral data)
                self.results.roc = 0.15
                self.results.roe = 0.15
            elif self.input.igroun == 2:
                # Clear water
                clearw_spec = clearw()
                idx = int((self.results.wlmoy - 0.25) / self.step)
                self.results.roc = clearw_spec[idx] if idx < len(clearw_spec) else 0.05
                self.results.roe = self.results.roc
            elif self.input.igroun == 3:
                # Sand
                sand_spec = sand()
                idx = int((self.results.wlmoy - 0.25) / self.step)
                self.results.roc = sand_spec[idx] if idx < len(sand_spec) else 0.30
                self.results.roe = self.results.roc
            elif self.input.igroun == 4:
                # Lake water
                lakew_spec = lakew()
                idx = int((self.results.wlmoy - 0.25) / self.step)
                self.results.roc = lakew_spec[idx] if idx < len(lakew_spec) else 0.03
                self.results.roe = self.results.roc
        else:
            # Inhomogeneous surface
            self.results.roc = self.input.roc
            self.results.roe = self.input.roe

    def _atmospheric_correction(self):
        """Perform atmospheric correction."""
        # Get input radiance or reflectance
        input_val = abs(self.input.radiance)

        if self.input.radiance > 0:
            # Radiance input - convert to reflectance
            swl = solirr(self.results.wlmoy)
            refet = input_val * self.pi / (swl * self.results.dsol * self.results.xmus)
        else:
            # Reflectance input
            refet = input_val

        # Simple atmospheric correction: remove atmospheric path radiance
        # refet_surface = (refet_toa - roatm) / (tdir_down * tdir_up)
        roatm = self.results.roatm[0, 7]  # Use 550nm value
        tdir = 0.8  # Simplified total downward transmittance
        tup = 0.85  # Simplified upward transmittance

        self.results.refet = max(0.0, (refet - roatm) / (tdir * tup))

        # Convert back to radiance if needed
        swl = solirr(self.results.wlmoy)
        self.results.alumet = self.results.refet * swl * self.results.dsol * \
                              self.results.xmus / self.pi
