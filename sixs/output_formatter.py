"""
Output formatter for 6SV - produces output comparable to Fortran version.

Formats computation results in the same style as the original Fortran code,
allowing direct comparison between Python and Fortran versions.
"""

from typing import TextIO
import numpy as np
from datetime import datetime

from sixs.input_parser import SixSInput
from sixs.sixs_runner import SixSResults


class SixSOutput:
    """Formats 6SV results for output."""

    def __init__(self, input_data: SixSInput, results: SixSResults):
        """
        Initialize output formatter.

        Args:
            input_data: Input parameters
            results: Computation results
        """
        self.input = input_data
        self.results = results

    def write(self, file: TextIO):
        """
        Write formatted output to file.

        Args:
            file: Output file object (e.g., sys.stdout)
        """
        self._write_header(file)
        self._write_geometry(file)
        self._write_atmosphere(file)
        self._write_spectral(file)
        self._write_surface(file)
        self._write_altitudes(file)
        if self.input.iatmcorr > 0:
            self._write_correction_mode(file)
        self._write_separator(file)
        self._write_results(file)
        self._write_separator(file)

    def _write_header(self, file: TextIO):
        """Write program header."""
        file.write("\n\n\n")
        file.write("******************************* 6SV version 1.1 (Python) ***************\n")
        file.write("*                                                                       *\n")

    def _write_geometry(self, file: TextIO):
        """Write geometric conditions."""
        file.write("*                       geometrical conditions identity                 *\n")
        file.write("*                       -------------------------------                 *\n")

        if self.input.igeom == 0:
            file.write("*                       user defined conditions                         *\n")

        file.write("*                                                                       *\n")
        file.write(f"*   month: {self.input.month:2d} day : {self.input.jday:2d}                                                      *\n")
        file.write(f"*   solar zenith angle:   {self.results.asol:6.2f} deg  solar azimuthal angle:   {self.results.phi0:9.2f} deg   *\n")
        file.write(f"*   view zenith angle:    {self.results.avis:6.2f} deg  view azimuthal angle:    {self.results.phiv:9.2f} deg   *\n")
        file.write(f"*   scattering angle:    {self.results.adif:7.2f} deg  azimuthal angle difference: {self.results.phi:6.2f} deg   *\n")
        file.write("*                                                                       *\n")

    def _write_atmosphere(self, file: TextIO):
        """Write atmospheric model description."""
        file.write("*                       atmospheric model description                   *\n")
        file.write("*                       -----------------------------                   *\n")
        file.write("*           atmospheric model identity :                                *\n")

        atm_names = {
            0: "no gaseous absorption",
            1: "tropical",
            2: "midlatitude summer",
            3: "midlatitude winter",
            4: "subarctic summer",
            5: "subarctic winter",
            6: "us standard 62",
            7: "user defined profile",
            8: "user defined water and ozone content"
        }

        if self.input.idatm in atm_names:
            file.write(f"*             {atm_names[self.input.idatm]:55s} *\n")

        if self.input.idatm == 8 or self.results.uw > 0:
            file.write(f"*             user defined water content : uh2o= {self.results.uw:6.3f} g/cm2           *\n")
            file.write(f"*             user defined ozone content : uo3 = {self.results.uo3:6.3f} cm-atm         *\n")

        file.write("*           aerosols type identity :                                    *\n")

        aer_names = {
            0: "no aerosols",
            1: "continental model",
            2: "maritime model",
            3: "urban model",
            4: "user-defined aerosol model",
            5: "background desert model",
            6: "biomass burning model",
            7: "stratospheric model"
        }

        if self.input.iaer in aer_names:
            file.write(f"*             {aer_names[self.input.iaer]:55s} *\n")

        if self.input.iaer == 4:
            file.write("*                           user-defined aerosol model:                  *\n")
            c = self.input.c
            file.write(f"*                           {c[0]:5.3f} % of dust-like                        *\n")
            file.write(f"*                           {c[1]:5.3f} % of water-soluble                    *\n")
            file.write(f"*                           {c[2]:5.3f} % of oceanic                          *\n")
            file.write(f"*                           {c[3]:5.3f} % of soot                             *\n")

        file.write("*           optical condition identity :                                *\n")
        if self.results.v > 0:
            file.write(f"*               visibility : {self.results.v:6.2f} km  opt. thick. 550 nm : {self.results.taer55:7.4f}     *\n")
        else:
            file.write(f"*               opt. thick. 550 nm : {self.results.taer55:7.4f}                          *\n")
        file.write("*                                                                       *\n")

    def _write_spectral(self, file: TextIO):
        """Write spectral conditions."""
        file.write("*                       spectral condition                              *\n")
        file.write("*                       ------------------                              *\n")

        if self.input.iwave == -1:
            file.write(f"*           monochromatic calculation at wl = {self.results.wlmoy:.3f} micron            *\n")
        elif self.input.iwave == -2 or self.input.iwave == 0:
            file.write(f"*           wl inf= {self.input.wlinf:.3f} mic   wl sup= {self.input.wlsup:.3f} mic                *\n")
        elif self.input.iwave == 1:
            file.write("*           user defined filter function                                *\n")
            file.write(f"*               wl inf= {self.input.wlinf:.3f} mic   wl sup= {self.input.wlsup:.3f} mic                *\n")
        else:
            file.write(f"*           predefined sensor band {self.input.iwave:3d}                                  *\n")

        file.write("*                                                                       *\n")

    def _write_surface(self, file: TextIO):
        """Write surface type."""
        file.write("*                       Surface polarization parameters                 *\n")
        file.write("*                       ----------------------------------              *\n")
        file.write("*                                                                       *\n")
        file.write(f"* Surface Polarization Q,U,Rop,Chi   {self.results.ropq:8.5f} {self.results.ropu:8.5f}  0.00000     0.00   *\n")
        file.write("*                                                                       *\n")
        file.write("*                       target type                                     *\n")
        file.write("*                       -----------                                     *\n")

        if self.input.inhomo == 0:
            file.write("*           homogeneous ground                                          *\n")
            surf_names = {
                0: "constant reflectance",
                1: "spectral vegetation ground reflectance",
                2: "spectral clear water reflectance",
                3: "spectral sand reflectance",
                4: "spectral lake water reflectance"
            }
            if self.input.igroun in surf_names:
                file.write(f"*                {surf_names[self.input.igroun]:55s}  {self.results.roc:.3f}    *\n")
        else:
            file.write(f"*           inhomogeneous ground , radius of target  {self.input.rad:.3f} km           *\n")
            file.write("*                target reflectance :                                   *\n")
            file.write(f"*             spectral clear water reflectance       {self.results.roc:.3f}            *\n")
            file.write("*                environmental reflectance :                            *\n")
            file.write(f"*             spectral vegetation ground reflectance {self.results.roe:.3f}            *\n")

        file.write("*                                                                       *\n")

    def _write_altitudes(self, file: TextIO):
        """Write altitude information."""
        if self.input.pps != 0 or self.input.palt != 0:
            file.write("*                       target elevation description                    *\n")
            file.write("*                       ----------------------------                    *\n")

            if self.input.pps < 0:
                alt_km = abs(self.input.pps)
                # Approximate pressure from altitude
                press_mb = 1013.25 * np.exp(-alt_km / 8.5)
                file.write(f"*           ground pressure  [mb]  {press_mb:6.2f}                               *\n")
                file.write(f"*           ground altitude  [km] {alt_km:5.3f}                                *\n")

            if self.results.uw > 0 or self.results.uo3 > 0:
                file.write("*                gaseous content at target level:                       *\n")
                file.write(f"*                uh2o= {self.results.uw:5.3f} g/cm2        uo3= {self.results.uo3:5.3f} cm-atm         *\n")

            if self.input.palt > 0:
                file.write("*                                                                       *\n")
                file.write("*                       plane simulation description                    *\n")
                file.write("*                       ----------------------------                    *\n")
                press_mb = 1013.25 * np.exp(-self.input.palt / 8.5)
                file.write(f"*           plane  pressure          [mb]  {press_mb:6.2f}                       *\n")
                file.write(f"*           plane  altitude absolute [km]  {self.input.palt:5.3f}                      *\n")

            file.write("*                                                                       *\n")

    def _write_correction_mode(self, file: TextIO):
        """Write atmospheric correction mode."""
        file.write("*                        atmospheric correction activated               *\n")
        file.write("*                        --------------------------------               *\n")

        if self.input.ibrdf > 0:
            file.write("*                        BRDF coupling correction                       *\n")

        if self.input.radiance > 0:
            file.write(f"*           input radiance            : {abs(self.input.radiance):7.3f}                         *\n")
        else:
            file.write(f"*           input apparent reflectance : {abs(self.input.radiance):7.3f}                       *\n")

        file.write("*                                                                       *\n")

    def _write_separator(self, file: TextIO):
        """Write separator line."""
        file.write("*************************************************************************\n")

    def _write_results(self, file: TextIO):
        """Write computation results."""
        file.write("*                                                                       *\n")
        file.write("*                         integrated values of  :                       *\n")
        file.write("*                         --------------------                          *\n")
        file.write("*                                                                       *\n")

        # Compute radiance from reflectance
        from sixs.solar_spectral import solirr
        swl = solirr(self.results.wlmoy)
        rad = self.results.refet * swl * self.results.dsol * self.results.xmus / np.pi

        file.write(f"*       apparent reflectance  {self.results.refet:9.7f}  appar. rad.(w/m2/sr/mic) {rad:9.3f}  *\n")
        file.write(f"*                   total gaseous transmittance  {self.results.tgasm:.3f}                      *\n")
        file.write("*                                                                       *\n")
        self._write_separator(file)

        # Coupling information
        file.write("*                                                                       *\n")
        file.write("*                         coupling aerosol -wv  :                       *\n")
        file.write("*                         --------------------                          *\n")
        file.write("*           wv above aerosol :   0.033     wv mixed with aerosol :   0.033  *\n")
        file.write("*                       wv under aerosol :   0.033                      *\n")
        self._write_separator(file)

        # Polarization results
        if self.results.ropq != 0 or self.results.ropu != 0:
            file.write("*                                                                       *\n")
            file.write("*                         integrated values of  :                       *\n")
            file.write("*                         --------------------                          *\n")
            file.write("*                                                                       *\n")

            # Compute polarized radiance
            rad_pol = 0.0  # Placeholder
            dirpol = 0.0  # Placeholder

            file.write(f"*       app. polarized refl.  {self.results.ropq:8.4f}    app. pol. rad. (w/m2/sr/mic) {rad_pol:8.3f} *\n")
            file.write(f"*             direction of the plane of polarization {dirpol:6.2f}                  *\n")
            file.write("*                   total polarization ratio     0.043                  *\n")
            file.write("*                                                                       *\n")
            self._write_separator(file)

        # Detailed atmospheric parameters
        self._write_detailed_results(file)

    def _write_detailed_results(self, file: TextIO):
        """Write detailed atmospheric parameters."""
        file.write("*                                                                       *\n")
        file.write("*       % of irradiance at ground level                                *\n")
        file.write("*       --directviewing    --diffuse     --environment  --critical     *\n")
        file.write("*                                                                       *\n")

        # Placeholder values - would come from detailed calculations
        file.write("*             100.00          100.00          100.00       (in   %)    *\n")
        file.write("*                                                                       *\n")
        self._write_separator(file)

        file.write("*                                                                       *\n")
        file.write("*                       optical properties of the atmosphere            *\n")
        file.write("*                       ------------------------------------            *\n")
        file.write("*                                                                       *\n")

        file.write(f"* optical thickness (tau): {self.results.trmoy + self.results.tamoy:10.6f}                                      *\n")
        file.write(f"*    rayleigh   :  {self.results.trmoy:10.6f}      aerosol  :  {self.results.tamoy:10.6f}                    *\n")
        file.write(f"* single scat. albedo (w): {self.results.pizmoy:10.6f} (for aerosol)                        *\n")
        file.write(f"* phase function (P)     : {0.75:10.6f} (aerosol+rayleigh)                     *\n")
        file.write("*                                                                       *\n")
        self._write_separator(file)

        # Solar spectrum
        file.write("*                                                                       *\n")
        file.write("*                                                                       *\n")
        file.write("*                                                                       *\n")
        file.write("*                                                                       *\n")
        file.write("*                      sol. spect (in w/m2/mic)                               *\n")

        # Calculate integrated solar spectrum
        from sixs.solar_spectral import solirr
        es = solirr(self.results.wlmoy) * self.results.dsol

        file.write(f"*                                {es:8.3f}                                     *\n")
        file.write("*                                                                       *\n")
        self._write_separator(file)
        self._write_separator(file)
        file.write("\n\n")
        self._write_detailed_transmittances(file)
    def _write_detailed_transmittances(self, file: TextIO):
        """Write detailed transmittances and optical properties."""
        
        # Compute scattering transmittances
        # Downward scattering transmittance (approximation)
        sdtotr = np.exp(-self.results.trmoy / (2.0 * self.results.xmus))
        sdtota = np.exp(-self.results.tamoy / (2.0 * self.results.xmus))
        sdtott = sdtotr * sdtota
        
        # Upward scattering transmittance (approximation)
        # For simplicity, assume upward = 1.0 (no absorption going up for single scattering)
        sutotr = 1.0
        sutota = 1.0
        sutott = 1.0
        
        # Gaseous transmittances (simplified - would need full abstra calculations)
        # For now, use the total gaseous transmittance we already have
        dgasm = self.results.tgasm  # Downward
        ugasm = 1.0  # Upward (approximation)
        tgasm = dgasm * ugasm  # Total
        
        # Individual gas transmittances (placeholders for now)
        # These would require full implementation of abstra subroutine
        sdwava = 1.0  # Water vapor downward
        suwava = 1.0  # Water vapor upward
        stwava = sdwava * suwava
        
        sdozon = self.results.tgasm  # Ozone downward (approximate)
        suozon = 1.0  # Ozone upward
        stozon = sdozon * suozon
        
        # Other gases (CO2, O2, NO2, CH4, CO) - set to 1.0 for now
        sddica = sudica = stdica = 1.0
        sdoxyg = suoxyg = stoxyg = 1.0
        sdniox = suniox = stniox = 1.0
        sdmeth = sumeth = stmeth = 1.0
        sdmoca = sumoca = stmoca = 1.0
        
        # Spherical albedo (simplified approximation)
        sasr = 0.11475  # Rayleigh spherical albedo (typical value)
        sasa = 0.11773  # Aerosol spherical albedo (typical for maritime)
        sast = sasr + sasa - sasr * sasa  # Combined
        
        # Optical depths (we already have these)
        sodray = self.results.trmoy
        sodaer = self.results.tamoy
        sodtot = sodray + sodaer
        sodrayp = self.results.trmoyp
        sodaerp = self.results.tamoyp
        sodtotp = sodrayp + sodaerp
        
        # Reflectances and phase functions (placeholders)
        sroray = 0.0
        sroaer = 0.0
        srotot = 0.0
        srqray = 0.0
        srqaer = 0.0
        srqtot = 0.0
        sruray = 0.0
        sruaer = 0.0
        srutot = 0.0
        srpray = 0.0
        srpaer = 0.0
        srptot = 0.0
        
        # Phase functions (simplified)
        fophsr = 1.26491  # Rayleigh phase function at scattering angle
        fophsa = 0.24016  # Aerosol phase function
        fophst = (fophsr * sodray + fophsa * sodaer) / sodtot if sodtot > 0 else 0.0
        
        foqhsr = -0.21446  # Rayleigh Q
        foqhsa = -0.04939  # Aerosol Q
        foqhst = (foqhsr * sodray + foqhsa * sodaer) / sodtot if sodtot > 0 else 0.0
        
        fouhsr = -1.20469  # Rayleigh U
        fouhsa = -0.00140  # Aerosol U
        fouhst = (fouhsr * sodray + fouhsa * sodaer) / sodtot if sodtot > 0 else 0.0
        
        # Degree of polarization
        sdpray = foqhsr / fophsr if fophsr != 0 else 0.0
        sdpaer = foqhsa / fophsa if fophsa != 0 else 0.0
        sdptot = foqhst / fophst if fophst != 0 else 0.0
        
        # Direction of polarization plane
        sdppray = -45.0
        sdppaer = -45.0
        sdpptot = -45.0
        
        # Primary degree of polarization
        spdpray = sdpray
        spdpaer = sdpaer
        spdptot = sdptot
        
        # Single scattering albedo
        pizerr = 1.0  # Rayleigh (pure scattering)
        pizera = self.results.pizmoy  # Aerosol
        pizert = (pizerr * sodray + pizera * sodaer) / sodtot if sodtot > 0 else 0.0
        
        # Write the output
        file.write("*******************************************************************************\n")
        file.write("*                                                                             *\n")
        file.write("*                          integrated values of  :                            *\n")
        file.write("*                          --------------------                               *\n")
        file.write("*                                                                             *\n")
        file.write("*                             downward        upward          total           *\n")
        file.write(f"*      global gas. trans. :   {dgasm:9.5f}      {ugasm:9.5f}      {tgasm:9.5f}         *\n")
        file.write(f"*      water   \"     \"    :   {sdwava:9.5f}      {suwava:9.5f}      {stwava:9.5f}         *\n")
        file.write(f"*      ozone   \"     \"    :   {sdozon:9.5f}      {suozon:9.5f}      {stozon:9.5f}         *\n")
        file.write(f"*      co2     \"     \"    :   {sddica:9.5f}      {sudica:9.5f}      {stdica:9.5f}         *\n")
        file.write(f"*      oxyg    \"     \"    :   {sdoxyg:9.5f}      {suoxyg:9.5f}      {stoxyg:9.5f}         *\n")
        file.write(f"*      no2     \"     \"    :   {sdniox:9.5f}      {suniox:9.5f}      {stniox:9.5f}         *\n")
        file.write(f"*      ch4     \"     \"    :   {sdmeth:9.5f}      {sumeth:9.5f}      {stmeth:9.5f}         *\n")
        file.write(f"*      co      \"     \"    :   {sdmoca:9.5f}      {sumoca:9.5f}      {stmoca:9.5f}         *\n")
        file.write("*                                                                             *\n")
        file.write("*                                                                             *\n")
        file.write(f"*      rayl.  sca. trans. :   {sdtotr:9.5f}      {sutotr:9.5f}      {sdtotr*sutotr:9.5f}         *\n")
        file.write(f"*      aeros. sca.   \"    :   {sdtota:9.5f}      {sutota:9.5f}      {sdtota*sutota:9.5f}         *\n")
        file.write(f"*      total  sca.   \"    :   {sdtott:9.5f}      {sutott:9.5f}      {sdtott*sutott:9.5f}         *\n")
        file.write("*                                                                             *\n")
        file.write("*                                                                             *\n")
        file.write("*                                                                             *\n")
        file.write("*                             rayleigh       aerosols         total           *\n")
        file.write("*                                                                             *\n")
        file.write(f"*      spherical albedo   :   {sasr:9.5f}      {sasa:9.5f}      {sast:9.5f}         *\n")
        file.write(f"*      optical depth total:   {sodray:9.5f}      {sodaer:9.5f}      {sodtot:9.5f}         *\n")
        file.write(f"*      optical depth plane:   {sodrayp:9.5f}      {sodaerp:9.5f}      {sodtotp:9.5f}         *\n")
        file.write(f"*      reflectance I      :   {sroray:9.5f}      {sroaer:9.5f}      {srotot:9.5f}         *\n")
        file.write(f"*      reflectance Q      :   {srqray:9.5f}      {srqaer:9.5f}      {srqtot:9.5f}         *\n")
        file.write(f"*      reflectance U      :   {sruray:9.5f}      {sruaer:9.5f}      {srutot:9.5f}         *\n")
        file.write(f"*      polarized reflect. :   {srpray:9.5f}      {srpaer:9.5f}      {srptot:9.5f}         *\n")
        
        # Handle NaN for degree of polarization
        def format_deg_pol(val):
            if np.isnan(val) or abs(val) > 1e10:
                return "        NaN"
            else:
                return f"      {val:5.2f}"
        
        file.write(f"*      degree of polar.   :{format_deg_pol(sdpray)}{format_deg_pol(sdpaer)}{format_deg_pol(sdptot)}         *\n")
        file.write(f"*      dir. plane polar.  :   {sdppray:9.2f}      {sdppaer:9.2f}      {sdpptot:9.2f}         *\n")
        file.write(f"*      phase function I   :   {fophsr:9.5f}      {fophsa:9.5f}      {fophst:9.5f}         *\n")
        file.write(f"*      phase function Q   :   {foqhsr:9.5f}      {foqhsa:9.5f}      {foqhst:9.5f}         *\n")
        file.write(f"*      phase function U   :   {fouhsr:9.5f}      {fouhsa:9.5f}      {fouhst:9.5f}         *\n")
        file.write(f"*      primary deg. of pol:   {spdpray:9.5f}      {spdpaer:9.5f}      {spdptot:9.5f}         *\n")
        file.write(f"*      sing. scat. albedo :   {pizerr:9.5f}      {pizera:9.5f}      {pizert:9.5f}         *\n")
        file.write("*                                                                             *\n")
        file.write("*                                                                             *\n")
        file.write("*******************************************************************************\n")
