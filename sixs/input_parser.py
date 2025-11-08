"""
Input parser for 6SV - reads the same input format as the Fortran version.

The input file format follows the structure defined in main.f:
1. Geometrical conditions (igeom)
2. Atmospheric model (idatm)
3. Aerosol model (iaer)
4. Spectral conditions (iwave)
5. Surface properties (ihomo/ibrdf)
6. Altitude/sensor parameters
7. Atmospheric correction mode
"""

from typing import TextIO, Dict, Any, List, Tuple, Optional
import numpy as np


class SixSInput:
    """Parses and stores 6SV input parameters."""

    def __init__(self):
        # Geometric conditions
        self.igeom = 0
        self.asol = 0.0  # Solar zenith angle (deg)
        self.phi0 = 0.0  # Solar azimuth angle (deg)
        self.avis = 0.0  # View zenith angle (deg)
        self.phiv = 0.0  # View azimuth angle (deg)
        self.month = 1
        self.jday = 1

        # Atmospheric model
        self.idatm = 0
        self.uw = 0.0  # Water vapor (g/cm²)
        self.uo3 = 0.0  # Ozone (cm-atm)
        self.z = np.zeros(34)  # Altitude (km)
        self.p = np.zeros(34)  # Pressure (mb)
        self.t = np.zeros(34)  # Temperature (K)
        self.wh = np.zeros(34)  # H2O density (g/m³)
        self.wo = np.zeros(34)  # O3 density (g/m³)

        # Aerosol model
        self.iaer = 0
        self.iaer_prof = 0
        self.c = np.zeros(4)  # Component fractions for iaer=4
        self.v = 0.0  # Visibility (km)
        self.taer55 = 0.0  # Aerosol optical thickness at 550nm
        self.num_z = 0
        self.alt_z = []
        self.taer55_z = []

        # Spectral conditions
        self.iwave = 0
        self.iinf = 1
        self.isup = 1501
        self.wl = 0.55  # Wavelength (µm)
        self.s = np.zeros(1501)  # Spectral response

        # Surface properties
        self.inhomo = 0  # 0=homogeneous, 1=inhomogeneous
        self.idirec = 0  # Directional effects
        self.ibrdf = 0  # BRDF model
        self.igroun = 0  # Ground reflectance type
        self.roc = 0.0  # Target reflectance
        self.roe = 0.0  # Environment reflectance
        self.rad = 0.0  # Target radius (km)

        # Altitude parameters
        self.idatmp = 0  # Target altitude mode
        self.pps = 0.0  # Target pressure/altitude
        self.palt = 0.0  # Sensor altitude (km)
        self.ftray = 1.0  # Rayleigh correction factor

        # Atmospheric correction
        self.iatmcorr = 0  # Atmospheric correction mode
        self.radiance = 0.0  # Input radiance/reflectance

        # Polarization
        self.ipol = 1  # Always enabled in 6SV 1.1

    @classmethod
    def from_file(cls, file: TextIO) -> 'SixSInput':
        """
        Parse 6SV input from a file object.

        Args:
            file: Input file object (e.g., sys.stdin)

        Returns:
            SixSInput object with parsed parameters
        """
        self = cls()
        lines = self._read_lines(file)
        line_idx = 0

        # 1. Geometrical conditions
        self.igeom = self._parse_int(lines[line_idx])
        line_idx += 1

        if self.igeom == 0:
            # User-defined angles
            vals = self._parse_floats(lines[line_idx])
            self.asol, self.phi0, self.avis, self.phiv = vals[:4]
            self.month, self.jday = int(vals[4]), int(vals[5])
            line_idx += 1
        elif 1 <= self.igeom <= 7:
            # Satellite-specific geometry (would need position calculations)
            raise NotImplementedError(f"Satellite geometry mode {self.igeom} not yet implemented")

        # 2. Atmospheric model
        self.idatm = self._parse_int(lines[line_idx])
        line_idx += 1

        if self.idatm == 8:
            # User-defined water vapor and ozone
            vals = self._parse_floats(lines[line_idx])
            self.uw, self.uo3 = vals[:2]
            line_idx += 1
        elif self.idatm == 7:
            # User-defined profile (34 levels)
            for k in range(34):
                vals = self._parse_floats(lines[line_idx])
                self.z[k], self.p[k], self.t[k], self.wh[k], self.wo[k] = vals[:5]
                line_idx += 1

        # 3. Aerosol model
        self.iaer = self._parse_int(lines[line_idx])
        line_idx += 1

        if self.iaer == 4:
            # User-defined component mix
            vals = self._parse_floats(lines[line_idx])
            self.c[:] = vals[:4]
            line_idx += 1
        elif self.iaer < 0:
            # User-defined aerosol profile
            self.iaer_prof = 1
            self.num_z = self._parse_int(lines[line_idx])
            line_idx += 1
            for i in range(self.num_z):
                vals = self._parse_floats(lines[line_idx])
                height, tau55, iaer_type = vals[0], vals[1], int(vals[2])
                self.alt_z.append(height)
                self.taer55_z.append(tau55)
                self.iaer = iaer_type  # Use type from last layer
                line_idx += 1
            self.taer55 = sum(self.taer55_z)
        elif self.iaer in [8, 9, 10, 11]:
            # Custom size distributions (not yet fully implemented)
            raise NotImplementedError(f"Aerosol model {self.iaer} not yet implemented")

        # 4. Aerosol optical thickness or visibility
        vals = self._parse_floats(lines[line_idx])
        if vals[0] < 0:
            self.v = abs(vals[0])  # Visibility in km
        else:
            self.taer55 = vals[0]  # AOT at 550nm
        line_idx += 1

        # 5. Target altitude
        self.pps = self._parse_float(lines[line_idx])
        line_idx += 1

        # 6. Sensor altitude (Fortran negates this: xpp=-xpp)
        self.palt = -self._parse_float(lines[line_idx])
        line_idx += 1

        # 7. Water vapor and ozone at sensor (if needed)
        if self.palt > 0 or self.pps != 0:
            vals = self._parse_floats(lines[line_idx])
            # Handle water vapor and ozone updates
            line_idx += 1

        # 8. AOT at sensor altitude (if above target)
        if self.palt > 0:
            vals = self._parse_floats(lines[line_idx])
            # Handle AOT at sensor
            line_idx += 1

        # 9. Spectral conditions
        self.iwave = self._parse_int(lines[line_idx])
        line_idx += 1

        if self.iwave == -1:
            # User-defined filter function
            vals = self._parse_floats(lines[line_idx])
            self.iinf, self.isup = int(vals[0]), int(vals[1])
            line_idx += 1
            # Read spectral response
            vals = self._parse_floats(lines[line_idx])
            self.s[self.iinf:self.isup+1] = vals[:(self.isup-self.iinf+1)]
            line_idx += 1

        # 10. Surface properties
        self.inhomo = self._parse_int(lines[line_idx])
        line_idx += 1

        if self.inhomo == 0:
            # Homogeneous surface
            self.roc = self._parse_float(lines[line_idx])
            line_idx += 1
        else:
            # Inhomogeneous surface
            vals = self._parse_floats(lines[line_idx])
            self.roc, self.roe, self.rad = vals[:3]
            line_idx += 1

        # 11. BRDF model
        self.ibrdf = self._parse_int(lines[line_idx])
        line_idx += 1

        # 12. Atmospheric correction mode
        vals = self._parse_floats(lines[line_idx])
        self.radiance = vals[0]
        if vals[0] > 0:
            self.iatmcorr = 1  # Radiance input
        elif vals[0] < 0:
            self.iatmcorr = 2  # Reflectance input
        line_idx += 1

        return self

    @staticmethod
    def _read_lines(file: TextIO) -> List[str]:
        """Read and clean input lines, removing comments."""
        lines = []
        for line in file:
            # Remove comments (anything after certain patterns)
            line = line.split('(')[0].strip()  # Remove text in parentheses
            if line and not line.startswith('#'):
                lines.append(line)
        return lines

    @staticmethod
    def _parse_int(line: str) -> int:
        """Parse the first integer from a line."""
        return int(line.split()[0])

    @staticmethod
    def _parse_float(line: str) -> float:
        """Parse the first float from a line."""
        return float(line.split()[0])

    @staticmethod
    def _parse_floats(line: str) -> List[float]:
        """Parse all floats from a line."""
        return [float(x) for x in line.split()]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"SixSInput(igeom={self.igeom}, idatm={self.idatm}, iaer={self.iaer}, "
                f"asol={self.asol:.2f}, avis={self.avis:.2f}, taer55={self.taer55:.4f})")
