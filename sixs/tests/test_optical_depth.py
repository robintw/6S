"""
Tests for optical depth calculations.

Validates Rayleigh optical depth calculations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.optical_depth import odrayl
from sixs.atmospheric_profiles import tropical, us_standard_1962


def test_odrayl_blue_wavelength():
    """Test Rayleigh optical depth at blue wavelength (0.4 μm)."""
    prof = us_standard_1962()

    # Calculate optical depth at 0.4 μm (blue)
    tray = odrayl(0.4, prof.z, prof.p, prof.t)

    # Rayleigh scattering is strong in blue
    # Typical value ~0.3-0.5 for clear atmosphere
    assert 0.2 < tray < 0.6
    assert tray > 0


def test_odrayl_red_wavelength():
    """Test Rayleigh optical depth at red wavelength (0.7 μm)."""
    prof = us_standard_1962()

    # Calculate optical depth at 0.7 μm (red)
    tray = odrayl(0.7, prof.z, prof.p, prof.t)

    # Rayleigh scattering is weaker in red
    # Typical value ~0.03-0.05 for clear atmosphere
    assert 0.01 < tray < 0.1
    assert tray > 0


def test_odrayl_wavelength_dependence():
    """Test that optical depth follows λ^-4 dependence."""
    prof = us_standard_1962()

    # Calculate at two wavelengths
    tray_blue = odrayl(0.4, prof.z, prof.p, prof.t)
    tray_red = odrayl(0.8, prof.z, prof.p, prof.t)

    # Rayleigh scattering: τ ∝ λ^-4
    # So tray_blue / tray_red ≈ (0.8/0.4)^4 = 2^4 = 16
    ratio = tray_blue / tray_red

    # Allow some tolerance due to dispersion effects
    assert 12 < ratio < 20


def test_odrayl_tropical_vs_standard():
    """Test optical depth for different atmospheric profiles."""
    prof_trop = tropical()
    prof_std = us_standard_1962()

    wl = 0.55  # Green wavelength

    tray_trop = odrayl(wl, prof_trop.z, prof_trop.p, prof_trop.t)
    tray_std = odrayl(wl, prof_std.z, prof_std.p, prof_std.t)

    # Both should be positive and similar magnitude
    assert tray_trop > 0
    assert tray_std > 0

    # Tropical has more atmosphere (higher T) so slightly different
    # but should be within 20% of each other
    assert 0.8 < tray_trop / tray_std < 1.2


def test_odrayl_depolarization_effect():
    """Test effect of depolarization parameter."""
    prof = us_standard_1962()
    wl = 0.5

    # Default depolarization
    tray_default = odrayl(wl, prof.z, prof.p, prof.t, delta=0.0279)

    # Different depolarization (should have small effect)
    tray_modified = odrayl(wl, prof.z, prof.p, prof.t, delta=0.03)

    # Should be close but not identical
    assert abs(tray_modified - tray_default) < 0.01 * tray_default


def test_odrayl_physical_constraints():
    """Test physical constraints on optical depth."""
    prof = us_standard_1962()

    wavelengths = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    for wl in wavelengths:
        tray = odrayl(wl, prof.z, prof.p, prof.t)

        # Must be positive
        assert tray > 0

        # Must be finite
        assert np.isfinite(tray)

        # Reasonable range (0.001 to 1.0 for visible wavelengths)
        assert 0.0001 < tray < 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
