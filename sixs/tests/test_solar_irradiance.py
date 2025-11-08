"""
Tests for solar irradiance functions.

Validates solar spectral irradiance and equivalent wavelength calculations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.solar_irradiance import solirr, equivwl


def test_solirr_visible_range():
    """Test solar irradiance in visible spectrum."""
    # Test at 0.5 μm (green light)
    irr = solirr(0.5)

    # Should be high in visible range (1000-2000 W/m²/μm)
    assert 1000 < irr < 2200
    assert np.isfinite(irr)


def test_solirr_uv():
    """Test solar irradiance in UV range."""
    # At 0.25 μm (UV boundary)
    irr_uv = solirr(0.25)

    # Should be lower than visible
    irr_vis = solirr(0.5)
    assert irr_uv < irr_vis


def test_solirr_ir():
    """Test solar irradiance in infrared."""
    # At 2.0 μm (near-IR)
    irr = solirr(2.0)

    # Should be positive but much lower than visible
    assert 0 < irr < 150
    assert np.isfinite(irr)


def test_solirr_out_of_range():
    """Test solar irradiance boundary handling."""
    # Below range (< 0.25 μm)
    irr_low = solirr(0.20)
    irr_min = solirr(0.25)

    # Should return boundary value
    assert irr_low == irr_min

    # Above range (> 4.0 μm)
    irr_high = solirr(5.0)

    # Should return boundary value (finite and positive)
    assert irr_high > 0
    assert np.isfinite(irr_high)


def test_solirr_consistency():
    """Test that irradiance decreases in general from visible to IR."""
    irr_blue = solirr(0.45)
    irr_green = solirr(0.55)
    irr_red = solirr(0.65)
    irr_nir = solirr(1.0)
    irr_swir = solirr(2.0)

    # All should be positive
    assert all([irr_blue > 0, irr_green > 0, irr_red > 0, irr_nir > 0, irr_swir > 0])

    # Near-IR should be less than visible
    assert irr_nir < irr_green

    # SWIR should be much less than visible
    assert irr_swir < irr_green / 10


def test_solirr_peak():
    """Test that solar irradiance peaks near visible range."""
    wavelengths = np.linspace(0.3, 2.0, 50)
    irradiances = np.array([solirr(wl) for wl in wavelengths])

    # Peak should be in visible range (0.4-0.7 μm)
    peak_idx = np.argmax(irradiances)
    peak_wl = wavelengths[peak_idx]

    # Solar peak is around 0.5 μm
    assert 0.4 < peak_wl < 0.7


def test_equivwl_monochromatic():
    """Test equivalent wavelength for narrow band."""
    # Create narrow Gaussian-like response
    s = np.zeros(1501)
    center_idx = 500  # Around 0.5 μm
    s[center_idx - 5:center_idx + 5] = 1.0

    wlinf = 0.25
    wlsup = 1.5

    wl_equiv = equivwl(s, wlinf, wlsup)

    # Should be near the center wavelength
    expected_wl = 0.25 + center_idx * 0.0025
    assert abs(wl_equiv - expected_wl) < 0.05


def test_equivwl_uniform():
    """Test equivalent wavelength for uniform response."""
    # Uniform response across band
    s = np.ones(1501)

    wlinf = 0.4
    wlsup = 0.7

    wl_equiv = equivwl(s, wlinf, wlsup)

    # For uniform response with solar weighting,
    # should be close to midpoint (slightly weighted to higher irradiance region)
    midpoint = (wlinf + wlsup) / 2.0

    # Should be within reasonable range of midpoint
    assert abs(wl_equiv - midpoint) < 0.1


def test_equivwl_bounds():
    """Test that equivalent wavelength is within band limits."""
    # Random response
    s = np.random.rand(1501) * 0.5 + 0.5

    wlinf = 0.5
    wlsup = 0.9

    wl_equiv = equivwl(s, wlinf, wlsup)

    # Must be within bounds
    assert wlinf <= wl_equiv <= wlsup


def test_equivwl_blue_vs_red():
    """Test that blue and red bands give appropriate equivalent wavelengths."""
    s = np.zeros(1501)

    # Blue band
    s[80:140] = 1.0  # Around 0.45-0.6 μm
    wl_blue = equivwl(s, 0.4, 0.7)

    # Red band
    s[:] = 0
    s[180:240] = 1.0  # Around 0.7-0.85 μm
    wl_red = equivwl(s, 0.6, 1.0)

    # Red should be longer wavelength than blue
    assert wl_red > wl_blue


def test_equivwl_zero_response():
    """Test equivalent wavelength when response is zero."""
    s = np.zeros(1501)

    wlinf = 0.5
    wlsup = 0.7

    wl_equiv = equivwl(s, wlinf, wlsup)

    # Should return midpoint when no valid data
    midpoint = (wlinf + wlsup) / 2.0
    assert wl_equiv == midpoint


def test_equivwl_realistic_sensor():
    """Test with realistic sensor-like spectral response."""
    # Create a triangular response (simplified sensor response)
    s = np.zeros(1501)
    center = 200  # At 0.75 μm: 0.25 + 200*0.0025 = 0.75
    width = 50

    for i in range(center - width, center + width):
        if 0 <= i < len(s):
            distance = abs(i - center)
            s[i] = 1.0 - distance / width

    wlinf = 0.6
    wlsup = 1.0

    wl_equiv = equivwl(s, wlinf, wlsup)

    # Should be close to center wavelength
    expected = 0.25 + center * 0.0025
    assert abs(wl_equiv - expected) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
