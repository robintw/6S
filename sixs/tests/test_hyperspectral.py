"""
Tests for hyperspectral sensor functions.

Validates spectral response data.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.hyperspectral import hypblue


def test_hypblue_modis_band3():
    """Test MODIS band 3 spectral response."""
    s, wlinf, wlsup = hypblue(1)

    # Check array size
    assert len(s) == 1501

    # Check wavelength bounds
    assert wlinf == 0.4375
    assert wlsup == 0.500

    # Check response is normalized (max should be close to 1 or small values)
    # MODIS band 3 has very small response values in the data
    assert np.max(s) <= 1.0

    # Check non-negative
    assert np.all(s >= 0.0)


def test_hypblue_etm_band1():
    """Test ETM+ band 1 spectral response."""
    s, wlinf, wlsup = hypblue(2)

    # Check array size
    assert len(s) == 1501

    # Check wavelength bounds
    assert wlinf == 0.435
    assert wlsup == 0.52

    # ETM+ band has strong response (near 1.0 in blue region)
    assert np.max(s) > 0.9

    # Check non-negative
    assert np.all(s >= 0.0)


def test_hypblue_invalid_band():
    """Test error handling for invalid band."""
    with pytest.raises(ValueError) as exc_info:
        hypblue(3)

    assert "Invalid band selector" in str(exc_info.value)


def test_hypblue_response_structure():
    """Test that response arrays have expected structure."""
    # Both bands should have most values as zero
    # with non-zero response only in the blue region

    for iwa in [1, 2]:
        s, wlinf, wlsup = hypblue(iwa)

        # Count non-zero elements
        nonzero_count = np.count_nonzero(s)

        # Should have some non-zero elements but not all
        assert 0 < nonzero_count < len(s)

        # Most elements should be zero (spectral band is narrow)
        zero_count = np.count_nonzero(s == 0.0)
        assert zero_count > 1400  # Most of 1501 should be zero


def test_hypblue_wavelength_ranges():
    """Test wavelength ranges are in blue region."""
    for iwa in [1, 2]:
        s, wlinf, wlsup = hypblue(iwa)

        # Both bands should be in blue region (roughly 0.4-0.5 μm)
        assert 0.4 <= wlinf <= 0.5
        assert 0.4 <= wlsup <= 0.6

        # Lower bound should be less than upper bound
        assert wlinf < wlsup


def test_hypblue_copy_independence():
    """Test that returned arrays are independent copies."""
    s1, _, _ = hypblue(1)
    s2, _, _ = hypblue(1)

    # Modify one array
    s1[100] = 999.0

    # Other should be unchanged
    assert s2[100] != 999.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
