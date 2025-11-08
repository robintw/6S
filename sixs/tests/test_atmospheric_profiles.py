"""
Tests for atmospheric profiles.

Validates that atmospheric profile data matches Fortran source.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.atmospheric_profiles import (
    tropical, midlatitude_summer, midlatitude_winter,
    subarctic_summer, subarctic_winter,
    us_standard_1962, get_profile
)


def test_tropical_profile():
    """Test tropical atmosphere profile structure."""
    prof = tropical()

    assert prof.name == "Tropical"
    assert len(prof.z) == 34
    assert len(prof.p) == 34
    assert len(prof.t) == 34
    assert len(prof.wh) == 34
    assert len(prof.wo) == 34

    # Test first level (surface)
    assert prof.z[0] == 0.0
    assert prof.p[0] == 1.013e+03  # Sea level pressure
    assert prof.t[0] == 3.000e+02  # 300 K (27°C - tropical)

    # Test physical constraints (excluding top-of-atmosphere sentinel)
    assert np.all(prof.p[:-1] > 0)  # Pressure positive (except last = 0)
    assert np.all(prof.t > 0)  # Temperature positive
    assert np.all(prof.wh >= 0)  # Water vapor non-negative
    assert np.all(prof.wo >= 0)  # Ozone non-negative

    # Test monotonic decrease
    assert np.all(np.diff(prof.p[:-2]) < 0)  # Pressure decreases with altitude


def test_subarctic_summer_profile():
    """Test subarctic summer profile."""
    prof = subarctic_summer()

    assert prof.name == "Subarctic Summer"
    assert len(prof.z) == 34
    assert prof.p[0] == 1.010e+03
    assert 286 < prof.t[0] < 288  # ~14°C typical for subarctic summer

    # Physical constraints
    assert np.all(prof.p[:-1] > 0)
    assert np.all(prof.t > 0)
    assert np.all(prof.wh >= 0)
    assert np.all(prof.wo >= 0)


def test_subarctic_winter_profile():
    """Test subarctic winter profile."""
    prof = subarctic_winter()

    assert prof.name == "Subarctic Winter"
    assert len(prof.z) == 34
    assert prof.p[0] == 1.013e+03
    assert 256 < prof.t[0] < 258  # ~-16°C typical for subarctic winter

    # Physical constraints
    assert np.all(prof.p[:-1] > 0)
    assert np.all(prof.t > 0)
    assert np.all(prof.wh >= 0)
    assert np.all(prof.wo >= 0)


def test_us_standard_profile():
    """Test US Standard 1962 profile."""
    prof = us_standard_1962()

    assert prof.name == "US Standard 1962"
    assert prof.p[0] == 1.013e+03  # Standard sea level
    assert 287 < prof.t[0] < 289  # ~15°C typical


def test_get_profile():
    """Test profile retrieval by name."""
    prof = get_profile('tropical')
    assert prof.name == "Tropical"

    prof = get_profile('SUBARCTIC_SUMMER')  # Case insensitive
    assert prof.name == "Subarctic Summer"

    prof = get_profile('subarctic_winter')
    assert prof.name == "Subarctic Winter"

    prof = get_profile('US_STANDARD_1962')  # Case insensitive
    assert prof.name == "US Standard 1962"

    with pytest.raises(ValueError):
        get_profile('invalid_profile_name')


def test_profile_completeness():
    """Test all 6 profiles have complete data."""
    profiles = [
        tropical(),
        midlatitude_summer(),
        midlatitude_winter(),
        subarctic_summer(),
        subarctic_winter(),
        us_standard_1962()
    ]

    for prof in profiles:
        # Check no NaN values
        assert not np.any(np.isnan(prof.z))
        assert not np.any(np.isnan(prof.p))
        assert not np.any(np.isnan(prof.t))
        assert not np.any(np.isnan(prof.wh))
        assert not np.any(np.isnan(prof.wo))

        # Check reasonable ranges (last element is sentinel value 99999)
        assert np.all(prof.z[:-1] >= 0)
        assert np.all(prof.z[:-1] < 110)  # Below 110 km (exclude sentinel)
        assert np.all(prof.p >= 0)
        assert np.all(prof.t >= 180)  # Above 180 K
        assert np.all(prof.t <= 320)  # Below 320 K


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
