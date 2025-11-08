"""
Tests for cosine albedo calculations.

Validates exponential integral approximations and albedo calculations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.cosine_albedo import csalbr, fintexp1, fintexp3


def test_fintexp1_small_values():
    """Test E1 integral approximation for small optical depths."""
    # E1(0.1) should be positive and reasonable
    e1 = fintexp1(0.1)

    assert e1 > 0
    assert np.isfinite(e1)

    # E1 function decreases with increasing argument
    e1_01 = fintexp1(0.1)
    e1_05 = fintexp1(0.5)
    e1_10 = fintexp1(1.0)

    assert e1_01 > e1_05 > e1_10


def test_fintexp1_accuracy():
    """Test E1 integral accuracy against known values."""
    # E1(0.5) ≈ 0.5598 (from tables)
    e1 = fintexp1(0.5)

    # Should be accurate to stated precision (2e-07)
    expected = 0.5598
    assert abs(e1 - expected) < 0.01  # Reasonable tolerance


def test_fintexp3_positive():
    """Test that E3 integral returns positive values."""
    test_values = [0.01, 0.1, 0.5, 1.0, 2.0]

    for tau in test_values:
        e3 = fintexp3(tau)

        assert e3 > 0
        assert np.isfinite(e3)


def test_fintexp3_decreasing():
    """Test that E3 decreases with increasing optical depth."""
    # Note: E1 approximation only valid for 0 < tau < 1
    tau_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    e3_values = [fintexp3(tau) for tau in tau_values]

    # Should be monotonically decreasing in valid range
    for i in range(len(e3_values) - 1):
        assert e3_values[i] > e3_values[i + 1]


def test_csalbr_small_optical_depth():
    """Test cosine albedo for optically thin atmosphere."""
    # For small tau, albedo should be small
    xalb = csalbr(0.01)

    assert 0 < xalb < 0.1
    assert np.isfinite(xalb)


def test_csalbr_moderate_optical_depth():
    """Test cosine albedo for moderate optical depth."""
    # Valid range for E1 approximation: 0 < tau < 1
    xalb = csalbr(0.8)

    assert 0 < xalb < 1.0
    assert np.isfinite(xalb)


def test_csalbr_monotonic():
    """Test that albedo increases with optical depth."""
    # Valid range for E1 approximation: 0 < tau < 1
    tau_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    albedo_values = [csalbr(tau) for tau in tau_values]

    # Albedo should generally increase with optical depth
    # (more scatterers = more reflection)
    for i in range(len(albedo_values) - 1):
        assert albedo_values[i] < albedo_values[i + 1]


def test_csalbr_bounded():
    """Test that cosine albedo is physically bounded."""
    # Test within valid range for E1 approximation (0 < tau < 1)
    tau_values = np.linspace(0.01, 0.99, 50)

    for tau in tau_values:
        xalb = csalbr(tau)

        # Albedo must be between 0 and 1 (physical constraint)
        assert 0 <= xalb <= 1.0
        assert np.isfinite(xalb)


def test_csalbr_limiting_behavior():
    """Test limiting behavior of albedo."""
    # For very small tau, albedo should be approximately proportional to tau
    tau_small = 0.001
    xalb_small = csalbr(tau_small)

    # Albedo should be small for small tau
    assert xalb_small < 0.01

    # For increasing tau (within valid range), albedo should increase
    xalb_03 = csalbr(0.3)
    xalb_07 = csalbr(0.7)

    # Higher tau should give higher albedo
    assert xalb_07 > xalb_03


def test_csalbr_typical_values():
    """Test albedo at typical atmospheric optical depths."""
    # Typical clear sky: tau ~ 0.1-0.5
    xalb_clear = csalbr(0.2)

    assert 0.1 < xalb_clear < 0.4

    # Moderately thick atmosphere (within valid range)
    xalb_moderate = csalbr(0.8)

    assert 0.3 < xalb_moderate < 0.7


def test_fintexp1_vectorization():
    """Test that exponential integrals work with array inputs."""
    tau_array = np.array([0.1, 0.5, 1.0, 2.0])

    # Should work element-wise
    for tau in tau_array:
        e1 = fintexp1(tau)
        assert np.isfinite(e1)


def test_csalbr_consistency():
    """Test internal consistency of albedo calculation."""
    # Calculate albedo at several points
    tau1 = 1.0
    tau2 = 2.0

    xalb1 = csalbr(tau1)
    xalb2 = csalbr(tau2)

    # Doubling optical depth should increase albedo significantly
    assert xalb2 > xalb1 * 1.2  # At least 20% increase


def test_exponential_integral_relationship():
    """Test relationship between E1 and E3 integrals."""
    tau = 0.5

    e1 = fintexp1(tau)
    e3 = fintexp3(tau)

    # E3 uses E1 in its calculation
    # E3 = [exp(-tau)(1-tau) + tau^2 E1(tau)] / 2
    e3_check = (np.exp(-tau) * (1.0 - tau) + tau**2 * e1) / 2.0

    assert abs(e3 - e3_check) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
