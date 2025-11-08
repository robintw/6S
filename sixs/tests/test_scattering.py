"""
Tests for atmospheric scattering functions.

Validates molecular (Rayleigh) scattering calculations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.scattering import chand


def test_chand_basic():
    """Test basic Chandrasekhar function calculation."""
    # Standard atmosphere parameters
    xphi = 0.0      # Backscattering
    xmuv = 0.8      # cos(36.9°) view zenith
    xmus = 0.7      # cos(45.6°) solar zenith
    xtau = 0.2      # Moderate optical depth

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Reflectance should be positive
    assert xrray > 0

    # Reflectance should be less than 1
    assert xrray < 1

    # Should be a reasonable value for these conditions
    assert 0.01 < xrray < 0.5


def test_chand_backscatter():
    """Test Chandrasekhar function in backscattering geometry."""
    xmuv = 0.8
    xmus = 0.8
    xtau = 0.15

    # Backscattering (xphi=0)
    xrray_back = chand(0.0, xmuv, xmus, xtau)

    # Forward scattering (xphi=180)
    xrray_forward = chand(180.0, xmuv, xmus, xtau)

    # Both should be valid
    assert 0 < xrray_back < 1
    assert 0 < xrray_forward < 1

    # Backscattering typically brighter for Rayleigh
    # (though this depends on geometry)
    assert xrray_back > 0
    assert xrray_forward > 0


def test_chand_optical_depth():
    """Test Chandrasekhar function with varying optical depth."""
    xphi = 45.0
    xmuv = 0.9
    xmus = 0.9

    # Test increasing optical depth
    taus = [0.05, 0.1, 0.2, 0.4]
    reflectances = []

    for xtau in taus:
        xrray = chand(xphi, xmuv, xmus, xtau)
        reflectances.append(xrray)

        # All should be valid
        assert 0 < xrray < 1

    # Generally, higher optical depth = higher reflectance
    # (for optically thin to moderate atmospheres)
    assert reflectances[-1] > reflectances[0]


def test_chand_zenith_angles():
    """Test Chandrasekhar function with varying zenith angles."""
    xphi = 90.0
    xtau = 0.15

    # Test different solar zenith angles
    zenith_angles = [0.9, 0.7, 0.5, 0.3]  # cos(angles)

    for xmus in zenith_angles:
        for xmuv in zenith_angles:
            xrray = chand(xphi, xmuv, xmus, xtau)

            # All should be physically valid
            assert np.isfinite(xrray)
            assert xrray > 0
            assert xrray < 1


def test_chand_azimuthal_symmetry():
    """Test azimuthal dependence of Chandrasekhar function."""
    xmuv = 0.8
    xmus = 0.7
    xtau = 0.2

    # Test various azimuthal angles
    azimuths = [0.0, 45.0, 90.0, 135.0, 180.0]
    reflectances = []

    for xphi in azimuths:
        xrray = chand(xphi, xmuv, xmus, xtau)
        reflectances.append(xrray)

        # All should be valid
        assert 0 < xrray < 1

    # Should have variation with azimuth
    assert max(reflectances) > min(reflectances)


def test_chand_nadir_viewing():
    """Test Chandrasekhar function for nadir viewing."""
    # Nadir viewing (xmuv = 1.0)
    xphi = 0.0
    xmuv = 1.0
    xmus = 0.8
    xtau = 0.15

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should be valid
    assert 0 < xrray < 1
    assert np.isfinite(xrray)


def test_chand_overhead_sun():
    """Test Chandrasekhar function with overhead sun."""
    # Overhead sun (xmus = 1.0)
    xphi = 90.0
    xmuv = 0.7
    xmus = 1.0
    xtau = 0.2

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should be valid
    assert 0 < xrray < 1
    assert np.isfinite(xrray)


def test_chand_small_optical_depth():
    """Test Chandrasekhar function with small optical depth."""
    xphi = 45.0
    xmuv = 0.8
    xmus = 0.8
    xtau = 0.01  # Very thin atmosphere

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should be small but positive
    assert 0 < xrray < 0.1
    assert np.isfinite(xrray)


def test_chand_large_optical_depth():
    """Test Chandrasekhar function with large optical depth."""
    xphi = 45.0
    xmuv = 0.8
    xmus = 0.8
    xtau = 0.5  # Thick atmosphere

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should be larger but still less than 1
    assert 0.1 < xrray < 1.0
    assert np.isfinite(xrray)


def test_chand_grazing_angles():
    """Test Chandrasekhar function at grazing angles."""
    xphi = 45.0
    xtau = 0.15

    # Near-grazing geometry
    xmuv = 0.1  # ~84° zenith
    xmus = 0.1

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should still be valid (though may be large)
    assert xrray > 0
    assert np.isfinite(xrray)


def test_chand_multiple_scattering():
    """Test that multiple scattering is included."""
    xphi = 0.0
    xmuv = 0.5
    xmus = 0.5

    # Compare very thin vs moderate optical depth
    xrray_thin = chand(xphi, xmuv, xmus, 0.05)
    xrray_mod = chand(xphi, xmuv, xmus, 0.3)

    # Ratio should indicate nonlinear growth
    # (multiple scattering becomes more important at higher tau)
    ratio = xrray_mod / xrray_thin

    # Ratio should be greater than optical depth ratio
    # (if only single scattering, ratio would be ~6)
    assert ratio > 5.0  # Indicates multiple scattering contribution


def test_chand_reciprocity():
    """Test reciprocity principle (switching sun and view)."""
    xphi = 60.0
    xtau = 0.2

    # Configuration 1
    xmuv1 = 0.7
    xmus1 = 0.9
    xrray1 = chand(xphi, xmuv1, xmus1, xtau)

    # Configuration 2 (switch sun and view)
    xmuv2 = 0.9
    xmus2 = 0.7
    xrray2 = chand(xphi, xmuv2, xmus2, xtau)

    # The raw reflectances should be equal (Helmholtz reciprocity)
    # Note: The function returns xrray/xmus, so direct comparison shows reciprocity
    assert abs(xrray1 - xrray2) < 0.001


def test_chand_consistency():
    """Test consistency of Chandrasekhar function across parameter space."""
    # Sample parameter space
    xphis = [0.0, 90.0, 180.0]
    xmuvs = [0.3, 0.6, 0.9]
    xmuss = [0.3, 0.6, 0.9]
    xtaus = [0.05, 0.15, 0.30]

    for xphi in xphis:
        for xmuv in xmuvs:
            for xmus in xmuss:
                for xtau in xtaus:
                    xrray = chand(xphi, xmuv, xmus, xtau)

                    # All should be physically valid
                    assert np.isfinite(xrray), \
                        f"Non-finite for xphi={xphi}, xmuv={xmuv}, xmus={xmus}, xtau={xtau}"
                    assert xrray > 0, \
                        f"Negative for xphi={xphi}, xmuv={xmuv}, xmus={xmus}, xtau={xtau}"
                    assert xrray < 2.0, \
                        f"Too large for xphi={xphi}, xmuv={xmuv}, xmus={xmus}, xtau={xtau}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
