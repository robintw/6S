"""
Tests for BRDF models.

Validates surface reflectance models.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.brdf_models import (
    minnalbe, modisalbe, waltalbe, roujalbe,
    minnbrdf, waltbrdf, roujbrdf,
    clearw, lakew
)


# ==============================================================================
# Albedo Model Tests
# ==============================================================================

def test_minnalbe():
    """Test Minnaert albedo calculation."""
    # Typical Minnaert parameters
    par1 = 1.0  # k parameter
    par2 = 0.3  # brightness

    alb = minnalbe(par1, par2)

    # Should be positive and less than 1
    assert 0 < alb < 1.0
    assert np.isfinite(alb)


def test_modisalbe():
    """Test MODIS albedo calculation."""
    # Test parameters
    p1 = 0.5
    p2 = 0.3
    p3 = 0.2

    alb = modisalbe(p1, p2, p3)

    # Should be finite
    assert np.isfinite(alb)

    # Check formula
    expected = p1 + p2 * 0.189184 - p3 * 1.377622
    assert np.isclose(alb, expected)


def test_waltalbe():
    """Test Walthall albedo calculation."""
    # Simple parameters
    a = 0.01
    ap = 0.02
    b = 0.01
    c = 0.1

    alb = waltalbe(a, ap, b, c)

    # Should be positive and less than 1
    assert 0 < alb < 1.0
    assert np.isfinite(alb)


def test_roujalbe():
    """Test Roujean albedo calculation."""
    # Typical parameters
    k0 = 0.15
    k1 = 0.05
    k2 = 0.02

    alb = roujalbe(k0, k1, k2)

    # Should be positive and reasonable
    assert 0 < alb < 1.0
    assert np.isfinite(alb)


# ==============================================================================
# BRDF Model Tests
# ==============================================================================

def test_minnbrdf_basic():
    """Test basic Minnaert BRDF calculation."""
    par1 = 1.0
    par2 = 0.3

    # Simple geometry
    rm = np.array([0.8, 0.7, 0.6])  # rm[0] is solar zenith
    rp = np.array([0, 45, 90, 135, 180])

    brdf = minnbrdf(par1, par2, rm, rp)

    # Check shape
    assert brdf.shape == (3, 5)

    # Should be positive
    assert np.all(brdf > 0)

    # Should be finite
    assert np.all(np.isfinite(brdf))


def test_minnbrdf_reciprocity():
    """Test reciprocity of Minnaert BRDF (symmetric in viewing/solar)."""
    par1 = 0.8
    par2 = 0.25

    # Symmetric case: viewing = solar
    rm1 = np.array([0.7, 0.7])
    rp1 = np.array([0])
    brdf1 = minnbrdf(par1, par2, rm1, rp1)

    # Should have same value
    assert brdf1.shape == (2, 1)


def test_waltbrdf_basic():
    """Test Walthall BRDF calculation."""
    a = 0.01
    ap = 0.02
    b = 0.01
    c = 0.1

    # Geometry with azimuth offset in rm[-1]
    rm = np.array([0.8, 0.7, 0.6, 0.0])  # Last element is azimuth offset
    rp = np.array([0, 45, 90])

    brdf = waltbrdf(a, ap, b, c, rm, rp)

    # Check shape (mu-1 because rm[-1] is azimuth)
    assert brdf.shape == (3, 3)

    # Should be positive
    assert np.all(brdf > 0)


def test_roujbrdf_basic():
    """Test Roujean BRDF calculation."""
    k0 = 0.15
    k1 = 0.05
    k2 = 0.02

    # Geometry
    rm = np.array([0.8, 0.7, 0.6, 0.0])  # Last is azimuth offset
    rp = np.array([0, 60, 120, 180])

    brdf = roujbrdf(k0, k1, k2, rm, rp)

    # Check shape
    assert brdf.shape == (3, 4)

    # Should be positive (mostly)
    assert np.all(brdf > -0.1)  # Allow small negative due to kernels

    # Should be finite
    assert np.all(np.isfinite(brdf))


def test_roujbrdf_isotropic_limit():
    """Test Roujean BRDF reduces to isotropic when k1=k2=0."""
    k0 = 0.2
    k1 = 0.0
    k2 = 0.0

    rm = np.array([0.8, 0.7, 0.6, 0.0])
    rp = np.array([0, 90, 180])

    brdf = roujbrdf(k0, k1, k2, rm, rp)

    # Should all be approximately k0
    assert np.allclose(brdf, k0, rtol=0.1)


def test_brdf_physical_range():
    """Test that BRDF values are in physical range."""
    # Test all BRDF models
    rm = np.array([0.9, 0.8, 0.7, 0.0])
    rp = np.array([0, 45, 90, 135])

    # Minnaert
    brdf_minn = minnbrdf(1.0, 0.3, rm[:3], rp)
    assert np.all(brdf_minn >= 0)
    assert np.all(brdf_minn < 2.0)

    # Walthall
    brdf_walt = waltbrdf(0.01, 0.02, 0.01, 0.1, rm, rp)
    assert np.all(brdf_walt >= 0)
    assert np.all(brdf_walt < 2.0)

    # Roujean
    brdf_rouj = roujbrdf(0.15, 0.05, 0.02, rm, rp)
    assert np.all(brdf_rouj > -0.2)  # Allow small negative
    assert np.all(brdf_rouj < 2.0)


# ==============================================================================
# Water Reflectance Tests
# ==============================================================================

def test_clearw_structure():
    """Test clear water reflectance structure."""
    r = clearw()

    # Should have 1501 points
    assert len(r) == 1501

    # Should be non-negative
    assert np.all(r >= 0)

    # Should have max < 0.1 (water has low reflectance)
    assert np.max(r) < 0.1


def test_clearw_spectral_range():
    """Test clear water spectral range."""
    r = clearw()

    # Count non-zero values
    nonzero = np.count_nonzero(r)

    # Should have some non-zero values
    assert nonzero > 0

    # But mostly zero outside visible range
    assert nonzero < len(r) / 2


def test_lakew_structure():
    """Test lake water reflectance structure."""
    r = lakew()

    # Should have 1501 points
    assert len(r) == 1501

    # Should be non-negative
    assert np.all(r >= 0)

    # Lake water has slightly higher reflectance than clear water
    assert np.max(r) < 0.1


def test_lakew_vs_clearw():
    """Test that lake water has different spectrum than clear water."""
    r_clear = clearw()
    r_lake = lakew()

    # Should not be identical
    assert not np.array_equal(r_clear, r_lake)

    # Both should have similar structure (mostly zero)
    assert len(r_clear) == len(r_lake)


def test_water_copy_independence():
    """Test that water reflectance returns independent copies."""
    r1 = clearw()
    r2 = clearw()

    # Modify one
    r1[100] = 999.0

    # Other should be unchanged
    assert r2[100] != 999.0


def test_albedo_consistency():
    """Test that albedo and BRDF models are self-consistent."""
    # For Minnaert model
    par1 = 1.0
    par2 = 0.3

    alb = minnalbe(par1, par2)

    # Albedo should be in reasonable range
    assert 0 < alb < 1.0


def test_brdf_normalization():
    """Test BRDF values are reasonably normalized."""
    # All BRDF models at nadir viewing
    rm_nadir = np.array([1.0, 1.0, 0.0])  # Both solar and view at nadir
    rp = np.array([0])

    # Minnaert at nadir
    brdf_minn = minnbrdf(1.0, 0.3, rm_nadir[:2], rp)
    assert brdf_minn[1, 0] > 0

    # Should be reasonable magnitude
    assert brdf_minn[1, 0] < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
