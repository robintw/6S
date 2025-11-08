"""
Tests for BRDF grid interpolation.

Validates BRDF interpolation onto Gaussian quadrature grids.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.brdf_grid import brdfgrid


def test_brdfgrid_basic():
    """Test basic BRDF grid interpolation."""
    # Simple grid
    angmu = np.linspace(-1.0, 1.0, 10)
    angphi = np.linspace(0, 180, 13)

    # Simple BRDF: constant value
    brdfdat = np.ones((10, 13)) * 0.5

    # Interpolation points
    rm = np.array([-0.5, 0.0, 0.5])
    rp = np.array([0, 45, 90, 135, 180])

    # Interpolate
    brdfint = brdfgrid(rm, rp, angmu, angphi, brdfdat)

    # Check output shape
    assert brdfint.shape == (3, 5)

    # For constant BRDF, all values should be ~0.5
    assert np.allclose(brdfint, 0.5, rtol=0.01)


def test_brdfgrid_linear_function():
    """Test interpolation of linear BRDF."""
    # Grid
    angmu = np.linspace(-1.0, 1.0, 10)
    angphi = np.linspace(0, 180, 13)

    # Linear BRDF: depends only on mu
    mu_grid, phi_grid = np.meshgrid(angmu, angphi, indexing='ij')
    brdfdat = 0.3 + 0.2 * mu_grid  # Linear in mu

    # Interpolation points
    rm = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    rp = np.array([30, 60, 90, 120, 150])

    # Interpolate
    brdfint = brdfgrid(rm, rp, angmu, angphi, brdfdat)

    # Check output shape
    assert brdfint.shape == (5, 5)

    # For linear function, interpolation should be accurate
    expected = 0.3 + 0.2 * rm[:, np.newaxis]
    assert np.allclose(brdfint, expected, rtol=0.05)


def test_brdfgrid_smooth_function():
    """Test interpolation of smooth BRDF."""
    # Grid
    angmu = np.linspace(-1.0, 1.0, 10)
    angphi = np.linspace(0, 180, 13)

    # Smooth BRDF: quadratic
    mu_grid, phi_grid = np.meshgrid(angmu, angphi, indexing='ij')
    brdfdat = 0.5 + 0.1 * mu_grid**2

    # Interpolation points within grid
    rm = np.linspace(-0.9, 0.9, 5)
    rp = np.linspace(10, 170, 8)

    # Interpolate
    brdfint = brdfgrid(rm, rp, angmu, angphi, brdfdat)

    # Check output shape
    assert brdfint.shape == (5, 8)

    # Check values are in reasonable range
    assert np.all(brdfint >= 0.4)
    assert np.all(brdfint <= 0.7)

    # Check values are finite
    assert np.all(np.isfinite(brdfint))


def test_brdfgrid_output_shape():
    """Test that output has correct shape."""
    angmu = np.linspace(-1.0, 1.0, 10)
    angphi = np.linspace(0, 180, 13)
    brdfdat = np.random.rand(10, 13)

    # Different sizes
    test_cases = [
        (5, 7),
        (3, 4),
        (10, 5),
    ]

    for nmu, nphi in test_cases:
        rm = np.linspace(-0.9, 0.9, nmu)
        rp = np.linspace(10, 170, nphi)

        brdfint = brdfgrid(rm, rp, angmu, angphi, brdfdat)

        assert brdfint.shape == (nmu, nphi)


def test_brdfgrid_positive_values():
    """Test that BRDF values remain positive."""
    # Grid
    angmu = np.linspace(-1.0, 1.0, 10)
    angphi = np.linspace(0, 180, 13)

    # Positive BRDF data
    brdfdat = 0.2 + 0.1 * np.random.rand(10, 13)

    # Interpolation points
    rm = np.linspace(-0.8, 0.8, 6)
    rp = np.linspace(20, 160, 9)

    # Interpolate
    brdfint = brdfgrid(rm, rp, angmu, angphi, brdfdat)

    # All values should be positive (or very close due to interpolation)
    assert np.all(brdfint >= -0.01)


def test_brdfgrid_boundary_points():
    """Test interpolation at boundary points."""
    # Grid
    angmu = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    angphi = np.array([0, 45, 90, 135, 180])

    # Simple BRDF
    brdfdat = np.ones((5, 5)) * 0.6

    # Interpolation at grid points
    rm = np.array([-1.0, 0.0, 1.0])
    rp = np.array([0, 90, 180])

    # Interpolate
    brdfint = brdfgrid(rm, rp, angmu, angphi, brdfdat)

    # At grid points, should match original data
    assert np.allclose(brdfint, 0.6, rtol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
