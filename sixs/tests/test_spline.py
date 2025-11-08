"""
Tests for spline interpolation functions.

Validates 1D and 2D cubic spline interpolation.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.spline import spline, splint, splie2, splin2


def test_spline_basic():
    """Test basic 1D spline interpolation."""
    # Simple linear function
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0

    # Calculate second derivatives
    y2 = spline(x, y, 1.0e30, 1.0e30)

    # For linear function, second derivatives should be ~0
    assert np.allclose(y2, 0.0, atol=1e-10)


def test_spline_quadratic():
    """Test spline on quadratic function."""
    x = np.linspace(0, 4, 20)
    y = x**2

    y2 = spline(x, y, 1.0e30, 1.0e30)

    # Interpolate at midpoints
    for i in range(len(x) - 1):
        x_mid = (x[i] + x[i + 1]) / 2.0
        y_interp = splint(x, y, y2, x_mid)
        y_expected = x_mid**2

        # Should be very accurate for smooth function
        assert abs(y_interp - y_expected) < 0.01


def test_splint_endpoints():
    """Test that splint returns exact values at knot points."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.sin(x)

    y2 = spline(x, y, 1.0e30, 1.0e30)

    # Interpolation at knot points should return exact values
    for i, xi in enumerate(x):
        y_interp = splint(x, y, y2, xi)
        assert abs(y_interp - y[i]) < 1e-10


def test_splint_monotonic():
    """Test interpolation of monotonic function."""
    x = np.linspace(0, 2 * np.pi, 50)
    y = np.exp(x)

    y2 = spline(x, y, 1.0e30, 1.0e30)

    # Test interpolation at many points
    x_test = np.linspace(0, 2 * np.pi, 200)
    y_expected = np.exp(x_test)

    for i, xi in enumerate(x_test):
        y_interp = splint(x, y, y2, xi)

        # Should be close to exponential (spline approximation)
        rel_error = abs((y_interp - y_expected[i]) / y_expected[i])
        assert rel_error < 0.01  # 1% error tolerance


def test_splie2_basic():
    """Test 2D spline initialization."""
    # Create a simple 2D function
    m, n = 5, 4
    x1 = np.linspace(0, 1, m)
    x2 = np.linspace(0, 1, n)

    # Create 2D function: f(x1, x2) = x1 + x2
    ya = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ya[i, j] = x1[i] + x2[j]

    # Initialize splines
    y2a = splie2(x2, ya)

    # Should return array of correct shape
    assert y2a.shape == (m, n)

    # For linear function, second derivatives should be ~0
    assert np.allclose(y2a, 0.0, atol=1e-10)


def test_splin2_basic():
    """Test 2D spline interpolation."""
    # Create 2D grid
    m, n = 6, 5
    x1 = np.linspace(0, 1, m)
    x2 = np.linspace(0, 1, n)

    # Create 2D function: f(x1, x2) = x1^2 + x2^2
    ya = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ya[i, j] = x1[i]**2 + x2[j]**2

    # Initialize splines
    y2a = splie2(x2, ya)

    # Test interpolation at grid points
    for i in range(m):
        for j in range(n):
            y_interp = splin2(x1, x2, ya, y2a, x1[i], x2[j])

            # Should match exact value at grid points
            assert abs(y_interp - ya[i, j]) < 1e-8


def test_splin2_intermediate_points():
    """Test 2D interpolation at points between grid nodes."""
    # Create 2D grid
    m, n = 10, 8
    x1 = np.linspace(0, 2, m)
    x2 = np.linspace(0, 3, n)

    # Create 2D function: f(x1, x2) = sin(x1) * cos(x2)
    ya = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ya[i, j] = np.sin(x1[i]) * np.cos(x2[j])

    # Initialize splines
    y2a = splie2(x2, ya)

    # Test at intermediate points
    test_points = [
        (0.5, 1.0),
        (1.0, 1.5),
        (1.5, 2.0),
        (0.3, 0.7),
    ]

    for x1_test, x2_test in test_points:
        y_interp = splin2(x1, x2, ya, y2a, x1_test, x2_test)
        y_expected = np.sin(x1_test) * np.cos(x2_test)

        # Spline approximation should be reasonably accurate
        rel_error = abs((y_interp - y_expected) / (abs(y_expected) + 1e-10))
        assert rel_error < 0.05  # 5% error tolerance


def test_splin2_product_function():
    """Test 2D interpolation on separable function."""
    # Create grid
    m, n = 8, 6
    x1 = np.linspace(1, 5, m)
    x2 = np.linspace(2, 4, n)

    # Separable function: f(x1, x2) = x1 * x2
    ya = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ya[i, j] = x1[i] * x2[j]

    # Initialize splines
    y2a = splie2(x2, ya)

    # Test interpolation
    x1_test = 3.0
    x2_test = 3.0
    y_interp = splin2(x1, x2, ya, y2a, x1_test, x2_test)
    y_expected = x1_test * x2_test

    assert abs(y_interp - y_expected) < 0.01


def test_spline_natural_boundary():
    """Test natural spline boundary conditions."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 1.5, 0.5])

    # Natural spline (second derivative = 0 at boundaries)
    y2 = spline(x, y, 1.0e30, 1.0e30)

    # Second derivatives at endpoints should be 0 (or very small)
    assert abs(y2[0]) < 1e-8
    assert abs(y2[-1]) < 1e-8


def test_spline_smooth_function():
    """Test spline interpolation on smooth function."""
    x = np.linspace(0, 2 * np.pi, 20)
    y = np.sin(x)

    y2 = spline(x, y, 1.0e30, 1.0e30)

    # Test at many intermediate points
    x_test = np.linspace(0, 2 * np.pi, 100)

    for xi in x_test:
        y_interp = splint(x, y, y2, xi)
        y_expected = np.sin(xi)

        # Should be accurate for smooth function
        assert abs(y_interp - y_expected) < 0.01


def test_splin2_constant_function():
    """Test 2D spline on constant function."""
    m, n = 5, 5
    x1 = np.linspace(0, 1, m)
    x2 = np.linspace(0, 1, n)

    # Constant function
    const = 3.14159
    ya = np.full((m, n), const)

    y2a = splie2(x2, ya)

    # Test at any point
    x1_test = 0.5
    x2_test = 0.7
    y_interp = splin2(x1, x2, ya, y2a, x1_test, x2_test)

    assert abs(y_interp - const) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
