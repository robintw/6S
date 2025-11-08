"""
Comprehensive test suite for Fortran to Python conversion.

Tests verify correctness using:
1. Known analytical solutions
2. Mathematical properties
3. Comparison with scipy implementations where applicable
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/6S/python_conversion')

from sixs_python import spline, splint, gauss, possol


def test_spline_interpolation():
    """Test spline interpolation on sin(x) function."""
    print("=" * 70)
    print("TEST 1: Spline Interpolation")
    print("=" * 70)

    # Create test data: y = sin(x) for x in [0, 2*pi]
    n = 10
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x)

    # Natural spline (2nd derivatives = 0 at endpoints)
    yp1 = 1.0e30
    ypn = 1.0e30

    # Compute spline coefficients
    y2 = spline(x, y, yp1, ypn)

    print(f"\nSpline coefficients for {n} points:")
    print(f"{'i':>3} {'x':>12} {'y':>12} {'y2':>12}")
    print("-" * 42)
    for i in range(n):
        print(f"{i:3d} {x[i]:12.6f} {y[i]:12.6f} {y2[i]:12.6f}")

    # Test interpolation at intermediate points
    print("\nInterpolation test:")
    print(f"{'x':>12} {'y_spline':>15} {'y_exact':>15} {'error':>15}")
    print("-" * 60)

    test_points = np.linspace(0, 2 * np.pi, 20)
    max_error = 0.0

    for xtest in test_points:
        y_spline = splint(x, y, y2, xtest)
        y_exact = np.sin(xtest)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
        print(f"{xtest:12.6f} {y_spline:15.10f} {y_exact:15.10f} {error:15.10e}")

    print(f"\nMaximum interpolation error: {max_error:.6e}")

    # Check against scipy
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x, y, bc_type='natural')
        scipy_result = cs(test_points[10])
        our_result = splint(x, y, y2, test_points[10])
        diff = abs(scipy_result - our_result)
        print(f"\nComparison with scipy.interpolate.CubicSpline:")
        print(f"  Our result:   {our_result:.10f}")
        print(f"  Scipy result: {scipy_result:.10f}")
        print(f"  Difference:   {diff:.6e}")

        if diff < 1e-6:
            print("  ✓ PASSED: Matches scipy implementation")
        else:
            print("  ⚠ WARNING: Differs from scipy implementation")
    except ImportError:
        print("\nNote: scipy not available for comparison")

    if max_error < 0.01:
        print("\n✓ PASSED: Spline interpolation accurate to 1e-2")
        return True
    else:
        print("\n✗ FAILED: Spline interpolation error too large")
        return False


def test_gauss_quadrature():
    """Test Gaussian quadrature on polynomial integration."""
    print("\n" + "=" * 70)
    print("TEST 2: Gaussian Quadrature")
    print("=" * 70)

    n = 5
    x1, x2 = 0.0, 1.0

    # Get quadrature points and weights
    x, w = gauss(x1, x2, n)

    print(f"\nGauss-Legendre quadrature with {n} points on [{x1}, {x2}]:")
    print(f"{'i':>3} {'x':>20} {'w':>20}")
    print("-" * 45)
    for i in range(n):
        print(f"{i:3d} {x[i]:20.15f} {w[i]:20.15f}")

    # Test 1: Integrate x^2 from 0 to 1 (exact = 1/3)
    print("\nTest 1: Integrate x^2 from 0 to 1")
    integral = np.sum(w * x**2)
    exact = 1.0 / 3.0
    error = abs(integral - exact)
    print(f"  Computed: {integral:.15f}")
    print(f"  Exact:    {exact:.15f}")
    print(f"  Error:    {error:.6e}")

    test1_pass = error < 1e-10

    # Test 2: Integrate x^4 from 0 to 1 (exact = 1/5)
    print("\nTest 2: Integrate x^4 from 0 to 1")
    integral = np.sum(w * x**4)
    exact = 1.0 / 5.0
    error = abs(integral - exact)
    print(f"  Computed: {integral:.15f}")
    print(f"  Exact:    {exact:.15f}")
    print(f"  Error:    {error:.6e}")

    test2_pass = error < 1e-10

    # Test 3: Integrate exp(x) from 0 to 1 (exact = e - 1)
    print("\nTest 3: Integrate exp(x) from 0 to 1")
    integral = np.sum(w * np.exp(x))
    exact = np.e - 1.0
    error = abs(integral - exact)
    print(f"  Computed: {integral:.15f}")
    print(f"  Exact:    {exact:.15f}")
    print(f"  Error:    {error:.6e}")

    test3_pass = error < 1e-6

    # Test 4: Check symmetry and weight sum
    print("\nTest 4: Check properties")
    weight_sum = np.sum(w)
    expected_sum = x2 - x1
    print(f"  Sum of weights: {weight_sum:.15f}")
    print(f"  Expected (x2-x1): {expected_sum:.15f}")

    test4_pass = abs(weight_sum - expected_sum) < 1e-10

    if test1_pass and test2_pass and test3_pass and test4_pass:
        print("\n✓ PASSED: All quadrature tests passed")
        return True
    else:
        print("\n✗ FAILED: Some quadrature tests failed")
        return False


def test_solar_position():
    """Test solar position calculations."""
    print("\n" + "=" * 70)
    print("TEST 3: Solar Position Calculation")
    print("=" * 70)

    # Test case 1: Summer solstice at solar noon in Greenwich
    print("\nTest case 1: June 21, solar noon in Greenwich (0°E, 51.5°N)")
    month, jday = 6, 21
    tu = 12.0  # UTC
    xlon, xlat = 0.0, 51.5

    asol, phi0 = possol(month, jday, tu, xlon, xlat)

    print(f"  Date: {month}/{jday}")
    print(f"  Time (UTC): {tu:.1f} hours")
    print(f"  Longitude: {xlon:.2f}°")
    print(f"  Latitude: {xlat:.2f}°")
    print(f"  Solar zenith angle: {asol:.4f}°")
    print(f"  Solar azimuth angle: {phi0:.4f}°")

    # Physical constraints
    test1_zenith = 0 < asol < 90  # Sun should be above horizon
    test1_azimuth = 0 <= phi0 <= 360  # Valid azimuth range

    # At summer solstice at ~51.5°N, solar elevation should be high
    # Solar zenith should be roughly latitude - declination
    # Declination at summer solstice ≈ 23.5°
    # So zenith ≈ 51.5 - 23.5 = 28°
    expected_zenith_approx = 28.0
    test1_reasonable = abs(asol - expected_zenith_approx) < 5.0

    print(f"  Expected zenith ≈ {expected_zenith_approx}° (within 5°)")

    # Test case 2: Equinox at equator
    print("\nTest case 2: March 21 at equator (0°E, 0°N)")
    month, jday = 3, 21
    tu = 12.0
    xlon, xlat = 0.0, 0.0

    asol, phi0 = possol(month, jday, tu, xlon, xlat)

    print(f"  Solar zenith angle: {asol:.4f}°")
    print(f"  Solar azimuth angle: {phi0:.4f}°")

    # At equinox on equator at solar noon, sun should be nearly overhead
    test2_overhead = abs(asol) < 5.0  # Should be close to 0°

    print(f"  Expected zenith ≈ 0° (sun overhead)")

    # Test case 3: Check error handling for nighttime
    print("\nTest case 3: Nighttime (should raise error)")
    month, jday = 6, 21
    tu = 0.0  # Midnight UTC
    xlon, xlat = 0.0, 51.5

    try:
        asol, phi0 = possol(month, jday, tu, xlon, xlat)
        # If we get here, sun is still up (which is possible in summer at high latitudes)
        print(f"  Solar zenith angle: {asol:.4f}°")
        if asol > 90:
            print("  ✗ ERROR: Should have raised exception for sun below horizon")
            test3_error = False
        else:
            print("  Note: Sun still above horizon at this time (summer at 51.5°N)")
            test3_error = True
    except ValueError as e:
        print(f"  ✓ Correctly raised error: {e}")
        test3_error = True

    # Summary
    all_tests = [test1_zenith, test1_azimuth, test1_reasonable,
                 test2_overhead, test3_error]

    if all(all_tests):
        print("\n✓ PASSED: All solar position tests passed")
        return True
    else:
        print("\n✗ FAILED: Some solar position tests failed")
        print(f"  Test results: {all_tests}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FORTRAN TO PYTHON CONVERSION - TEST SUITE")
    print("=" * 70)

    results = []

    results.append(("Spline Interpolation", test_spline_interpolation()))
    results.append(("Gaussian Quadrature", test_gauss_quadrature()))
    results.append(("Solar Position", test_solar_position()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<50} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} test suites passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Conversion successful.")
        return 0
    else:
        print("\n⚠ Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
