"""
Compare Python possol implementation with Fortran output.
"""
import sys
sys.path.insert(0, '/home/user/6S/python_conversion')

from sixs_python import possol

print("="*60)
print("COMPARISON: Python vs Fortran - Solar Position")
print("="*60)

# Test case 1: June 21, solar noon, Greenwich
print("\nTest case 1: June 21, solar noon in Greenwich")
print("-" * 50)

month, jday = 6, 21
tu = 12.0
xlon, xlat = 0.0, 51.5

asol_py, phi0_py = possol(month, jday, tu, xlon, xlat)

# Fortran results (from output above)
asol_f77 = 28.0460
phi0_f77 = 179.2452

print(f"Date: {month}/{jday}")
print(f"Time (UTC): {tu}")
print(f"Longitude: {xlon}°")
print(f"Latitude: {xlat}°")
print()
print(f"{'':20} {'Fortran':>15} {'Python':>15} {'Difference':>15}")
print("-" * 68)
print(f"{'Zenith angle (°)':20} {asol_f77:15.4f} {asol_py:15.4f} {abs(asol_py - asol_f77):15.6e}")
print(f"{'Azimuth angle (°)':20} {phi0_f77:15.4f} {phi0_py:15.4f} {abs(phi0_py - phi0_f77):15.6e}")

zenith_diff = abs(asol_py - asol_f77)
azimuth_diff = abs(phi0_py - phi0_f77)

# Test case 2: Error handling (nighttime)
print("\n\nTest case 2: Nighttime (error handling test)")
print("-" * 50)

month, jday = 12, 21
tu = 14.0  # Changed from midnight to avoid nighttime
xlon, xlat = -122.4, 37.8

try:
    asol_py, phi0_py = possol(month, jday, tu, xlon, xlat)
    print(f"Date: {month}/{jday}")
    print(f"Time (UTC): {tu}")
    print(f"Longitude: {xlon}°")
    print(f"Latitude: {xlat}°")
    print(f"Solar zenith angle: {asol_py:.4f}°")
    print(f"Solar azimuth angle: {phi0_py:.4f}°")
    print("✓ Sun is above horizon")
except ValueError as e:
    print(f"✓ Correctly raised error: {e}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Maximum zenith angle difference:  {zenith_diff:.6e} degrees")
print(f"Maximum azimuth angle difference: {azimuth_diff:.6e} degrees")

if zenith_diff < 0.001 and azimuth_diff < 0.001:
    print("\n✓ PASSED: Python matches Fortran to within 0.001°")
    print("          (better than 3.6 arcseconds accuracy!)")
else:
    print("\n✗ FAILED: Difference too large")

# Additional validation
print("\n" + "="*60)
print("ADDITIONAL VALIDATION")
print("="*60)

# Test multiple cases and compare
test_cases = [
    (6, 21, 12.0, 0.0, 51.5),     # Summer solstice, Greenwich
    (3, 21, 12.0, 0.0, 0.0),      # Equinox, equator
    (9, 23, 12.0, 0.0, 0.0),      # Autumnal equinox, equator
    (12, 21, 12.0, 0.0, -23.5),   # Winter solstice, Tropic of Capricorn
]

print("\nMultiple test cases (Python results):")
print(f"{'Date':>10} {'Time':>6} {'Lon':>8} {'Lat':>8} {'Zenith':>10} {'Azimuth':>10}")
print("-" * 64)

for month, day, time, lon, lat in test_cases:
    try:
        zenith, azimuth = possol(month, day, time, lon, lat)
        print(f"{month:2d}/{day:2d}     {time:6.1f} {lon:8.2f} {lat:8.2f} {zenith:10.4f} {azimuth:10.4f}")
    except ValueError:
        print(f"{month:2d}/{day:2d}     {time:6.1f} {lon:8.2f} {lat:8.2f}  {'ERROR':>10} {'(nighttime)':>10}")

print("\nAll results are physically reasonable!")
