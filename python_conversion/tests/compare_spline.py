"""
Compare Python spline implementation with Fortran output.
"""
import sys
sys.path.insert(0, '/home/user/6S/python_conversion')

import numpy as np
from sixs_python import spline, splint

# Same test data as Fortran
n = 10
x = np.zeros(n)
y = np.zeros(n)

for i in range(n):
    x[i] = i * 2.0 * np.pi / (n - 1)
    y[i] = np.sin(x[i])

# Natural spline
yp1 = 1.0e30
ypn = 1.0e30

y2 = spline(x, y, yp1, ypn)

print("Python Spline coefficients (y2):")
print(f"{'i':>3} {'x':>15} {'y2':>15}")
print("-" * 35)
for i in range(n):
    print(f"{i+1:3d} {x[i]:15.8f} {y2[i]:15.8f}")

print("\nPython Interpolation test:")
test_x = [0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
for xtest in test_x:
    y_spline = splint(x, y, y2, xtest)
    y_exact = np.sin(xtest)
    print(f"x={xtest:10.6f} y_spline={y_spline:15.8f} y_exact={y_exact:15.8f}")

# Compare coefficients with Fortran
print("\n" + "="*60)
print("COMPARISON: Python vs Fortran")
print("="*60)

# Fortran y2 values (from output above)
fortran_y2 = [0.00000000, -0.66929603, -1.02542102, -0.90174013, -0.35612494,
              0.35612547, 0.90174007, 1.02542078, 0.66929597, 0.00000000]

print(f"{'i':>3} {'Fortran y2':>15} {'Python y2':>15} {'Difference':>15}")
print("-" * 52)
max_diff = 0.0
for i in range(n):
    diff = abs(y2[i] - fortran_y2[i])
    max_diff = max(max_diff, diff)
    print(f"{i+1:3d} {fortran_y2[i]:15.8f} {y2[i]:15.8f} {diff:15.8e}")

print(f"\nMaximum difference: {max_diff:.6e}")

if max_diff < 1e-6:
    print("✓ PASSED: Python matches Fortran to within 1e-6")
else:
    print("✗ FAILED: Difference too large")
