"""
Compare Python gauss implementation with Fortran output.
"""
import sys
sys.path.insert(0, '/home/user/6S/python_conversion')

import numpy as np
from sixs_python import gauss

# Same test as Fortran
n = 5
x1 = 0.0
x2 = 1.0

x, w = gauss(x1, x2, n)

print("Python Gauss-Legendre quadrature:")
print(f"n = {n}\n")
print(f"{'i':>3} {'x':>20} {'w':>20}")
print("-" * 45)
for i in range(n):
    print(f"{i+1:3d} {x[i]:20.15f} {w[i]:20.15f}")

# Integration test
integral = np.sum(w * x**2)
exact = 1.0 / 3.0
error = abs(integral - exact)

print("\nIntegration test: integral of x^2 from 0 to 1")
print(f"Computed: {integral:.15f}")
print(f"Exact:    {exact:.15f}")
print(f"Error:    {error:.6e}")

# Compare with Fortran values
print("\n" + "="*60)
print("COMPARISON: Python vs Fortran")
print("="*60)

# Fortran values (from output above)
fortran_x = [0.046910077333450, 0.230765342712402, 0.500000000000000,
             0.769234657287598, 0.953089952468872]
fortran_w = [0.118463441729546, 0.239314332604408, 0.284444451332092,
             0.239314332604408, 0.118463441729546]

print("\nAbscissas (x):")
print(f"{'i':>3} {'Fortran':>20} {'Python':>20} {'Difference':>15}")
print("-" * 62)
max_diff_x = 0.0
for i in range(n):
    diff = abs(x[i] - fortran_x[i])
    max_diff_x = max(max_diff_x, diff)
    print(f"{i+1:3d} {fortran_x[i]:20.15f} {x[i]:20.15f} {diff:15.8e}")

print("\nWeights (w):")
print(f"{'i':>3} {'Fortran':>20} {'Python':>20} {'Difference':>15}")
print("-" * 62)
max_diff_w = 0.0
for i in range(n):
    diff = abs(w[i] - fortran_w[i])
    max_diff_w = max(max_diff_w, diff)
    print(f"{i+1:3d} {fortran_w[i]:20.15f} {w[i]:20.15f} {diff:15.8e}")

print(f"\nMaximum difference in x: {max_diff_x:.6e}")
print(f"Maximum difference in w: {max_diff_w:.6e}")

# Compare integration results
fortran_integral = 0.333333332819166
fortran_error = 0.51416765e-09
print(f"\nIntegration comparison:")
print(f"  Fortran integral: {fortran_integral:.15f}")
print(f"  Python integral:  {integral:.15f}")
print(f"  Difference:       {abs(integral - fortran_integral):.6e}")

if max_diff_x < 1e-10 and max_diff_w < 1e-10:
    print("\n✓ PASSED: Python matches Fortran to within 1e-10")
else:
    print(f"\n✗ FAILED: Difference too large")
