"""
6S Radiative Transfer Code - Python Conversion
A proof-of-concept conversion of Fortran 77 code to Python
using NumPy and Numba for performance.
"""

from .spline import spline, splint
from .gauss import gauss
from .possol import possol

__version__ = "0.1.0"
__all__ = ['spline', 'splint', 'gauss', 'possol']
