"""
Custom exceptions for 6S radiative transfer code.

Provides specific exception types for different error conditions.
"""


class SixSError(Exception):
    """Base exception for all 6S errors."""
    pass


class InputError(SixSError):
    """Exception raised for invalid input parameters."""
    pass


class ConvergenceError(SixSError):
    """Exception raised when iterative computation fails to converge."""
    pass


class PhysicalError(SixSError):
    """Exception raised when physical constraints are violated."""
    pass


class GeometryError(SixSError):
    """Exception raised for invalid geometric configurations."""
    pass


class AtmosphericError(SixSError):
    """Exception raised for atmospheric model errors."""
    pass


__all__ = [
    'SixSError',
    'InputError',
    'ConvergenceError',
    'PhysicalError',
    'GeometryError',
    'AtmosphericError',
]
