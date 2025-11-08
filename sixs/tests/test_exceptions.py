"""
Tests for exception classes.

Validates custom exception hierarchy.
"""

import pytest
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.exceptions import (
    SixSError, InputError, ConvergenceError,
    PhysicalError, GeometryError, AtmosphericError
)


def test_exception_hierarchy():
    """Test that all exceptions inherit from SixSError."""
    assert issubclass(InputError, SixSError)
    assert issubclass(ConvergenceError, SixSError)
    assert issubclass(PhysicalError, SixSError)
    assert issubclass(GeometryError, SixSError)
    assert issubclass(AtmosphericError, SixSError)


def test_exception_inheritance():
    """Test that SixSError inherits from Exception."""
    assert issubclass(SixSError, Exception)


def test_raise_input_error():
    """Test raising InputError."""
    with pytest.raises(InputError) as exc_info:
        raise InputError("Invalid wavelength: must be positive")

    assert "Invalid wavelength" in str(exc_info.value)


def test_raise_convergence_error():
    """Test raising ConvergenceError."""
    with pytest.raises(ConvergenceError) as exc_info:
        raise ConvergenceError("Failed to converge after 100 iterations")

    assert "Failed to converge" in str(exc_info.value)


def test_catch_specific_exception():
    """Test catching specific exception type."""
    try:
        raise GeometryError("Invalid zenith angle")
    except GeometryError as e:
        assert "zenith angle" in str(e)
    except Exception:
        pytest.fail("Should have caught GeometryError specifically")


def test_catch_base_exception():
    """Test catching via base SixSError."""
    try:
        raise PhysicalError("Temperature below absolute zero")
    except SixSError as e:
        assert "Temperature" in str(e)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
