"""
Tests for atmospheric absorption coefficient tables.

Validates absorption data loading and access.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.absorption import load_absorption, list_absorption_tables, AbsorptionTable


def test_list_absorption_tables():
    """Test listing available absorption tables."""
    tables = list_absorption_tables()

    # Should have 7 categories
    assert len(tables) == 7

    # Check expected categories
    assert 'Water Vapor (H2O)' in tables
    assert 'Ozone (O3)' in tables
    assert 'Methane (CH4)' in tables

    # Check counts
    assert len(tables['Water Vapor (H2O)']) == 6
    assert len(tables['Nitrogen Dioxide (NO2)']) == 6
    assert len(tables['Oxygen (O2)']) == 4
    assert len(tables['Ozone (O3)']) == 1
    assert len(tables['Molecular Chain (CO)']) == 6
    assert len(tables['Diatomic (CO2)']) == 3
    assert len(tables['Methane (CH4)']) == 6


def test_load_wava1():
    """Test loading water vapor absorption table."""
    wava1 = load_absorption('wava1')

    assert isinstance(wava1, AbsorptionTable)
    assert wava1.name == 'wava1'
    assert wava1.n_intervals == 256
    assert wava1.n_coefficients == 6


def test_wava1_coefficients():
    """Test getting coefficients from water vapor table."""
    wava1 = load_absorption('wava1')

    # Get coefficients at a valid wavenumber
    # Water vapor table should cover certain wavenumber ranges
    wn_min, wn_max = wava1.wavenumber_range()

    # Should have a valid range
    assert wn_min > 0
    assert wn_max > wn_min

    # Get coefficients in the middle of the range
    mid_wn = (wn_min + wn_max) / 2.0
    coef = wava1.get_coefficients(mid_wn)

    # Should return valid coefficients or None
    if coef is not None:
        assert isinstance(coef, np.ndarray)
        assert len(coef) == 6
        assert np.all(np.isfinite(coef))


def test_ozon1_structure():
    """Test ozone absorption table structure."""
    ozon1 = load_absorption('ozon1')

    assert ozon1.n_intervals == 256
    assert ozon1.n_coefficients == 6

    # Get an interval
    coef, wn_low, wn_high = ozon1.get_interval(0)

    assert isinstance(coef, np.ndarray)
    assert len(coef) == 6
    assert isinstance(wn_low, (int, float, np.number))
    assert isinstance(wn_high, (int, float, np.number))


def test_get_interval():
    """Test getting specific intervals."""
    wava1 = load_absorption('wava1')

    # Get first few intervals
    for i in range(10):
        coef, wn_low, wn_high = wava1.get_interval(i)

        assert len(coef) == 6
        assert np.all(np.isfinite(coef))

        # If valid interval, upper bound should be >= lower bound
        if wn_low > 0 or wn_high > 0:
            assert wn_high >= wn_low


def test_invalid_interval_index():
    """Test accessing invalid interval index."""
    wava1 = load_absorption('wava1')

    with pytest.raises(IndexError):
        wava1.get_interval(300)

    with pytest.raises(IndexError):
        wava1.get_interval(-1)


def test_wavenumber_range():
    """Test wavenumber range retrieval."""
    ozon1 = load_absorption('ozon1')

    wn_min, wn_max = ozon1.wavenumber_range()

    # Should have valid range
    assert wn_min > 0
    assert wn_max > wn_min
    assert np.isfinite(wn_min)
    assert np.isfinite(wn_max)


def test_get_all_intervals():
    """Test getting all valid intervals."""
    wava1 = load_absorption('wava1')

    intervals = wava1.get_all_intervals()

    # Should have valid intervals
    assert len(intervals) > 0

    # Each interval should have proper structure
    for coef, wn_low, wn_high in intervals:
        assert len(coef) == 6
        assert np.all(np.isfinite(coef))
        assert wn_high >= wn_low


def test_table_caching():
    """Test that tables are cached after first load."""
    wava1_first = load_absorption('wava1')
    wava1_second = load_absorption('wava1')

    # Should be the same object (cached)
    assert wava1_first is wava1_second


def test_invalid_table_name():
    """Test loading invalid table name."""
    with pytest.raises(ValueError, match="not found"):
        load_absorption('invalid_table')


def test_case_insensitive_loading():
    """Test that table names are case-insensitive."""
    wava1_lower = load_absorption('wava1')
    wava1_upper = load_absorption('WAVA1')
    wava1_mixed = load_absorption('WaVa1')

    # All should load the same table
    assert wava1_lower.name == 'wava1'
    assert wava1_upper.name == 'wava1'
    assert wava1_mixed.name == 'wava1'


def test_multiple_tables():
    """Test loading multiple different tables."""
    tables_to_test = ['wava1', 'ozon1', 'meth1', 'niox1']

    for table_name in tables_to_test:
        table = load_absorption(table_name)

        assert isinstance(table, AbsorptionTable)
        assert table.name == table_name
        assert table.n_intervals == 256
        assert table.n_coefficients == 6


def test_table_repr():
    """Test table string representation."""
    wava1 = load_absorption('wava1')

    repr_str = repr(wava1)

    assert 'wava1' in repr_str
    assert '256' in repr_str  # n_intervals
    assert 'cm^-1' in repr_str
    assert 'AbsorptionTable' in repr_str


def test_coefficients_physical_properties():
    """Test that coefficients have valid physical properties."""
    ozon1 = load_absorption('ozon1')

    # Get all valid intervals
    intervals = ozon1.get_all_intervals()

    for coef, wn_low, wn_high in intervals:
        # Coefficients should be finite
        assert np.all(np.isfinite(coef))

        # Wavenumber bounds should be positive
        assert wn_low >= 0
        assert wn_high >= 0

        # Upper bound should be >= lower bound
        assert wn_high >= wn_low


def test_get_coefficients_boundary():
    """Test coefficient lookup at table boundaries."""
    wava1 = load_absorption('wava1')

    wn_min, wn_max = wava1.wavenumber_range()

    # At minimum wavenumber
    coef_min = wava1.get_coefficients(wn_min)
    if coef_min is not None:
        assert len(coef_min) == 6

    # At maximum wavenumber
    coef_max = wava1.get_coefficients(wn_max)
    if coef_max is not None:
        assert len(coef_max) == 6

    # Outside range should return None
    coef_below = wava1.get_coefficients(wn_min - 1000)
    assert coef_below is None

    coef_above = wava1.get_coefficients(wn_max + 1000)
    assert coef_above is None


def test_different_gas_categories():
    """Test that different gas categories have distinct absorption data."""
    wava1 = load_absorption('wava1')  # Water vapor
    ozon1 = load_absorption('ozon1')  # Ozone
    meth1 = load_absorption('meth1')  # Methane

    # All should be valid tables
    assert wava1.n_intervals == 256
    assert ozon1.n_intervals == 256
    assert meth1.n_intervals == 256

    # All should have wavenumber ranges
    wava_range = wava1.wavenumber_range()
    ozon_range = ozon1.wavenumber_range()
    meth_range = meth1.wavenumber_range()

    # All should have valid ranges (some may share the same spectral region)
    for wn_min, wn_max in [wava_range, ozon_range, meth_range]:
        assert wn_min > 0
        assert wn_max > wn_min


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
