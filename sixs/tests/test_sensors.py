"""
Tests for sensor spectral response functions.

Validates sensor data loading and access.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.sensors import load_sensor, list_sensors, SensorData


def test_list_sensors():
    """Test listing available sensors."""
    sensors = list_sensors()

    # Should have 16 sensors
    assert len(sensors) == 16

    # Check expected sensors are present
    expected = ['modis', 'avhrr', 'meris', 'seawifs', 'polder', 'aatsr',
                'hrv', 'etm', 'ali', 'aster', 'mas', 'mss', 'vgt',
                'viirs', 'goes', 'gli']

    for sensor_name in expected:
        assert sensor_name in sensors


def test_load_modis():
    """Test loading MODIS sensor."""
    modis = load_sensor('modis')

    assert isinstance(modis, SensorData)
    assert modis.name == 'modis'
    assert modis.num_bands == 8
    assert modis.array_size == 1501


def test_modis_bands():
    """Test MODIS spectral response bands."""
    modis = load_sensor('modis')

    # Get band 1
    band1 = modis.get_band(1)

    assert isinstance(band1, np.ndarray)
    assert len(band1) == 1501

    # Values should be normalized 0-1
    assert np.all(band1 >= 0)
    assert np.all(band1 <= 1.0)

    # Band should have some non-zero values
    assert np.any(band1 > 0)


def test_modis_all_bands():
    """Test all MODIS bands load correctly."""
    modis = load_sensor('modis')

    all_bands = modis.get_all_bands()

    # Should have 8 bands
    assert len(all_bands) == 8

    # All bands should have correct shape
    for band_num, response in all_bands.items():
        assert len(response) == 1501
        assert np.all(response >= 0)
        # Some bands may have values slightly > 1.0 (e.g., 1.0141 due to calibration)
        assert np.all(response <= 1.05)


def test_sensor_caching():
    """Test that sensors are cached after first load."""
    modis1 = load_sensor('modis')
    modis2 = load_sensor('modis')

    # Should be the same object (cached)
    assert modis1 is modis2


def test_invalid_sensor():
    """Test loading invalid sensor name."""
    with pytest.raises(ValueError, match="not found"):
        load_sensor('invalid_sensor')


def test_invalid_band_number():
    """Test accessing invalid band number."""
    modis = load_sensor('modis')

    # MODIS has 8 bands
    with pytest.raises(ValueError, match="out of range"):
        modis.get_band(0)

    with pytest.raises(ValueError, match="out of range"):
        modis.get_band(9)


def test_wavelength_bounds():
    """Test wavelength bounds retrieval."""
    modis = load_sensor('modis')

    # Get bounds for band 1
    wl_lower, wl_upper = modis.wavelength_bounds(1)

    # Should have valid bounds (or None if not defined)
    if wl_lower is not None:
        assert isinstance(wl_lower, (int, float))
        assert wl_lower > 0

    if wl_upper is not None:
        assert isinstance(wl_upper, (int, float))
        assert wl_upper > 0

        # Upper should be greater than lower
        if wl_lower is not None:
            assert wl_upper > wl_lower


def test_multiple_sensors():
    """Test loading multiple different sensors."""
    sensors_to_test = ['modis', 'seawifs', 'viirs', 'aster']

    for sensor_name in sensors_to_test:
        sensor = load_sensor(sensor_name)

        assert isinstance(sensor, SensorData)
        assert sensor.name == sensor_name
        assert sensor.num_bands > 0

        # Try to get first band
        band1 = sensor.get_band(1)
        assert len(band1) == 1501


def test_gli_30_bands():
    """Test GLI sensor with 30 bands."""
    gli = load_sensor('gli')

    assert gli.num_bands == 30

    # Get all bands
    all_bands = gli.get_all_bands()
    assert len(all_bands) == 30


def test_sensor_repr():
    """Test sensor string representation."""
    modis = load_sensor('modis')

    repr_str = repr(modis)

    assert 'modis' in repr_str
    assert '8' in repr_str  # num_bands
    assert 'SensorData' in repr_str


def test_case_insensitive_loading():
    """Test that sensor names are case-insensitive."""
    modis1 = load_sensor('MODIS')
    modis2 = load_sensor('modis')
    modis3 = load_sensor('MoDiS')

    # All should load the same sensor
    assert modis1.name == 'modis'
    assert modis2.name == 'modis'
    assert modis3.name == 'modis'


def test_spectral_response_physical_properties():
    """Test that spectral responses have physical properties."""
    modis = load_sensor('modis')

    for band_num in range(1, 9):
        response = modis.get_band(band_num)

        # Should be finite
        assert np.all(np.isfinite(response))

        # Should be non-negative (spectral response is physical quantity)
        assert np.all(response >= 0)

        # Should have at least some non-zero values (band has coverage)
        assert np.any(response > 0)

        # Peak response should be close to 1.0 (relative response, may slightly exceed)
        assert np.max(response) <= 1.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
