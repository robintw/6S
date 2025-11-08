"""
Sensor spectral response functions.

Provides access to spectral response data for 16 satellite sensors,
extracted from original Fortran DATA statements.

Sensors:
    - MODIS: 8 bands
    - AVHRR: 16 bands
    - MERIS: 15 bands
    - SeaWiFS: 8 bands
    - POLDER: 8 bands
    - AATSR: 8 bands
    - HRV: 8 bands
    - ETM (Landsat): 6 bands
    - ALI: 9 bands
    - ASTER: 10 bands
    - MAS: 10 bands
    - MSS (Landsat): 4 bands
    - VGT (SPOT): 4 bands
    - VIIRS: 16 bands
    - GOES: 2 bands
    - GLI: 30 bands

Usage:
    >>> sensor = load_sensor('modis')
    >>> sensor.get_band(1)  # Returns band 1 spectral response (1501 points)
    >>> sensor.wavelength_bounds(1)  # Returns (lower, upper) wavelength bounds
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional


class SensorData:
    """
    Container for sensor spectral response data.

    Attributes
    ----------
    name : str
        Sensor name (lowercase)
    num_bands : int
        Number of spectral bands
    array_size : int
        Number of spectral points (usually 1501)
    wavelength_range : str
        Wavelength coverage description
    wavelength_step : float
        Wavelength step size in micrometers
    """

    def __init__(self, name: str, npz_file: Path, metadata_file: Path):
        """
        Load sensor data from NPZ and JSON files.

        Parameters
        ----------
        name : str
            Sensor name
        npz_file : Path
            Path to NPZ file with spectral response data
        metadata_file : Path
            Path to JSON metadata file
        """
        self.name = name

        # Load spectral response arrays
        self._data = np.load(npz_file)

        # Load metadata
        with open(metadata_file, 'r') as f:
            self._metadata = json.load(f)

        self.num_bands = self._metadata['num_bands']
        self.array_size = self._metadata['array_size']
        self.wavelength_range = self._metadata['wavelength_range']
        self.wavelength_step = self._metadata['wavelength_step']

    def get_band(self, band_num: int) -> np.ndarray:
        """
        Get spectral response for a specific band.

        Parameters
        ----------
        band_num : int
            Band number (1-indexed)

        Returns
        -------
        response : ndarray
            Spectral response values (length 1501)
            Normalized to 0.0-1.0 range

        Raises
        ------
        ValueError
            If band number is out of range
        """
        if band_num < 1 or band_num > self.num_bands:
            raise ValueError(
                f"Band {band_num} out of range for {self.name} "
                f"(valid: 1-{self.num_bands})"
            )

        key = f'band_{band_num}'
        if key not in self._data:
            raise ValueError(f"Band {band_num} data not found for {self.name}")

        return self._data[key]

    def wavelength_bounds(self, band_num: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Get wavelength bounds for a specific band.

        Parameters
        ----------
        band_num : int
            Band number (1-indexed)

        Returns
        -------
        bounds : tuple
            (lower_wavelength, upper_wavelength) in micrometers
            Either value may be None if not defined

        Raises
        ------
        ValueError
            If band number is out of range
        """
        if band_num < 1 or band_num > self.num_bands:
            raise ValueError(
                f"Band {band_num} out of range for {self.name} "
                f"(valid: 1-{self.num_bands})"
            )

        key = f'band_{band_num}'
        if key in self._metadata['bands']:
            band_meta = self._metadata['bands'][key]
            return (band_meta['wavelength_lower'], band_meta['wavelength_upper'])

        return (None, None)

    def get_all_bands(self) -> Dict[int, np.ndarray]:
        """
        Get spectral response for all bands.

        Returns
        -------
        bands : dict
            Dictionary mapping band_num -> spectral_response
        """
        bands = {}
        for i in range(1, self.num_bands + 1):
            try:
                bands[i] = self.get_band(i)
            except ValueError:
                # Skip bands without data
                pass
        return bands

    def __repr__(self):
        return (
            f"SensorData(name='{self.name}', "
            f"num_bands={self.num_bands}, "
            f"wavelength_range='{self.wavelength_range}')"
        )


# Cache loaded sensors
_SENSOR_CACHE: Dict[str, SensorData] = {}


def load_sensor(sensor_name: str) -> SensorData:
    """
    Load sensor spectral response data.

    Parameters
    ----------
    sensor_name : str
        Sensor name (case-insensitive)
        Valid sensors: modis, avhrr, meris, seawifs, polder, aatsr,
                      hrv, etm, ali, aster, mas, mss, vgt, viirs, goes, gli

    Returns
    -------
    sensor : SensorData
        Sensor data object

    Raises
    ------
    ValueError
        If sensor name is not recognized or files not found

    Examples
    --------
    >>> modis = load_sensor('modis')
    >>> band1 = modis.get_band(1)
    >>> print(band1.shape)
    (1501,)
    >>> bounds = modis.wavelength_bounds(1)
    >>> print(f"Band 1: {bounds[0]}-{bounds[1]} μm")
    """
    sensor_name = sensor_name.lower()

    # Check cache
    if sensor_name in _SENSOR_CACHE:
        return _SENSOR_CACHE[sensor_name]

    # Find data files
    data_dir = Path(__file__).parent / 'data' / 'sensors'
    npz_file = data_dir / f'{sensor_name}_spectral_response.npz'
    json_file = data_dir / f'{sensor_name}_metadata.json'

    if not npz_file.exists():
        raise ValueError(
            f"Sensor '{sensor_name}' not found. "
            f"Expected file: {npz_file}"
        )

    if not json_file.exists():
        raise ValueError(
            f"Metadata for '{sensor_name}' not found. "
            f"Expected file: {json_file}"
        )

    # Load and cache
    sensor = SensorData(sensor_name, npz_file, json_file)
    _SENSOR_CACHE[sensor_name] = sensor

    return sensor


def list_sensors() -> list:
    """
    List all available sensors.

    Returns
    -------
    sensors : list
        List of sensor names
    """
    data_dir = Path(__file__).parent / 'data' / 'sensors'

    if not data_dir.exists():
        return []

    sensors = []
    for npz_file in data_dir.glob('*_spectral_response.npz'):
        sensor_name = npz_file.stem.replace('_spectral_response', '')
        sensors.append(sensor_name)

    return sorted(sensors)


__all__ = [
    'SensorData',
    'load_sensor',
    'list_sensors',
]
