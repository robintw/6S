"""
Atmospheric absorption coefficient tables.

Provides access to absorption data for atmospheric gases extracted from
original Fortran DATA statements.

Gas Categories:
    - Water Vapor (H2O): WAVA1-6 (6 tables)
    - Nitrogen Dioxide (NO2): NIOX1-6 (6 tables)
    - Oxygen (O2): OXYG3-6 (4 tables)
    - Ozone (O3): OZON1 (1 table)
    - Molecular Chain (CO): MOCA1-6 (6 tables)
    - Diatomic (CO2): DICA1-3 (3 tables)
    - Methane (CH4): METH1-6 (6 tables)

Each table contains:
    - 256 spectral intervals
    - 6 absorption coefficients per interval
    - Wavenumber bounds (cm^-1) for each interval

Usage:
    >>> table = load_absorption('wava1')
    >>> coef = table.get_coefficients(2500)  # At wavenumber 2500 cm^-1
    >>> bounds = table.wavenumber_range()
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List


class AbsorptionTable:
    """
    Container for atmospheric absorption coefficient data.

    Attributes
    ----------
    name : str
        Table name (e.g., 'wava1', 'ozon1')
    description : str
        Description from original Fortran file
    n_intervals : int
        Number of spectral intervals (usually 256)
    n_coefficients : int
        Number of coefficients per interval (6)
    """

    def __init__(self, name: str, npz_file: Path, metadata_file: Path):
        """
        Load absorption table from NPZ and JSON files.

        Parameters
        ----------
        name : str
            Table name
        npz_file : Path
            Path to NPZ file with absorption data
        metadata_file : Path
            Path to JSON metadata file
        """
        self.name = name

        # Load absorption data
        data = np.load(npz_file)
        self._coefficients = data['coefficients']
        self._wn_lower = data['wavenumber_lower']
        self._wn_upper = data['wavenumber_upper']

        # Load metadata
        with open(metadata_file, 'r') as f:
            self._metadata = json.load(f)

        self.description = self._metadata['description']
        self.n_intervals = self._metadata['n_intervals']
        self.n_coefficients = self._metadata['n_coefficients']

        # Find valid intervals (non-zero bounds)
        self._valid = (self._wn_lower > 0) | (self._wn_upper > 0)

    def get_coefficients(self, wavenumber: float) -> Optional[np.ndarray]:
        """
        Get absorption coefficients for a given wavenumber.

        Parameters
        ----------
        wavenumber : float
            Wavenumber in cm^-1

        Returns
        -------
        coefficients : ndarray or None
            Array of 6 absorption coefficients, or None if wavenumber
            is outside the table range

        Notes
        -----
        Returns coefficients for the spectral interval that contains
        the given wavenumber. If wavenumber falls in a gap between
        intervals, returns None.
        """
        # Find the interval containing this wavenumber
        for i in range(len(self._wn_lower)):
            if not self._valid[i]:
                continue

            wn_low = self._wn_lower[i]
            wn_high = self._wn_upper[i]

            if wn_low <= wavenumber <= wn_high:
                return self._coefficients[i]

        return None

    def get_interval(self, index: int) -> Tuple[np.ndarray, float, float]:
        """
        Get absorption data for a specific interval.

        Parameters
        ----------
        index : int
            Interval index (0-255)

        Returns
        -------
        coefficients : ndarray
            Array of 6 absorption coefficients
        wn_lower : float
            Lower wavenumber bound (cm^-1)
        wn_upper : float
            Upper wavenumber bound (cm^-1)

        Raises
        ------
        IndexError
            If index is out of range
        """
        if index < 0 or index >= len(self._coefficients):
            raise IndexError(
                f"Interval {index} out of range (0-{len(self._coefficients)-1})"
            )

        return (
            self._coefficients[index],
            self._wn_lower[index],
            self._wn_upper[index]
        )

    def wavenumber_range(self) -> Tuple[float, float]:
        """
        Get the full wavenumber range covered by this table.

        Returns
        -------
        wn_min : float
            Minimum wavenumber (cm^-1)
        wn_max : float
            Maximum wavenumber (cm^-1)
        """
        valid_lower = self._wn_lower[self._valid]
        valid_upper = self._wn_upper[self._valid]

        if len(valid_lower) == 0:
            return (0.0, 0.0)

        return (np.min(valid_lower), np.max(valid_upper))

    def get_all_intervals(self) -> List[Tuple[np.ndarray, float, float]]:
        """
        Get all valid spectral intervals.

        Returns
        -------
        intervals : list
            List of (coefficients, wn_lower, wn_upper) tuples for
            all valid intervals
        """
        intervals = []
        for i in range(len(self._coefficients)):
            if self._valid[i]:
                intervals.append(self.get_interval(i))
        return intervals

    def __repr__(self):
        wn_min, wn_max = self.wavenumber_range()
        return (
            f"AbsorptionTable(name='{self.name}', "
            f"intervals={self.n_intervals}, "
            f"range={wn_min:.0f}-{wn_max:.0f} cm^-1)"
        )


# Cache loaded tables
_ABSORPTION_CACHE: Dict[str, AbsorptionTable] = {}


def load_absorption(table_name: str) -> AbsorptionTable:
    """
    Load atmospheric absorption table.

    Parameters
    ----------
    table_name : str
        Table name (case-insensitive)
        Valid tables:
        - Water vapor: wava1, wava2, wava3, wava4, wava5, wava6
        - NO2: niox1, niox2, niox3, niox4, niox5, niox6
        - O2: oxyg3, oxyg4, oxyg5, oxyg6
        - O3: ozon1
        - CO: moca1, moca2, moca3, moca4, moca5, moca6
        - CO2: dica1, dica2, dica3
        - CH4: meth1, meth2, meth3, meth4, meth5, meth6

    Returns
    -------
    table : AbsorptionTable
        Absorption table object

    Raises
    ------
    ValueError
        If table name is not recognized or files not found

    Examples
    --------
    >>> wava1 = load_absorption('wava1')
    >>> coef = wava1.get_coefficients(2500)
    >>> print(coef)
    [0.62007e-01, 0.24365e+01, ...]
    """
    table_name = table_name.lower()

    # Check cache
    if table_name in _ABSORPTION_CACHE:
        return _ABSORPTION_CACHE[table_name]

    # Find data files
    data_dir = Path(__file__).parent / 'data' / 'absorption'
    npz_file = data_dir / f'{table_name}_absorption.npz'
    json_file = data_dir / f'{table_name}_metadata.json'

    if not npz_file.exists():
        raise ValueError(
            f"Absorption table '{table_name}' not found. "
            f"Expected file: {npz_file}"
        )

    if not json_file.exists():
        raise ValueError(
            f"Metadata for '{table_name}' not found. "
            f"Expected file: {json_file}"
        )

    # Load and cache
    table = AbsorptionTable(table_name, npz_file, json_file)
    _ABSORPTION_CACHE[table_name] = table

    return table


def list_absorption_tables() -> Dict[str, List[str]]:
    """
    List all available absorption tables by category.

    Returns
    -------
    tables : dict
        Dictionary mapping category name to list of table names
    """
    data_dir = Path(__file__).parent / 'data' / 'absorption'

    if not data_dir.exists():
        return {}

    # Categorize tables
    categories = {
        'Water Vapor (H2O)': [],
        'Nitrogen Dioxide (NO2)': [],
        'Oxygen (O2)': [],
        'Ozone (O3)': [],
        'Molecular Chain (CO)': [],
        'Diatomic (CO2)': [],
        'Methane (CH4)': [],
    }

    for npz_file in data_dir.glob('*_absorption.npz'):
        table_name = npz_file.stem.replace('_absorption', '')

        if table_name.startswith('wava'):
            categories['Water Vapor (H2O)'].append(table_name)
        elif table_name.startswith('niox'):
            categories['Nitrogen Dioxide (NO2)'].append(table_name)
        elif table_name.startswith('oxyg'):
            categories['Oxygen (O2)'].append(table_name)
        elif table_name.startswith('ozon'):
            categories['Ozone (O3)'].append(table_name)
        elif table_name.startswith('moca'):
            categories['Molecular Chain (CO)'].append(table_name)
        elif table_name.startswith('dica'):
            categories['Diatomic (CO2)'].append(table_name)
        elif table_name.startswith('meth'):
            categories['Methane (CH4)'].append(table_name)

    # Sort each category
    for category in categories:
        categories[category].sort()

    return categories


__all__ = [
    'AbsorptionTable',
    'load_absorption',
    'list_absorption_tables',
]
