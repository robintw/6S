#!/usr/bin/env python3
"""
Script to convert Fortran gas absorption DATA statements to Python numpy arrays.

This script automates the conversion of WAVA, DICA, OXYG, OZON, NIOX, METH, MOCA
Fortran files containing absorption coefficient lookup tables.

Usage:
    python convert_gas_tables.py <fortran_file> <output_file>

Example:
    python convert_gas_tables.py src/WAVA1.f sixs/wava_tables.py
"""

import re
import sys
from pathlib import Path


def parse_fortran_data(fortran_file):
    """
    Parse Fortran DATA statements and extract coefficient arrays.

    Parameters
    ----------
    fortran_file : Path
        Path to Fortran source file

    Returns
    -------
    str
        Subroutine name
    list
        List of coefficient rows (each row has 8 values)
    """
    with open(fortran_file, 'r') as f:
        content = f.read()

    # Extract subroutine name
    match = re.search(r'SUBROUTINE\s+(\w+)', content, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find SUBROUTINE in {fortran_file}")

    subroutine_name = match.group(1).lower()

    # Extract all DATA statement values
    # Pattern: DATA ((ACR(K,J),K=1,8),J= ...) / ... /
    data_pattern = r'DATA\s+\(\(ACR\(K,J\),K=1,8\),J=.*?\)/\s*(.*?)\/'
    matches = re.findall(data_pattern, content, re.DOTALL)

    if not matches:
        raise ValueError(f"Could not find DATA statements in {fortran_file}")

    # Parse all coefficient values
    all_values = []
    for match in matches:
        # Remove comments and line continuations
        values_str = match.replace('&', '').strip()
        # Split by commas and clean up
        values = []
        for line in values_str.split('\n'):
            # Remove Fortran continuation 'A' at start
            line = re.sub(r'^\s*A\s+', '', line)
            # Extract numbers (including scientific notation)
            numbers = re.findall(r'[-+]?\d+\.?\d*[eE]?[-+]?\d*', line)
            values.extend(numbers)
        all_values.extend(values)

    # Group into rows of 8
    num_rows = len(all_values) // 8
    rows = []
    for i in range(num_rows):
        row = [float(all_values[i * 8 + j]) for j in range(8)]
        rows.append(row)

    return subroutine_name, rows


def generate_python_module(subroutine_name, rows, output_file):
    """
    Generate Python module with absorption coefficients.

    Parameters
    ----------
    subroutine_name : str
        Name of the function
    rows : list
        Coefficient data (shape: n x 8)
    output_file : Path
        Output Python file path
    """
    # Determine gas type from name
    gas_map = {
        'wava': 'Water Vapor',
        'dica': 'Carbon Dioxide',
        'oxyg': 'Oxygen',
        'ozon': 'Ozone',
        'niox': 'Nitrous Oxide',
        'meth': 'Methane',
        'moca': 'Carbon Monoxide',
    }

    gas_prefix = subroutine_name[:4]
    gas_name = gas_map.get(gas_prefix, 'Unknown Gas')

    # Extract spectral region
    region_num = subroutine_name[-1]

    code = f'''"""
Absorption coefficients for {gas_name} (spectral region {region_num}).

This module contains absorption coefficient lookup tables for
atmospheric gaseous absorption calculations.

Converted from Fortran {subroutine_name.upper()}.f

Functions:
    {subroutine_name}: Get absorption coefficients for spectral index
"""

import numpy as np


# Absorption coefficient table (256 x 8)
# Each row: [a0, a1, a2, a3, a4, a5, wl_min, wl_max]
ACR = np.array([
'''

    # Write coefficient data
    for row in rows:
        code += '    ['
        code += ', '.join(f'{val:.5e}' for val in row)
        code += '],\n'

    code += '''])


def {subroutine_name}(a, inu):
    """
    Get absorption coefficients for spectral index.

    Parameters
    ----------
    a : ndarray
        Output array for coefficients, shape (8,)
    inu : int
        Spectral index (1-256)

    Returns
    -------
    ndarray
        Absorption coefficients [a0, a1, a2, a3, a4, a5, wl_min, wl_max]
    """
    if inu < 1 or inu > 256:
        raise ValueError(f"Spectral index {{inu}} out of range [1, 256]")

    a[:] = ACR[inu - 1, :]
    return a


__all__ = ['{subroutine_name}', 'ACR']
'''

    with open(output_file, 'w') as f:
        f.write(code)

    print(f"Generated {output_file} ({len(rows)} rows)")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    fortran_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not fortran_file.exists():
        print(f"Error: {fortran_file} not found")
        sys.exit(1)

    print(f"Converting {fortran_file} -> {output_file}")
    subroutine_name, rows = parse_fortran_data(fortran_file)
    generate_python_module(subroutine_name, rows, output_file)
    print("Conversion complete!")


if __name__ == '__main__':
    main()
