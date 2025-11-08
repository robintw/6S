#!/usr/bin/env python3
"""
Extract atmospheric absorption coefficient data from Fortran files.

Parses Fortran DATA statements from absorption files (WAVA, NIOX, OXYG, etc.)
and extracts absorption coefficients to NPZ files.

Each absorption table has the structure:
- acr(8, 256) array:
  - Columns 1-6: Absorption coefficients
  - Columns 7-8: Wavenumber bounds (lower, upper) in cm^-1

Usage:
    python extract_absorption_data.py
"""

import re
import numpy as np
import json
from pathlib import Path


def parse_fortran_data_array(content, array_name='acr', rows=256, cols=8):
    """
    Parse Fortran DATA statement for a 2D array.

    Parameters
    ----------
    content : str
        Full file content
    array_name : str
        Array variable name (e.g., 'acr')
    rows : int
        Number of rows in array
    cols : int
        Number of columns in array

    Returns
    -------
    data : ndarray
        Extracted array data (rows x cols)
    """
    # Find all DATA statements for this array
    # Pattern: data ((acr(k,j),k=1,8),j=  1,  8) / ... /
    pattern = rf'data\s*\(\s*\({array_name}\s*\(k,j\)\s*,\s*k=1,\s*{cols}\s*\)\s*,\s*j=\s*(\d+),\s*(\d+)\s*\)\s*/(.+?)(?=data\s*\(|subroutine|end\s|return|\Z)'

    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)

    # Initialize array
    data = np.zeros((rows, cols), dtype=np.float64)

    for match in matches:
        j_start = int(match[0]) - 1  # Convert to 0-indexed
        j_end = int(match[1]) - 1
        data_content = match[2]

        # Parse the values
        values = parse_values(data_content)

        # Fill the array
        # Values are organized as (k=1..cols for j=j_start), (k=1..cols for j=j_start+1), ...
        expected_count = (j_end - j_start + 1) * cols

        if len(values) >= expected_count:
            for j_offset, j_idx in enumerate(range(j_start, j_end + 1)):
                for k_idx in range(cols):
                    value_idx = j_offset * cols + k_idx
                    if value_idx < len(values):
                        data[j_idx, k_idx] = values[value_idx]
        else:
            print(f"  Warning: Expected {expected_count} values but got {len(values)} for j={j_start+1}..{j_end+1}")

    return data


def parse_values(data_str):
    """
    Parse Fortran DATA values from a string.

    Handles:
    - Continuation lines (starting with 'a' or 'A')
    - Scientific notation (e.g., 0.12345e-02)
    - Trailing slashes and comments
    """
    # Join continuation lines
    lines = data_str.split('\n')
    joined = ""
    for line in lines:
        line = line.strip()
        if line.startswith(('A', 'a', '&')):
            # Continuation line
            joined += " " + line[1:].strip()
        else:
            joined += " " + line

    # Remove trailing slash and any comments
    if '/' in joined:
        slash_idx = joined.rfind('/')
        joined = joined[:slash_idx]

    # Split by comma and parse
    values = []
    tokens = [t.strip() for t in joined.split(',')]

    for token in tokens:
        if not token:
            continue

        # Remove any trailing slashes or comments
        token = token.split('/')[0].strip()

        if not token:
            continue

        # Parse float value
        try:
            value = float(token)
            values.append(value)
        except ValueError:
            # Skip non-numeric tokens
            pass

    return values


def extract_absorption_file(fortran_file):
    """
    Extract absorption coefficient data from a Fortran file.

    Returns dict with file info and coefficient data.
    """
    with open(fortran_file, 'r') as f:
        content = f.read()

    # Determine gas/molecule name from filename
    file_name = Path(fortran_file).stem.lower()

    # Extract comment at top to understand wavenumber range
    comment_match = re.search(r'c\s+(.+?)\((.+?cm-1\))', content, re.IGNORECASE)
    description = comment_match.group(0).strip() if comment_match else f"{file_name} absorption"

    # Extract the acr array (8 columns, 256 rows)
    data = parse_fortran_data_array(content, 'acr', rows=256, cols=8)

    # Check if data was extracted
    if np.all(data == 0):
        print(f"Warning: No data extracted for {file_name}")
        return None

    # Split into coefficients and wavenumber bounds
    coefficients = data[:, :6]  # First 6 columns
    wn_lower = data[:, 6]       # Column 7: lower wavenumber bound
    wn_upper = data[:, 7]       # Column 8: upper wavenumber bound

    # Find valid spectral intervals (non-zero bounds)
    valid_intervals = (wn_lower > 0) | (wn_upper > 0)
    n_valid = np.sum(valid_intervals)

    print(f"  Extracted {n_valid} spectral intervals")

    return {
        'name': file_name,
        'description': description,
        'n_intervals': n_valid,
        'coefficients': coefficients,
        'wavenumber_lower': wn_lower,
        'wavenumber_upper': wn_upper,
    }


def save_absorption_data(absorption_data, output_dir):
    """Save absorption data to NPZ and JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = absorption_data['name']

    # Save absorption coefficients as NPZ (efficient for large arrays)
    npz_data = {
        'coefficients': absorption_data['coefficients'],
        'wavenumber_lower': absorption_data['wavenumber_lower'],
        'wavenumber_upper': absorption_data['wavenumber_upper'],
    }

    npz_file = output_dir / f'{name}_absorption.npz'
    np.savez_compressed(npz_file, **npz_data)
    print(f"  Saved to: {npz_file}")

    # Save metadata as JSON
    metadata = {
        'name': name,
        'description': absorption_data['description'],
        'n_intervals': int(absorption_data['n_intervals']),
        'n_coefficients': 6,
        'coefficient_names': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
        'wavenumber_unit': 'cm^-1',
    }

    json_file = output_dir / f'{name}_metadata.json'
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Extract all absorption data files."""
    # Absorption files by category
    file_groups = {
        'Water Vapor': ['WAVA1', 'WAVA2', 'WAVA3', 'WAVA4', 'WAVA5', 'WAVA6'],
        'Nitrogen Dioxide': ['NIOX1', 'NIOX2', 'NIOX3', 'NIOX4', 'NIOX5', 'NIOX6'],
        'Oxygen': ['OXYG3', 'OXYG4', 'OXYG5', 'OXYG6'],
        'Ozone': ['OZON1'],
        'Molecular Chain': ['MOCA1', 'MOCA2', 'MOCA3', 'MOCA4', 'MOCA5', 'MOCA6'],
        'Diatomic': ['DICA1', 'DICA2', 'DICA3'],
        'Methane': ['METH1', 'METH2', 'METH3', 'METH4', 'METH5', 'METH6'],
    }

    src_dir = Path('/home/user/6S/src')
    output_dir = Path('/home/user/6S/sixs/data/absorption')

    success_count = 0
    fail_count = 0

    for category, files in file_groups.items():
        print(f"\n{category}:")
        print("=" * 60)

        for file_base in files:
            fortran_file = src_dir / f'{file_base}.f'

            if not fortran_file.exists():
                print(f"Warning: File not found: {fortran_file}")
                fail_count += 1
                continue

            print(f"Processing {file_base}...")

            try:
                absorption_data = extract_absorption_file(fortran_file)

                if absorption_data:
                    save_absorption_data(absorption_data, output_dir)
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                print(f"Error processing {file_base}: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1

    total_files = sum(len(files) for files in file_groups.values())

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Success: {success_count}/{total_files}")
    print(f"Failed: {fail_count}/{total_files}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
