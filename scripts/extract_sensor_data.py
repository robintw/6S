#!/usr/bin/env python3
"""
Extract sensor spectral response data from Fortran files.

Parses Fortran DATA statements from sensor files (MODIS, AVHRR, etc.)
and extracts spectral response functions to NPZ files.

Usage:
    python extract_sensor_data.py
"""

import re
import numpy as np
import json
from pathlib import Path


def parse_fortran_data_statement(lines):
    """
    Parse Fortran DATA statement into Python list.

    Handles continuations (lines starting with 'A' or 'a') and
    compact notation like '144*0.' meaning 144 zeros.
    """
    # Join all continuation lines
    data_str = ""
    for line in lines:
        line = line.strip()
        if line.startswith(('A', 'a', '&')):
            # Continuation line
            data_str += " " + line[1:].strip()
        else:
            data_str += " " + line

    # Remove trailing slash and any comments after it
    # Comments in Fortran start with 'c' or 'C'
    if '/' in data_str:
        # Find last slash (DATA statement terminator)
        slash_idx = data_str.rfind('/')
        data_str = data_str[:slash_idx]

    # Parse the data
    values = []

    # Split by commas
    tokens = [t.strip() for t in data_str.split(',')]

    for token in tokens:
        if not token or token == '/':
            continue

        # Remove any trailing slashes or comments from this token
        token = token.split('/')[0].strip()

        if not token:
            continue

        # Handle compact notation like "144*0."
        if '*' in token:
            parts = token.split('*')
            if len(parts) == 2:
                try:
                    count = int(parts[0].strip())
                    value = float(parts[1].strip())
                    values.extend([value] * count)
                except ValueError:
                    # Skip malformed tokens
                    pass
        else:
            try:
                values.append(float(token))
            except ValueError:
                # Skip non-numeric tokens
                pass

    return values


def extract_sensor_file(fortran_file):
    """
    Extract sensor spectral response data from a Fortran file.

    Returns dict with sensor name, bands, and wavelength bounds.
    """
    with open(fortran_file, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Determine sensor name from filename
    sensor_name = Path(fortran_file).stem.lower()

    # Find number of bands from array declaration
    # Look for patterns like: real sr(8,1501)
    array_match = re.search(r'real\s+sr\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', content, re.IGNORECASE)

    if not array_match:
        print(f"Warning: Could not determine array size for {sensor_name}")
        return None

    num_bands = int(array_match.group(1))
    array_size = int(array_match.group(2))

    print(f"Processing {sensor_name}: {num_bands} bands, {array_size} spectral points")

    # Extract DATA statements for each band
    bands = {}

    for band_idx in range(1, num_bands + 1):
        # Find DATA statement for this band
        # Pattern: DATA (SR(N,L),L=1,1501)/ ... /
        pattern = rf'DATA\s*\(\s*SR\s*\(\s*{band_idx}\s*,\s*L\s*\)\s*,\s*L\s*=\s*1\s*,\s*\d+\s*\)\s*/(.+?)(?=DATA|wli|wls|do\s+\d+|\Z)'

        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

        if match:
            data_content = match.group(1)
            # Split into lines and parse
            data_lines = data_content.split('\n')
            values = parse_fortran_data_statement(data_lines)

            if len(values) == array_size:
                bands[band_idx] = values
                print(f"  Band {band_idx}: {len(values)} values extracted")
            else:
                print(f"  Warning: Band {band_idx} has {len(values)} values, expected {array_size}")
                # Pad or truncate to correct size
                if len(values) < array_size:
                    values.extend([0.0] * (array_size - len(values)))
                else:
                    values = values[:array_size]
                bands[band_idx] = values

    # Extract wavelength bounds
    wli = {}
    wls = {}

    # Look for wli(N)=value and wls(N)=value
    wli_matches = re.findall(r'wli\s*\(\s*(\d+)\s*\)\s*=\s*([\d.]+)', content, re.IGNORECASE)
    wls_matches = re.findall(r'wls\s*\(\s*(\d+)\s*\)\s*=\s*([\d.]+)', content, re.IGNORECASE)

    for idx, value in wli_matches:
        wli[int(idx)] = float(value)

    for idx, value in wls_matches:
        wls[int(idx)] = float(value)

    return {
        'sensor': sensor_name,
        'num_bands': num_bands,
        'array_size': array_size,
        'bands': bands,
        'wavelength_lower': wli,
        'wavelength_upper': wls,
    }


def save_sensor_data(sensor_data, output_dir):
    """Save sensor data to NPZ and JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sensor_name = sensor_data['sensor']

    # Save spectral response data as NPZ (efficient for large arrays)
    npz_data = {}
    for band_idx, values in sensor_data['bands'].items():
        npz_data[f'band_{band_idx}'] = np.array(values, dtype=np.float64)

    npz_file = output_dir / f'{sensor_name}_spectral_response.npz'
    np.savez_compressed(npz_file, **npz_data)
    print(f"Saved spectral response to: {npz_file}")

    # Save metadata as JSON
    metadata = {
        'sensor': sensor_name,
        'num_bands': sensor_data['num_bands'],
        'array_size': sensor_data['array_size'],
        'wavelength_range': '0.25-4.0 µm',
        'wavelength_step': 0.0025,
        'bands': {}
    }

    for band_idx in sensor_data['bands'].keys():
        metadata['bands'][f'band_{band_idx}'] = {
            'wavelength_lower': sensor_data['wavelength_lower'].get(band_idx),
            'wavelength_upper': sensor_data['wavelength_upper'].get(band_idx),
        }

    json_file = output_dir / f'{sensor_name}_metadata.json'
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {json_file}")


def main():
    """Extract all sensor data files."""
    # Sensor files to process (from FILE_ORDER_PLAN.md Wave 2.2)
    sensor_files = [
        'MODIS.f',
        'AVHRR.f',
        'MERIS.f',
        'SEAWIFS.f',
        'POLDER.f',
        'AATSR.f',
        'HRV.f',
        'ETM.f',
        'ALI.f',
        'ASTER.f',
        'MAS.f',
        'MSS.f',
        'VGT.f',
        'VIIRS.f',
        'GOES.f',
        'GLI.f',
    ]

    src_dir = Path('/home/user/6S/src')
    output_dir = Path('/home/user/6S/sixs/data/sensors')

    success_count = 0
    fail_count = 0

    for sensor_file in sensor_files:
        fortran_path = src_dir / sensor_file

        if not fortran_path.exists():
            print(f"Warning: File not found: {fortran_path}")
            fail_count += 1
            continue

        try:
            sensor_data = extract_sensor_file(fortran_path)

            if sensor_data:
                save_sensor_data(sensor_data, output_dir)
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"Error processing {sensor_file}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Success: {success_count}/{len(sensor_files)}")
    print(f"Failed: {fail_count}/{len(sensor_files)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
