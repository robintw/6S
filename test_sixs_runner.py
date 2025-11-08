#!/usr/bin/env python3
"""
Test script for 6SV Python implementation.

This script demonstrates how to:
1. Parse a 6SV input file using SixSInput
2. Run the radiative transfer computation using SixSRunner
3. Format and display results using SixSOutput

Usage:
    python test_sixs_runner.py [input_file]

If no input file is provided, defaults to 'test_input.txt'
"""

import sys
from pathlib import Path

from sixs.input_parser import SixSInput
from sixs.sixs_runner import SixSRunner
from sixs.output_formatter import SixSOutput


def main():
    """Main test function."""
    # Get input file path
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = Path(__file__).parent / "test_input.txt"

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading input from: {input_file}")
    print("=" * 80)

    # Parse input file
    try:
        with open(input_file, 'r') as f:
            input_data = SixSInput.from_file(f)
        print(f"✓ Input parsed successfully")
        print(f"  Geometry: Solar zenith={input_data.asol}°, View zenith={input_data.avis}°")
        print(f"  Atmosphere: Model {input_data.idatm}")
        print(f"  Aerosol: Model {input_data.iaer}, Visibility={input_data.v} km")
        print(f"  Spectral: Mode {input_data.iwave}, Wavelength={input_data.wl} µm")
        print(f"  Surface: Type {input_data.igroun}, Reflectance={input_data.roc}")
    except Exception as e:
        print(f"✗ Failed to parse input: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("Running 6SV computation...")
    print("=" * 80)

    # Run computation
    try:
        runner = SixSRunner(input_data)
        results = runner.run()
        print(f"✓ Computation completed successfully")
        print(f"  Mean wavelength: {results.wlmoy:.4f} µm")
        print(f"  Rayleigh optical depth: {results.trmoy:.6f}")
        print(f"  Aerosol optical depth (550nm): {results.taer55:.6f}")
        print(f"  Integrated reflectance: {results.refet:.6f}")
        print(f"  Integrated radiance: {results.alumet:.6f}")
    except Exception as e:
        print(f"✗ Computation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("Formatting output...")
    print("=" * 80)

    # Format and write output
    try:
        output = SixSOutput(input_data, results)
        output.write(sys.stdout)
        print(f"\n✓ Output written successfully")
    except Exception as e:
        print(f"✗ Failed to write output: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
