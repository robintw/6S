#!/usr/bin/env python3
"""
6SV (Second Simulation of a Satellite Signal in the Solar Spectrum, Vector) - Python version

This is a Python port of the 6SV Fortran code for atmospheric correction of satellite imagery.

Main program that reads input files in the same format as the Fortran version,
performs radiative transfer calculations, and outputs results.

Usage:
    python sixs_main.py < input_file.txt > output.txt
"""

import sys
import numpy as np
from sixs.input_parser import SixSInput
from sixs.output_formatter import SixSOutput
from sixs.sixs_runner import SixSRunner


def main():
    """
    Main entry point for 6SV Python version.

    Reads input from stdin (same format as Fortran version),
    performs atmospheric correction calculations,
    and writes results to stdout.
    """
    try:
        # Parse input from stdin
        print("Parsing input...", file=sys.stderr)
        input_data = SixSInput.from_file(sys.stdin)

        # Run 6S calculations
        print("Running 6SV calculations...", file=sys.stderr)
        runner = SixSRunner(input_data)
        results = runner.run()

        # Format and output results
        print("Formatting output...", file=sys.stderr)
        output = SixSOutput(input_data, results)
        output.write(sys.stdout)

        print("Done!", file=sys.stderr)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
