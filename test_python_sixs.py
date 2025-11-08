#!/usr/bin/env python3
"""
Simple test to verify the Python 6SV implementation can run.

This test creates a minimal input and verifies the computation pipeline works.
"""

import sys
from io import StringIO

from sixs.input_parser import SixSInput
from sixs.sixs_runner import SixSRunner
from sixs.output_formatter import SixSOutput


def test_basic_run():
    """Test basic 6SV run with minimal input."""

    # User-provided input
    input_text = """0
32.000000 264.000000 23.000000 190.000000 7 14
2
2
0.500000
0.000000
0.000000
-1
0.500000
0
0
0
0.3
-1
"""

    print("Testing Python 6SV implementation...")
    print("=" * 70)

    try:
        # Parse input
        print("\n1. Parsing input...")
        input_file = StringIO(input_text)
        input_data = SixSInput.from_file(input_file)
        print(f"   ✓ Input parsed: {input_data}")

        # Run computations
        print("\n2. Running computations...")
        runner = SixSRunner(input_data)
        results = runner.run()
        print(f"   ✓ Computations complete")
        print(f"   - Wavelength: {results.wlmoy:.3f} µm")
        print(f"   - Solar zenith: {results.asol:.2f}°")
        print(f"   - View zenith: {results.avis:.2f}°")
        print(f"   - Scattering angle: {results.adif:.2f}°")
        print(f"   - AOT@550nm: {results.taer55:.4f}")
        print(f"   - Rayleigh OT: {results.trmoy:.4f}")

        # Format output
        print("\n3. Formatting output...")
        output = SixSOutput(input_data, results)
        output_file = StringIO()
        output.write(output_file)
        print(f"   ✓ Output formatted ({len(output_file.getvalue())} chars)")

        # Show full output
        print("\n4. Full output:")
        print("-" * 70)
        print(output_file.getvalue())
        print("-" * 70)

        print("\n✓ All tests passed!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_basic_run())
