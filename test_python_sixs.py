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

    # Create minimal input (similar to Example_In_1.txt but simplified)
    input_text = """0
40.0 100.0 45.0 50.0 7 23
8
3.0 3.5
4
0.25 0.25 0.25 0.25
0.5
-0.2
-3.3
-1.5 -3.5
0.25
11
1
2 1 0.5
1
-0.1
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
