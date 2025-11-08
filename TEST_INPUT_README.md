# Test Input File Documentation

This document describes the contents of `test_input.txt` for the 6SV Python implementation.

## Input File: test_input.txt

```
0
32.000000 264.000000 23.000000 190.000000 7 14
2
2
0
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
```

## Line-by-Line Explanation

### Line 1: Geometric Conditions (igeom = 0)
```
0
```
- **igeom = 0**: User-defined geometry (manual angle specification)

### Line 2: Viewing Geometry and Date
```
32.000000 264.000000 23.000000 190.000000 7 14
```
- **asol = 32.0°**: Solar zenith angle
- **phi0 = 264.0°**: Solar azimuth angle
- **avis = 23.0°**: View zenith angle (satellite/sensor)
- **phiv = 190.0°**: View azimuth angle
- **month = 7**: July
- **jday = 14**: 14th day of the month

### Line 3: Atmospheric Model (idatm = 2)
```
2
```
- **idatm = 2**: Midlatitude summer atmosphere
  - Water vapor: ~2.93 g/cm²
  - Ozone: ~0.319 cm-atm

### Line 4: Aerosol Model (iaer = 2)
```
2
```
- **iaer = 2**: Maritime aerosol model
  - Typical oceanic/coastal conditions
  - Moderate hygroscopic growth

### Line 5: Aerosol Optical Thickness Mode
```
0
```
- **0**: Use visibility to determine AOT (next line)

### Line 6: Visibility
```
0.500000
```
- **v = 0.5 km**: Very low visibility (dense aerosol conditions)
  - Will be converted to aerosol optical thickness at 550nm via oda550()

### Line 7: Target Altitude (pps)
```
0.000000
```
- **pps = 0.0**: Target at sea level (no altitude adjustment)

### Line 8: Sensor Altitude (palt)
```
0.000000
```
- **palt = 0.0**: Sensor at ground level (not satellite observation)

### Line 9: Spectral Mode (iwave = -1)
```
-1
```
- **iwave = -1**: Monochromatic calculation at a specific wavelength

### Line 10: Wavelength
```
0.500000
```
- **wl = 0.5 µm**: 500 nm (green light, middle of visible spectrum)

### Line 11: Surface Homogeneity (inhomo = 0)
```
0
```
- **inhomo = 0**: Homogeneous surface (uniform reflectance)

### Line 12: Directional Effects (idirec = 0)
```
0
```
- **idirec = 0**: Lambertian surface (no BRDF, isotropic reflectance)

### Line 13: Ground Type (igroun = 0)
```
0
```
- **igroun = 0**: User-defined constant reflectance (next line)

### Line 14: Surface Reflectance
```
0.3
```
- **roc = 0.3**: Surface reflectance of 30%
  - Typical of light soil, vegetation, or urban areas

### Line 15: Atmospheric Correction Mode (iatmcorr = -1)
```
-1
```
- **iatmcorr = -1**: No atmospheric correction
  - Forward calculation only (simulate TOA signal)

## Expected Computation Flow

1. **Geometry**: Compute scattering angle from solar and view angles
2. **Atmosphere**: Load midlatitude summer profile
3. **Aerosols**: Load maritime model, convert visibility (0.5 km) to AOT
4. **Spectral**: Monochromatic at 500 nm
5. **Surface**: 30% Lambertian reflectance
6. **Radiative Transfer**: Run discrete ordinate method (discom)
7. **Output**: Compute atmospheric reflectance, transmittances, and TOA signal

## Running the Test

```bash
# Run with default test_input.txt
python test_sixs_runner.py

# Run with custom input file
python test_sixs_runner.py my_input.txt
```

## Expected Outputs

The computation should produce:

- **Atmospheric reflectance** (roatm): Contribution from Rayleigh + aerosol scattering
- **Transmittances** (dtdir, dtdif, utdir, utdif): Direct and diffuse, up and down
- **Surface-leaving radiance**: Combined effect of surface + atmosphere
- **TOA reflectance**: What a satellite would observe

For this scenario (dense aerosols, 0.5 km visibility):
- High atmospheric scattering
- Significant path radiance
- Strong aerosol signal masking surface
