# Radiative Transfer Implementation Status

## Required for Complete Output

The detailed transmittances and optical properties output page requires values from the full discrete ordinates radiative transfer solver.

###Values that require full RT implementation:

From `interp` subroutine (INTERP.f) which interpolates pre-computed RT results:
- `dtotr`, `dtota`, `dtott` - Downward transmittances (Rayleigh, aerosol, total)
- `utotr`, `utota`, `utott` - Upward transmittances
- `fophsr`, `fophsa`, `fophst` - Phase functions I (Rayleigh, aerosol, total)
- `foqhsr`, `foqhsa`, `foqhst` - Phase functions Q
- `fouhsr`, `fouhsa`, `fouhst` - Phase functions U
- `asray`, `asaer`, `astot` - Spherical albedos
- `rorayl`, `roaero`, `romix` - Atmospheric reflectances

These values come from:
1. Discrete ordinates solver (`os` subroutine in OS.f)
2. Pre-computed lookup tables for different aerosol types
3. Spectral interpolation based on wavelength

## Current Implementation

Currently using optical thicknesses (`tray`, `taer`) which are computed correctly:
- Rayleigh optical thickness via `odrayl`
- Aerosol optical thickness via `oda550` or direct input

## Required Implementation

To match Fortran 6S exactly, need to implement:
1. **OS.f** - Discrete ordinates radiative transfer solver (~2000 lines)
2. **INTERP.f** - Spectral interpolation of RT results
3. Pre-computed lookup tables for standard aerosol models
4. **ABSTRA.f** - Gaseous absorption transmittances for H2O, O3, CO2, O2, NO2, CH4, CO

This is a major undertaking representing the core radiative transfer engine of 6S.
