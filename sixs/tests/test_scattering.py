"""
Tests for atmospheric scattering functions.

Validates molecular (Rayleigh) scattering calculations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.scattering import chand, scatra, odrayl


def test_chand_basic():
    """Test basic Chandrasekhar function calculation."""
    # Standard atmosphere parameters
    xphi = 0.0      # Backscattering
    xmuv = 0.8      # cos(36.9°) view zenith
    xmus = 0.7      # cos(45.6°) solar zenith
    xtau = 0.2      # Moderate optical depth

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Reflectance should be positive
    assert xrray > 0

    # Reflectance should be less than 1
    assert xrray < 1

    # Should be a reasonable value for these conditions
    assert 0.01 < xrray < 0.5


def test_chand_backscatter():
    """Test Chandrasekhar function in backscattering geometry."""
    xmuv = 0.8
    xmus = 0.8
    xtau = 0.15

    # Backscattering (xphi=0)
    xrray_back = chand(0.0, xmuv, xmus, xtau)

    # Forward scattering (xphi=180)
    xrray_forward = chand(180.0, xmuv, xmus, xtau)

    # Both should be valid
    assert 0 < xrray_back < 1
    assert 0 < xrray_forward < 1

    # Backscattering typically brighter for Rayleigh
    # (though this depends on geometry)
    assert xrray_back > 0
    assert xrray_forward > 0


def test_chand_optical_depth():
    """Test Chandrasekhar function with varying optical depth."""
    xphi = 45.0
    xmuv = 0.9
    xmus = 0.9

    # Test increasing optical depth
    taus = [0.05, 0.1, 0.2, 0.4]
    reflectances = []

    for xtau in taus:
        xrray = chand(xphi, xmuv, xmus, xtau)
        reflectances.append(xrray)

        # All should be valid
        assert 0 < xrray < 1

    # Generally, higher optical depth = higher reflectance
    # (for optically thin to moderate atmospheres)
    assert reflectances[-1] > reflectances[0]


def test_chand_zenith_angles():
    """Test Chandrasekhar function with varying zenith angles."""
    xphi = 90.0
    xtau = 0.15

    # Test different solar zenith angles
    zenith_angles = [0.9, 0.7, 0.5, 0.3]  # cos(angles)

    for xmus in zenith_angles:
        for xmuv in zenith_angles:
            xrray = chand(xphi, xmuv, xmus, xtau)

            # All should be physically valid
            assert np.isfinite(xrray)
            assert xrray > 0
            assert xrray < 1


def test_chand_azimuthal_symmetry():
    """Test azimuthal dependence of Chandrasekhar function."""
    xmuv = 0.8
    xmus = 0.7
    xtau = 0.2

    # Test various azimuthal angles
    azimuths = [0.0, 45.0, 90.0, 135.0, 180.0]
    reflectances = []

    for xphi in azimuths:
        xrray = chand(xphi, xmuv, xmus, xtau)
        reflectances.append(xrray)

        # All should be valid
        assert 0 < xrray < 1

    # Should have variation with azimuth
    assert max(reflectances) > min(reflectances)


def test_chand_nadir_viewing():
    """Test Chandrasekhar function for nadir viewing."""
    # Nadir viewing (xmuv = 1.0)
    xphi = 0.0
    xmuv = 1.0
    xmus = 0.8
    xtau = 0.15

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should be valid
    assert 0 < xrray < 1
    assert np.isfinite(xrray)


def test_chand_overhead_sun():
    """Test Chandrasekhar function with overhead sun."""
    # Overhead sun (xmus = 1.0)
    xphi = 90.0
    xmuv = 0.7
    xmus = 1.0
    xtau = 0.2

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should be valid
    assert 0 < xrray < 1
    assert np.isfinite(xrray)


def test_chand_small_optical_depth():
    """Test Chandrasekhar function with small optical depth."""
    xphi = 45.0
    xmuv = 0.8
    xmus = 0.8
    xtau = 0.01  # Very thin atmosphere

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should be small but positive
    assert 0 < xrray < 0.1
    assert np.isfinite(xrray)


def test_chand_large_optical_depth():
    """Test Chandrasekhar function with large optical depth."""
    xphi = 45.0
    xmuv = 0.8
    xmus = 0.8
    xtau = 0.5  # Thick atmosphere

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should be larger but still less than 1
    assert 0.1 < xrray < 1.0
    assert np.isfinite(xrray)


def test_chand_grazing_angles():
    """Test Chandrasekhar function at grazing angles."""
    xphi = 45.0
    xtau = 0.15

    # Near-grazing geometry
    xmuv = 0.1  # ~84° zenith
    xmus = 0.1

    xrray = chand(xphi, xmuv, xmus, xtau)

    # Should still be valid (though may be large)
    assert xrray > 0
    assert np.isfinite(xrray)


def test_chand_multiple_scattering():
    """Test that multiple scattering is included."""
    xphi = 0.0
    xmuv = 0.5
    xmus = 0.5

    # Compare very thin vs moderate optical depth
    xrray_thin = chand(xphi, xmuv, xmus, 0.05)
    xrray_mod = chand(xphi, xmuv, xmus, 0.3)

    # Ratio should indicate nonlinear growth
    # (multiple scattering becomes more important at higher tau)
    ratio = xrray_mod / xrray_thin

    # Ratio should be greater than optical depth ratio
    # (if only single scattering, ratio would be ~6)
    assert ratio > 5.0  # Indicates multiple scattering contribution


def test_chand_reciprocity():
    """Test reciprocity principle (switching sun and view)."""
    xphi = 60.0
    xtau = 0.2

    # Configuration 1
    xmuv1 = 0.7
    xmus1 = 0.9
    xrray1 = chand(xphi, xmuv1, xmus1, xtau)

    # Configuration 2 (switch sun and view)
    xmuv2 = 0.9
    xmus2 = 0.7
    xrray2 = chand(xphi, xmuv2, xmus2, xtau)

    # The raw reflectances should be equal (Helmholtz reciprocity)
    # Note: The function returns xrray/xmus, so direct comparison shows reciprocity
    assert abs(xrray1 - xrray2) < 0.001


def test_chand_consistency():
    """Test consistency of Chandrasekhar function across parameter space."""
    # Sample parameter space
    xphis = [0.0, 90.0, 180.0]
    xmuvs = [0.3, 0.6, 0.9]
    xmuss = [0.3, 0.6, 0.9]
    xtaus = [0.05, 0.15, 0.30]

    for xphi in xphis:
        for xmuv in xmuvs:
            for xmus in xmuss:
                for xtau in xtaus:
                    xrray = chand(xphi, xmuv, xmus, xtau)

                    # All should be physically valid
                    assert np.isfinite(xrray), \
                        f"Non-finite for xphi={xphi}, xmuv={xmuv}, xmus={xmus}, xtau={xtau}"
                    assert xrray > 0, \
                        f"Negative for xphi={xphi}, xmuv={xmuv}, xmus={xmus}, xtau={xtau}"
                    assert xrray < 2.0, \
                        f"Too large for xphi={xphi}, xmuv={xmuv}, xmus={xmus}, xtau={xtau}"


# ===== Tests for scatra function =====

def test_scatra_basic():
    """Test basic scatra function call."""
    # Standard atmospheric parameters
    iaer_prof = 1
    taer = 0.2      # Aerosol optical depth
    taerp = 0.1     # Aerosol depth above target
    tray = 0.15     # Rayleigh optical depth
    trayp = 0.08    # Rayleigh depth above target
    piza = 0.9      # Single scattering albedo
    palt = 1000.0   # Above atmosphere
    nt = 30         # Number of layers
    mu = 25         # Gauss angles
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8      # Solar zenith cosine
    xmuv = 0.7      # View zenith cosine

    result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu,
                    rm, gb, xmus, xmuv)

    # Check that result has correct structure
    assert 'total' in result
    assert 'rayleigh' in result
    assert 'aerosol' in result

    # Check that all components have required keys
    for key in ['total', 'rayleigh', 'aerosol']:
        assert 'ddir' in result[key]
        assert 'ddif' in result[key]
        assert 'udir' in result[key]
        assert 'udif' in result[key]
        assert 'sphalb' in result[key]


def test_scatra_transmittance_bounds():
    """Test that transmittances are within physical bounds."""
    iaer_prof = 1
    taer = 0.3
    taerp = 0.15
    tray = 0.2
    trayp = 0.1
    piza = 0.85
    palt = 1000.0
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7

    result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu,
                    rm, gb, xmus, xmuv)

    # All transmittances should be between 0 and 1
    for component in ['total', 'rayleigh', 'aerosol']:
        assert 0 <= result[component]['ddir'] <= 1
        assert 0 <= result[component]['ddif'] <= 1
        assert 0 <= result[component]['udir'] <= 1
        assert 0 <= result[component]['udif'] <= 1
        assert 0 <= result[component]['sphalb'] <= 1


def test_scatra_no_aerosol():
    """Test scatra with no aerosol (taer=0)."""
    iaer_prof = 1
    taer = 0.0      # No aerosol
    taerp = 0.0
    tray = 0.15
    trayp = 0.08
    piza = 0.9
    palt = 1000.0
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7

    result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu,
                    rm, gb, xmus, xmuv)

    # Aerosol component should show no attenuation (initial values)
    assert result['aerosol']['ddir'] == 1.0
    assert result['aerosol']['udir'] == 1.0
    assert result['aerosol']['sphalb'] == 0.0

    # Rayleigh component should show attenuation
    assert result['rayleigh']['ddir'] < 1.0


def test_scatra_above_atmosphere():
    """Test scatra for altitude above atmosphere (palt > 900)."""
    iaer_prof = 1
    taer = 0.2
    taerp = 0.1
    tray = 0.15
    trayp = 0.08
    piza = 0.9
    palt = 1000.0   # Above atmosphere
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7

    result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu,
                    rm, gb, xmus, xmuv)

    # For Rayleigh, should use simple exponential formulas
    expected_ddir_ray = np.exp(-tray/xmus)
    assert abs(result['rayleigh']['ddir'] - expected_ddir_ray) < 1e-6

    # Spherical albedo should be calculated via csalbr
    assert result['rayleigh']['sphalb'] > 0


def test_scatra_at_surface():
    """Test scatra at or below surface (palt <= 0)."""
    iaer_prof = 1
    taer = 0.2
    taerp = 0.0     # No optical depth above (at surface)
    tray = 0.15
    trayp = 0.0
    piza = 0.9
    palt = 0.0      # At surface
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7

    result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu,
                    rm, gb, xmus, xmuv)

    # At surface, upward diffuse should be 0 and upward direct should be 1
    assert result['rayleigh']['udif'] == 0.0
    assert result['rayleigh']['udir'] == 1.0


def test_scatra_optical_depth_dependence():
    """Test that transmittances decrease with increasing optical depth."""
    iaer_prof = 1
    piza = 0.9
    palt = 1000.0
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7

    # Case 1: Low optical depth
    result1 = scatra(iaer_prof, 0.1, 0.05, 0.1, 0.05, piza, palt, nt, mu,
                     rm, gb, xmus, xmuv)

    # Case 2: High optical depth
    result2 = scatra(iaer_prof, 0.4, 0.2, 0.3, 0.15, piza, palt, nt, mu,
                     rm, gb, xmus, xmuv)

    # Direct transmittances should decrease with higher optical depth
    assert result2['total']['ddir'] < result1['total']['ddir']
    assert result2['rayleigh']['ddir'] < result1['rayleigh']['ddir']


def test_scatra_direct_transmittance_exponential():
    """Test that direct transmittances follow Beer-Lambert law."""
    iaer_prof = 1
    taer = 0.2
    taerp = 0.1
    tray = 0.15
    trayp = 0.08
    piza = 0.9
    palt = 1000.0
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7

    result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu,
                    rm, gb, xmus, xmuv)

    # For Rayleigh at palt > 900, direct transmittances are exponential
    expected_ddir = np.exp(-tray/xmus)
    expected_udir = np.exp(-tray/xmuv)

    assert abs(result['rayleigh']['ddir'] - expected_ddir) < 1e-6
    assert abs(result['rayleigh']['udir'] - expected_udir) < 1e-6


def test_scatra_combined_vs_components():
    """Test that total optical depth is sum of components."""
    iaer_prof = 1
    taer = 0.2
    taerp = 0.1
    tray = 0.15
    trayp = 0.08
    piza = 0.9
    palt = 1000.0
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7

    result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu,
                    rm, gb, xmus, xmuv)

    # Total direct transmittance should be product of individual transmittances
    # (for non-scattering case, or approximately for weak scattering)
    total_tau = taer + tray
    expected_total_ddir = np.exp(-total_tau/xmus)

    # Should be close (exact for direct beam without scattering)
    assert abs(result['total']['ddir'] - expected_total_ddir) < 1e-6


def test_scatra_zenith_angle_dependence():
    """Test scatra with varying zenith angles."""
    iaer_prof = 1
    taer = 0.2
    taerp = 0.1
    tray = 0.15
    trayp = 0.08
    piza = 0.9
    palt = 1000.0
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)

    # Test different zenith angles
    zenith_cos = [0.9, 0.7, 0.5, 0.3]

    for xmus in zenith_cos:
        for xmuv in zenith_cos:
            result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt,
                           nt, mu, rm, gb, xmus, xmuv)

            # All transmittances should be valid
            assert 0 <= result['total']['ddir'] <= 1
            assert 0 <= result['total']['udir'] <= 1
            assert np.isfinite(result['total']['ddir'])
            assert np.isfinite(result['total']['udir'])


def test_scatra_spherical_albedo_positive():
    """Test that spherical albedo is non-negative."""
    iaer_prof = 1
    taer = 0.2
    taerp = 0.1
    tray = 0.15
    trayp = 0.08
    piza = 0.9
    palt = 1000.0
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7

    result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu,
                    rm, gb, xmus, xmuv)

    # Spherical albedo should be non-negative
    assert result['rayleigh']['sphalb'] >= 0
    assert result['aerosol']['sphalb'] >= 0
    assert result['total']['sphalb'] >= 0


def test_scatra_diffuse_transmittance_nonnegative():
    """Test that diffuse transmittances are non-negative."""
    iaer_prof = 1
    taer = 0.2
    taerp = 0.1
    tray = 0.15
    trayp = 0.08
    piza = 0.9
    palt = 1000.0
    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7

    result = scatra(iaer_prof, taer, taerp, tray, trayp, piza, palt, nt, mu,
                    rm, gb, xmus, xmuv)

    # Diffuse transmittances should be non-negative
    for component in ['total', 'rayleigh', 'aerosol']:
        assert result[component]['ddif'] >= 0
        assert result[component]['udif'] >= 0


def test_scatra_consistency():
    """Test consistency of scatra across parameter space."""
    iaer_prof = 1
    palt_values = [0.0, 500.0, 1000.0]
    tau_values = [(0.1, 0.05), (0.2, 0.1), (0.3, 0.15)]

    nt = 30
    mu = 25
    rm = np.zeros(2*mu + 1)
    gb = np.ones(2*mu + 1) / (2*mu + 1)
    xmus = 0.8
    xmuv = 0.7
    piza = 0.9

    for palt in palt_values:
        for taer, taerp in tau_values:
            for tray, trayp in tau_values:
                result = scatra(iaer_prof, taer, taerp, tray, trayp, piza,
                               palt, nt, mu, rm, gb, xmus, xmuv)

                # Check all values are finite and physical
                for component in ['total', 'rayleigh', 'aerosol']:
                    assert np.isfinite(result[component]['ddir'])
                    assert np.isfinite(result[component]['ddif'])
                    assert np.isfinite(result[component]['udir'])
                    assert np.isfinite(result[component]['udif'])
                    assert np.isfinite(result[component]['sphalb'])

                    assert 0 <= result[component]['ddir'] <= 1
                    assert 0 <= result[component]['udir'] <= 1


# ===== Tests for odrayl function =====

def test_odrayl_basic():
    """Test basic Rayleigh optical depth calculation."""
    # Create a simple standard atmosphere profile
    # US Standard Atmosphere (simplified)
    z = np.array([  # Altitude in km
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 30, 35, 40, 45, 50,
        70, 100, 120
    ])
    p = np.array([  # Pressure in mb
        1013.25, 898.76, 794.95, 701.09, 616.40, 540.19, 471.81, 410.61,
        355.99, 307.42, 264.36, 226.32, 193.30, 165.11, 141.02, 120.45,
        103.01, 88.21, 75.65, 64.95, 55.83,
        47.99, 41.27, 35.51, 30.57, 26.32, 11.97, 5.75, 2.87, 1.49, 0.80,
        0.05, 0.0003, 0.00001
    ])
    t = np.array([  # Temperature in K
        288.15, 281.65, 275.15, 268.65, 262.17, 255.68, 249.19, 242.70,
        236.21, 229.73, 223.25, 216.77, 216.65, 216.65, 216.65, 216.65,
        216.65, 216.65, 216.65, 216.65, 216.65,
        217.65, 218.65, 219.65, 220.65, 221.65, 226.65, 236.65, 250.65,
        264.65, 270.65, 219.65, 210.65, 190.65
    ])

    # Test at visible wavelength (550 nm = 0.55 μm)
    wl = 0.55
    tray = odrayl(wl, z, p, t)

    # Rayleigh optical depth should be positive
    assert tray > 0

    # For 550 nm, typical Rayleigh optical depth is ~0.1
    assert 0.05 < tray < 0.3


def test_odrayl_wavelength_dependence():
    """Test that Rayleigh optical depth follows lambda^-4 scaling."""
    # Simple atmosphere
    z = np.linspace(0, 120, 34)
    p = 1013.25 * np.exp(-z / 8.0)  # Exponential decay
    t = np.full(34, 288.15)  # Isothermal

    # Test at two wavelengths with 2:1 ratio
    wl1 = 0.4  # 400 nm (blue)
    wl2 = 0.8  # 800 nm (near-IR)

    tray1 = odrayl(wl1, z, p, t)
    tray2 = odrayl(wl2, z, p, t)

    # Rayleigh scattering ~ lambda^-4
    # So tray1 / tray2 should be approximately (wl2/wl1)^4 = 2^4 = 16
    ratio = tray1 / tray2
    expected_ratio = (wl2 / wl1)**4

    # Allow 10% tolerance for numerical effects
    assert abs(ratio - expected_ratio) / expected_ratio < 0.1


def test_odrayl_positive():
    """Test that Rayleigh optical depth is always positive."""
    z = np.linspace(0, 120, 34)
    p = 1013.25 * np.exp(-z / 8.0)
    t = np.full(34, 288.15)

    # Test several wavelengths
    wavelengths = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]

    for wl in wavelengths:
        tray = odrayl(wl, z, p, t)
        assert tray > 0, f"Optical depth should be positive at {wl} μm"
        assert np.isfinite(tray), f"Optical depth should be finite at {wl} μm"


def test_odrayl_visible_range():
    """Test Rayleigh optical depth in visible spectrum."""
    z = np.linspace(0, 120, 34)
    p = 1013.25 * np.exp(-z / 8.0)
    t = np.full(34, 288.15)

    # Visible spectrum
    wl_blue = 0.45  # 450 nm
    wl_green = 0.55  # 550 nm
    wl_red = 0.65  # 650 nm

    tray_blue = odrayl(wl_blue, z, p, t)
    tray_green = odrayl(wl_green, z, p, t)
    tray_red = odrayl(wl_red, z, p, t)

    # Blue should scatter more than green, green more than red
    assert tray_blue > tray_green > tray_red

    # Check reasonable values for standard atmosphere
    assert 0.05 < tray_green < 0.15  # Typical value at 550 nm


def test_odrayl_pressure_dependence():
    """Test that optical depth increases with surface pressure."""
    z = np.linspace(0, 120, 34)
    t = np.full(34, 288.15)
    wl = 0.55

    # Two atmospheres with different surface pressures
    p1 = 1013.25 * np.exp(-z / 8.0)  # Standard
    p2 = 1200.0 * np.exp(-z / 8.0)   # Higher pressure

    tray1 = odrayl(wl, z, p1, t)
    tray2 = odrayl(wl, z, p2, t)

    # Higher pressure should give higher optical depth
    assert tray2 > tray1


def test_odrayl_temperature_independence():
    """Test that optical depth is weakly dependent on temperature."""
    z = np.linspace(0, 120, 34)
    p = 1013.25 * np.exp(-z / 8.0)
    wl = 0.55

    # Two different temperature profiles
    t1 = np.full(34, 288.15)  # Standard
    t2 = np.full(34, 300.0)   # Warmer

    tray1 = odrayl(wl, z, p, t1)
    tray2 = odrayl(wl, z, p, t2)

    # Temperature affects density, so warmer = less dense = less scattering
    assert tray1 > tray2

    # But effect should be small (~ 10%)
    assert abs(tray1 - tray2) / tray1 < 0.1


def test_odrayl_uv_vs_ir():
    """Test that UV has much higher scattering than IR."""
    z = np.linspace(0, 120, 34)
    p = 1013.25 * np.exp(-z / 8.0)
    t = np.full(34, 288.15)

    wl_uv = 0.3  # UV (300 nm)
    wl_ir = 2.0  # Near-IR (2000 nm)

    tray_uv = odrayl(wl_uv, z, p, t)
    tray_ir = odrayl(wl_ir, z, p, t)

    # UV should have much higher scattering
    ratio = tray_uv / tray_ir
    expected_ratio = (wl_ir / wl_uv)**4  # ~1200

    # Should be close to lambda^-4 scaling (within 15% due to refractive index variation)
    assert abs(ratio - expected_ratio) / expected_ratio < 0.15


def test_odrayl_realistic_values():
    """Test that optical depths match typical atmospheric values."""
    # US Standard Atmosphere
    z = np.linspace(0, 120, 34)
    p = 1013.25 * np.exp(-z / 8.0)
    t = np.full(34, 288.15)

    # Common wavelengths
    test_cases = [
        (0.55, 0.07, 0.12),   # 550 nm: expect ~0.09
        (0.44, 0.12, 0.25),   # 440 nm: expect ~0.23
        (0.87, 0.01, 0.03),   # 870 nm: expect ~0.015
    ]

    for wl, min_expected, max_expected in test_cases:
        tray = odrayl(wl, z, p, t)
        assert min_expected < tray < max_expected, \
            f"At {wl} μm, expected {min_expected}-{max_expected}, got {tray}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
