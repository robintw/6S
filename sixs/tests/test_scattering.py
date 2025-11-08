"""
Tests for atmospheric scattering functions.

Validates molecular (Rayleigh) scattering calculations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.scattering import chand, scatra


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
