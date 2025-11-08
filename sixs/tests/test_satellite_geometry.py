"""
Tests for satellite geometry functions.

Validates satellite-specific geometry calculations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/6S')

from sixs.geometry import posspo, poslan, posge, posgw, posmto, posnoa


def test_posspo_basic():
    """Test SPOT satellite geometry calculation."""
    # Standard test case
    month = 6
    jday = 15
    tu = 12.0  # Noon
    xlon = 0.0  # Greenwich
    xlat = 45.0  # Mid-latitude

    asol, phi0, avis, phiv = posspo(month, jday, tu, xlon, xlat)

    # Solar angles should be reasonable
    assert 0 <= asol <= 90
    assert 0 <= phi0 <= 360

    # Viewing angles should be zero (nadir)
    assert avis == 0.0
    assert phiv == 0.0


def test_posspo_nadir_viewing():
    """Test that SPOT always returns nadir viewing geometry."""
    # Test multiple locations and times
    test_cases = [
        (1, 1, 10.0, 0.0, 0.0),      # Equator, Greenwich
        (6, 21, 12.0, 100.0, 30.0),  # Summer solstice
        (12, 21, 14.0, -120.0, -45.0), # Winter solstice, southern hemisphere
    ]

    for month, jday, tu, xlon, xlat in test_cases:
        asol, phi0, avis, phiv = posspo(month, jday, tu, xlon, xlat)

        # SPOT is always nadir viewing
        assert avis == 0.0
        assert phiv == 0.0

        # Solar angles should be computed
        assert np.isfinite(asol)
        assert np.isfinite(phi0)


def test_poslan_basic():
    """Test Landsat satellite geometry calculation."""
    # Standard test case
    month = 7
    jday = 4
    tu = 14.0
    xlon = -77.0  # Washington DC longitude
    xlat = 38.9   # Washington DC latitude

    asol, phi0, avis, phiv = poslan(month, jday, tu, xlon, xlat)

    # Solar angles should be reasonable
    assert 0 <= asol <= 90
    assert 0 <= phi0 <= 360

    # Viewing angles should be zero (nadir)
    assert avis == 0.0
    assert phiv == 0.0


def test_poslan_nadir_viewing():
    """Test that Landsat always returns nadir viewing geometry."""
    # Test multiple locations (ensuring sun is above horizon)
    test_cases = [
        (3, 20, 12.0, 0.0, 0.0),       # Equinox, equator, noon
        (6, 21, 12.0, 0.0, 45.0),      # Summer solstice, mid-latitude
        (12, 1, 12.0, 0.0, -30.0),     # Southern hemisphere summer
    ]

    for month, jday, tu, xlon, xlat in test_cases:
        asol, phi0, avis, phiv = poslan(month, jday, tu, xlon, xlat)

        # Landsat is always nadir viewing
        assert avis == 0.0
        assert phiv == 0.0

        # Solar angles should be valid
        assert np.isfinite(asol)
        assert np.isfinite(phi0)


def test_posspo_vs_poslan_consistency():
    """Test that SPOT and Landsat give consistent solar geometry."""
    # Both should give same solar angles for same location/time
    # (they both just call possol)
    month = 6
    jday = 15
    tu = 12.0
    xlon = 50.0
    xlat = 25.0

    asol_spot, phi0_spot, _, _ = posspo(month, jday, tu, xlon, xlat)
    asol_lan, phi0_lan, _, _ = poslan(month, jday, tu, xlon, xlat)

    # Solar angles should be identical
    assert abs(asol_spot - asol_lan) < 1e-10
    assert abs(phi0_spot - phi0_lan) < 1e-10


def test_satellite_geometry_edge_cases():
    """Test satellite geometry at extreme latitudes."""
    # Polar regions
    test_cases = [
        (6, 21, 12.0, 0.0, 85.0),    # Near north pole, summer
        (12, 21, 12.0, 0.0, -85.0),  # Near south pole, summer
        (6, 21, 0.0, 180.0, 89.0),   # Very close to pole
    ]

    for month, jday, tu, xlon, xlat in test_cases:
        asol_spot, phi0_spot, avis_spot, phiv_spot = posspo(month, jday, tu, xlon, xlat)
        asol_lan, phi0_lan, avis_lan, phiv_lan = poslan(month, jday, tu, xlon, xlat)

        # All values should be finite
        assert np.isfinite(asol_spot) and np.isfinite(asol_lan)
        assert np.isfinite(phi0_spot) and np.isfinite(phi0_lan)

        # Viewing angles still zero
        assert avis_spot == 0.0 and avis_lan == 0.0
        assert phiv_spot == 0.0 and phiv_lan == 0.0


def test_satellite_geometry_return_types():
    """Test that return values are correct types."""
    asol, phi0, avis, phiv = posspo(6, 15, 12.0, 0.0, 45.0)

    # All should be floats
    assert isinstance(asol, (float, np.floating))
    assert isinstance(phi0, (float, np.floating))
    assert isinstance(avis, (float, np.floating))
    assert isinstance(phiv, (float, np.floating))


def test_posge_basic():
    """Test GOES East satellite geometry calculation."""
    # Center pixel coordinates
    month = 6
    jday = 15
    tu = 12.0
    nc = 6500  # Near center column
    nl = 8665  # Near center line

    asol, phi0, avis, phiv, xlon, xlat = posge(month, jday, tu, nc, nl)

    # All values should be finite
    assert np.isfinite(asol)
    assert np.isfinite(phi0)
    assert np.isfinite(avis)
    assert np.isfinite(phiv)
    assert np.isfinite(xlon)
    assert np.isfinite(xlat)

    # Solar zenith should be reasonable
    assert 0 <= asol <= 90

    # Viewing zenith should be small for near-center pixels
    assert 0 <= avis < 90

    # Latitude should be close to equator for center pixel
    assert abs(xlat) < 10  # Should be near 0 for center

    # Longitude should be close to GOES East position (75°W)
    assert -100 < xlon < -50


def test_posgw_basic():
    """Test GOES West satellite geometry calculation."""
    # Center pixel coordinates
    month = 6
    jday = 15
    tu = 21.0  # 21:00 UTC = daytime at 135°W
    nc = 6500  # Near center column
    nl = 8665  # Near center line

    asol, phi0, avis, phiv, xlon, xlat = posgw(month, jday, tu, nc, nl)

    # All values should be finite
    assert np.isfinite(asol)
    assert np.isfinite(phi0)
    assert np.isfinite(avis)
    assert np.isfinite(phiv)
    assert np.isfinite(xlon)
    assert np.isfinite(xlat)

    # Solar zenith should be reasonable
    assert 0 <= asol <= 90

    # Viewing zenith should be small for near-center pixels
    assert 0 <= avis < 90

    # Latitude should be close to equator for center pixel
    assert abs(xlat) < 10

    # Longitude should be close to GOES West position (135°W)
    assert -160 < xlon < -110


def test_goes_viewing_angles():
    """Test that GOES satellites have non-zero viewing angles."""
    # Off-center pixel (not nadir)
    month = 6
    jday = 15
    nc = 8000  # Off-center column
    nl = 10000  # Off-center line

    # Use appropriate times for each satellite
    asol_e, phi0_e, avis_e, phiv_e, xlon_e, xlat_e = posge(month, jday, 12.0, nc, nl)
    asol_w, phi0_w, avis_w, phiv_w, xlon_w, xlat_w = posgw(month, jday, 21.0, nc, nl)

    # Viewing angles should be non-zero for off-center pixels
    assert avis_e > 0
    assert avis_w > 0

    # Viewing angles should be reasonable
    assert 0 < avis_e < 90
    assert 0 < avis_w < 90

    # Viewing azimuth should be defined
    assert 0 <= phiv_e <= 360
    assert 0 <= phiv_w <= 360


def test_goes_edge_pixels():
    """Test GOES geometry at edge of visible disk."""
    # Pixels closer to edge (but still valid)
    month = 6
    jday = 15
    tu = 12.0

    # Test a few off-center but valid pixels
    test_pixels = [
        (7000, 9000),
        (6000, 9000),
        (7000, 8000),
    ]

    for nc, nl in test_pixels:
        # Should not raise error for these pixels
        asol, phi0, avis, phiv, xlon, xlat = posge(month, jday, tu, nc, nl)

        # All should be valid
        assert np.isfinite(asol)
        assert np.isfinite(avis)
        assert -180 <= xlon <= 180
        assert -90 <= xlat <= 90


def test_goes_invalid_pixel():
    """Test that invalid pixels raise appropriate error."""
    month = 6
    jday = 15
    tu = 12.0

    # Pixel way outside visible disk
    nc = 20000
    nl = 20000

    with pytest.raises(ValueError, match="outside Earth disk"):
        posge(month, jday, tu, nc, nl)


def test_goes_east_west_different_longitudes():
    """Test that GOES East and West give different longitudes."""
    month = 6
    jday = 15
    nc = 6500
    nl = 8665

    # Use appropriate times for each satellite
    _, _, _, _, xlon_e, xlat_e = posge(month, jday, 12.0, nc, nl)
    _, _, _, _, xlon_w, xlat_w = posgw(month, jday, 21.0, nc, nl)

    # Latitudes should be similar (same pixel position)
    assert abs(xlat_e - xlat_w) < 1.0

    # Longitudes should be different (60° apart for sat positions)
    # GOES East at 75°W, GOES West at 135°W
    assert abs(xlon_e - xlon_w) > 50  # Should differ significantly


def test_posmto_basic():
    """Test Meteosat satellite geometry calculation."""
    # Center pixel coordinates
    month = 6
    jday = 15
    tu = 12.0  # Noon UTC = noon at 0°E
    nc = 2500  # Near center column
    nl = 1250  # Near center line

    asol, phi0, avis, phiv, xlon, xlat = posmto(month, jday, tu, nc, nl)

    # All values should be finite
    assert np.isfinite(asol)
    assert np.isfinite(phi0)
    assert np.isfinite(avis)
    assert np.isfinite(phiv)
    assert np.isfinite(xlon)
    assert np.isfinite(xlat)

    # Solar zenith should be reasonable
    assert 0 <= asol <= 90

    # Viewing zenith should be small for near-center pixels
    assert 0 <= avis < 90

    # Latitude should be close to equator for center pixel
    assert abs(xlat) < 10

    # Longitude should be close to Meteosat position (0°E)
    assert -30 < xlon < 30


def test_posmto_viewing_angles():
    """Test that Meteosat has non-zero viewing angles for off-center pixels."""
    # Off-center pixel
    month = 6
    jday = 15
    tu = 12.0
    nc = 3000  # Off-center column
    nl = 1500  # Off-center line

    asol, phi0, avis, phiv, xlon, xlat = posmto(month, jday, tu, nc, nl)

    # Viewing angle should be non-zero for off-center pixels
    assert avis > 0

    # Viewing angles should be reasonable
    assert 0 < avis < 90
    assert 0 <= phiv <= 360


def test_posmto_invalid_pixel():
    """Test that invalid Meteosat pixels raise appropriate error."""
    month = 6
    jday = 15
    tu = 12.0

    # Pixel way outside visible disk
    nc = 10000
    nl = 10000

    with pytest.raises(ValueError, match="outside Earth disk"):
        posmto(month, jday, tu, nc, nl)


def test_meteosat_vs_goes_consistency():
    """Test that geostationary satellites use consistent algorithm."""
    # All three satellites should give similar viewing angles
    # for pixels at the same angular offset from center
    month = 6
    jday = 15

    # Calculate pixel offsets that are proportional
    # GOES: center at (6498.5, 8665.5)
    # Meteosat: center at (2500.5, 1250.5)

    # Use small offset from center for all
    offset_frac = 0.01  # 1% offset

    # GOES East
    nc_goes = 6498.5 + offset_frac * 6498.5
    nl_goes = 8665.5 + offset_frac * 8665.5
    _, _, avis_goes, _, _, _ = posge(month, jday, 12.0, int(nc_goes), int(nl_goes))

    # Meteosat
    nc_met = 2500.5 + offset_frac * 2500.5
    nl_met = 1250.5 + offset_frac * 1250.5
    _, _, avis_met, _, _, _ = posmto(month, jday, 12.0, int(nc_met), int(nl_met))

    # Both should have small but non-zero viewing angles
    assert 0 < avis_goes < 5
    assert 0 < avis_met < 5


def test_all_geostationary_satellites():
    """Test all three geostationary satellites at once."""
    month = 6
    jday = 15

    # Use appropriate times and center pixels
    asol_ge, phi0_ge, avis_ge, phiv_ge, xlon_ge, xlat_ge = posge(
        month, jday, 12.0, 6500, 8665)
    asol_gw, phi0_gw, avis_gw, phiv_gw, xlon_gw, xlat_gw = posgw(
        month, jday, 21.0, 6500, 8665)
    asol_mt, phi0_mt, avis_mt, phiv_mt, xlon_mt, xlat_mt = posmto(
        month, jday, 12.0, 2500, 1250)

    # All should have small viewing angles at center
    assert avis_ge < 2
    assert avis_gw < 2
    assert avis_mt < 2

    # All should be near equator
    assert abs(xlat_ge) < 5
    assert abs(xlat_gw) < 5
    assert abs(xlat_mt) < 5

    # Longitudes should match satellite positions
    assert -80 < xlon_ge < -70   # GOES East at 75°W
    assert -140 < xlon_gw < -130  # GOES West at 135°W
    assert -5 < xlon_mt < 5       # Meteosat at 0°E


def test_posnoa_basic():
    """Test NOAA polar-orbiting satellite geometry calculation."""
    # NOAA-6 parameters for a typical equator crossing
    month = 6
    jday = 15
    tu = 14.0           # Universal time (14:00 UTC)
    nc = 1024           # Nadir pixel (center)
    xlonan = -75.0      # Longitude of ascending node
    hna = 14.0          # Hour at ascending node (equator crossing)

    asol, phi0, avis, phiv, xlon, xlat = posnoa(month, jday, tu, nc, xlonan, hna)

    # All values should be finite
    assert np.isfinite(asol)
    assert np.isfinite(phi0)
    assert np.isfinite(avis)
    assert np.isfinite(phiv)
    assert np.isfinite(xlon)
    assert np.isfinite(xlat)

    # Solar zenith should be reasonable
    assert 0 <= asol <= 180

    # At nadir, viewing angle should be close to zero
    assert avis < 5

    # At ascending node at equator crossing time, should be near equator
    assert abs(xlat) < 10


def test_posnoa_nadir_viewing():
    """Test that NOAA nadir pixel has zero viewing azimuth."""
    month = 6
    jday = 15
    tu = 12.0
    nc = 1024  # Nadir pixel
    xlonan = 0.0
    hna = 12.0

    asol, phi0, avis, phiv, xlon, xlat = posnoa(month, jday, tu, nc, xlonan, hna)

    # Nadir should have zero viewing azimuth
    assert phiv == 0.0

    # Viewing zenith should be very small at nadir
    assert avis < 1.0


def test_posnoa_off_nadir():
    """Test NOAA off-nadir viewing geometry."""
    month = 6
    jday = 15
    tu = 12.0
    xlonan = 0.0
    hna = 12.0

    # Test several off-nadir pixels
    test_pixels = [1500, 1700, 500]

    for nc in test_pixels:
        asol, phi0, avis, phiv, xlon, xlat = posnoa(month, jday, tu, nc, xlonan, hna)

        # Off-nadir should have non-zero viewing angle
        assert avis > 0

        # Viewing angle should be reasonable (within scan range)
        assert 0 < avis < 60  # Max scan is ±55.385°

        # Viewing azimuth should be defined
        assert np.isfinite(phiv)


def test_posnoa_edge_pixels():
    """Test NOAA at edge of scan swath."""
    month = 6
    jday = 15
    tu = 12.0
    xlonan = 0.0
    hna = 12.0

    # Edge pixels (max scan angle ~55°)
    nc_edge1 = 1  # Left edge
    nc_edge2 = 2048  # Right edge

    asol1, phi01, avis1, phiv1, xlon1, xlat1 = posnoa(month, jday, tu, nc_edge1, xlonan, hna)
    asol2, phi02, avis2, phiv2, xlon2, xlat2 = posnoa(month, jday, tu, nc_edge2, xlonan, hna)

    # Both edges should have large viewing angles
    assert avis1 > 40
    assert avis2 > 40

    # Viewing angles should be similar at both edges
    assert abs(avis1 - avis2) < 5

    # Longitudes should be different (opposite sides)
    assert xlon1 != xlon2


def test_posnoa_orbital_mechanics():
    """Test NOAA orbital mechanics calculations."""
    month = 6
    jday = 21  # Summer solstice - better sun coverage
    xlonan = 0.0
    hna = 12.0
    nc = 1024

    # Test at different times (different positions in orbit)
    # Use times close to ascending node to ensure daylight
    times = [12.0, 12.1, 12.2, 12.3]
    latitudes = []

    for tu in times:
        try:
            _, _, _, _, _, xlat = posnoa(month, jday, tu, nc, xlonan, hna)
            latitudes.append(xlat)
        except ValueError:
            # Sun below horizon - skip this time
            pass

    # Should have gotten some valid results
    assert len(latitudes) > 0

    # All latitudes should be valid
    for lat in latitudes:
        assert -90 <= lat <= 90


def test_posnoa_am_vs_pm_platform():
    """Test NOAA AM vs PM platform differences."""
    month = 6
    jday = 15
    tu = 12.0
    nc = 1024
    xlonan = 0.0
    hna = 12.0

    # AM platform (campm = +1)
    asol_am, phi0_am, avis_am, phiv_am, xlon_am, xlat_am = posnoa(
        month, jday, tu, nc, xlonan, hna, campm=1.0)

    # PM platform (campm = -1)
    asol_pm, phi0_pm, avis_pm, phiv_pm, xlon_pm, xlat_pm = posnoa(
        month, jday, tu, nc, xlonan, hna, campm=-1.0)

    # Solar angles might differ due to different ground locations
    # But viewing angles at nadir should be similar
    assert abs(avis_am - avis_pm) < 1.0

    # Locations may differ due to orbital direction
    # All values should be valid
    assert -180 <= xlon_am <= 180
    assert -180 <= xlon_pm <= 180
    assert -90 <= xlat_am <= 90
    assert -90 <= xlat_pm <= 90


def test_posnoa_scan_symmetry():
    """Test that NOAA scan is symmetric around nadir."""
    month = 6
    jday = 15
    tu = 12.0
    xlonan = 0.0
    hna = 12.0

    # Symmetric pixels around nadir (1024)
    offset = 200
    nc_left = 1024 - offset
    nc_right = 1024 + offset

    _, _, avis_left, _, _, _ = posnoa(month, jday, tu, nc_left, xlonan, hna)
    _, _, avis_right, _, _, _ = posnoa(month, jday, tu, nc_right, xlonan, hna)

    # Viewing angles should be similar for symmetric pixels
    assert abs(avis_left - avis_right) < 1.0


def test_posnoa_latitude_bounds():
    """Test that NOAA polar orbit reaches high latitudes."""
    # Test at a time when satellite is far from equator
    month = 6
    jday = 21  # Summer solstice
    xlonan = 0.0
    hna = 12.0
    nc = 1024

    # Sample various times to find high latitudes (daytime observations only)
    max_lat = 0
    for tu in np.linspace(11.5, 13.5, 30):
        try:
            _, _, _, _, _, xlat = posnoa(month, jday, tu, nc, xlonan, hna)
            max_lat = max(max_lat, abs(xlat))
        except ValueError:
            # Sun below horizon - skip this time
            continue

    # Polar orbits should reach moderate to high latitudes during daytime
    # Conservative threshold since we're filtering for daylight only
    assert max_lat > 30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
