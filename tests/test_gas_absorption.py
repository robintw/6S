"""
Test gas absorption coefficient loading.

Tests the Python gas_absorption module to verify:
- NPZ data files load correctly
- Absorption coefficients have expected structure
- All gas/region combinations are accessible
"""

import numpy as np
import pytest
from sixs.gas_absorption import (
    get_absorption_coefficients,
    wava1, wava2, wava3, wava4, wava5, wava6,
    dica1, dica2, dica3,
    oxyg3, oxyg4, oxyg5, oxyg6,
    ozon1,
    niox1, niox2, niox3, niox4, niox5, niox6,
    meth1, meth2, meth3, meth4, meth5, meth6,
    moca1, moca2, moca3, moca4, moca5, moca6,
)


class TestGasAbsorptionLoader:
    """Test loading of gas absorption data."""

    def test_wava1_loads(self):
        """Test water vapor region 1 loads."""
        a = np.zeros(8)
        result = wava1(a, 1)
        assert result is not None
        assert result.shape == (8,)
        # Check that we have wavenumber bounds
        assert a[6] > 0 or a[7] > 0, "Should have wavenumber bounds"

    def test_get_absorption_coefficients_water_vapor(self):
        """Test getting water vapor coefficients."""
        # H2O (idgaz=1) in region 1
        coeffs = get_absorption_coefficients(1, 1, 1)
        assert coeffs is not None
        assert coeffs.shape == (8,)
        assert len(coeffs) == 8
        # First 6 are absorption coefficients, last 2 are wavenumber bounds
        assert coeffs[6] >= 0  # Lower wavenumber bound
        assert coeffs[7] >= 0  # Upper wavenumber bound

    def test_get_absorption_coefficients_co2(self):
        """Test getting CO2 coefficients."""
        # CO2 (idgaz=2) in region 1
        coeffs = get_absorption_coefficients(2, 1, 1)
        assert coeffs is not None
        assert coeffs.shape == (8,)

    def test_get_absorption_coefficients_o2(self):
        """Test getting O2 coefficients."""
        # O2 (idgaz=3) is not in region 1, should return None
        coeffs = get_absorption_coefficients(3, 1, 1)
        assert coeffs is None

        # O2 (idgaz=3) is in region 3
        coeffs = get_absorption_coefficients(3, 3, 1)
        assert coeffs is not None
        assert coeffs.shape == (8,)

    def test_get_absorption_coefficients_o3(self):
        """Test getting O3 coefficients."""
        # O3 (idgaz=4) in region 1
        coeffs = get_absorption_coefficients(4, 1, 1)
        assert coeffs is not None
        assert coeffs.shape == (8,)

    def test_all_wava_regions(self):
        """Test all water vapor regions load."""
        a = np.zeros(8)
        wava_funcs = [wava1, wava2, wava3, wava4, wava5, wava6]
        for i, func in enumerate(wava_funcs, 1):
            result = func(a, 1)
            assert result is not None, f"WAVA{i} should load"
            assert result.shape == (8,)

    def test_all_niox_regions(self):
        """Test all N2O regions load."""
        a = np.zeros(8)
        niox_funcs = [niox1, niox2, niox3, niox4, niox5, niox6]
        for i, func in enumerate(niox_funcs, 1):
            result = func(a, 1)
            assert result is not None, f"NIOX{i} should load"
            assert result.shape == (8,)

    def test_all_meth_regions(self):
        """Test all methane regions load."""
        a = np.zeros(8)
        meth_funcs = [meth1, meth2, meth3, meth4, meth5, meth6]
        for i, func in enumerate(meth_funcs, 1):
            result = func(a, 1)
            assert result is not None, f"METH{i} should load"
            assert result.shape == (8,)

    def test_all_moca_regions(self):
        """Test all CO regions load."""
        a = np.zeros(8)
        moca_funcs = [moca1, moca2, moca3, moca4, moca5, moca6]
        for i, func in enumerate(moca_funcs, 1):
            result = func(a, 1)
            assert result is not None, f"MOCA{i} should load"
            assert result.shape == (8,)

    def test_dica_regions(self):
        """Test CO2 regions load."""
        a = np.zeros(8)
        dica_funcs = [dica1, dica2, dica3]
        for i, func in enumerate(dica_funcs, 1):
            result = func(a, 1)
            assert result is not None, f"DICA{i} should load"
            assert result.shape == (8,)

    def test_oxyg_regions(self):
        """Test O2 regions load."""
        a = np.zeros(8)
        # O2 is only in regions 3-6
        oxyg_funcs = [oxyg3, oxyg4, oxyg5, oxyg6]
        for i, func in enumerate(oxyg_funcs, 3):
            result = func(a, 1)
            assert result is not None, f"OXYG{i} should load"
            assert result.shape == (8,)

    def test_ozon_region(self):
        """Test O3 region loads."""
        a = np.zeros(8)
        result = ozon1(a, 1)
        assert result is not None
        assert result.shape == (8,)

    def test_invalid_spectral_index(self):
        """Test that invalid spectral indices raise errors."""
        with pytest.raises(ValueError, match="out of range"):
            get_absorption_coefficients(1, 1, 0)

        with pytest.raises(ValueError, match="out of range"):
            get_absorption_coefficients(1, 1, 257)

    def test_invalid_gas_id(self):
        """Test that invalid gas IDs raise errors."""
        with pytest.raises(ValueError, match="Invalid gas ID"):
            get_absorption_coefficients(0, 1, 1)

        with pytest.raises(ValueError, match="Invalid gas ID"):
            get_absorption_coefficients(8, 1, 1)

    def test_coefficient_structure(self):
        """Test that coefficients have expected structure."""
        coeffs = get_absorption_coefficients(1, 1, 10)
        if coeffs is not None:
            # Check shape
            assert len(coeffs) == 8

            # Wavenumber bounds should be ordered
            wn_lower = coeffs[6]
            wn_upper = coeffs[7]
            if wn_lower > 0 and wn_upper > 0:
                assert wn_upper >= wn_lower, "Upper bound should be >= lower bound"

    def test_caching(self):
        """Test that data loading is cached."""
        # First call loads data
        coeffs1 = get_absorption_coefficients(1, 1, 1)

        # Second call should use cache (faster)
        coeffs2 = get_absorption_coefficients(1, 1, 1)

        # Should be identical
        assert np.array_equal(coeffs1, coeffs2)

    def test_different_spectral_indices(self):
        """Test loading different spectral indices."""
        # Load several different indices
        indices_to_test = [1, 50, 100, 150, 200, 256]

        for idx in indices_to_test:
            coeffs = get_absorption_coefficients(1, 1, idx)
            assert coeffs is not None
            assert coeffs.shape == (8,)


class TestIntegrationWithAbstra:
    """Test integration with ABSTRA module."""

    def test_abstra_can_load_coefficients(self):
        """Test that ABSTRA can load gas absorption coefficients."""
        from sixs.abstra import abstra

        # Create dummy atmospheric profile
        z = np.linspace(0, 100, 34)
        p = 1013.25 * np.exp(-z / 8.5)
        t = 288.15 - 6.5 * z / 1000
        wh = np.exp(-z / 2.0) * 10
        wo = np.exp(-z / 5.0) * 0.01

        # Test with tropical atmosphere
        result = abstra(
            idatm=1,  # Tropical atmosphere
            wl=0.55,  # 550 nm
            xmus=0.5,  # 60 degree solar zenith
            xmuv=1.0,  # Nadir viewing
            uw=1.424,  # Water vapor
            uo3=0.344,  # Ozone
            uwus=1.424,
            uo3us=0.344,
            idatmp=0,  # Ground level
            uwpl=0.0,
            uo3pl=0.0,
            uwusp=1.424,
            uo3usp=0.344,
            z=z,
            p=p,
            t=t,
            wh=wh,
            wo=wo,
            zpl=z,
            ppl=p,
            tpl=t,
            whpl=wh,
            wopl=wo,
        )

        # Check that transmittances are calculated
        assert 'dtwava' in result
        assert 'utozon' in result
        assert 'ttdica' in result

        # Transmittances should be between 0 and 1
        for key, value in result.items():
            if key.startswith(('dt', 'ut', 'tt')):
                assert 0.0 <= value <= 1.0, f"{key} should be in [0, 1], got {value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
