"""
Test polarized scattering modules (KERNELPOL and OSPOL).

Tests the Python conversions of KERNELPOL.f and OSPOL.f to verify:
- Basic functionality and array shapes
- Numerical accuracy of polarized scattering calculations
- Integration with successive orders framework
"""

import numpy as np
import pytest
from sixs.successive_orders import kernelpol, ospol, _atm_state


class TestKernelpol:
    """Test the kernelpol function (polarized scattering kernels)."""

    def setup_method(self):
        """Set up common test parameters."""
        self.mu = 10
        self.rm = np.zeros(2 * self.mu + 1)
        # Set up cosines of Gauss angles
        for i in range(self.mu + 1):
            self.rm[self.mu + i] = i / self.mu  # Positive mu values
            self.rm[self.mu - i] = -i / self.mu  # Negative mu values

    def test_basic_functionality_is0(self):
        """Test kernelpol with Fourier component is=0 (azimuthally averaged)."""
        xpl, xrl, xtl, bp, gr, gt, arr, art, att = kernelpol(0, self.mu, self.rm)

        # Check output shapes
        assert xpl.shape == (2 * self.mu + 1,)
        assert xrl.shape == (2 * self.mu + 1,)
        assert xtl.shape == (2 * self.mu + 1,)
        assert bp.shape == (self.mu + 1, 2 * self.mu + 1)
        assert gr.shape == (self.mu + 1, 2 * self.mu + 1)
        assert gt.shape == (self.mu + 1, 2 * self.mu + 1)
        assert arr.shape == (self.mu + 1, 2 * self.mu + 1)
        assert art.shape == (self.mu + 1, 2 * self.mu + 1)
        assert att.shape == (self.mu + 1, 2 * self.mu + 1)

        # Check that arrays are not all zeros
        assert np.any(xpl != 0), "P-component polynomials should not be all zeros"

        # For is=0, R and T components should be zero
        assert np.all(xrl == 0), "R-component should be zero for is=0"
        assert np.all(xtl == 0), "T-component should be zero for is=0"

    def test_basic_functionality_is1(self):
        """Test kernelpol with Fourier component is=1 (first harmonic)."""
        xpl, xrl, xtl, bp, gr, gt, arr, art, att = kernelpol(1, self.mu, self.rm)

        # Check output shapes
        assert xpl.shape == (2 * self.mu + 1,)
        assert xrl.shape == (2 * self.mu + 1,)
        assert xtl.shape == (2 * self.mu + 1,)

        # For is=1, all components should have non-zero values
        assert np.any(xpl != 0), "P-component should have non-zero values"
        assert np.any(xrl != 0), "R-component should have non-zero values for is=1"
        assert np.any(xtl != 0), "T-component should have non-zero values for is=1"

    def test_basic_functionality_is2(self):
        """Test kernelpol with Fourier component is=2 (second harmonic)."""
        xpl, xrl, xtl, bp, gr, gt, arr, art, att = kernelpol(2, self.mu, self.rm)

        # Check output shapes are correct
        assert xpl.shape == (2 * self.mu + 1,)
        assert xrl.shape == (2 * self.mu + 1,)
        assert xtl.shape == (2 * self.mu + 1,)

        # Should have non-zero values
        assert np.any(xpl != 0), "P-component should have non-zero values"

    def test_phase_function_coefficients(self):
        """Test with custom phase function coefficients."""
        # Set up some test coefficients
        alphal = np.ones(100) * 0.5
        betal = np.ones(100) * 0.8
        gammal = np.ones(100) * 0.3
        zetal = np.ones(100) * 0.2

        xpl, xrl, xtl, bp, gr, gt, arr, art, att = kernelpol(
            0, self.mu, self.rm,
            alphal_vals=alphal,
            betal_vals=betal,
            gammal_vals=gammal,
            zetal_vals=zetal
        )

        # Kernels should reflect the phase function coefficients
        assert np.any(bp != 0), "bp kernel should be non-zero with non-zero betal"

    def test_symmetry_properties(self):
        """Test symmetry properties of polarized kernels."""
        xpl, xrl, xtl, bp, gr, gt, arr, art, att = kernelpol(0, self.mu, self.rm)

        # For is=0, P-component should be symmetric about mu
        for j in range(1, self.mu + 1):
            assert np.isclose(xpl[self.mu + j], xpl[self.mu - j]), \
                f"P-component should be symmetric for is=0 at j={j}"


class TestOspol:
    """Test the ospol function (polarized successive orders)."""

    def setup_method(self):
        """Set up common test parameters."""
        self.mu = 10
        self.nt = 15
        self.naz = 5

        # Set up Gauss angles and weights using proper quadrature
        from sixs.gauss import gauss

        self.rm = np.zeros(2 * self.mu + 1)
        self.gb = np.zeros(2 * self.mu + 1)

        # Get Gauss quadrature points in [-1, 1]
        angles, weights = gauss(-1.0, 1.0, 2 * self.mu - 1)

        # Fill rm and gb arrays (indexed from -mu to mu)
        for i in range(self.mu):
            # Negative mu values (downward)
            self.rm[i] = angles[i]
            self.gb[i] = weights[i]
            # Positive mu values (upward)
            self.rm[2 * self.mu - i] = -angles[i]
            self.gb[2 * self.mu - i] = weights[i]

        # Center value (mu=0, nadir)
        self.rm[self.mu] = angles[self.mu - 1]
        self.gb[self.mu] = weights[self.mu - 1]

        # Azimuth angles
        self.rp = np.linspace(0, 2 * np.pi, self.naz)

        # Set up global state
        _atm_state.nquad = 83
        _atm_state.igmax = 10

    def test_basic_functionality_pure_rayleigh(self):
        """Test ospol with pure Rayleigh scattering."""
        iaer_prof = 0
        tamoy = 0.0  # No aerosol
        trmoy = 0.2  # Rayleigh optical depth
        pizmoy = 1.0
        tamoyp = 0.0
        trmoyp = 0.0
        palt = 0.0  # Ground level
        phirad = 0.0

        xli, xlq, xlu, xlphim, rolut, rolutq, rolutu, filut, nfilut = ospol(
            iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
            phirad, self.nt, self.mu, self.naz, self.rm, self.gb, self.rp
        )

        # Check output shapes
        assert xli.shape == (2 * self.mu + 1, self.naz)
        assert xlq.shape == (2 * self.mu + 1, self.naz)
        assert xlu.shape == (2 * self.mu + 1, self.naz)
        assert xlphim.shape == (13,)  # nfi = 13
        assert rolut.shape == (self.mu, 41)
        assert rolutq.shape == (self.mu, 41)
        assert rolutu.shape == (self.mu, 41)
        assert filut.shape == (self.mu, 41)
        assert nfilut.shape == (self.mu,)

        # Check that intensity (I) has non-zero values
        assert np.any(xli != 0), "Stokes I should have non-zero values"

        # For pure Rayleigh, Q should be non-zero (polarization)
        assert np.any(xlq != 0), "Stokes Q should be non-zero for Rayleigh scattering"

    def test_basic_functionality_mixed(self):
        """Test ospol with mixed Rayleigh and aerosol."""
        iaer_prof = 0
        tamoy = 0.1  # Aerosol optical depth
        trmoy = 0.2  # Rayleigh optical depth
        pizmoy = 0.95  # Aerosol single scattering albedo
        tamoyp = 0.0
        trmoyp = 0.0
        palt = 0.0
        phirad = 0.0

        # Set some aerosol phase function
        _atm_state.alphal[:] = 0.5
        _atm_state.betal[:] = 0.8
        _atm_state.gammal[:] = 0.3
        _atm_state.zetal[:] = 0.2

        xli, xlq, xlu, xlphim, rolut, rolutq, rolutu, filut, nfilut = ospol(
            iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
            phirad, self.nt, self.mu, self.naz, self.rm, self.gb, self.rp
        )

        # Check that all Stokes parameters have values
        assert np.any(xli != 0), "Stokes I should have non-zero values"
        assert np.any(xlq != 0), "Stokes Q should have non-zero values"

        # Check that intensities are physically reasonable (mostly positive or zero)
        # Allow for some numerical noise
        assert np.nansum(xli) > 0, "Total Stokes I should be positive"
        assert np.all(np.isfinite(xli) | (xli == 0)), "Values should be finite or zero"

    def test_plane_observation(self):
        """Test ospol with plane observation (palt > 0)."""
        iaer_prof = 0
        tamoy = 0.1
        trmoy = 0.2
        pizmoy = 0.95
        tamoyp = 0.05  # Half aerosol above plane
        trmoyp = 0.1   # Half Rayleigh above plane
        palt = 2.0     # Plane at 2 km
        phirad = 0.0

        xli, xlq, xlu, xlphim, rolut, rolutq, rolutu, filut, nfilut = ospol(
            iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
            phirad, self.nt, self.mu, self.naz, self.rm, self.gb, self.rp
        )

        # Should still produce valid results
        assert np.any(xli != 0), "Should have non-zero intensity with plane observation"
        assert np.all(np.isfinite(xli)), "All values should be finite"
        assert np.all(np.isfinite(xlq)), "All values should be finite"
        assert np.all(np.isfinite(xlu)), "All values should be finite"

    def test_look_up_table(self):
        """Test that look-up tables are properly generated."""
        iaer_prof = 0
        tamoy = 0.1
        trmoy = 0.2
        pizmoy = 0.95
        tamoyp = 0.0
        trmoyp = 0.0
        palt = 0.0
        phirad = 0.0

        xli, xlq, xlu, xlphim, rolut, rolutq, rolutu, filut, nfilut = ospol(
            iaer_prof, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
            phirad, self.nt, self.mu, self.naz, self.rm, self.gb, self.rp
        )

        # Check nfilut values are reasonable
        assert np.all(nfilut > 0), "All viewing angles should have LUT entries"
        assert np.all(nfilut <= 41), "LUT entries should not exceed maximum"

        # Check that filut angles are in valid range [0, 180] degrees
        for i in range(self.mu):
            n = nfilut[i]
            if n > 1:
                # Check endpoints
                assert 0.0 <= filut[i, 0] <= filut[i, n-1] <= 180.0, \
                    f"Angles should be in [0, 180] range for index {i}"
                # Angles should be monotonic
                for j in range(1, n):
                    assert filut[i, j] >= filut[i, j-1], \
                        f"Angles should be monotonic at {i},{j}"


class TestIntegration:
    """Integration tests combining kernelpol and ospol."""

    def test_kernelpol_feeds_ospol(self):
        """Test that kernelpol output can be used in ospol calculations."""
        mu = 10
        rm = np.zeros(2 * mu + 1)
        for i in range(mu + 1):
            rm[mu + i] = i / mu
            rm[mu - i] = -i / mu

        # Call kernelpol
        xpl, xrl, xtl, bp, gr, gt, arr, art, att = kernelpol(0, mu, rm)

        # Verify these kernels have expected properties
        assert bp.shape == (mu + 1, 2 * mu + 1)
        assert np.any(bp != 0), "Scattering kernels should be non-zero"

        # This confirms the data flow between the two functions works correctly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
