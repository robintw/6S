# TODO: Enable Radiative Transfer in Runner

## Current Status
- discom (discrete ordinates) IS implemented in `sixs/discrete_ordinate_computation.py`
- aeroso (aerosol properties) IS implemented in `sixs/aeroso.py`
- Both are commented out in `sixs/sixs_runner.py` lines 18-19
- `_run_radiative_transfer()` uses placeholder values instead of calling discom

## Steps to Enable

### 1. In sixs/sixs_runner.py, line 18-19:
Uncomment:
```python
from sixs.discrete_ordinate_computation import discom
from sixs.aeroso import aeroso
```

### 2. In sixs/sixs_runner.py, SixSResults class (around line 28):
Add missing fields:
```python
# Spherical albedos
sphal: np.ndarray = None  # shape (3, 20)
trayl: np.ndarray = None  # shape (20,)
traypl: np.ndarray = None  # shape (20,)

# Stokes parameters
rqatm: np.ndarray = None  # shape (3, 20)
ruatm: np.ndarray = None  # shape (3, 20)

# Accumulated values for output
sdtotr: float = 0.0
sdtota: float = 0.0
sdtott: float = 0.0
sutotr: float = 0.0
sutota: float = 0.0
sutott: float = 0.0
sasr: float = 0.0
sasa: float = 0.0
sast: float = 0.0
fophsr: float = 0.0
fophsa: float = 0.0
fophst: float = 0.0
foqhsr: float = 0.0
foqhsa: float = 0.0
foqhst: float = 0.0
fouhsr: float = 0.0
fouhsa: float = 0.0
fouhst: float = 0.0
```

### 3. In `__post_init__`:
Initialize new arrays:
```python
if self.sphal is None:
    self.sphal = np.zeros((3, 20))
if self.trayl is None:
    self.trayl = np.zeros(20)
if self.traypl is None:
    self.traypl = np.zeros(20)
if self.rqatm is None:
    self.rqatm = np.zeros((3, 20))
if self.ruatm is None:
    self.ruatm = np.zeros((3, 20))
```

### 4. In `_run_radiative_transfer()` method:
Replace placeholder code (lines 370-383) with actual discom call:

```python
# Get aerosol optical properties
if self.input.iaer != 0:
    ext, ome, gasym, phase, qhase, uhase = aeroso(
        self.input.iaer,
        self.input.c,
        self.results.xmud  # scattering angle cosine
    )
else:
    ext = np.ones(20)
    ome = np.zeros(20)
    gasym = np.zeros(20)
    phase = np.zeros(20)
    qhase = np.zeros(20)
    uhase = np.zeros(20)

# Call discrete ordinates solver
roatm, rqatm, ruatm, dtdir, dtdif, utdir, utdif, sphal, trayl, traypl = discom(
    idatmp=self.idatmp,
    iaer=self.input.iaer,
    iaer_prof=self.input.iaer_prof,
    xmus=self.results.xmus,
    xmuv=self.results.xmuv,
    phi=self.results.phi,
    taer55=self.results.taer55,
    taer55p=self.taer55p,  # Need to set this
    palt=self.input.palt,
    phirad=self.phirad,
    nt=self.nt,
    mu=self.mu,
    np=self.np_angles,
    rm=rm,
    gb=gb,
    rp=rp,
    ftray=self.ftray,
    ipol=1,  # Polarization enabled
    xlm1=self.xlm1,
    xlm2=self.xlm2,
    nfi=self.nfi,
    ext=ext,
    ome=ome,
    gasym=gasym,
    phase=phase,
    qhase=qhase,
    uhase=uhase,
    wldis=self.wldis,
    wlinf=0.25,
    wlsup=4.0,
    alphal=self.alphal,
    betal=self.betal,
    gammal=self.gammal,
    zetal=self.zetal,
    phasel=self.phasel,
    qhasel=self.qhasel,
    uhasel=self.uhasel,
    nquad=self.nquad,
    alt_z=self.input.alt_z if hasattr(self.input, 'alt_z') else None,
    taer_z=self.input.taer_z if hasattr(self.input, 'taer_z') else None,
    taer55_z=self.input.taer55_z if hasattr(self.input, 'taer55_z') else None,
    num_z=self.input.num_z if hasattr(self.input, 'num_z') else 0
)

# Store results
self.results.roatm = roatm
self.results.rqatm = rqatm
self.results.ruatm = ruatm
self.results.dtdir = dtdir
self.results.dtdif = dtdif
self.results.utdir = utdir
self.results.utdif = utdif
self.results.sphal = sphal
self.results.trayl = trayl
self.results.traypl = traypl
```

### 5. Add spectral accumulation method:
After RT, need to accumulate values over spectrum (for monochromatic, this is trivial):
```python
def _accumulate_spectral_values(self):
    """Accumulate values for spectral integration (or use single wavelength)."""
    # For monochromatic (iwave=-1), no integration needed
    # Just use values at the single wavelength

    # Find wavelength index
    wl_idx = int((self.results.wlmoy - 0.25) / 0.0025)
    if wl_idx < 0 or wl_idx >= 20:
        wl_idx = int((0.55 - 0.25) / 0.0025)  # Default to 550nm

    # Extract values at this wavelength
    self.results.sdtotr = self.results.dtdir[0, wl_idx] + self.results.dtdif[0, wl_idx]
    self.results.sdtota = self.results.dtdir[2, wl_idx] + self.results.dtdif[2, wl_idx]
    self.results.sdtott = self.results.dtdir[1, wl_idx] + self.results.dtdif[1, wl_idx]

    self.results.sutotr = self.results.utdir[0, wl_idx] + self.results.utdif[0, wl_idx]
    self.results.sutota = self.results.utdir[2, wl_idx] + self.results.utdif[2, wl_idx]
    self.results.sutott = self.results.utdir[1, wl_idx] + self.results.utdif[1, wl_idx]

    self.results.sasr = self.results.sphal[0, wl_idx]
    self.results.sasa = self.results.sphal[2, wl_idx]
    self.results.sast = self.results.sphal[1, wl_idx]

    # Phase functions would come from interp subroutine
    # For now, use typical values
    self.results.fophsr = 0.75 * (1 + self.results.xmud**2)  # Rayleigh
    self.results.fophsa = 0.24  # Aerosol (needs proper calculation)
    self.results.fophst = (self.results.fophsr * self.results.trmoy +
                          self.results.fophsa * self.results.tamoy) / (self.results.trmoy + self.results.tamoy)
```

### 6. Call accumulation in run() method:
After `_run_radiative_transfer()`:
```python
self._accumulate_spectral_values()
```

### 7. Update output_formatter.py:
Uncomment the call to `_write_detailed_transmittances()` and use the actual accumulated values from results instead of computing them locally.

## Testing
After making these changes:
1. Run `python test_python_sixs.py`
2. Compare output with Fortran 6S
3. Verify transmittances and optical properties match
