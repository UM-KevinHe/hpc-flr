# `paraflr` (Python)

A Python build of **ParaFLR** — parallel Firth-corrected logistic regression
with high-dimensional provider effects. It calls the **same C++ core as the R
package** (`cpp/firth_core.hpp`, compiled here via pybind11), so an R fit and a
Python fit of the same data agree to machine precision (see `tests/parity_with_r.py`).

Use this if you work in Python; use [`../paraflr`](../paraflr) if you work in R.
Neither depends on the other.

## Install

### Prebuilt wheel — no compiler needed (recommended)

Wheels for **Windows, macOS (Intel + Apple Silicon), and Linux** are attached to
each [GitHub Release](https://github.com/UM-KevinHe/hpc-flr/releases). They need
only **Python ≥ 3.8 and NumPy** — no C++ toolchain, no Armadillo, no BLAS
(everything is bundled inside the wheel). Download the `.whl` matching your OS
and Python version from the Releases page, then:

```bash
pip install paraflr-0.1.0-cp311-cp311-macosx_11_0_arm64.whl   # example filename
```

Or let `pip` pick the right wheel from a release automatically:

```bash
pip install paraflr --find-links https://github.com/UM-KevinHe/hpc-flr/releases/expanded_assets/v0.1.0
```

### From source

Building from source additionally needs a C++ toolchain, **Armadillo** headers,
and a BLAS/LAPACK:

| Platform | Install the build prerequisites |
|---|---|
| macOS | `brew install armadillo` (BLAS/LAPACK from the Accelerate framework) |
| Debian/Ubuntu | `sudo apt-get install libarmadillo-dev` |

```bash
pip install ./paraflr-py          # from the repository root
```

or from this directory: `pip install .`. To develop in place:
`python setup.py build_ext --inplace`.

OpenMP is off by default (the `threads` argument then runs the serial path,
which gives identical results). To enable multi-threaded fitting:

```bash
PARAFLR_OPENMP=1 pip install ./paraflr-py    # macOS also needs `brew install libomp`
```

## Usage

```python
import numpy as np
import paraflr

# Y : binary outcome (n,)   Z : covariates (n, p)   ID : provider id (n,)
fit = paraflr.logis_firth(Y, Z, ID, threads=4)

fit["beta"]        # covariate effects (order matches Z's columns)
fit["gamma"]       # provider effects
fit["prov_ids"]    # provider id for each gamma

# Test the first provider's effect against the population median
paraflr.test_gamma_single(fit, method="score")
paraflr.test_gamma_single(fit, method="lrt", firth=True)
```

The API mirrors the R package's `logis_firth()` and `test_gamma.single()`.

## Verifying R/Python agreement

```bash
python tests/parity_with_r.py
# set PARAFLR_RLIB=/path/to/Rlib if paraflr (R) is in a non-default library
```

Fits the same data in both languages and checks the coefficients match. It
skips (does not fail) when R or the R `paraflr` package is not installed.
