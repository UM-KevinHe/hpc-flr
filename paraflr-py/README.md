# `paraflr` (Python)

A Python build of **ParaFLR** — parallel Firth-corrected logistic regression
with high-dimensional provider effects. It calls the **same C++ core as the R
package** (`cpp/firth_core.hpp`, compiled here via pybind11), so an R fit and a
Python fit of the same data agree to machine precision (see `tests/parity_with_r.py`).

Use this if you work in Python; use [`../paraflr`](../paraflr) if you work in R.
Neither depends on the other.

## Requirements

A C++ toolchain plus **Armadillo** (headers) and a BLAS/LAPACK. No CRAN/PyPI —
you install from source in this repo.

| Platform | Install the prerequisites |
|---|---|
| macOS | `brew install armadillo` (BLAS/LAPACK come from the Accelerate framework) |
| Debian/Ubuntu | `sudo apt-get install libarmadillo-dev` |

Python ≥ 3.8 and NumPy. The build uses `pybind11` (pulled in automatically).

## Install

### Prebuilt wheel (no compiler needed)

If a release provides a wheel for your platform, `pip` picks it automatically —
no C++ toolchain, no Armadillo, no BLAS. Wheels are built for Windows, macOS
(Intel + Apple Silicon), and Linux by the `wheels` GitHub Actions workflow and
attached to each GitHub Release:

```bash
pip install paraflr --no-index --find-links https://github.com/UM-KevinHe/hpc-flr/releases/latest
```

(or download the matching `.whl` from the Releases page and `pip install` it).

### From source

Needs a C++ toolchain, Armadillo headers, and a BLAS/LAPACK (see Requirements):

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
