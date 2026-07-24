"""Build the paraflr C++ core as a Python extension via pybind11.

Requires a C++ toolchain and Armadillo (headers) plus a BLAS/LAPACK:
  * macOS:  brew install armadillo         (BLAS/LAPACK = Accelerate framework)
  * Debian: apt-get install libarmadillo-dev
We use Armadillo header-only (ARMA_DONT_USE_WRAPPER) and link BLAS/LAPACK
directly — only dense chol/solve/inv are used, which BLAS/LAPACK alone provide.
This avoids the system libarmadillo runtime, whose optional SuperLU/ARPACK
links are brittle across package upgrades.

CI overrides (used to build portable wheels; a normal local install ignores
them and falls back to the platform default):
  PARAFLR_ARMADILLO_INCLUDE  extra include dir holding <armadillo>
  PARAFLR_BLAS_INCLUDE       extra include dir for the BLAS headers
  PARAFLR_BLAS_LIBDIR        library dir for the BLAS/LAPACK to link
  PARAFLR_BLAS_LIBS          comma-separated lib names (e.g. "openblas")
  PARAFLR_BLAS_FRAMEWORK     macOS framework to link (e.g. "Accelerate")
  PARAFLR_OPENMP=1           enable OpenMP (off by default; serial == same result)
"""
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _env_list(name):
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


# --- BLAS / LAPACK -----------------------------------------------------------
# If the environment names a BLAS explicitly (CI does, for a self-contained
# wheel), use it; otherwise fall back to the platform default for a local
# source install.
blas_libs = _env_list("PARAFLR_BLAS_LIBS")
blas_framework = os.environ.get("PARAFLR_BLAS_FRAMEWORK")

extra_compile, extra_link, libraries = [], [], []
if blas_libs or blas_framework:
    libraries += blas_libs
    if blas_framework:
        extra_link += ["-framework", blas_framework]
elif sys.platform == "darwin":
    extra_link += ["-framework", "Accelerate"]       # macOS BLAS/LAPACK
else:
    libraries += ["lapack", "blas"]                   # Linux (openblas also works)

if os.environ.get("PARAFLR_OPENMP") == "1":
    if sys.platform == "darwin":
        extra_compile += ["-Xpreprocessor", "-fopenmp"]
        extra_link += ["-lomp"]
    elif sys.platform == "win32":
        extra_compile += ["/openmp"]
    else:
        extra_compile += ["-fopenmp"]
        extra_link += ["-fopenmp"]

# --- include / library dirs --------------------------------------------------
include_dirs = ["cpp"]
for env in ("PARAFLR_ARMADILLO_INCLUDE", "PARAFLR_BLAS_INCLUDE"):
    d = os.environ.get(env)
    if d:
        include_dirs.append(d)
include_dirs += [d for d in ("/usr/local/include", "/opt/homebrew/include",
                             "/usr/include") if os.path.isdir(d)]

library_dirs = []
if os.environ.get("PARAFLR_BLAS_LIBDIR"):
    library_dirs.append(os.environ["PARAFLR_BLAS_LIBDIR"])
library_dirs += [d for d in ("/usr/local/lib", "/opt/homebrew/lib", "/usr/lib",
                             "/usr/lib64", "/usr/lib/x86_64-linux-gnu")
                 if os.path.isdir(d)]

# Force Armadillo to call the external BLAS/LAPACK we linked, regardless of
# what its bundled config.hpp happens to enable (downloaded source headers may
# ship with these commented out).
define_macros = [("ARMA_DONT_USE_WRAPPER", None),
                 ("ARMA_USE_LAPACK", None),
                 ("ARMA_USE_BLAS", None)]

ext = Pybind11Extension(
    "paraflr._core",
    sources=["cpp/bindings_py.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    cxx_std=14,
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
