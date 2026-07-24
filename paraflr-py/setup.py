"""Build the paraflr C++ core as a Python extension via pybind11.

Requires a C++ toolchain and Armadillo (headers + library) on the system:
  * macOS:  brew install armadillo
  * Debian: apt-get install libarmadillo-dev
Armadillo pulls in a BLAS/LAPACK; on macOS that is the Accelerate framework.
OpenMP is optional and off by default (the `threads` argument then runs the
serial path, giving identical results); set PARAFLR_OPENMP=1 to enable it.
"""
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Common locations for Armadillo headers / libs (Homebrew, MacPorts, apt).
_INC = ["cpp", "/usr/local/include", "/opt/homebrew/include", "/usr/include"]
_LIB = ["/usr/local/lib", "/opt/homebrew/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu"]

# Use Armadillo header-only (ARMA_DONT_USE_WRAPPER) and link BLAS/LAPACK
# directly. This avoids depending on the system libarmadillo runtime (whose
# optional SuperLU/ARPACK links are brittle across package upgrades); we only
# use dense chol/solve/inv, which BLAS/LAPACK alone provide.
extra_compile = []
extra_link = []
libraries = []
if sys.platform == "darwin":
    extra_link += ["-framework", "Accelerate"]     # macOS BLAS/LAPACK
else:
    libraries += ["lapack", "blas"]                 # Linux (openblas also works)
if os.environ.get("PARAFLR_OPENMP") == "1":
    if sys.platform == "darwin":
        extra_compile += ["-Xpreprocessor", "-fopenmp"]
        extra_link += ["-lomp"]
    else:
        extra_compile += ["-fopenmp"]
        extra_link += ["-fopenmp"]

ext = Pybind11Extension(
    "paraflr._core",
    sources=["cpp/bindings_py.cpp"],
    include_dirs=[d for d in _INC if os.path.isdir(d)],
    library_dirs=[d for d in _LIB if os.path.isdir(d)],
    libraries=libraries,
    define_macros=[("ARMA_DONT_USE_WRAPPER", None)],
    cxx_std=14,
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
