#!/usr/bin/env python3
"""Download a prebuilt OpenBLAS (BLAS + LAPACK) for Windows and extract it to
``<dest>`` so the wheel build can link it. macOS uses the Accelerate framework
and Linux installs openblas from the package manager, so this is Windows-only.

    python fetch_openblas_windows.py <dest_dir>

The extracted tree has include\\, lib\\ (import library) and bin\\ (the DLL,
which delvewheel bundles into the wheel). Override the release if needed:
    PARAFLR_OPENBLAS_URL=https://.../OpenBLAS-x.y.z-x64.zip
"""
import os
import sys
import tempfile
import zipfile
import urllib.request

VERSION = os.environ.get("PARAFLR_OPENBLAS_VERSION", "0.3.27")
URL = os.environ.get(
    "PARAFLR_OPENBLAS_URL",
    f"https://github.com/OpenMathLib/OpenBLAS/releases/download/"
    f"v{VERSION}/OpenBLAS-{VERSION}-x64.zip")


def main():
    dest = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "openblas")
    if os.path.isdir(os.path.join(dest, "include")) and \
       os.path.isdir(os.path.join(dest, "lib")):
        print("openblas already present at", dest)
        return
    os.makedirs(dest, exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), "openblas.zip")
    print("downloading", URL)
    urllib.request.urlretrieve(URL, tmp)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dest)
    print("openblas ready at", dest)
    for d in ("include", "lib", "bin"):
        print("  ", d, "->", os.path.isdir(os.path.join(dest, d)))


if __name__ == "__main__":
    main()
