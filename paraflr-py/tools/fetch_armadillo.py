#!/usr/bin/env python3
"""Download Armadillo's (header-only) sources and place its ``include/`` dir at
``<dest>/include`` — for building the paraflr wheel in CI, where Armadillo is
not installed. We only need the headers: the extension uses Armadillo in
header-only mode (ARMA_DONT_USE_WRAPPER) and links BLAS/LAPACK itself.

    python fetch_armadillo.py <dest_dir>

Override the version or URL if a mirror is down:
    PARAFLR_ARMADILLO_VERSION=12.8.4
    PARAFLR_ARMADILLO_URL=https://.../armadillo-12.8.4.tar.xz
"""
import os
import sys
import tarfile
import tempfile
import shutil
import urllib.request

VERSION = os.environ.get("PARAFLR_ARMADILLO_VERSION", "12.8.4")
URLS = [u for u in (
    os.environ.get("PARAFLR_ARMADILLO_URL", ""),
    # GitLab archive is a reliable, direct tarball (SourceForge serves an HTML
    # mirror page to non-interactive downloaders, so it is only a fallback).
    f"https://gitlab.com/conradsnicta/armadillo-code/-/archive/{VERSION}/armadillo-code-{VERSION}.tar.gz",
    f"https://downloads.sourceforge.net/project/arma/armadillo-{VERSION}.tar.xz",
) if u]


def _extract(tar_path, dest):
    with tarfile.open(tar_path) as t:
        try:
            t.extractall(dest, filter="data")   # Python >= 3.12
        except TypeError:
            t.extractall(dest)


def _download(url, dest_file):
    # A browser User-Agent avoids the 403 some mirrors (SourceForge) return to
    # the default urllib agent; follow redirects to the actual mirror.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if data[:16].lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise RuntimeError("got an HTML page, not an archive (mirror redirect?)")
    if len(data) < 100_000:
        raise RuntimeError(f"suspiciously small download ({len(data)} bytes)")
    with open(dest_file, "wb") as f:
        f.write(data)


def main():
    dest = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "arma")
    inc = os.path.join(dest, "include")
    if os.path.isdir(os.path.join(inc, "armadillo_bits")):
        print("armadillo headers already present at", inc)
        return
    os.makedirs(dest, exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), "armadillo.tar.xz")

    last = None
    for url in URLS:
        try:
            print("downloading", url)
            _download(url, tmp)
            _extract(tmp, dest)
            sub = next(d for d in os.listdir(dest)
                       if d.startswith("armadillo")
                       and os.path.isdir(os.path.join(dest, d, "include")))
            src_inc = os.path.join(dest, sub, "include")
            if os.path.exists(inc):
                shutil.rmtree(inc)
            shutil.move(src_inc, inc)
            print("armadillo headers ready at", inc)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            print("  failed:", e)
    raise SystemExit(f"could not fetch Armadillo {VERSION}: {last}")


if __name__ == "__main__":
    main()
