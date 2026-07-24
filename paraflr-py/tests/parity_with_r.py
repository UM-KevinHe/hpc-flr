"""R vs Python parity: the same data fitted by paraflr (R) and paraflr (Python)
must agree to numerical tolerance, since both call the identical C++ core.

    python tests/parity_with_r.py
    PARAFLR_RLIB=/path/to/Rlib python tests/parity_with_r.py   # if paraflr is in a custom R lib

Skips (does not fail) when R or the R paraflr package is unavailable.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paraflr

TOL = 1e-6
_HERE = os.path.dirname(os.path.abspath(__file__))


def make_data(seed: int = 1):
    rng = np.random.default_rng(seed)
    m = 15
    sizes = rng.integers(20, 60, size=m)
    ID = np.repeat(np.arange(1, m + 1), sizes)
    n = int(ID.size)
    Z = rng.standard_normal((n, 3))
    gamma_true = rng.normal(0.0, 0.5, m)
    beta_true = np.array([0.4, -0.3, 0.2])
    eta = np.repeat(gamma_true, sizes) + Z @ beta_true
    Y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return Y, Z, ID


def fit_in_r(Y, Z, ID):
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None, "Rscript not found"
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "data.csv")
    out_path = os.path.join(d, "out.json")
    p = Z.shape[1]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "y"] + [f"z{i + 1}" for i in range(p)])
        for i in range(Y.shape[0]):
            w.writerow([int(ID[i]), int(Y[i])] + [repr(float(x)) for x in Z[i]])
    proc = subprocess.run(
        [rscript, "--no-save", "--no-restore", os.path.join(_HERE, "_fit_in_r.R"),
         csv_path, out_path],
        capture_output=True, text=True)
    if not os.path.exists(out_path):
        return None, (proc.stderr or "R produced no output")[:800]
    with open(out_path) as f:
        return json.load(f), None


def main():
    Y, Z, ID = make_data()
    py = paraflr.logis_firth(Y, Z, ID, threads=1)
    r, err = fit_in_r(Y, Z, ID)
    if r is None:
        print(f"SKIP: R parity unavailable ({err}).")
        print(f"Python fit ran: {len(py['gamma'])} providers, "
              f"beta={py['beta']}, iters={py['iters']}")
        return 0

    # Align gamma by provider id (both are sorted by id); beta by position.
    r_gamma = np.array([r["gamma"][str(int(pid))] if isinstance(r["gamma"], dict)
                        else r["gamma"][i]
                        for i, pid in enumerate(py["prov_ids"])], dtype=float)
    r_beta = np.array(list(r["beta"].values()) if isinstance(r["beta"], dict)
                      else r["beta"], dtype=float)

    dg = float(np.max(np.abs(py["gamma"] - r_gamma)))
    db = float(np.max(np.abs(py["beta"] - r_beta)))
    dl = abs(py["neg2Loglkd"] - float(r["neg2Loglkd"]))
    print(f"max |gamma_py - gamma_R| = {dg:.3e}")
    print(f"max |beta_py  - beta_R|  = {db:.3e}")
    print(f"|neg2Loglkd diff|        = {dl:.3e}")
    ok = dg < TOL and db < TOL
    print("PARITY OK" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
