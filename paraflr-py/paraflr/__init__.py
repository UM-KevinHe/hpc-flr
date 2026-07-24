"""paraflr (Python) — parallel Firth-corrected logistic regression.

Python front end to the same C++ core the R package uses (``src/firth_core.hpp``,
identical math), so a Python fit agrees with the R fit to numerical tolerance.

    import numpy as np, paraflr
    fit = paraflr.logis_firth(Y, Z, ID, threads=4)
    fit["beta"]; fit["gamma"]
    paraflr.test_gamma_single(fit, method="score")
"""
from __future__ import annotations

import math
import numpy as np

from . import _core

__all__ = ["logis_firth", "test_gamma_single"]
__version__ = "0.1.0"


def _erfc(x: float) -> float:
    return math.erfc(x)


def _two_sided_p(z: float) -> float:
    # 2 * pnorm(-|z|)
    return _erfc(abs(z) / math.sqrt(2.0))


def _chisq1_sf(x: float) -> float:
    # P(chi^2_1 > x) = erfc(sqrt(x/2))
    if x <= 0:
        return 1.0
    return _erfc(math.sqrt(x / 2.0))


def logis_firth(Y, Z, ID, cutoff=0, max_iter=10000, tol=1e-5, bound=10.0,
                backtrack=False, threads=1, z_names=None):
    """Fit the model with provider-specific intercepts and no global intercept.

    Mirrors the R ``logis_firth()``: records are sorted by provider, providers
    with fewer than ``cutoff`` records are dropped, then the shared C++ core
    runs the Firth-corrected fit. Returns a dict with ``beta`` (named by
    covariate), ``gamma`` (named by provider id), and ``neg2Loglkd``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    ID = np.asarray(ID).ravel()
    n0, p = Z.shape
    if Y.shape[0] != n0 or ID.shape[0] != n0:
        raise ValueError("Y, Z, ID must have the same number of rows")
    if z_names is None:
        z_names = [f"Z{i + 1}" for i in range(p)]

    # Sort by provider id (stable → same within-provider row order as R's order()).
    order = np.argsort(ID, kind="stable")
    Ys, Zs, IDs = Y[order], Z[order], ID[order]

    # Run lengths on the sorted ids, then the cutoff filter.
    prov_ids, counts = _rle(IDs)
    if cutoff > 0:
        size_long = np.repeat(counts, counts)
        keep = size_long >= cutoff
        Ys, Zs, IDs = Ys[keep], Zs[keep], IDs[keep]
        prov_ids, counts = _rle(IDs)

    n_prov = counts.astype(float)
    m = len(n_prov)
    n_obs = Ys.shape[0]

    ybar = Ys.mean()
    gamma0 = np.full(m, math.log(ybar / (1.0 - ybar)))
    beta0 = np.zeros(p)

    fit = _core.logis_firth_prov(
        Ys, np.asfortranarray(Zs), n_prov, gamma0, beta0,
        n_obs, m, threads, tol, max_iter, bound, backtrack)
    gamma = np.asarray(fit["gamma"], dtype=float)
    beta = np.asarray(fit["beta"], dtype=float)

    gamma_obs = np.repeat(gamma, counts)
    eta = gamma_obs + Zs @ beta
    neg2 = -2.0 * np.sum(eta * Ys - np.log1p(np.exp(eta)))

    return {
        "beta": beta,
        "gamma": gamma,
        "neg2Loglkd": float(neg2),
        "z_names": list(z_names),
        "prov_ids": prov_ids,
        "iters": int(fit["iters"]),
        # kept for test_gamma_single (sorted / filtered, provider-contiguous):
        "_Y": Ys, "_Z": np.ascontiguousarray(Zs), "_n_prov": n_prov,
    }


def test_gamma_single(fit, method="wald", null="median", alpha=0.05, firth=False):
    """Test the first provider's effect. Mirrors R ``test_gamma.single()``.

    ``method`` in {"wald", "score", "lrt"}. Returns a dict with the estimate,
    the statistic, the p-value, and a flag in {-1, 0, 1, None}.
    """
    if method not in ("wald", "score", "lrt"):
        raise ValueError("method must be 'wald', 'score', or 'lrt'")
    Y = fit["_Y"]
    Z = fit["_Z"]
    n_prov = fit["_n_prov"]
    gamma = np.asarray(fit["gamma"], dtype=float)
    beta = np.asarray(fit["beta"], dtype=float)
    gamma1 = float(gamma[0])
    if null == "median":
        gamma_null = float(np.median(gamma))
    elif np.isscalar(null):
        gamma_null = float(null)
    else:
        raise ValueError("null must be 'median' or a number")

    stat = None
    pval = None
    Zf = np.asfortranarray(Z)

    if method == "wald":
        gamma_obs = np.repeat(gamma, n_prov.astype(int))
        probs = 1.0 / (1.0 + np.exp(-(gamma_obs + Z @ beta)))
        se = np.asarray(_core.wald_gamma(Zf, probs, n_prov, np.array([1.0])))
        if se.size == 0 or not np.isfinite(se[0]):
            stat, pval = float("nan"), float("nan")
        else:
            stat = (gamma1 - gamma_null) / se[0]
            pval = _two_sided_p(stat)
    elif method == "score":
        k = int(n_prov[0])
        prob_null = 1.0 / (1.0 + np.exp(-(gamma_null + Z[:k] @ beta)))
        stat = float(np.sum(Y[:k] - prob_null) /
                     math.sqrt(np.sum(prob_null * (1.0 - prob_null))))
        pval = _two_sided_p(stat)
    else:  # lrt
        gamma_test = gamma.copy()
        gamma_test[0] = gamma_null
        if firth:
            full = _core.loglkd_firth(Y, Zf, n_prov, gamma, beta)
            nullv = _core.loglkd_firth(Y, Zf, n_prov, gamma_test, beta)
        else:
            Z_beta = Z @ beta
            full = _core.loglkd(Y, Z_beta, np.repeat(gamma, n_prov.astype(int)))
            nullv = _core.loglkd(Y, Z_beta, np.repeat(gamma_test, n_prov.astype(int)))
        stat = -2.0 * (nullv - full)
        pval = _chisq1_sf(stat)

    if pval is None or math.isnan(pval):
        flag = None
    elif pval < alpha:
        if method == "lrt":
            flag = 1 if (gamma1 - gamma_null) > 0 else -1
        else:
            flag = 1 if stat > 0 else -1
    else:
        flag = 0

    return {"gamma_est": gamma1, "stat": stat, "p": pval, "flag": flag}


def _rle(sorted_ids):
    """Run-length groups of an already-sorted id array. Returns (ids, counts)."""
    if sorted_ids.shape[0] == 0:
        return np.array([]), np.array([], dtype=int)
    change = np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1]))
    ids = sorted_ids[change]
    idx = np.flatnonzero(change)
    counts = np.diff(np.concatenate((idx, [sorted_ids.shape[0]])))
    return ids, counts
