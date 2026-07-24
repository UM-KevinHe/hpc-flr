"""Pure-Python backend for the agent tools — no R in the loop.

Same six tools as the R scripts in ``r_scripts/``, computing every number
through the ``paraflr`` **Python** package (``paraflr-py``), which shares the
C++ core with the R package. Each function returns the same result shape its
R counterpart does, so the agent loop, the trace, and the reports are
identical whichever backend runs.

Data files: this backend reads Python-native ``.npz`` (keys ``Y``/``Z``/``ID``)
and ``.csv``. Use the R backend for ``.rda``/``.rds``. ``*_expr`` arguments are
interpreted as field names (an ``.npz`` key, or a CSV column; ``z_expr`` may be
a comma-separated column list), not R code — ``inspect_data`` reports the right
ones so the model uses them verbatim, exactly as on the R side.
"""
from __future__ import annotations

import csv
import math
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _paraflr():
    try:
        import paraflr  # the paraflr-py package
        return paraflr
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "The Python backend needs the 'paraflr' Python package. Install it "
            "with `pip install ./paraflr-py` from the repo root.") from e


# --- data loading ------------------------------------------------------------

def _read_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    cols: Dict[str, np.ndarray] = {}
    for j, name in enumerate(header):
        vals = [r[j] for r in body]
        try:
            cols[name] = np.array([float(v) for v in vals])
        except ValueError:
            cols[name] = np.array(vals, dtype=object)
    return cols


def _load_ns(data_path: Optional[str]) -> Tuple[Dict[str, Any], str]:
    if not data_path:
        raise ValueError("data_path is required")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".npz":
        d = np.load(data_path, allow_pickle=True)
        return {k: d[k] for k in d.files}, "npz"
    if ext == ".csv":
        return _read_csv(data_path), "csv"
    raise ValueError(
        f"The Python backend reads .npz and .csv (got '{ext}'). Use the R "
        f"backend for .rda/.rds, or convert the file to CSV.")


def _resolve_fields(ns, y_expr, z_expr, id_expr):
    if not y_expr or not z_expr or not id_expr:
        raise ValueError("y_expr, z_expr and id_expr are all required")
    y = np.asarray(ns[y_expr], dtype=float).ravel()
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y_expr must resolve to a 0/1 binary outcome vector")
    idv = np.asarray(ns[id_expr]).ravel()
    if "," in str(z_expr):
        cols = [c.strip() for c in z_expr.split(",")]
        Z = np.column_stack([np.asarray(ns[c], dtype=float) for c in cols])
        z_names = cols
    else:
        Z = np.asarray(ns[z_expr], dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        z_names = ns["z_names"].tolist() if "z_names" in ns else None
    if not (len(y) == Z.shape[0] == len(idv)):
        raise ValueError(
            f"Field lengths differ: y has {len(y)}, z has {Z.shape[0]} rows, "
            f"id has {len(idv)}. The three must describe the same records.")
    return y, Z, idv, z_names


def _fit(pf, y, Z, idv, payload, z_names):
    return pf.logis_firth(
        y, Z, idv,
        cutoff=float(payload.get("cutoff", 0)),
        max_iter=int(payload.get("max_iter", 10000)),
        tol=float(payload.get("tol", 1e-5)),
        bound=float(payload.get("bound", 10)),
        threads=int(payload.get("threads", 1)),
        z_names=z_names)


# --- tools -------------------------------------------------------------------

def _fit_flr(payload: Dict[str, Any]) -> Dict[str, Any]:
    pf = _paraflr()
    ns, _ = _load_ns(payload.get("data_path"))
    y, Z, idv, z_names = _resolve_fields(
        ns, payload.get("y_expr"), payload.get("z_expr"), payload.get("id_expr"))
    threads = int(payload.get("threads", 1))
    t0 = time.perf_counter()
    fit = _fit(pf, y, Z, idv, payload, z_names)
    elapsed = time.perf_counter() - t0

    gamma = np.asarray(fit["gamma"], dtype=float)
    beta = np.asarray(fit["beta"], dtype=float)
    names = [str(x) for x in fit["prov_ids"]]
    zn = fit.get("z_names") or [f"Z{i + 1}" for i in range(Z.shape[1])]
    m = len(gamma)

    gamma_out = ({n: float(g) for n, g in zip(names, gamma)} if m <= 200 else None)
    order = np.argsort(gamma)
    lo, hi = order[:5], order[-5:][::-1]
    q = np.quantile(gamma, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "status": "ok",
        "beta": {n: float(b) for n, b in zip(zn, beta)},
        "gamma": gamma_out,
        "gamma_summary": {
            "n_providers": m,
            "median": float(np.median(gamma)),
            "quantiles": {k: round(float(v), 6) for k, v in zip(
                ["0%", "25%", "50%", "75%", "100%"], q)},
            "extremes": {
                "lowest": {names[i]: float(gamma[i]) for i in lo},
                "highest": {names[i]: float(gamma[i]) for i in hi}},
        },
        "neg2Loglkd": float(fit["neg2Loglkd"]),
        "n_obs": int(len(fit["_Y"])),
        "n_covariates": int(Z.shape[1]),
        "n_providers": m,
        "event_rate": round(float(np.mean(fit["_Y"])), 4),
        "threads": threads,
        "elapsed_sec": round(elapsed, 3),
    }


def _rotate(fit, provider):
    names = [str(x) for x in fit["prov_ids"]]
    if str(provider) not in names:
        shown = ", ".join(names[:20])
        raise ValueError(
            f"Provider '{provider}' not found. The fitted providers are: "
            f"{shown}" + (f" ... ({len(names)} total)" if len(names) > 20 else ""))
    k = names.index(str(provider))
    n_prov = np.asarray(fit["_n_prov"], dtype=int)
    starts = np.concatenate(([0], np.cumsum(n_prov)))
    prov_order = [k] + [i for i in range(len(names)) if i != k]
    row_idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in prov_order])
    out = dict(fit)
    out["_Y"] = np.asarray(fit["_Y"])[row_idx]
    out["_Z"] = np.asarray(fit["_Z"])[row_idx]
    out["_n_prov"] = n_prov[prov_order].astype(float)
    out["gamma"] = np.asarray(fit["gamma"])[prov_order]
    out["prov_ids"] = np.asarray(fit["prov_ids"])[prov_order]
    return out


def _test_provider(payload: Dict[str, Any]) -> Dict[str, Any]:
    pf = _paraflr()
    ns, _ = _load_ns(payload.get("data_path"))
    y, Z, idv, z_names = _resolve_fields(
        ns, payload.get("y_expr"), payload.get("z_expr"), payload.get("id_expr"))
    method = str(payload.get("method", "score")).strip().lower()
    if method not in ("wald", "score", "lrt"):
        raise ValueError(
            f"method must be one of 'wald', 'score', 'lrt' (got '{method}')")
    firth = bool(payload.get("firth", False))
    alpha = float(payload.get("alpha", 0.05))
    null_arg = payload.get("null", "median")

    fit = _fit(pf, y, Z, idv, payload, z_names)
    names = [str(x) for x in fit["prov_ids"]]
    provider = str(payload["provider"]) if payload.get("provider") is not None else names[0]
    fit = _rotate(fit, provider)

    res = pf.test_gamma_single(fit, method=method, null=null_arg,
                               alpha=alpha, firth=firth)
    flag = res["flag"]
    verdict = ("undetermined" if flag is None else
               "worse than expected (significantly above the null)" if flag > 0 else
               "better than expected (significantly below the null)" if flag < 0 else
               "not significantly different from the null")
    is_median = isinstance(null_arg, str)
    gamma = np.asarray(fit["gamma"], dtype=float)
    return {
        "status": "ok",
        "provider": provider,
        "method": method,
        "firth": firth,
        "null": "median" if is_median else float(null_arg),
        "null_value": float(np.median(gamma)) if is_median else float(null_arg),
        "alpha": alpha,
        "gamma_est": float(res["gamma_est"]),
        "stat": None if res["stat"] is None else float(res["stat"]),
        "p_value": None if res["p"] is None else float(res["p"]),
        "flag": flag,
        "verdict": verdict,
        "n_obs": int(len(fit["_Y"])),
        "n_providers": len(names),
        "provider_n_obs": int(fit["_n_prov"][0]),
        "threads": int(payload.get("threads", 1)),
    }


def _benchmark_threads(payload: Dict[str, Any]) -> Dict[str, Any]:
    pf = _paraflr()
    ns, _ = _load_ns(payload.get("data_path"))
    y, Z, idv, z_names = _resolve_fields(
        ns, payload.get("y_expr"), payload.get("z_expr"), payload.get("id_expr"))
    tl = payload.get("threads_list") or [1, 2, 4]
    threads_list, seen = [], set()
    for t in tl:
        t = int(t)
        if t >= 1 and t not in seen:
            seen.add(t)
            threads_list.append(t)
    if not threads_list:
        raise ValueError("threads_list must contain at least one count >= 1")
    repeats = max(1, int(payload.get("repeats", 1)))

    runs: List[Dict[str, Any]] = []
    beta_ref = None
    max_diff = 0.0
    for th in threads_list:
        times = []
        beta = None
        for _ in range(repeats):
            p = dict(payload)
            p["threads"] = th
            t0 = time.perf_counter()
            fit = _fit(pf, y, Z, idv, p, z_names)
            times.append(time.perf_counter() - t0)
            beta = np.asarray(fit["beta"], dtype=float)
        if beta_ref is None:
            beta_ref = beta
        else:
            max_diff = max(max_diff, float(np.max(np.abs(beta - beta_ref))))
        runs.append({"threads": th, "elapsed_sec": round(min(times), 4),
                     "elapsed_sec_all": [round(t, 4) for t in times]})

    base = runs[0]["elapsed_sec"]
    for r in runs:
        r["speedup"] = round(base / r["elapsed_sec"], 3) if r["elapsed_sec"] else None
        r["efficiency"] = (round((base / r["elapsed_sec"]) / r["threads"], 3)
                           if r["elapsed_sec"] else None)
    fastest = min(runs, key=lambda r: r["elapsed_sec"])
    return {
        "status": "ok",
        "parallel_speedup_observed": bool(fastest["speedup"] and fastest["speedup"] > 1.2),
        "detected_cores": os.cpu_count(),
        "threads_list": threads_list,
        "repeats": repeats,
        "runs": runs,
        "baseline_threads": runs[0]["threads"],
        "fastest_threads": fastest["threads"],
        "best_speedup": fastest["speedup"],
        "max_beta_diff": float(f"{max_diff:.6g}"),
        "estimates_agree": max_diff < 1e-8,
        "n_obs": int(len(y)),
        "n_providers": int(len(np.unique(idv))),
        "n_covariates": int(Z.shape[1]),
        "note": ("OpenMP is off in this Python build unless PARAFLR_OPENMP=1, so "
                 "thread counts share the serial path: identical estimates, flat "
                 "timings. That is a build report, not a scaling result."),
    }


def _simulate_provider_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    m = int(payload.get("n_providers", 50))
    n_per = int(payload.get("n_per_provider", 80))
    p = int(payload.get("n_covariates", 5))
    event_rate = float(payload.get("event_rate", 0.2))
    gamma_sd = float(payload.get("gamma_sd", 0.5))
    seed = int(payload.get("seed", 2026))
    out_path = payload.get("out_path") or os.path.join(
        tempfile.gettempdir(), f"SimulatedProviders_{seed}.npz")
    if m < 2:
        raise ValueError("n_providers must be at least 2")
    if n_per < 2:
        raise ValueError("n_per_provider must be at least 2")
    if not (0.0 < event_rate < 1.0):
        raise ValueError("event_rate must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    sizes = np.maximum(2, rng.poisson(n_per, m))
    n = int(sizes.sum())
    ID = np.repeat([f"P{i + 1:03d}" for i in range(m)], sizes)
    Z = rng.standard_normal((n, p))
    z_names = [f"x{i + 1}" for i in range(p)]
    beta = np.round(np.linspace(0.5, -0.5, p), 3)
    gamma = rng.normal(0.0, gamma_sd, m)

    def marginal(o):
        eta = np.repeat(gamma, sizes) + o + Z @ beta
        return float(np.mean(1.0 / (1.0 + np.exp(-eta)))) - event_rate

    lo, hi = -20.0, 20.0
    for _ in range(200):  # bisection for the intercept shift
        mid = 0.5 * (lo + hi)
        if marginal(mid) > 0:
            hi = mid
        else:
            lo = mid
    gamma = gamma + 0.5 * (lo + hi)
    eta = np.repeat(gamma, sizes) + Z @ beta
    Y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, Y=Y, Z=Z, ID=ID, beta=beta, gamma=gamma,
             z_names=np.array(z_names, dtype=object), seed=seed)

    # providers with zero events
    zero = int(sum(Y[ID == pid].sum() == 0 for pid in np.unique(ID)))
    return {
        "status": "ok",
        "data_path": os.path.abspath(out_path),
        "object_name": os.path.splitext(os.path.basename(out_path))[0],
        "y_expr": "Y", "z_expr": "Z", "id_expr": "ID",
        "n_obs": n, "n_providers": m, "n_covariates": p,
        "event_rate": round(float(np.mean(Y)), 4),
        "providers_with_no_events": zero,
        "true_beta": {n_: float(b) for n_, b in zip(z_names, beta)},
        "seed": seed,
    }


def _describe(x) -> Any:
    a = np.asarray(x)
    if a.ndim == 2:
        return f"matrix {a.shape[0]}x{a.shape[1]} ({a.dtype})"
    if a.dtype.kind in "fiu":
        v = a.ravel()
        if np.all(np.isin(v, (0, 1))):
            return f"numeric 0/1 (length {v.size}, {int(v.sum())} ones)"
        return f"numeric (length {v.size})"
    return f"{a.dtype} (length {a.size}, {len(np.unique(a))} unique)"


def _guess_mapping(ns) -> Optional[Dict[str, Any]]:
    # binary 1-D vector -> y ; 2-D array -> z ; id-like 1-D vector -> id.
    def is_binary(k):
        a = np.asarray(ns[k])
        return a.ndim == 1 and a.dtype.kind in "fiu" and a.size > 1 and \
            np.all(np.isin(a, (0, 1)))

    def is_id(k, n):
        a = np.asarray(ns[k])
        if a.ndim != 1 or a.size != n:
            return False
        u = len(np.unique(a))
        return 1 < u < n / 2

    keys = list(ns.keys())
    y_pref = [k for k in ("Y", "y", "status", "event", "outcome") if k in keys and is_binary(k)]
    y = y_pref[0] if y_pref else next((k for k in keys if is_binary(k)), None)
    if y is None:
        return None
    n = np.asarray(ns[y]).size
    z_cands = [k for k in keys if np.asarray(ns[k]).ndim == 2 and np.asarray(ns[k]).shape[0] == n]
    if not z_cands:
        return None
    z = next((k for k in ("Z", "z", "X", "x") if k in z_cands), z_cands[0])
    id_cands = [k for k in keys if k != y and is_id(k, n)]
    if not id_cands:
        return None
    idk = next((k for k in ("ID", "id", "provider", "prov", "cluster") if k in id_cands), id_cands[0])
    Z = np.asarray(ns[z])
    return {"y_expr": y, "z_expr": z, "id_expr": idk, "n_obs": int(n),
            "n_covariates": int(Z.shape[1]),
            "n_providers": int(len(np.unique(ns[idk]))),
            "event_rate": round(float(np.mean(np.asarray(ns[y], dtype=float))), 4)}


def _inspect_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    data_path = payload.get("data_path")
    ns, _ = _load_ns(data_path)
    structure = {k: _describe(v) for k, v in ns.items()}
    return {"status": "ok", "data_path": data_path,
            "top_level_names": list(ns.keys()),
            "structure": structure, "mapping": _guess_mapping(ns)}


_QUESTIONS_EN = [
    {"key": "data", "ask": "Where is your data, and which fields hold the binary outcome, the covariates, and the provider identifier? (An .npz or .csv path is enough — I will inspect it.)",
     "why": "Every paraflr tool needs a 0/1 outcome, a covariate matrix, and one provider id per record."},
    {"key": "goal", "ask": "Do you want the fitted provider effects for the whole population, or a hypothesis test for one specific provider?",
     "why": "Estimation is fit_flr; flagging a provider against a null is test_provider."},
    {"key": "test_choice", "ask": "If you want a test: which one — Wald, score, or penalised likelihood-ratio — and against what null (the population median, or a fixed value)?",
     "why": "The three tests differ most for small or low-event providers."},
    {"key": "scale", "ask": "How large is the dataset, and how many cores may I use?",
     "why": "logis_firth() parallelises over OpenMP threads; benchmark_threads reports the scaling."},
]

_QUESTIONS_ZH = [
    {"key": "data", "ask": "数据在哪里？哪些字段是二分类结局、协变量矩阵和医院/中心编号？（给出 .npz 或 .csv 路径即可）",
     "why": "所有 paraflr 工具都需要 0/1 结局、协变量矩阵和逐条记录的 provider id。"},
    {"key": "goal", "ask": "您是想估计所有 provider 的效应，还是对某一个 provider 做假设检验？",
     "why": "前者用 fit_flr，后者用 test_provider。"},
    {"key": "test_choice", "ask": "如果做检验：用 Wald、score 还是惩罚似然比 (LRT)？零假设是总体中位数还是固定值？",
     "why": "在小医院、低事件率情形下三种检验差异最大。"},
    {"key": "scale", "ask": "数据规模大约多少？可以用几个核？",
     "why": "logis_firth() 通过 OpenMP 线程并行；benchmark_threads 可以给出加速比。"},
]


def _start_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    lang = str(payload.get("user_language", "en")).lower()
    zh = lang in ("zh", "zh-cn", "zh_cn", "chinese", "中文")
    return {
        "status": "ok",
        "language": "zh" if zh else "en",
        "questions": _QUESTIONS_ZH if zh else _QUESTIONS_EN,
        "tools": {
            "fit_flr": "Fit Firth-corrected logistic regression with one effect per provider (paraflr.logis_firth).",
            "test_provider": "Wald / score / penalised-LRT test of one provider effect (paraflr.test_gamma_single).",
            "benchmark_threads": "Time the same fit across OpenMP thread counts and check the estimates agree.",
            "simulate_provider_data": "Generate a clustered binary dataset when you have none to hand."},
        "note": "Ask these questions, then call the matching tool. Do not fit anything until the data fields are known.",
    }


_TOOLS = {
    "inspect_data": _inspect_data,
    "start_analysis": _start_analysis,
    "simulate_provider_data": _simulate_provider_data,
    "fit_flr": _fit_flr,
    "test_provider": _test_provider,
    "benchmark_threads": _benchmark_threads,
}


def run_python(tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one tool in Python. Mirror of ``runner.run_r`` for the Python backend."""
    fn = _TOOLS.get(tool)
    if fn is None:
        return {"status": "error", "message": f"Unknown tool: {tool!r}",
                "class": "UnknownTool", "where": "paraflr_agent.backend_py"}
    try:
        return fn(payload)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"{type(e).__name__}: {e}",
                "class": type(e).__name__,
                "where": f"paraflr_agent.backend_py.{tool}"}
