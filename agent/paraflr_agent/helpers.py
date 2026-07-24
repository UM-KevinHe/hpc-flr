"""Deterministic post-processing applied around every tool call.

Two jobs live here:

  * resolving arguments the model routinely gets wrong or leaves out (the
    null hypothesis for a provider test, a thread count larger than the
    machine has) — done in Python, recorded in the trace, so the resolution
    is auditable rather than a matter of model discretion;
  * shrinking what the model sees between turns, so a fit with 200 provider
    effects does not cost 5K tokens on every subsequent turn.

Notices are attached under ``_notice_*`` keys. They are for the model and
the user, and are never read back by any code path.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


# --- argument resolution -----------------------------------------------------

def resolve_null(null: Optional[str],
                 null_value: Optional[float]) -> Tuple[Any, Optional[str]]:
    """Turn the schema's (``null``, ``null_value``) pair into R's ``null`` arg.

    The schema splits these because a single parameter that accepts either
    the string ``"median"`` or a number is exactly the kind of union type a
    7B model fumbles. Returns ``(r_null, notice)``.
    """
    if null in (None, "", "median"):
        if null_value is not None:
            return float(null_value), (
                f"A fixed null of {null_value} was supplied, so the provider "
                f"effect was tested against it rather than against the "
                f"population median.")
        return "median", None
    if null == "value":
        if null_value is None:
            return "median", (
                "null was set to 'value' but no null_value was given, so the "
                "test fell back to the population median.")
        return float(null_value), None
    # Anything else: try to read it as a number before giving up.
    try:
        return float(null), None
    except (TypeError, ValueError):
        return "median", (
            f"Unrecognised null {null!r}; tested against the population "
            f"median instead.")


def clamp_threads(threads: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
    """Clamp a requested thread count to what the machine actually has.

    Oversubscribing OpenMP threads slows the fit down instead of speeding it
    up, and a model asked to "use lots of cores" will happily request 64 on
    an 8-core laptop.
    """
    if threads is None:
        return None, None
    try:
        t = int(threads)
    except (TypeError, ValueError):
        return 1, f"threads={threads!r} is not an integer; ran single-threaded."
    if t < 1:
        return 1, f"threads={t} is not valid; ran single-threaded."
    ncores = os.cpu_count() or 1
    if t > ncores:
        return ncores, (f"Requested {t} threads but this machine has {ncores} "
                        f"cores; ran with {ncores}.")
    return t, None


def attach_notice(result: Dict[str, Any], key: str,
                  notice: Optional[str]) -> Dict[str, Any]:
    """Attach a ``_notice_<key>`` to a successful result (no-op otherwise)."""
    if not notice or not isinstance(result, dict):
        return result
    out = dict(result)
    out["_notice_" + key] = notice
    return out


def attach_firth_note(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag likelihood-ratio tests run without the Firth penalty.

    On the data this package is for — small providers, rare events — the
    unpenalised LRT is the variant with the known small-sample problem, and
    it is also the default. Saying so once, in the result, keeps the choice
    visible to whoever reads the answer.
    """
    if not isinstance(result, dict) or result.get("status") != "ok":
        return result
    if result.get("method") != "lrt" or result.get("firth"):
        return result
    n = result.get("provider_n_obs")
    return attach_notice(
        result, "lrt_unpenalised",
        f"This is the unpenalised likelihood-ratio test"
        + (f" on a provider with {n} records" if n else "")
        + ". For small or low-event providers, rerun with firth=true for the "
          "penalised version.")


# --- token shrinking ---------------------------------------------------------

_ALWAYS_KEEP = {
    "status", "class", "where", "message", "remediation", "offending_args",
    "n_obs", "n_covariates", "n_providers", "event_rate", "threads",
    "elapsed_sec", "neg2Loglkd", "beta",
    "provider", "method", "firth", "null", "null_value", "alpha",
    "gamma_est", "stat", "p_value", "flag", "verdict", "provider_n_obs",
    "data_path", "object_name", "y_expr", "z_expr", "id_expr",
    "best_speedup", "fastest_threads", "baseline_threads", "max_beta_diff",
    "estimates_agree", "parallel_speedup_observed", "detected_cores",
    "providers_with_no_events", "seed", "language",
}


def compress_tool_result_for_llm(result):
    """Return a token-shrunk copy of a tool result for the LLM loop.

    The input is not mutated: the full result is what goes to the caller and
    into ``repro.R``; only the model's copy is shrunk.
    """
    if not isinstance(result, dict):
        return result
    out = {}
    for k, v in result.items():
        if k.startswith("_notice_") or k in _ALWAYS_KEEP:
            out[k] = v
        elif k == "gamma" and isinstance(v, dict):
            out[k] = (f"[{len(v)} provider effects; see gamma_summary, or "
                      f"trace.json for the full vector]")
        elif isinstance(v, list) and len(v) > 8:
            if all(isinstance(x, (int, float)) for x in v):
                out[k] = (f"[vector len={len(v)}, range "
                          f"[{min(v):.4g}, {max(v):.4g}]; full values in "
                          f"trace.json]")
            else:
                out[k] = list(v[:3]) + ["..."] + list(v[-2:])
        elif isinstance(v, dict):
            out[k] = compress_tool_result_for_llm(v)
        else:
            out[k] = v
    return out
