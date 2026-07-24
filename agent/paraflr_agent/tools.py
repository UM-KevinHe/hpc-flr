"""Tool registry, argument resolution, tool subsetting, and the dispatcher.

Six tools sit on top of the two functions paraflr exports:

    fit_flr            paraflr::logis_firth
    test_provider      paraflr::test_gamma.single  (wald / score / lrt)
    benchmark_threads  logis_firth timed across OpenMP thread counts
    inspect_data       structure + a guess at the y / z / id mapping
    simulate_provider_data  clustered binary data, for when there is none
    start_analysis     the wizard, for requests too vague to route

``dispatch(name, **kwargs)`` is the only way a tool ever runs — from the
agent loop or a direct call. The language model never reaches R except
through here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from . import helpers
from . import runner

# --- registry ----------------------------------------------------------------

TOOL_REGISTRY: Dict[str, str] = {
    "inspect_data":           "inspect_data.R",
    "start_analysis":         "start_analysis.R",
    "simulate_provider_data": "simulate_provider_data.R",
    "fit_flr":                "fit_flr.R",
    "test_provider":          "test_provider.R",
    "benchmark_threads":      "benchmark_threads.R",
}

# Tools that fit something and therefore need a data file.
ANALYSIS_TOOLS: FrozenSet[str] = frozenset(
    {"fit_flr", "test_provider", "benchmark_threads"})

# Always exposed: inspect_data is the documented recovery path when a field
# name is wrong, and start_analysis is the escape hatch for a request the
# exposed subset cannot satisfy.
_ALWAYS_EXPOSED: FrozenSet[str] = frozenset({"inspect_data", "start_analysis"})

_SCHEMAS_PATH = Path(__file__).resolve().parent / "schemas.json"


def load_schemas() -> List[dict]:
    """Return the OpenAI-format tool schema list (one entry per tool)."""
    with open(_SCHEMAS_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- tool subsetting ---------------------------------------------------------
#
# Small models route better with fewer options, and some options are simply
# inadmissible: you cannot fit without a data file, and you do not need to
# simulate data once you have some. Both facts are known to the harness
# before the model is asked anything, so they are enforced by what gets sent
# rather than by a prompt rule the model may ignore.

def select_tool_schemas(
    all_schemas: List[dict],
    inspect_result: Optional[dict],
    has_data_file: Optional[bool] = None,
    task_override: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Return ``(subset, decision_metadata)`` for one request.

    ``inspect_result`` is the auto-inspection of the user's data file, or
    ``None`` when there was none — either because no file was supplied or
    because context injection is switched off. ``has_data_file`` separates
    those two cases; leave it unset and it is inferred from the inspection.
    Keeping them apart matters for the ablation: turning off injection must
    not also hide the fitting tools, or the cell would measure two layers at
    once.

    ``task_override`` ("fit" / "test" / "benchmark") narrows to a single
    analysis tool; supply it when the caller already knows the task — a UI
    where the user picked it from a menu, or a harness pinning the
    configuration so only the routing is under test.
    """
    inspected = isinstance(inspect_result, dict) and \
        inspect_result.get("status") == "ok"
    if has_data_file is None:
        has_data_file = inspected
    mapping_found = bool(inspected and inspect_result.get("mapping"))

    keep = set(_ALWAYS_EXPOSED)
    reason: str
    if not has_data_file:
        keep.add("simulate_provider_data")
        reason = "no_data_file"
    elif not inspected:
        # There is a file but we have not looked inside it; the model will
        # have to work the field names out for itself.
        keep |= ANALYSIS_TOOLS
        reason = "data_file_not_inspected"
    elif mapping_found:
        keep |= ANALYSIS_TOOLS
        reason = "data_inspected"
    else:
        # The file loaded but no y/z/id triple was recognisable. Expose
        # everything and let the model (or the user) sort it out; blocking a
        # legitimate analysis is worse than a wider choice.
        keep |= ANALYSIS_TOOLS | {"simulate_provider_data"}
        reason = "data_inspected_but_mapping_ambiguous"

    task_map = {"fit": "fit_flr", "test": "test_provider",
                "benchmark": "benchmark_threads"}
    if task_override:
        picked = task_map.get(task_override)
        if picked is None:
            raise ValueError(
                f"task_override must be one of {sorted(task_map)}, "
                f"got {task_override!r}")
        keep = set(_ALWAYS_EXPOSED) | {picked}
        reason = f"task_override:{task_override}"

    subset = [s for s in all_schemas if s["function"]["name"] in keep]
    return subset, {
        "reason": reason,
        "mapping_found": mapping_found,
        "task_override": task_override,
        "n_exposed": len(subset),
        "n_total": len(all_schemas),
        "names": [s["function"]["name"] for s in subset],
    }


# --- ASCII gate --------------------------------------------------------------
#
# Qwen 2.5-7B falls back to its Chinese training distribution when it is
# unsure of a field name, emitting things like `data$医院编号`. Those are
# never valid R identifiers, so the failure is guaranteed — we may as well
# catch it here and tell the model exactly how to recover, rather than let it
# read an R parse error and guess again. Only R-expression arguments are
# checked: `data_path` may legitimately contain any script, and so may a
# provider id.

_EXPR_ARGS: FrozenSet[str] = frozenset({"y_expr", "z_expr", "id_expr"})


def _validate_ascii(name: str, kwargs: Dict[str, Any]) -> dict:
    offenders = {}
    for arg, val in kwargs.items():
        if arg in _EXPR_ARGS and isinstance(val, str):
            try:
                val.encode("ascii")
            except UnicodeEncodeError:
                offenders[arg] = val
    if not offenders:
        return {}
    return {
        "status": "error",
        "class": "NonAsciiIdentifier",
        "where": "paraflr_agent.tools.dispatch",
        "message": (
            f"Tool {name!r} was called with non-ASCII characters in R field "
            f"expressions: {offenders}. R identifiers must be ASCII. Do not "
            f"guess field names: call `inspect_data` on the data file first, "
            f"then retry with the exact names it reports."),
        "offending_args": offenders,
        "remediation": "call_inspect_data_then_retry",
    }


# --- argument resolution -----------------------------------------------------

def resolve_args(name: str,
                 kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Return ``(payload_for_R, notices)``.

    This is where the harness makes the deterministic decisions that the
    model should not be trusted with: the null hypothesis for a test, a
    thread count the machine cannot honour, the spelling of a method name.
    ``agent.py`` records the returned payload as ``effective_args``, so a
    trace shows both what the model asked for and what actually ran.
    """
    payload = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    notices: Dict[str, str] = {}

    if name in ("fit_flr", "test_provider"):
        threads, note = helpers.clamp_threads(payload.get("threads"))
        if threads is not None:
            payload["threads"] = threads
        if note:
            notices["threads"] = note

    if name == "benchmark_threads":
        raw = payload.get("threads_list")
        if raw:
            clamped, notes = [], []
            for t in raw:
                c, note = helpers.clamp_threads(t)
                if c is not None:
                    clamped.append(c)
                if note:
                    notes.append(note)
            # De-duplicate while keeping order: clamping 4/8/16 on a 4-core
            # box would otherwise time the same run three times.
            seen, uniq = set(), []
            for t in clamped:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            payload["threads_list"] = uniq
            if notes:
                notices["threads"] = notes[0]

    if name == "test_provider":
        r_null, note = helpers.resolve_null(payload.pop("null", None),
                                            payload.pop("null_value", None))
        payload["null"] = r_null
        if note:
            notices["null"] = note
        if isinstance(payload.get("method"), str):
            payload["method"] = payload["method"].strip().lower()

    return payload, notices


# --- dispatch ----------------------------------------------------------------

def dispatch(name: str, backend: str = "r", **kwargs: Any) -> dict:
    """Run one tool by name and return its result dict.

    ``backend`` selects the computation layer: ``"r"`` shells out to the R
    scripts (paraflr R package), ``"python"`` calls the paraflr Python package
    in process (``backend_py``). Both return the same result shape, so the
    agent loop, trace, and reports are backend-agnostic. Everything before the
    call — unknown-tool rejection, the ASCII gate, argument resolution — and
    the notices after it are shared.

    Order: reject unknown tools, gate non-ASCII identifiers, resolve
    arguments, call the backend, then attach any notices the resolution
    produced.
    """
    if name not in TOOL_REGISTRY:
        return {"status": "error",
                "message": f"Unknown tool: {name!r}. "
                           f"Known: {sorted(TOOL_REGISTRY)}",
                "class": "UnknownTool",
                "where": "paraflr_agent.tools.dispatch"}

    ascii_err = _validate_ascii(name, kwargs)
    if ascii_err:
        return ascii_err

    payload, notices = resolve_args(name, kwargs)
    if backend == "python":
        from . import backend_py
        result = backend_py.run_python(name, payload)
    elif backend == "r":
        result = runner.run_r(TOOL_REGISTRY[name], payload)
    else:
        return {"status": "error",
                "message": f"Unknown backend: {backend!r} (use 'r' or 'python').",
                "class": "UnknownBackend",
                "where": "paraflr_agent.tools.dispatch"}

    if name == "test_provider":
        result = helpers.attach_firth_note(result)
    for key, note in notices.items():
        result = helpers.attach_notice(result, key, note)
    return result
