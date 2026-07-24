"""Shared infrastructure for the paraflr agent evaluation.

Everything below drives the same frozen ``paraflr_agent`` over two
representative analyses — one estimation, one hypothesis test — on the two
bundled datasets. "Which tool is correct" is fixed per representative as
``expected_tool``, so scoring never depends on reading the model's prose.

Endpoint configuration comes from the environment, so the same scripts run
against a local vLLM server or a hosted API::

    PARAFLR_MODEL_ENDPOINT   e.g. http://localhost:8000/v1
    PARAFLR_MODEL_NAME       e.g. qwen2.5-7b-awq
    OPENAI_API_KEY           vLLM ignores it; hosted APIs need a real key
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR))

from paraflr_agent import ParaFLRAgent, dispatch  # noqa: E402

DATA_DIR = _AGENT_DIR / "data"


def endpoint_cfg():
    return dict(
        model_endpoint=os.environ.get("PARAFLR_MODEL_ENDPOINT",
                                      "http://localhost:8000/v1"),
        model_name=os.environ.get("PARAFLR_MODEL_NAME", "qwen2.5-7b-awq"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )


# The two representatives. `manual` is the verified direct dispatch — the
# ceiling, and the reference the agent's own numbers must reproduce exactly.
REPS = [
    {
        "id": "fit_moderate",
        "data": "ExampleProviders.rda",
        "expected_tool": "fit_flr",
        "task": "fit",
        "query": (
            "I have provider profiling data at {data_path} — a binary outcome "
            "per record, five risk adjusters, and a provider id. Fit the "
            "Firth-corrected model with one effect per provider and tell me "
            "the covariate effects. Use 2 threads."
        ),
        "manual": {
            "y_expr": "ExampleProviders$Y",
            "z_expr": "ExampleProviders$Z",
            "id_expr": "ExampleProviders$ID",
            "threads": 2,
        },
    },
    {
        "id": "test_rare_provider",
        "data": "ExampleProviders_rare.rda",
        "expected_tool": "test_provider",
        "task": "test",
        "query": (
            "Data at {data_path}: 40 small facilities with a rare adverse "
            "outcome. Is facility P004 significantly different from the "
            "median facility? Use the score test."
        ),
        "manual": {
            "y_expr": "ExampleProviders_rare$Y",
            "z_expr": "ExampleProviders_rare$Z",
            "id_expr": "ExampleProviders_rare$ID",
            "provider": "P004",
            "method": "score",
            "null": "median",
        },
    },
]


def data_path(rep):
    return str(DATA_DIR / rep["data"])


def reference_result(rep):
    """The verified direct call, with no model in the loop."""
    return dispatch(rep["expected_tool"], data_path=data_path(rep),
                    **rep["manual"])


def _signature(result):
    """The part of a result two runs must agree on, as a canonical string.

    Timings and thread counts are deliberately excluded: they vary run to
    run and are not what fidelity is about. What must match exactly are the
    estimates and the test statistics.
    """
    if not isinstance(result, dict) or result.get("status") != "ok":
        return None
    if "beta" in result:
        return json.dumps({"beta": result["beta"],
                           "neg2Loglkd": result.get("neg2Loglkd")},
                          sort_keys=True)
    if "p_value" in result:
        return json.dumps({k: result.get(k) for k in
                           ("provider", "method", "firth", "null_value",
                            "gamma_est", "stat", "p_value", "flag")},
                          sort_keys=True)
    return None


def bit_match(agent_result, reference):
    a, b = _signature(agent_result), _signature(reference)
    return (a is not None) and (a == b)


def tools_called(resp):
    """Tool names the model itself called (the harness's auto-inspect aside)."""
    return [e.tool for e in resp.trace.events
            if not e.effective_args.get("_auto_prepend")]


def classify_outcome(resp, rep):
    """correct / silently_wrong / crashed / refused.

    ``silently_wrong`` is the outcome that matters: a plausible but incorrect
    tool that ran cleanly and produced numbers nobody flagged.
    """
    expected = rep["expected_tool"]
    evs = [e for e in resp.trace.events
           if not e.effective_args.get("_auto_prepend")]
    if any(e.tool == expected and e.status == "ok" for e in evs):
        return "correct"
    if any(e.tool != expected and e.status == "ok"
           and e.tool not in ("inspect_data", "start_analysis") for e in evs):
        return "silently_wrong"
    if any(e.status != "ok" for e in evs):
        return "crashed"
    if resp.error:
        return "crashed"
    return "refused"


def usage(resp):
    t = resp.trace
    return {
        "prompt_tokens": t.prompt_tokens_total,
        "completion_tokens": t.completion_tokens_total,
        "llm_turns": t.llm_turns,
        "total_latency_ms": t.total_latency_ms,
        "n_tools_exposed": (t.tools_exposed or {}).get("n_exposed"),
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print("wrote", path)
