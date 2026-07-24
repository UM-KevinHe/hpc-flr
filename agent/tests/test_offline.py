"""Offline tests — no GPU, no vLLM, no network.

The language model is replaced by a stub that returns a scripted sequence of
tool calls, so the whole agent loop runs: context injection, tool subsetting,
the dispatch guard, R execution, the trace, and the generated ``repro.R``
(which is then actually run through Rscript). What is NOT tested here is
whether a real model routes correctly — that is what ``eval/run_routing.py``
is for, and it needs an endpoint.

Run either way::

    python tests/test_offline.py
    pytest tests/test_offline.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

from paraflr_agent import ParaFLRAgent, dispatch, load_schemas, select_tool_schemas  # noqa: E402
from paraflr_agent import helpers, tools as tools_mod  # noqa: E402
from paraflr_agent.runner import find_rscript  # noqa: E402

DATA = str(AGENT_DIR / "data" / "ExampleProviders.rda")
DATA_RARE = str(AGENT_DIR / "data" / "ExampleProviders_rare.rda")
MAPPING = {"y_expr": "ExampleProviders$Y",
           "z_expr": "ExampleProviders$Z",
           "id_expr": "ExampleProviders$ID"}


# --- the stub model ----------------------------------------------------------

class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, i, name, args):
        self.id, self.type = "call_%d" % i, "function"
        self.function = _Fn(name, json.dumps(args))


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Completion:
    def __init__(self, msg):
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"prompt_tokens": 100,
                                    "completion_tokens": 20})()


class _Completions:
    def __init__(self, turns):
        self.turns, self.calls = list(turns), []

    def create(self, **kwargs):
        # Snapshot the message list: the agent appends to the same list every
        # turn, so keeping the reference would show every call the final
        # conversation rather than what it was actually sent.
        self.calls.append(dict(kwargs, messages=list(kwargs["messages"])))
        if not self.turns:
            return _Completion(_Msg(content="(stub ran out of turns)"))
        return self.turns.pop(0)


class StubClient:
    """Returns a scripted sequence of assistant turns."""

    def __init__(self, turns):
        self.chat = type("Chat", (), {"completions": _Completions(turns)})()

    @property
    def calls(self):
        return self.chat.completions.calls


def tool_turn(name, args, i=1):
    return _Completion(_Msg(content="", tool_calls=[_ToolCall(i, name, args)]))


def text_turn(text):
    return _Completion(_Msg(content=text))


def _agent(turns, **kw):
    return ParaFLRAgent(model_endpoint="stub://none", model_name="stub",
                        client=StubClient(turns), **kw)


# --- argument resolution (pure Python, no R) ---------------------------------

def test_thread_clamp():
    ncores = os.cpu_count() or 1
    clamped, note = helpers.clamp_threads(ncores + 64)
    assert clamped == ncores and "cores" in note
    assert helpers.clamp_threads(None) == (None, None)
    assert helpers.clamp_threads(0)[0] == 1


def test_null_resolution():
    assert helpers.resolve_null(None, None) == ("median", None)
    assert helpers.resolve_null("value", -1.5)[0] == -1.5
    # A fixed value without the flag is still honoured, with a notice saying so.
    val, note = helpers.resolve_null("median", -2.0)
    assert val == -2.0 and note
    # 'value' with nothing to use falls back rather than failing.
    val, note = helpers.resolve_null("value", None)
    assert val == "median" and note


def test_ascii_gate():
    out = dispatch("fit_flr", data_path=DATA, y_expr="数据$Y",
                   z_expr=MAPPING["z_expr"], id_expr=MAPPING["id_expr"])
    assert out["status"] == "error"
    assert out["class"] == "NonAsciiIdentifier"
    assert out["remediation"] == "call_inspect_data_then_retry"


def test_unknown_tool():
    assert dispatch("fit_cox")["class"] == "UnknownTool"


def test_benchmark_threads_deduplicated():
    ncores = os.cpu_count() or 1
    payload, notices = tools_mod.resolve_args(
        "benchmark_threads", {"threads_list": [1, ncores + 1, ncores + 2]})
    assert payload["threads_list"] == [1, ncores] or ncores == 1
    assert "threads" in notices


def test_tool_subsetting():
    schemas = load_schemas()
    assert len(schemas) == len(tools_mod.TOOL_REGISTRY)

    _, meta = select_tool_schemas(schemas, None)
    assert meta["reason"] == "no_data_file"
    assert "fit_flr" not in meta["names"]          # nothing to fit yet
    assert "simulate_provider_data" in meta["names"]

    ins = dispatch("inspect_data", data_path=DATA)
    _, meta = select_tool_schemas(schemas, ins)
    assert meta["mapping_found"] and "fit_flr" in meta["names"]
    assert "simulate_provider_data" not in meta["names"]   # data in hand

    _, meta = select_tool_schemas(schemas, ins, task_override="test")
    assert set(meta["names"]) == {"inspect_data", "start_analysis",
                                  "test_provider"}

    # A data file that was never inspected (context injection off) must still
    # expose the fitting tools — otherwise the ablation's no_structure cell
    # would be measuring two layers at once.
    _, meta = select_tool_schemas(schemas, None, has_data_file=True)
    assert meta["reason"] == "data_file_not_inspected"
    assert "fit_flr" in meta["names"]
    assert "simulate_provider_data" not in meta["names"]


def test_no_structure_cell_still_exposes_fitting_tools():
    agent = _agent([text_turn("Which field is the outcome?")],
                   scaffold={"inject_structure": False})
    resp = agent.query("Fit the model.", data_path=DATA)
    sent = agent._client.calls[0]
    assert "DATA STRUCTURE" not in sent["messages"][-1]["content"]
    assert DATA in sent["messages"][-1]["content"]     # the path, at least
    exposed = {s["function"]["name"] for s in sent["tools"]}
    assert "fit_flr" in exposed
    assert resp.trace.tools_exposed["reason"] == "data_file_not_inspected"


# --- the R tools -------------------------------------------------------------

def test_inspect_finds_mapping():
    out = dispatch("inspect_data", data_path=DATA)
    assert out["status"] == "ok"
    assert out["mapping"]["y_expr"] == MAPPING["y_expr"]
    assert out["mapping"]["n_providers"] == 50


def test_fit_runs():
    out = dispatch("fit_flr", data_path=DATA, threads=2, **MAPPING)
    assert out["status"] == "ok"
    assert len(out["beta"]) == 5
    assert out["n_providers"] == 50
    # Recovers the simulation truth to within sampling error.
    assert abs(out["beta"]["x1"] - 0.5) < 0.15


def test_fit_is_deterministic():
    a = dispatch("fit_flr", data_path=DATA, threads=1, **MAPPING)
    b = dispatch("fit_flr", data_path=DATA, threads=4, **MAPPING)
    assert a["beta"] == b["beta"], "thread count must not move the estimate"


def test_test_provider_methods_agree_on_estimate():
    ests = {}
    for method in ("wald", "score", "lrt"):
        out = dispatch("test_provider", data_path=DATA, provider="P007",
                       method=method, **MAPPING)
        assert out["status"] == "ok"
        assert 0 <= out["p_value"] <= 1
        ests[method] = out["gamma_est"]
    assert len(set(ests.values())) == 1, "same fit, so same point estimate"


def test_test_provider_unknown_provider_errors_helpfully():
    out = dispatch("test_provider", data_path=DATA, provider="NOPE", **MAPPING)
    assert out["status"] == "error"
    assert "not found" in out["message"]


def test_firth_notice_on_unpenalised_lrt():
    out = dispatch("test_provider", data_path=DATA_RARE, method="lrt",
                   y_expr="ExampleProviders_rare$Y",
                   z_expr="ExampleProviders_rare$Z",
                   id_expr="ExampleProviders_rare$ID")
    assert "_notice_lrt_unpenalised" in out


def test_benchmark_reports_agreement():
    out = dispatch("benchmark_threads", data_path=DATA, threads_list=[1, 2],
                   **MAPPING)
    assert out["status"] == "ok"
    assert out["estimates_agree"] is True
    assert len(out["runs"]) == 2


# --- the agent loop ----------------------------------------------------------

def test_context_injection_and_exposed_tools():
    agent = _agent([tool_turn("fit_flr", dict(data_path=DATA, **MAPPING)),
                    text_turn("The covariate effects are as follows.")])
    resp = agent.query("Fit the model.", data_path=DATA)

    sent = agent._client.calls[0]
    user_msg = sent["messages"][-1]["content"]
    assert "DATA STRUCTURE" in user_msg
    assert "FIELD MAPPING" in user_msg
    assert MAPPING["y_expr"] in user_msg
    exposed = {s["function"]["name"] for s in sent["tools"]}
    assert exposed == {"inspect_data", "start_analysis", "fit_flr",
                       "test_provider", "benchmark_threads"}
    assert sent["temperature"] == 0.0

    assert resp.error is None
    assert resp.text.startswith("The covariate effects")
    # Two events: the harness's own inspection, then the model's fit.
    assert [e.tool for e in resp.trace.events] == ["inspect_data", "fit_flr"]
    assert resp.trace.events[0].effective_args["_auto_prepend"] is True
    assert resp.trace.events[1].status == "ok"
    assert len(resp.trace.events[1].result_summary["beta"]) == 5


def test_scaffolding_off_skips_injection():
    agent = _agent([text_turn("I would need the field names.")],
                   scaffold={"all": False})
    resp = agent.query("Fit the model.", data_path=DATA)
    sent = agent._client.calls[0]
    assert "DATA STRUCTURE" not in sent["messages"][-1]["content"]
    assert len(sent["tools"]) == 6            # no subsetting either
    assert resp.trace.events == []            # no auto-inspection


def test_dispatch_guard_refuses_unexposed_tool():
    # simulate_provider_data is not exposed once a data file is in hand.
    agent = _agent([tool_turn("simulate_provider_data", {"n_providers": 10}),
                    text_turn("Sorry, let me use the data you gave me.")])
    resp = agent.query("Make me some data.", data_path=DATA)
    ev = resp.trace.events[-1]
    assert ev.tool == "simulate_provider_data"
    assert ev.status == "error"
    assert "not available" in ev.error_message


def test_dispatch_guard_off_lets_it_through():
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "Sim.rda")
        agent = _agent([tool_turn("simulate_provider_data",
                                  {"n_providers": 5, "n_per_provider": 20,
                                   "out_path": out}),
                        text_turn("Generated.")],
                       scaffold={"dispatch_guard": False})
        resp = agent.query("Make me some data.", data_path=DATA)
        assert resp.trace.events[-1].status == "ok"
        assert Path(out).exists()


def test_model_args_are_recorded_alongside_effective_args():
    """The trace must show both what the model asked for and what ran."""
    agent = _agent([tool_turn("fit_flr",
                              dict(data_path=DATA, threads=9999, **MAPPING)),
                    text_turn("Done.")])
    resp = agent.query("Fit it on all the cores.", data_path=DATA)
    ev = resp.trace.events[-1]
    assert ev.llm_args["threads"] == 9999
    assert ev.effective_args["threads"] == (os.cpu_count() or 1)


def test_max_turns_guard():
    agent = _agent([tool_turn("fit_flr", dict(data_path=DATA, **MAPPING))] * 4,
                   max_turns=3)
    resp = agent.query("Fit the model.", data_path=DATA)
    assert resp.error and "Max turns" in resp.error


# --- traces and repro.R ------------------------------------------------------

def _run_rscript(path):
    return subprocess.run(
        [find_rscript(), "--no-save", "--no-restore", "--no-init-file", path],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)


def test_trace_and_repro_for_a_fit():
    agent = _agent([tool_turn("fit_flr",
                              dict(data_path=DATA, threads=2, **MAPPING)),
                    text_turn("Fitted.")])
    resp = agent.query("Fit the model with 2 threads.", data_path=DATA)

    with tempfile.TemporaryDirectory() as td:
        trace_path = str(Path(td) / "trace.json")
        repro_path = str(Path(td) / "repro.R")
        resp.save_trace(trace_path)
        resp.write_repro_r(repro_path)

        saved = json.loads(Path(trace_path).read_text())
        assert saved["events"][1]["tool"] == "fit_flr"
        assert saved["tools_exposed"]["n_exposed"] == 5

        proc = _run_rscript(repro_path)
        assert proc.returncode == 0, proc.stderr
        # The generated script recomputes the coefficients by calling paraflr
        # directly — no Python, no model.
        assert "x1" in proc.stdout


def test_repro_for_a_provider_test():
    args = dict(data_path=DATA, provider="P007", method="lrt", firth=True,
                **MAPPING)
    agent = _agent([tool_turn("test_provider", args), text_turn("Tested.")])
    resp = agent.query("Penalised LRT for P007.", data_path=DATA)
    direct = dispatch("test_provider", **args)

    with tempfile.TemporaryDirectory() as td:
        repro_path = str(Path(td) / "repro.R")
        resp.write_repro_r(repro_path)
        proc = _run_rscript(repro_path)
        assert proc.returncode == 0, proc.stderr
        # The p-value the direct call produced must appear in the replay.
        assert "%.6f" % direct["p_value"] in proc.stdout or \
               "%.5f" % direct["p_value"] in proc.stdout, proc.stdout


def test_repro_skips_failed_steps():
    agent = _agent([tool_turn("fit_flr", dict(data_path=DATA, y_expr="nope$Y",
                                              z_expr=MAPPING["z_expr"],
                                              id_expr=MAPPING["id_expr"])),
                    text_turn("That failed.")])
    resp = agent.query("Fit it.", data_path=DATA)
    assert resp.trace.events[-1].status == "error"
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "repro.R")
        resp.write_repro_r(p)
        text = Path(p).read_text()
        assert "skipped, this call errored" in text
        assert _run_rscript(p).returncode == 0


# --- plain-python runner -----------------------------------------------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print("FAIL  %s: %s: %s" % (name, type(e).__name__, e))
    print("\n%d passed, %d failed, %d total"
          % (len(tests) - len(failed), len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
