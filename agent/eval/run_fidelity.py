"""E2 — fidelity: the agent's numbers versus a direct call.

For each representative, run the verified direct dispatch (no model) and the
agent (plain-language request, configuration bound) on the same data, and
check the estimates are identical — not close, identical. The model routes;
paraflr computes; so a mismatch is a bug in the harness, not a statistical
result. That is why this is shown once per representative rather than
averaged over runs.

Each run also writes a trace and a standalone ``repro.R`` that recomputes the
same numbers by calling paraflr directly, with no Python and no model.

    python run_fidelity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import eval_common as E


def _bound_context(rep):
    """The exact configuration the agent must use, as a UI would supply it."""
    m = rep["manual"]
    fields = ("outcome = {y}, covariates = {z}, provider id = {i}".format(
        y=m["y_expr"], z=m["z_expr"], i=m["id_expr"]))
    if rep["expected_tool"] == "fit_flr":
        return ("Use exactly this configuration: {}; threads = {}."
                .format(fields, m.get("threads", 1)))
    return ("Use exactly this configuration: {}; provider = {}; test = {}; "
            "null = population median; alpha = 0.05."
            .format(fields, m["provider"], m["method"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_out/fidelity.json")
    ap.add_argument("--trace-dir", default="eval_out")
    args = ap.parse_args()
    Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
    cfg = E.endpoint_cfg()

    rows = []
    for rep in E.REPS:
        dp = E.data_path(rep)

        # (1) the verified direct call — no model in the loop.
        ref = E.reference_result(rep)
        ref_status = ref.get("status") if isinstance(ref, dict) else None
        print("{:20s} direct {:18s} status: {}".format(
            rep["id"], rep["expected_tool"], ref_status))

        # (2) the agent, from a plain-language request, on the same data.
        agent = E.ParaFLRAgent(**cfg)
        resp = agent.query(rep["query"].format(data_path=dp), data_path=dp,
                           task_override=rep["task"],
                           bound_context=_bound_context(rep))
        agent_result = next((tr["result"] for tr in resp.tool_results
                             if tr["tool"] == rep["expected_tool"]), None)

        trace_path = "{}/fidelity_{}_trace.json".format(args.trace_dir, rep["id"])
        repro_path = "{}/fidelity_{}_repro.R".format(args.trace_dir, rep["id"])
        resp.save_trace(trace_path)
        resp.write_repro_r(repro_path)

        row = {
            "rep": rep["id"],
            "expected_tool": rep["expected_tool"],
            "outcome": E.classify_outcome(resp, rep),
            "tools_called": E.tools_called(resp),
            "identical_to_direct_call": E.bit_match(agent_result, ref),
            "direct_status": ref_status,
            "trace": trace_path,
            "repro_r": repro_path,
            **E.usage(resp),
        }
        rows.append(row)
        print("{:20s} outcome: {} | identical to direct call: {}".format(
            rep["id"], row["outcome"], row["identical_to_direct_call"]))

    E.save_json({
        "experiment": "E2_fidelity",
        "note": ("Determinism shown once per representative: the agent's "
                 "estimates must equal a direct paraflr call exactly. Each "
                 "repro.R recomputes them with no model and no Python."),
        "rows": rows}, args.out)


if __name__ == "__main__":
    main()
