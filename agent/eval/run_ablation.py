"""E3 — scaffolding ablation.

The model is held fixed; the harness layers are switched off one at a time.
The question is whether the scaffolding does any work, or whether a 7B model
with six tools would have got there anyway. temperature=0, so one run per
cell settles it.

Cells:
  full               everything on — the deployed configuration
  no_structure       no auto-inspection, no injected field mapping; the model
                     is told the file path and must name the fields itself
  no_subset          all six tools exposed regardless of what the data allows
  no_dispatch_guard  a tool call the model was never offered is executed
                     rather than refused
  raw_all_off        every layer off: path only, all tools, no guard
  bare_prompt        raw, plus a generic assistant prompt in place of the
                     routing prompt — roughly "a 7B model with six tools"

    python run_ablation.py
"""
from __future__ import annotations

import argparse

import eval_common as E

BARE_PROMPT = "You are a helpful assistant. Use the available tools."

CELLS = {
    "full":              dict(scaffold=None),
    "no_structure":      dict(scaffold={"inject_structure": False}),
    "no_subset":         dict(scaffold={"subset_tools": False}),
    "no_dispatch_guard": dict(scaffold={"dispatch_guard": False}),
    "raw_all_off":       dict(scaffold={"all": False}),
    "bare_prompt":       dict(scaffold={"all": False},
                              system_prompt=BARE_PROMPT),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_out/ablation.json")
    args = ap.parse_args()
    cfg = E.endpoint_cfg()

    rows = []
    for rep in E.REPS:
        dp = E.data_path(rep)
        q = rep["query"].format(data_path=dp)
        for cell, kw in CELLS.items():
            agent = E.ParaFLRAgent(**cfg,
                                   system_prompt=kw.get("system_prompt"),
                                   scaffold=kw.get("scaffold"))
            resp = agent.query(q, data_path=dp)
            row = {
                "rep": rep["id"],
                "cell": cell,
                "outcome": E.classify_outcome(resp, rep),
                "tools_called": E.tools_called(resp),
                # The failure this catches: right tool, invented field name.
                "tool_errors": [e.error_message for e in resp.trace.events
                                if e.status != "ok" and e.error_message],
                "agent_error": resp.error,
                **E.usage(resp),
            }
            rows.append(row)
            print("{:20s} {:18s} -> {}".format(rep["id"], cell, row["outcome"]))

    E.save_json({"experiment": "E3_scaffolding_ablation",
                 "cells": list(CELLS), "rows": rows}, args.out)


if __name__ == "__main__":
    main()
