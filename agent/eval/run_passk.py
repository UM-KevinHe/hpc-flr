"""E4 — reliability at temperature > 0.

The deployment decodes greedily (temperature=0), which makes a run
reproducible from its trace. This experiment asks what that costs or saves:
each representative is run k times at a sampling temperature, and we record
how often the agent still lands the right estimator. pass^k — every one of k
runs correct — is the quantity that matters for an analysis someone will act
on, and it decays fast in k unless per-run accuracy is essentially 1.

    python run_passk.py --k 8 --temperature 0.7
"""
from __future__ import annotations

import argparse

import eval_common as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="eval_out/passk.json")
    args = ap.parse_args()
    cfg = E.endpoint_cfg()

    rows = []
    for rep in E.REPS:
        dp = E.data_path(rep)
        q = rep["query"].format(data_path=dp)
        outcomes = []
        for i in range(args.k):
            agent = E.ParaFLRAgent(**cfg, temperature=args.temperature)
            resp = agent.query(q, data_path=dp)
            oc = E.classify_outcome(resp, rep)
            outcomes.append(oc)
            print("{:20s} run {}/{} -> {}".format(rep["id"], i + 1, args.k, oc))
        n_ok = sum(1 for o in outcomes if o == "correct")
        p = n_ok / args.k
        rows.append({"rep": rep["id"], "k": args.k,
                     "temperature": args.temperature,
                     "n_correct": n_ok, "pass_at_1": p,
                     "pass_hat_k": p ** args.k,
                     "outcomes": outcomes})
        print("{:20s} pass@1 = {}/{}".format(rep["id"], n_ok, args.k))

    E.save_json({
        "experiment": "E4_passk",
        "note": ("pass_hat_k = (empirical per-run success rate)^k, a plain "
                 "reliability proxy; with large k use the unbiased pass^k "
                 "estimator instead. The point of the table is how quickly "
                 "reliability decays once decoding is not deterministic."),
        "rows": rows}, args.out)


if __name__ == "__main__":
    main()
