"""E1 — routing benchmark.

Single-turn routing: given a plain-language request and no data file, does the
model call the right tool, with the right arguments? No R runs here, so the
whole benchmark finishes in about a minute regardless of dataset size.

Two numbers come out of it:

  * tool accuracy — the first tool called is an acceptable one. This is the
    routing decision itself.
  * argument accuracy — over the subset of queries that name a statistic, a
    null, or a penalty, whether those arguments came through correctly.
    Choosing `test_provider` and then running a Wald test when the user asked
    for a score test is a wrong answer that tool accuracy alone would score
    as a hit.

Scoring is deterministic (temperature=0), so rerunning reproduces the number
exactly; more queries, not more seeds, is what tightens the estimate.

    export PARAFLR_MODEL_ENDPOINT=http://localhost:8000/v1 \\
           PARAFLR_MODEL_NAME=qwen2.5-7b-awq OPENAI_API_KEY=EMPTY
    python run_routing.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import eval_common as E                        # puts the agent on sys.path
from paraflr_agent import load_schemas, SYSTEM_PROMPT_ROUTING  # noqa: E402

# `expect` lists the acceptable first tools. `expect_args` (optional) is the
# subset of arguments that must match for the call to count as right.
QUERIES = [
    # --- estimation -----------------------------------------------------
    {"id": "F01", "axis": "fit_plain",
     "query": "I have 50 hospitals with a binary complication indicator, 5 risk adjusters, and a hospital id per record. Fit the Firth-corrected logistic model with a separate effect per hospital.",
     "expect": ["fit_flr"]},
    {"id": "F02", "axis": "fit_covariate_effects",
     "query": "What are the adjusted covariate effects in my provider profiling data? I want the risk-adjustment coefficients, not a test.",
     "expect": ["fit_flr"]},
    {"id": "F03", "axis": "fit_provider_effects",
     "query": "Give me the estimated provider effects for all 300 dialysis facilities in my dataset.",
     "expect": ["fit_flr"]},
    {"id": "F04", "axis": "fit_threads",
     "query": "Fit the model on my claims data using 8 cores.",
     "expect": ["fit_flr"], "expect_args": {"threads": 8}},
    {"id": "F05", "axis": "fit_large_not_benchmark",
     "query": "I have 4 million records across 1,200 providers. Fit the provider-effect model; it needs to actually finish, so use as many threads as you can.",
     "expect": ["fit_flr"]},
    {"id": "F06", "axis": "fit_separation",
     "query": "Several of my small centres have zero events, so plain logistic regression blows up. Fit the bias-reduced version with provider intercepts.",
     "expect": ["fit_flr"]},
    {"id": "F07", "axis": "fit_cutoff",
     "query": "Fit the model but drop any provider with fewer than 25 records.",
     "expect": ["fit_flr"], "expect_args": {"cutoff": 25}},
    {"id": "F08", "axis": "fit_rerun_tolerance",
     "query": "Refit the provider model with a tighter convergence tolerance of 1e-8.",
     "expect": ["fit_flr"], "expect_args": {"tol": 1e-08}},
    {"id": "F09", "axis": "fit_wording_estimate",
     "query": "Estimate the hospital-specific intercepts and the covariate coefficients for my readmission data.",
     "expect": ["fit_flr"]},
    {"id": "F10", "axis": "fit_no_test_wanted",
     "query": "Just run the model. I do not need any hypothesis testing yet.",
     "expect": ["fit_flr"]},

    # --- testing one provider -------------------------------------------
    {"id": "T01", "axis": "test_default",
     "query": "Is provider P012 significantly different from the median provider?",
     "expect": ["test_provider"]},
    {"id": "T02", "axis": "test_score_explicit",
     "query": "Run a score test for provider P003 against the median provider effect.",
     "expect": ["test_provider"], "expect_args": {"method": "score"}},
    {"id": "T03", "axis": "test_wald_explicit",
     "query": "Give me the Wald test and p-value for facility P021.",
     "expect": ["test_provider"], "expect_args": {"method": "wald"}},
    {"id": "T04", "axis": "test_lrt_explicit",
     "query": "Use a likelihood ratio test to check whether centre P008 differs from the median.",
     "expect": ["test_provider"], "expect_args": {"method": "lrt"}},
    {"id": "T05", "axis": "test_lrt_firth",
     "query": "Provider P015 has only 3 events. Use the Firth-penalised likelihood ratio test for it.",
     "expect": ["test_provider"], "expect_args": {"method": "lrt", "firth": True}},
    {"id": "T06", "axis": "test_penalised_wording",
     "query": "Run the penalised LRT for hospital P002 against the population median.",
     "expect": ["test_provider"], "expect_args": {"method": "lrt", "firth": True}},
    {"id": "T07", "axis": "test_fixed_null",
     "query": "Test whether provider P009's effect differs from a fixed value of -1.5 rather than from the median.",
     "expect": ["test_provider"], "expect_args": {"null": "value", "null_value": -1.5}},
    {"id": "T08", "axis": "test_outlier_wording",
     "query": "Is hospital P044 an outlier in my provider profiling data?",
     "expect": ["test_provider"]},
    {"id": "T09", "axis": "test_flag_wording",
     "query": "Should facility P030 be flagged as a poor performer? I want a p-value.",
     "expect": ["test_provider"]},
    {"id": "T10", "axis": "test_worse_than_average",
     "query": "My regulator wants to know whether centre P007 performs worse than average after risk adjustment.",
     "expect": ["test_provider"]},
    {"id": "T11", "axis": "test_alpha",
     "query": "Test provider P011 at the 1% significance level.",
     "expect": ["test_provider"], "expect_args": {"alpha": 0.01}},
    {"id": "T12", "axis": "test_rao_wording",
     "query": "I would like Rao's efficient score statistic for provider P005.",
     "expect": ["test_provider"], "expect_args": {"method": "score"}},
    {"id": "T13", "axis": "test_fit_then_test",
     "query": "Fit the model and then tell me whether provider P019 stands out from the median.",
     "expect": ["test_provider"]},
    {"id": "T14", "axis": "test_numeric_id",
     "query": "Test provider 42 against the median using the Wald statistic.",
     "expect": ["test_provider"], "expect_args": {"method": "wald"}},

    # --- performance ----------------------------------------------------
    {"id": "B01", "axis": "bench_scaling",
     "query": "How well does this scale? Time the fit at 1, 2, 4 and 8 threads.",
     "expect": ["benchmark_threads"], "expect_args": {"threads_list": [1, 2, 4, 8]}},
    {"id": "B02", "axis": "bench_how_many_cores",
     "query": "How many cores should I actually use for this dataset?",
     "expect": ["benchmark_threads"]},
    {"id": "B03", "axis": "bench_speedup",
     "query": "What speedup do I get from OpenMP parallelism on my data?",
     "expect": ["benchmark_threads"]},
    {"id": "B04", "axis": "bench_how_long",
     "query": "Roughly how long will a fit take on this data, and does adding threads help?",
     "expect": ["benchmark_threads"]},
    {"id": "B05", "axis": "bench_repeats",
     "query": "Benchmark the fit across thread counts, 3 repeats each so the timings are stable.",
     "expect": ["benchmark_threads"], "expect_args": {"repeats": 3}},
    {"id": "B06", "axis": "bench_agreement",
     "query": "Do the coefficients come out the same whether I run on 1 thread or 8?",
     "expect": ["benchmark_threads"]},

    # --- simulation -----------------------------------------------------
    {"id": "S01", "axis": "sim_plain",
     "query": "I do not have data yet. Generate a clustered binary dataset with provider effects so I can try this out.",
     "expect": ["simulate_provider_data"]},
    {"id": "S02", "axis": "sim_sized",
     "query": "Simulate 200 providers with about 60 records each and 6 covariates.",
     "expect": ["simulate_provider_data"],
     "expect_args": {"n_providers": 200, "n_per_provider": 60, "n_covariates": 6}},
    {"id": "S03", "axis": "sim_rare",
     "query": "Make me a test dataset with a 2% event rate so that some providers have no events at all.",
     "expect": ["simulate_provider_data"], "expect_args": {"event_rate": 0.02}},
    {"id": "S04", "axis": "sim_seed",
     "query": "Generate example provider data with seed 123 so I can reproduce it.",
     "expect": ["simulate_provider_data"], "expect_args": {"seed": 123}},

    # --- inspection and the wizard --------------------------------------
    {"id": "I01", "axis": "inspect_path",
     "query": "What is inside /data/claims_2024.rds?",
     "expect": ["inspect_data"]},
    {"id": "I02", "axis": "inspect_fields",
     "query": "Open /home/user/providers.rda and tell me which fields it has.",
     "expect": ["inspect_data"]},
    {"id": "I03", "axis": "inspect_csv",
     "query": "I am not sure my file is formatted right. Check /tmp/hospital_data.csv.",
     "expect": ["inspect_data"]},
    {"id": "W01", "axis": "wizard_vague",
     "query": "Help me with my provider data.",
     "expect": ["start_analysis"]},
    {"id": "W02", "axis": "wizard_no_goal",
     "query": "I work on hospital quality measurement. Where do I start with this package?",
     "expect": ["start_analysis"]},
    {"id": "W03", "axis": "wizard_zh",
     "query": "我想做医院绩效分析，能帮我看看吗？",
     "expect": ["start_analysis"], "expect_args": {"user_language": "zh"}},
]


def _args_match(emitted, expected):
    """True if every expected argument is present and equal in `emitted`.

    Numbers are compared with a tolerance so that 1e-8 and 0.00000001 agree;
    lists must match element for element.
    """
    for key, want in expected.items():
        if key not in emitted:
            return False
        got = emitted[key]
        if isinstance(want, bool):
            if bool(got) is not want:
                return False
        elif isinstance(want, (int, float)):
            try:
                if abs(float(got) - float(want)) > 1e-12 * max(1.0, abs(want)):
                    return False
            except (TypeError, ValueError):
                return False
        elif isinstance(want, list):
            if list(got or []) != want:
                return False
        elif str(got).strip().lower() != str(want).strip().lower():
            return False
    return True


def run_one(client, model, tools, q):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_ROUTING},
                {"role": "user", "content": q["query"]}]
    first_tool, all_tools, first_args, err = None, [], {}, None
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
            tool_choice="auto", temperature=0.0, timeout=60)
        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None) or []
        all_tools = [tc.function.name for tc in tcs]
        if tcs:
            first_tool = all_tools[0]
            try:
                first_args = json.loads(tcs[0].function.arguments or "{}")
            except json.JSONDecodeError:
                first_args = {}
    except Exception as e:  # noqa: BLE001
        err = str(e)[:300]

    hit = first_tool in q["expect"]
    expect_args = q.get("expect_args")
    arg_hit = None
    if expect_args is not None:
        arg_hit = bool(hit and _args_match(first_args, expect_args))
    return {"id": q["id"], "axis": q["axis"], "query": q["query"],
            "expect": q["expect"], "expect_args": expect_args,
            "first_tool": first_tool, "first_args": first_args,
            "all_tools": all_tools, "error": err,
            "latency_s": round(time.time() - t0, 2),
            "hit": hit, "arg_hit": arg_hit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_out/routing.json")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cfg = E.endpoint_cfg()
    from openai import OpenAI
    client = OpenAI(base_url=cfg["model_endpoint"], api_key=cfg["api_key"])
    tools = load_schemas()

    print("=== routing benchmark: {} queries, {} tools, temp=0 ===".format(
        len(QUERIES), len(tools)))
    rows = []
    for i, q in enumerate(QUERIES, 1):
        r = run_one(client, cfg["model_name"], tools, q)
        rows.append(r)
        mark = "OK" if r["hit"] else "XX"
        if r["hit"] and r["arg_hit"] is False:
            mark = "~args"
        print("[{:3d}/{}] {:4s} {:26s} -> {:24s} {} ({}s)".format(
            i, len(QUERIES), q["id"], q["axis"][:26],
            r["first_tool"] or "NO_TOOL", mark, r["latency_s"]))

    n_hit = sum(1 for r in rows if r["hit"])
    scored_args = [r for r in rows if r["arg_hit"] is not None]
    n_arg_hit = sum(1 for r in scored_args if r["arg_hit"])

    E.save_json({
        "experiment": "E1_routing_benchmark",
        "n": len(QUERIES),
        "n_hit": n_hit,
        "tool_accuracy": round(n_hit / len(QUERIES), 4),
        "n_arg_scored": len(scored_args),
        "n_arg_hit": n_arg_hit,
        "arg_accuracy": (round(n_arg_hit / len(scored_args), 4)
                         if scored_args else None),
        "note": ("Single-turn routing from a verbal request, no data file, no R "
                 "execution, temperature=0. Argument accuracy is scored only "
                 "over the queries that name a statistic, a null, a penalty, or "
                 "a size, and requires the tool to be right as well."),
        "rows": rows}, args.out)

    print("\nTool accuracy:     {}/{} = {:.1f}%".format(
        n_hit, len(QUERIES), 100 * n_hit / len(QUERIES)))
    if scored_args:
        print("Argument accuracy: {}/{} = {:.1f}% (of the argument-scored subset)"
              .format(n_arg_hit, len(scored_args),
                      100 * n_arg_hit / len(scored_args)))


if __name__ == "__main__":
    main()
