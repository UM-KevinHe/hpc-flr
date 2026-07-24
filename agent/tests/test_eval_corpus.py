"""Endpoint-free checks on the evaluation layer itself.

The four experiments in ``agent/eval`` need a served model, so they cannot run
here. What CAN be checked without one is that they would ask the right
questions: that every expected tool exists, that every expected argument is a
real parameter of that tool's schema (with a valid enum value), that the
representatives' data files and manual arguments are valid, and that the
fidelity comparison does what it claims.

A typo in `expect_args` would otherwise sit quietly in the corpus and be
scored as a model failure forever.

    python tests/test_eval_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR / "eval"))
sys.path.insert(0, str(AGENT_DIR / "tests"))

import eval_common as E  # noqa: E402
import run_routing  # noqa: E402
from paraflr_agent import dispatch, load_schemas  # noqa: E402
from test_offline import _agent, text_turn, tool_turn  # noqa: E402

SCHEMAS = {s["function"]["name"]: s["function"] for s in load_schemas()}


def _params(tool):
    return SCHEMAS[tool]["parameters"]["properties"]


# --- the routing corpus ------------------------------------------------------

def test_query_ids_are_unique():
    ids = [q["id"] for q in run_routing.QUERIES]
    assert len(ids) == len(set(ids))


def test_expected_tools_exist():
    for q in run_routing.QUERIES:
        for tool in q["expect"]:
            assert tool in SCHEMAS, "%s expects unknown tool %s" % (q["id"], tool)


def test_expected_args_are_real_parameters():
    for q in run_routing.QUERIES:
        expect_args = q.get("expect_args")
        if not expect_args:
            continue
        # An argument expectation only makes sense against a single tool.
        assert len(q["expect"]) == 1, q["id"]
        props = _params(q["expect"][0])
        for key, want in expect_args.items():
            assert key in props, "%s: %s has no parameter %r" % (
                q["id"], q["expect"][0], key)
            enum = props[key].get("enum")
            if enum is not None:
                assert want in enum, "%s: %r not in %s" % (q["id"], want, enum)


def test_corpus_covers_every_tool():
    covered = {t for q in run_routing.QUERIES for t in q["expect"]}
    assert covered == set(SCHEMAS), "uncovered tools: %s" % (
        set(SCHEMAS) - covered)


def test_args_matcher():
    m = run_routing._args_match
    assert m({"method": "score"}, {"method": "score"})
    assert m({"method": "Score "}, {"method": "score"})       # case / spacing
    assert m({"tol": 1e-08}, {"tol": 0.00000001})             # numeric equality
    assert m({"threads_list": [1, 2, 4]}, {"threads_list": [1, 2, 4]})
    assert m({"firth": True, "method": "lrt"}, {"method": "lrt", "firth": True})
    assert not m({"method": "wald"}, {"method": "score"})
    assert not m({}, {"method": "score"})                     # missing entirely
    assert not m({"firth": False}, {"firth": True})
    assert not m({"threads_list": [1, 2]}, {"threads_list": [1, 2, 4]})


# --- the representatives -----------------------------------------------------

def test_rep_definitions():
    for rep in E.REPS:
        assert rep["expected_tool"] in SCHEMAS
        assert Path(E.data_path(rep)).exists(), (
            "missing %s — run Rscript agent/data/make_example_data.R"
            % rep["data"])
        assert "{data_path}" in rep["query"]
        props = _params(rep["expected_tool"])
        for key in rep["manual"]:
            assert key in props, "%s: no parameter %r" % (rep["id"], key)


def test_reference_results_run():
    for rep in E.REPS:
        ref = E.reference_result(rep)
        assert ref["status"] == "ok", ref.get("message")
        assert E.bit_match(ref, ref)


def test_bit_match_is_not_vacuous():
    """A different configuration must NOT compare equal."""
    rep = E.REPS[1]                                    # the provider test
    ref = E.reference_result(rep)
    other = dict(rep["manual"])
    other["method"] = "wald"                           # a different statistic
    alt = dispatch(rep["expected_tool"], data_path=E.data_path(rep), **other)
    assert alt["status"] == "ok"
    assert not E.bit_match(alt, ref)

    # And an error result never counts as a match.
    assert not E.bit_match({"status": "error", "message": "x"}, ref)
    assert not E.bit_match(None, ref)


def test_fidelity_path_with_a_stub_model():
    """Run the fidelity comparison end to end, with a stub in place of Qwen.

    This is what run_fidelity.py does per representative, minus the served
    model: the agent's own tool result must equal the direct call exactly.
    """
    for rep in E.REPS:
        dp = E.data_path(rep)
        ref = E.reference_result(rep)
        agent = _agent([tool_turn(rep["expected_tool"],
                                  dict(data_path=dp, **rep["manual"])),
                        text_turn("Here are the results.")])
        resp = agent.query(rep["query"].format(data_path=dp), data_path=dp,
                           task_override=rep["task"])
        agent_result = next(tr["result"] for tr in resp.tool_results
                            if tr["tool"] == rep["expected_tool"])
        assert E.classify_outcome(resp, rep) == "correct"
        assert E.tools_called(resp) == [rep["expected_tool"]]
        assert E.bit_match(agent_result, ref), rep["id"]
        assert E.usage(resp)["llm_turns"] == 2


def test_outcome_taxonomy():
    rep = E.REPS[0]
    dp = E.data_path(rep)

    # A wrong-but-clean tool call is the failure the taxonomy is built around.
    agent = _agent([tool_turn("benchmark_threads",
                              dict(data_path=dp, threads_list=[1],
                                   y_expr=rep["manual"]["y_expr"],
                                   z_expr=rep["manual"]["z_expr"],
                                   id_expr=rep["manual"]["id_expr"])),
                    text_turn("Benchmarked.")])
    resp = agent.query("x", data_path=dp)
    assert E.classify_outcome(resp, rep) == "silently_wrong"

    # No tool call at all.
    agent = _agent([text_turn("You could use glm() for this.")])
    resp = agent.query("x", data_path=dp)
    assert E.classify_outcome(resp, rep) == "refused"

    # A tool that ran and errored.
    agent = _agent([tool_turn("fit_flr", dict(data_path=dp, y_expr="nope$Y",
                                              z_expr=rep["manual"]["z_expr"],
                                              id_expr=rep["manual"]["id_expr"])),
                    text_turn("That failed.")])
    resp = agent.query("x", data_path=dp)
    assert E.classify_outcome(resp, rep) == "crashed"


def test_save_json_roundtrip(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "out" / "x.json")
        E.save_json({"experiment": "smoke", "rows": []}, p)
        assert json.loads(Path(p).read_text())["experiment"] == "smoke"


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
