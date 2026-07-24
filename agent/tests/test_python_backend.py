"""Python-backend smoke test — the agent's six tools computed through the
paraflr **Python** package (``paraflr-py``), no R in the loop.

Skips if the paraflr Python package is not importable (build it with
``pip install ./paraflr-py`` from the repo root, or run
``python setup.py build_ext --inplace`` in ``paraflr-py/``).

Run either way::

    python tests/test_python_backend.py
    pytest tests/test_python_backend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent
REPO = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(REPO / "paraflr-py"))

pytest.importorskip(
    "paraflr", reason="paraflr-py not built; `pip install ./paraflr-py`")

from paraflr_agent import tools  # noqa: E402


def test_python_backend_end_to_end(tmp_path):
    out = str(tmp_path / "sim.npz")
    sim = tools.dispatch("simulate_provider_data", backend="python",
                         n_providers=6, n_per_provider=30, n_covariates=3,
                         seed=11, out_path=out)
    assert sim["status"] == "ok"
    assert (sim["y_expr"], sim["z_expr"], sim["id_expr"]) == ("Y", "Z", "ID")
    dp = sim["data_path"]

    ins = tools.dispatch("inspect_data", backend="python", data_path=dp)
    assert ins["status"] == "ok" and ins["mapping"]
    m = ins["mapping"]

    fit = tools.dispatch("fit_flr", backend="python", data_path=dp,
                         y_expr=m["y_expr"], z_expr=m["z_expr"],
                         id_expr=m["id_expr"], threads=1)
    assert fit["status"] == "ok"
    assert len(fit["beta"]) == 3 and fit["n_providers"] == 6

    prov = next(iter(fit["gamma"]))
    for method in ("wald", "score", "lrt"):
        t = tools.dispatch("test_provider", backend="python", data_path=dp,
                           y_expr=m["y_expr"], z_expr=m["z_expr"],
                           id_expr=m["id_expr"], method=method, provider=prov)
        assert t["status"] == "ok" and t["provider"] == prov
        assert t["method"] == method

    bm = tools.dispatch("benchmark_threads", backend="python", data_path=dp,
                        y_expr=m["y_expr"], z_expr=m["z_expr"],
                        id_expr=m["id_expr"], threads_list=[1, 2])
    assert bm["status"] == "ok" and bm["estimates_agree"]


def test_python_backend_rejects_rda(tmp_path):
    # .rda is the R backend's format; the Python backend says so clearly.
    f = tmp_path / "x.rda"
    f.write_bytes(b"not really an rda")
    r = tools.dispatch("inspect_data", backend="python", data_path=str(f))
    assert r["status"] == "error" and "npz" in r["message"].lower()


def test_unknown_backend():
    r = tools.dispatch("fit_flr", backend="martian")
    assert r["status"] == "error" and r["class"] == "UnknownBackend"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
