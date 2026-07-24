# `paraflr` agent

A small natural-language front end to the `paraflr` package: a locally
served open-weights model (Qwen 2.5-7B by default) routes a plain-language
request to one of six tools, and paraflr computes every reported number.
The model chooses and explains; it never calculates.

The computation runs on **either backend** — the `paraflr` R package or the
`paraflr` Python package (`paraflr-py`), chosen with one argument. Every run
leaves a `trace.json` of what was called with which arguments; the R backend
also emits a standalone `repro.R`.

This is a convenience layer; `paraflr` is complete and usable without it.

```
agent/
  paraflr_agent/      the agent package (Python)
    r_scripts/        R backend: one R script per tool; JSON in, JSON out
    backend_py.py     Python backend: the same six tools via paraflr-py
    schemas.json      the six tool schemas sent to the model
  data/               example datasets + the script that builds them
  tests/              offline tests — no GPU, no network
```

## Install

The harness is Python; the computation backend is your choice — install at
least one.

```bash
pip install -r agent/requirements.txt      # the harness itself

R CMD INSTALL paraflr                       # R backend (reads .rda/.rds/.csv)
pip install ./paraflr-py                    # Python backend (reads .npz/.csv)
```

For the R backend, build the bundled example datasets:

```bash
Rscript agent/data/make_example_data.R
```

`ExampleProviders.rda` is the ordinary case (50 providers, 20% event rate);
`ExampleProviders_rare.rda` has 40 small providers and a 4% event rate.

## Serve a model

Any OpenAI-compatible endpoint works, e.g. vLLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --served-model-name qwen2.5-7b-awq \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

```bash
export PARAFLR_MODEL_ENDPOINT=http://localhost:8000/v1
export PARAFLR_MODEL_NAME=qwen2.5-7b-awq
export OPENAI_API_KEY=EMPTY
```

## Use

```python
from paraflr_agent import ParaFLRAgent

agent = ParaFLRAgent(model_endpoint="http://localhost:8000/v1",
                     model_name="qwen2.5-7b-awq")

resp = agent.query(
    "Is facility P004 significantly different from the median? "
    "It only has a handful of events, so use the penalised likelihood ratio test.",
    data_path="agent/data/ExampleProviders_rare.rda")

print(resp.text)
resp.save_trace("trace.json")
resp.write_repro_r("repro.R")
```

The tools are also callable directly, with no model in the loop:

```python
from paraflr_agent import dispatch

dispatch("fit_flr", data_path="agent/data/ExampleProviders.rda",
         y_expr="ExampleProviders$Y", z_expr="ExampleProviders$Z",
         id_expr="ExampleProviders$ID", threads=4)
```

## Backends: R or Python

The same agent drives either backend; pass `backend=` (default `"r"`). No R is
needed for the Python backend.

```python
# Python backend — computes through paraflr-py, no R in the loop.
agent = ParaFLRAgent(model_endpoint="http://localhost:8000/v1",
                     model_name="qwen2.5-7b-awq", backend="python")

# or a direct call:
from paraflr_agent import dispatch
sim = dispatch("simulate_provider_data", backend="python", n_providers=8)
dispatch("fit_flr", backend="python", data_path=sim["data_path"],
         y_expr="Y", z_expr="Z", id_expr="ID", threads=4)
```

| | R backend (`backend="r"`, default) | Python backend (`backend="python"`) |
|---|---|---|
| computes via | `paraflr` R package | `paraflr` Python package (`paraflr-py`) |
| data files | `.rda`, `.rds`, `.csv` | `.npz`, `.csv` |
| `*_expr` args | R expressions (e.g. `Data$Y`) | field names (an `.npz` key / CSV column) |
| reproducer | `repro.R` | — |

## The six tools

| Tool | What it runs |
|---|---|
| `fit_flr` | `logis_firth()` — provider effects and covariate effects |
| `test_provider` | `test_gamma.single()` — Wald / score / penalised LRT for one provider |
| `benchmark_threads` | the same fit across OpenMP thread counts |
| `inspect_data` | file structure, plus a guess at which fields are the outcome, covariates, and provider id |
| `simulate_provider_data` | clustered binary data with provider effects, for when there is none |
| `start_analysis` | the wizard, for requests too vague to route |

## Tests

```bash
pytest agent/tests/
```

No GPU and no network: a stub replaces the model. `test_offline.py` covers the
R backend; `test_python_backend.py` runs the six tools through the Python
backend (and skips if `paraflr-py` is not built).
