# `paraflr` agent

A small natural-language front end to the `paraflr` package: a locally
served open-weights model (Qwen 2.5-7B by default) routes a plain-language
request to one of six tools, and **paraflr computes every reported number**.
The model chooses and explains; it never calculates.

The computation runs on **either backend** — the `paraflr` **R** package or
the `paraflr` **Python** package (`paraflr-py`), chosen with one argument.
Both call the same C++ core, so the numbers are identical (see
[Backends](#backends-r-or-python) below). Every run leaves a `trace.json` of
what was called with which arguments; the R backend also emits a standalone
`repro.R` that recomputes the numbers with no model in the loop.

This is a convenience layer, not part of the method. `paraflr` is complete
and usable without it.

```
agent/
  paraflr_agent/      the agent package (Python)
    r_scripts/        R backend: one R script per tool; JSON in, JSON out
    backend_py.py     Python backend: the same six tools via paraflr-py
    schemas.json      the six tool schemas sent to the model
  data/               simulated example datasets + the script that builds them
  eval/               the four evaluation experiments
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

The example datasets are simulated: the Medicare claims data behind the paper
cannot be redistributed. `ExampleProviders.rda` is the ordinary case (50
providers, 20% event rate); `ExampleProviders_rare.rda` has 40 small providers
and a 4% event rate, so a third of them see no events at all — which is where
the Firth correction matters and where the three tests disagree.

## Serve a model

Any OpenAI-compatible endpoint works. The runs behind the evaluation used
vLLM on one A40:

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
resp.write_repro_r("repro.R")      # runs on its own, with just paraflr
```

The tools are also callable directly, with no model in the loop — this is the
path the evaluation uses as its reference:

```python
from paraflr_agent import dispatch

dispatch("fit_flr", data_path="agent/data/ExampleProviders.rda",
         y_expr="ExampleProviders$Y", z_expr="ExampleProviders$Z",
         id_expr="ExampleProviders$ID", threads=4)
```

## Backends: R or Python

The same agent drives either computation backend; pass `backend=` (default
`"r"`). No R is needed for the Python backend.

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

Both backends compile the same C++ core, so a fit on one matches the other to
machine precision (`paraflr-py/tests/parity_with_r.py`). The model's routing —
and everything the harness does below — is backend-agnostic.

## The six tools

| Tool | What it runs |
|---|---|
| `fit_flr` | `logis_firth()` — provider effects and covariate effects |
| `test_provider` | `test_gamma.single()` — Wald / score / penalised LRT for one provider |
| `benchmark_threads` | the same fit across OpenMP thread counts, with a check that the estimates agree |
| `inspect_data` | file structure, plus a guess at which fields are the outcome, covariates, and provider id |
| `simulate_provider_data` | clustered binary data with provider effects, for when there is none |
| `start_analysis` | the wizard, for requests too vague to route |

`test_gamma.single()` tests the first provider block of a fitted object.
`test_provider` re-indexes the fitted object — records and `gamma` together —
so any requested provider can be tested. No estimate changes; the block
structure the test relies on is preserved.

## What the harness does before the model sees anything

Four layers, each of which can be switched off individually (that is what the
ablation does):

1. **Context injection.** The data file is inspected first, in R, and the
   resulting structure — plus a derived `y_expr` / `z_expr` / `id_expr`
   mapping — goes into the request. A 7B model asked to call `inspect_data`
   first will often skip it and invent a plausible field name instead; making
   this architectural removes the question.
2. **Tool subsetting.** Only admissible tools are exposed: no fitting without
   a data file, no simulating with one.
3. **Dispatch guard.** A tool the model was not offered is refused with a
   message listing what it may call, rather than executed.
4. **Argument resolution.** Thread counts above the machine's core count are
   clamped, the null hypothesis is resolved from the schema's two fields,
   method names are normalised, and non-ASCII R identifiers are rejected with
   instructions to re-inspect. All of it is recorded in the trace as
   `effective_args` alongside the model's own `llm_args`, so a reader can see
   both what was asked for and what ran.

Decoding is greedy (`temperature=0`): routing is deterministic and a run can
be reproduced from its trace.

## Tests

```bash
pytest agent/tests/
```

No GPU and no network: a stub replaces the model, so the loop, the tools, the
trace, and the generated `repro.R` are all exercised — the last by running it
through `Rscript` and checking the numbers come back. `test_offline.py` covers
the R backend; `test_python_backend.py` runs the six tools through the Python
backend (and skips if `paraflr-py` is not built). Whether a real model routes
correctly is a separate question, and the subject of `agent/eval`.
