# hpc-flr

High-performance computing implementations of Firth bias-reduced logistic
regression (FLR) with high-dimensional provider effects, for provider profiling
with massive clustered healthcare data.

This repository provides:

| Path | Language | What it is |
|------|----------|------------|
| [`paraflr/`](paraflr) | R + C++ (RcppArmadillo/OpenMP) | **ParaFLR (R)**: shared-memory parallel FLR fitting, plus Wald / score / penalized likelihood-ratio tests for provider effects. |
| [`paraflr-py/`](paraflr-py) | Python + C++ (pybind11/Armadillo) | **ParaFLR (Python)**: the same estimator, for Python users. |
| [`disflr/`](disflr) | Scala (Apache Spark) | **DisFLR**: distributed FLR for data exceeding single-node memory — an exact iterative variant (`MultiShot`) and a communication-efficient divide-and-combine variant (`OneShot`). |

`paraflr` (R) and `paraflr-py` (Python) provide the same estimator — install
whichever fits your environment; `disflr` targets the distributed setting.

The optional [`agent/`](agent) directory is a natural-language front end to
`paraflr`, running on either the R or the Python backend.

## `paraflr` (R package)

### Installation

```r
# install.packages("remotes")
remotes::install_github("UM-KevinHe/hpc-flr", subdir = "paraflr")
```

A C++ toolchain is required; OpenMP enables multi-threaded fitting (`threads > 1`)
and is used automatically when available.

### Usage

```r
library(paraflr)

# Y   : binary outcome vector
# Z   : covariate matrix (named columns)
# ID  : provider identifier for each record
fit <- logis_firth(Y, Z, ID, threads = 4)

fit$beta           # covariate effects
fit$gamma          # provider effects (named by provider ID)

# Test a provider effect against the population median
test_gamma.single(fit, methods = "score")
test_gamma.single(fit, methods = "lrt", firth = TRUE)
```

`logis_firth()` fits the model with provider-specific intercepts and no global
intercept; records are sorted by provider internally.

## `paraflr` (Python package)

For Python users — the same estimator as the R package (shared C++ core).

**Prebuilt wheel (recommended, no compiler needed).** Wheels for Windows, macOS
(Intel + Apple Silicon), and Linux are attached to each
[release](https://github.com/UM-KevinHe/hpc-flr/releases); download the one for
your OS + Python version and install it (BLAS and everything else is bundled):

```bash
pip install paraflr-0.2.0-cp311-cp311-macosx_11_0_arm64.whl   # example filename
```

Or let `pip` pick the right wheel for your OS + Python automatically:

```bash
pip install paraflr --find-links https://github.com/UM-KevinHe/hpc-flr/releases/expanded_assets/v0.2.0
```

**From source** (needs a C++ toolchain plus Armadillo — `brew install armadillo`
or `apt-get install libarmadillo-dev`):

```bash
pip install ./paraflr-py
```

```python
import paraflr

# Y : binary outcome (n,)   Z : covariates (n, p)   ID : provider id (n,)
fit = paraflr.logis_firth(Y, Z, ID, threads=4)

fit["beta"]        # covariate effects
fit["gamma"]       # provider effects

paraflr.test_gamma_single(fit, method="score")
paraflr.test_gamma_single(fit, method="lrt", firth=True)
```

## `disflr` (Scala / Spark)

`disflr/` is an sbt project (Spark 3.5.0 / Scala 2.12, matching Databricks
Runtime 14.3 LTS); the `MultiShot` and `OneShot` objects live in
`src/main/scala/`. Spark and Breeze are `provided` dependencies (supplied by the
cluster). Build a thin JAR and attach it to a Spark cluster, or paste the
objects into a notebook.

```bash
cd disflr && sbt package   # -> target/scala-2.12/disflr_2.12-0.1.0.jar
```

```scala
// Binomial outcome: successes out of trials, per record, grouped by provider
val res = MultiShot.run(spark, df,
  successCol = "successes", trialsCol = "trials",
  groupCol = "provider_id", numPartitions = 50)

res.beta    // covariate effects, keyed by feature name
res.gamma   // provider effects, keyed by provider id
```

`OneShot.run(...)` has the same signature. For Bernoulli data, supply a trials
column equal to `1`.

## Data

Runs on any clustered binary/binomial dataset: a binary outcome, a covariate
matrix, and a provider identifier per record.
