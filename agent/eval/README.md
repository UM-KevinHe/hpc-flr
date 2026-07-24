# Agent evaluation

Four experiments, all driving the same frozen `paraflr_agent` over two
representative analyses. They answer four separate questions, and none of
them asks a language model to grade another language model — every score is
either a tool name, an exact numeric comparison, or a p-value.

## The system under evaluation

A **verification-scaffolded** agent: an open-weights model served locally
(Qwen 2.5-7B-AWQ via vLLM) selects and configures an estimator, and the
`paraflr` package computes every reported value. The model routes and
explains; it computes nothing. Decoding is deterministic (`temperature = 0`),
so rerunning any experiment below reproduces it exactly.

## The two representatives

| id | data | correct tool | why this one |
|---|---|---|---|
| `fit_moderate` | `ExampleProviders.rda` — 50 providers, 20% event rate | `fit_flr` | the ordinary estimation request |
| `test_rare_provider` | `ExampleProviders_rare.rda` — 40 small providers, 4% event rate, 14 with no events | `test_provider` | the case the package exists for, and the one where picking the wrong test silently changes the answer |

## Experiments

| Script | Question | Output |
|---|---|---|
| `run_routing.py` | **E1** Given a plain-language request and no data file, does the model call the right tool — and with the right arguments? 40 queries, one model call each, no R execution. | `eval_out/routing.json` |
| `run_fidelity.py` | **E2** Are the agent's numbers identical to a direct `paraflr` call on the same data? Also writes the trace and a standalone `repro.R` per representative. | `eval_out/fidelity.json`, `eval_out/fidelity_*_{trace.json,repro.R}` |
| `run_ablation.py` | **E3** Does the scaffolding do any work, or would a 7B model with six tools have got there anyway? Layers switched off one at a time, model and data fixed. | `eval_out/ablation.json` |
| `run_passk.py` | **E4** How reliable is routing once decoding is not deterministic? k runs per representative at `temperature > 0`. | `eval_out/passk.json` |

E1 measures **argument accuracy** separately from tool accuracy, over the
queries that name a statistic, a null, a penalty, or a size. Routing to
`test_provider` and then running a Wald test when the user asked for a score
test is a wrong answer that tool accuracy alone scores as a hit — and on
small providers the two tests genuinely disagree, so the distinction is not
bookkeeping.

E3's outcome taxonomy is the one that matters most:

- `correct` — the right tool ran cleanly
- `silently_wrong` — a plausible but wrong tool ran cleanly and produced
  numbers nobody flagged
- `crashed` — a tool ran and errored, or the loop failed
- `refused` — no tool was called at all

`silently_wrong` is the failure mode a scaffolded harness exists to prevent,
and the reason correctness is scored from the trace rather than from prose.

## Running them

```bash
export PARAFLR_MODEL_ENDPOINT=http://localhost:8000/v1
export PARAFLR_MODEL_NAME=qwen2.5-7b-awq
export OPENAI_API_KEY=EMPTY          # vLLM ignores the value

cd agent/eval
python run_routing.py                # ~1 min, no R
python run_fidelity.py               # writes traces + repro scripts
python run_ablation.py               # 12 runs
python run_passk.py --k 8 --temperature 0.7
```

Everything except `run_routing.py` needs R with `paraflr` installed and the
bundled datasets built (`Rscript agent/data/make_example_data.R`).

Results land in `eval_out/`. Each `*_trace.json` is an audit trail: every
tool call, the arguments the model emitted, the arguments actually
dispatched after the harness resolved them, the status, and a result summary.
Each `*_repro.R` replays the run against `paraflr` directly, with no model in
the loop — so the fidelity claim can be checked by running it, not by reading
JSON.

## Running on a cluster (Slurm + vLLM)

The paper's runs were on an A40 GPU partition (`spgpu`) of a Slurm cluster.
Everything runs **on the compute node**: the E2–E4 harnesses shell out to R
for real Firth fits, which must not run on a login node, and on-node the
endpoint is plain `localhost` — no SSH tunnel.

Placeholders below — fill them from your own site (keep the real values in a
private file, not in this repo):

| placeholder | meaning |
|---|---|
| `<login-host>` | cluster login host, e.g. `you@cluster.example.edu` |
| `<account>` | your Slurm allocation |
| `$SCRATCH` | your scratch dir holding `envs/vllm-env`, `models/qwen2.5-7b-awq`, and the `hpc-flr` checkout |

Two scripts are provided:

- [`start_vllm.sbatch`](start_vllm.sbatch) — serve Qwen as a batch job.
- [`run_eval.sh`](run_eval.sh) — set `PARAFLR_*` and run the four experiments.

### 0. Log in and copy the repo up

```bash
ssh <login-host>          # 2FA per connection at many sites
export SCRATCH=/path/to/your/scratch          # set to your real scratch dir
```

Get the code onto scratch one of two ways:

```bash
# (a) rsync the working tree from your machine (run LOCALLY):
rsync -avz --exclude='.git' --exclude='.Rproj.user' --exclude='__pycache__' \
      --exclude='*.o' --exclude='*.so' --exclude='eval_out' --exclude='*.local.md' \
      /path/to/local/hpc-flr/  <login-host>:$SCRATCH/hpc-flr/

# (b) OR, if agent/ is pushed to GitHub, on the cluster:
cd "$SCRATCH" && git clone https://github.com/UM-KevinHe/hpc-flr.git   # or `git pull`
```

### 1. One-time setup on the cluster

`paraflr` is a C++/OpenMP package and must be **compiled on a node**, not
copied from a laptop. Install it into a scratch-backed personal R library (the
system R lib is read-only):

```bash
# once, ever — makes R_LIBS_USER stick for every future R/Rscript session:
mkdir -p "$SCRATCH/Rlibs"
echo "R_LIBS_USER=$SCRATCH/Rlibs" >> ~/.Renviron

module load R/4.5.1        # match the cluster's current R (see note below)
cd "$SCRATCH/hpc-flr"
R CMD INSTALL paraflr
Rscript agent/data/make_example_data.R
```

`openai` ships with vLLM, so if you use the same venv there is no pip step.

### 2. Serve the model and run the eval (all on the compute node)

```bash
# grab an A40 (spgpu; the plain `gpu` partition is V100 — too small):
salloc --account=<account> --partition=spgpu --gres=gpu:a40:1 \
       --mem=64G --cpus-per-task=8 --time=4:00:00

# these modules do NOT persist across salloc — reload every session:
module load python/3.11.5 cuda/12.8.2 R/4.5.1
source "$SCRATCH/envs/vllm-env/bin/activate"

# start vLLM in the background (don't paste a multi-line block containing
# `tmux new` — the paste gets truncated and vLLM never starts):
vllm serve "$SCRATCH/models/qwen2.5-7b-awq" \
    --port 8000 --served-model-name qwen2.5-7b-awq \
    --tool-call-parser hermes --enable-auto-tool-choice \
    --max-model-len 32768 --gpu-memory-utilization 0.5 --enforce-eager \
    > ~/vllm.log 2>&1 &
#   (or run it in `tmux new -s vllm`, then Ctrl-b d)

# readiness is NOT a non-empty log (it block-buffers). It is this returning JSON:
curl -s http://localhost:8000/v1/models
#   and `nvidia-smi --query-gpu=memory.used --format=csv,noheader` climbing.
#   Model load takes 3–5 min. If it HANGS at CUDA-graph with the GPU idle:
#   `pkill -9 -f vllm; sleep 3` and rerun (the --enforce-eager above prevents it).

cd "$SCRATCH/hpc-flr/agent/eval"
./run_eval.sh                                            # all four experiments

scancel <jobid>                                          # when done — GPU bills hourly
```

`run_routing.py` (E1) is the one experiment that runs no R; if you only want
E1, it can also be driven from a laptop over `ssh -N -L 8000:<node>:8000
<login-host>` with `./run_eval.sh routing`. E2–E4 must stay on the compute node.

### If `salloc` is unavailable, use the batch script

`sbatch start_vllm.sbatch` serves the model as a job; then `srun --jobid=<id>
--pty bash` onto the node, load `R/4.5.1`, activate the venv, and
`./run_eval.sh` there. Same on-node rule.

> Cluster-specific R version: match whatever your site's shared R libraries
> were built under (on the paper's cluster that was **R/4.5.1**, and R/4.4
> failed to load them). Check with `module avail R`.
