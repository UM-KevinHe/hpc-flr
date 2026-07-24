#!/bin/bash
# =============================================================================
# Run all four paraflr agent experiments against a running vLLM endpoint.
#
# On an HPC cluster, run this ON THE SAME COMPUTE NODE as vLLM: the E2-E4
# harnesses shell out to R (real Firth fits) and must not run on a login node,
# and on-node the endpoint is just localhost:8000 with no tunnel. The shell
# needs BOTH R on PATH and openai importable:
#   module load R/4.5.1
#   source "$SCRATCH/envs/vllm-env/bin/activate"
# (put your real scratch path in a private file, not in this repo.)
#
#   ./run_eval.sh                 # all four experiments
#   ./run_eval.sh routing         # just E1 (pure Python; the only one that can
#                                 #          also run off-node, e.g. over a tunnel)
#
# Prerequisites, ONE TIME on the cluster (see README.md "Running on a cluster"):
#   * ~/.Renviron points R_LIBS_USER at scratch, and `R CMD INSTALL paraflr` done
#   * the example datasets built       (Rscript agent/data/make_example_data.R)
#   * openai present in the venv        (it ships with vLLM)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export PARAFLR_MODEL_ENDPOINT="${PARAFLR_MODEL_ENDPOINT:-http://localhost:8000/v1}"
export PARAFLR_MODEL_NAME="${PARAFLR_MODEL_NAME:-qwen2.5-7b-awq}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "endpoint: $PARAFLR_MODEL_ENDPOINT   model: $PARAFLR_MODEL_NAME"

run_e1() { echo "=== E1 routing ===";  python run_routing.py; }
run_e2() { echo "=== E2 fidelity ===";  python run_fidelity.py; }
run_e3() { echo "=== E3 ablation ===";  python run_ablation.py; }
run_e4() { echo "=== E4 pass^k ===";    python run_passk.py --k 8 --temperature 0.7; }

case "${1:-all}" in
  routing)  run_e1 ;;
  fidelity) run_e2 ;;
  ablation) run_e3 ;;
  passk)    run_e4 ;;
  all)      run_e1; run_e2; run_e3; run_e4 ;;
  *) echo "usage: $0 [all|routing|fidelity|ablation|passk]" >&2; exit 2 ;;
esac

echo "=== done. results in $(pwd)/eval_out/ ==="
