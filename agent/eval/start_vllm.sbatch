#!/bin/bash
# =============================================================================
# Serve Qwen-2.5-7B-AWQ on a GPU node of a Slurm cluster (an A40 partition) for
# the paraflr agent.
#
# FILL IN the placeholders below before submitting:
#   --account   your Slurm allocation
#   --mail-user your email (or delete the two --mail-* lines)
#   SCRATCH     your scratch dir holding the venv + model (see the notes at the
#               top of run_eval.sh; keep the real path in a private file, not
#               in this repo)
#
#   sbatch start_vllm.sbatch
#   squeue -u $USER                      # note the NODELIST
#   tail -f vllm_<jobid>.log             # DON'T trust an empty log — it
#                                        # block-buffers; readiness is really
#                                        # `curl -s http://localhost:8000/v1/models`
#                                        # returning JSON (see run_eval / README).
#
# Then run the eval ON THE SAME COMPUTE NODE (the E2-E4 harnesses shell out to
# R and must NOT run on a login node):
#   srun --jobid=<this jobid> --pty bash     # hop onto the node
#   module load R/4.5.1
#   source "$SCRATCH/envs/vllm-env/bin/activate"
#   cd "$SCRATCH/hpc-flr/agent/eval" && ./run_eval.sh
#
# The endpoint is just localhost:8000 on-node — no SSH tunnel. `scancel` the
# job when done; the GPU bills by the hour.
# =============================================================================
#SBATCH --job-name=vllm_paraflr
#SBATCH --account=<slurm-account>
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=<you>@example.edu
#SBATCH --output=vllm_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Your scratch dir holding envs/vllm-env and models/qwen2.5-7b-awq. Override
# with the real value here or via `export PARAFLR_SCRATCH=...` before sbatch.
SCRATCH="${PARAFLR_SCRATCH:-/path/to/your/scratch}"
MODEL="$SCRATCH/models/qwen2.5-7b-awq"

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
echo "=== run the eval on this node:  srun --jobid=$SLURM_JOB_ID --pty bash ==="

module load python/3.11.5 cuda/12.8.2
source "$SCRATCH/envs/vllm-env/bin/activate"

# --enforce-eager skips CUDA-graph capture (a small throughput cost) and avoids
# a startup hang seen without it.
vllm serve "$MODEL" \
  --served-model-name qwen2.5-7b-awq \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.5 \
  --enforce-eager \
  --host 0.0.0.0 --port 8000
