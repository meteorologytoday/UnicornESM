#!/bin/bash
#
# DIAGNOSTIC RERUN of the batch-1076 blow-up (2026-09-20).
# Restarts from a COPY of checkpoint/batch_01074 in output_T31/real_topo_diag/,
# so production output_T31/real_topo/ is never touched. Runs batch 1075 (clean,
# gives a healthy baseline trace) then 1076 (expected to explode).
#
# Paired with temporary instrumentation in veros_case_setup.py:
#   settings.enable_nan_checks = True   + the DIAG after_timestep trace.
# Revert both once the question is answered.
#
#SBATCH --job-name=JEM-diag-1076
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --partition=gpu
#SBATCH --qos=gpumedium
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch.diag.%j.out
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --time=02:00:00

nvidia-smi

echo "pwd: $(pwd)"
eval "$( /home/tienyiao/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
conda activate jax

source /home/tienyiao/.bashrc_jcm_v2
export PYTHONPATH=/home/tienyiao/projects_local/EarthSystemGrids.py:$PYTHONPATH

export PYTHONHASHSEED=0
# Deliberately NOT setting --xla_gpu_deterministic_ops: too slow, and this run
# only needs to localise the first non-finite field, not reproduce bit-for-bit.
#export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

echo "PYTHONPATH=$PYTHONPATH"

JOB_ID=${SLURM_JOBID:-NO_JOB_ID}

grid_folder=part2_output_0p2

# --checkpoint-interval 100000 => no checkpoint is ever written, so the copied
# batch_01074 restart point cannot be pruned by main_forward.py's cleanup.
time python3 main_forward.py \
    --total-simulation-days $(( 365 * 1000 ))       \
    --simulation-interval-days 73                   \
    --checkpoint-interval 100000                    \
    --truncation-number 31                          \
    --simulation-name real_topo_diag                \
    --jcm-timestep-min 30                           \
    --veros-timestep-min 60                         \
    --max-rerun-attempts 0                          \
    --grid-folder $grid_folder                      \
    --ocean-surface-thickness 30                    \
    --explode-log explode.diag.${JOB_ID}.log
