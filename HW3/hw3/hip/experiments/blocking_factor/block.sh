#!/bin/bash

# ==============================================================================
#  AMD Blocking Factor Analysis & Data Collection
#  - Uses rocprof-compute for deep analysis
#  - Parses output to CSV for Python plotting
# ==============================================================================

source /home/pp25/share/venv/bin/activate

# === BF = 8 ===
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute profile -n bf8_run -- ./hw3-2_bf8 ../../../testcases-amd/c21.1 temp.out
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute analyze -p workloads/bf8_run/MI100/

# === BF = 16 ===
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute profile -n bf16_run -- ./hw3-2_bf16 ../../../testcases-amd/c21.1 temp.out
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute analyze -p workloads/bf16_run/MI100/

# === BF = 32 ===
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute profile -n bf32_run -- ./hw3-2_bf32 ../../../testcases-amd/c21.1 temp.out
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute analyze -p workloads/bf32_run/MI100/

# === BF = 64 ===
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute profile -n bf64_run -- ./hw3-2_bf64 ../../../testcases-amd/c21.1 temp.out
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute analyze -p workloads/bf64_run/MI100/

