#!/bin/bash

# 1. Setup Environment
source /home/pp25/share/venv/bin/activate

# 2. Compile
make clean
make all

# 3. Create Metrics File
echo "pmc: FETCH_SIZE WRITE_SIZE SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE" > metrics.txt

# 4. Run AMD Experiments (Force Absolute Paths)
CWD=$(pwd)
METRICS="$CWD/metrics.txt"

echo "Running AMD Group 0..."
srun -p amd -N1 -n1 --gres=gpu:1 rocprof -d "$CWD" -i "$METRICS" -o rocprof_g0.csv "$CWD/hw3-2-group0-hip" ../../../testcases-amd/c21.1 /dev/null

echo "Running AMD Group 1..."
srun -p amd -N1 -n1 --gres=gpu:1 rocprof -d "$CWD" -i "$METRICS" -o rocprof_g1.csv "$CWD/hw3-2-group1-hip" ../../../testcases-amd/c21.1 /dev/null

# 5. Run NVIDIA Experiments
echo "Running NVIDIA Group 0..."
srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --csv --metrics gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput,achieved_occupancy --log-file nvprof_g0.csv ./hw3-2-group0-cuda ../../../testcases/c21.1 /dev/null

echo "Running NVIDIA Group 1..."
srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --csv --metrics gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput,achieved_occupancy --log-file nvprof_g1.csv ./hw3-2-group1-cuda ../../../testcases/c21.1 /dev/null

echo "Done!"
