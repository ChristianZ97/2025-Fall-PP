#!/bin/bash

# ==============================================================================
#  Advanced Profiling Script for Blocked Floyd-Warshall
#  - Separates performance run (for GOPS) and profiling run (for metrics)
#    to get accurate performance data.
# ==============================================================================

# --- Configuration ---
OUTPUT_CSV="bf_perf_results_final.csv"
TEMP_OUT_FILE="temp_bf_output.bin"

# --- Check for Testcase Directory Argument ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_testcase_directory>"
    echo "Example: $0 ../../testcases"
    exit 1
fi

TESTCASE_DIR=$1
if [ ! -d "$TESTCASE_DIR" ]; then
    echo "Error: Testcase directory not found at '$TESTCASE_DIR'"
    exit 1
fi

# --- Select a single, representative testcase for this experiment ---
# Using c21.1 as it's large enough but not too slow for profiling.
TESTCASE_PATH="$TESTCASE_DIR/c21.1"
if [ ! -f "$TESTCASE_PATH" ]; then
    echo "Error: Testcase 'c21.1' not found in '$TESTCASE_DIR'."
    echo "Please use a directory containing the required testcases."
    exit 1
fi

TC_NAME=$(basename "$TESTCASE_PATH")
# Read vertex count 'V' from the binary testcase file
V=$(od -t d4 -N 4 "$TESTCASE_PATH" | head -n 1 | awk '{print $2}')
echo "Using testcase: $TC_NAME (V = $V)"

# --- Step 1: Compilation ---
echo "Step 1: Compiling all blocking factor versions..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# --- Step 2: Initialize CSV Output File ---
echo "Step 2: Preparing results file..."
echo "Version,BF,Testcase,V,TotalTime_ms,ComputeTime_ms,GOPS,gld_GBs,gst_GBs,shld_GBs,shst_GBs" > "$OUTPUT_CSV"

# --- Step 3: Run Experiments for Each Blocking Factor ---
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw3-2_bf*" | sort -V))
SRUN_CMD="srun -p nvidia -N1 -n1 --gres=gpu:1 --time=00:05:00" # Increased time for safety

echo "Step 3: Starting experiment runs..."
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    # Extract blocking factor from filename (e.g., hw3-2_bf64 -> 64)
    bf=$(echo "$exe_name" | grep -o 'bf[0-9]*' | sed 's/bf//')

    echo "=================================================="
    echo "Processing: $exe_name (Blocking Factor = $bf)"
    echo "=================================================="

    # --- STAGE 1: Performance Run (for accurate GOPS) ---
    echo "  -> Stage 1: Running for pure performance measurement..."
    PERF_LOG="perf_${exe_name}.log"
    # Execute without nvprof to get real timing
    $SRUN_CMD ./$exe_name "$TESTCASE_PATH" "$TEMP_OUT_FILE" > "$PERF_LOG" 2>&1

    # Parse the [PROF_RESULT] line from the standard run
    perf_line=$(grep "\[PROF_RESULT\]" "$PERF_LOG")
    if [[ -z "$perf_line" ]]; then
        echo "  [ERROR] No [PROF_RESULT] found in performance run for $exe_name. Skipping."
        continue
    fi
    
    total_ms=$(echo "$perf_line" | cut -d',' -f2)
    compute_ms=$(echo "$perf_line" | cut -d',' -f3)
    
    # Calculate GOPS using the accurate timing
    gops=$(python3 -c "V=$V; t_ms=$compute_ms; print(f'{2 * (V**3) / (t_ms / 1000.0) / 1e9:.4f}')")
    echo "     Real Compute Time: ${compute_ms} ms"
    echo "     Calculated GOPS: ${gops}"

    # --- STAGE 2: Profiling Run (for detailed metrics) ---
    echo "  -> Stage 2: Running with nvprof for detailed metrics..."
    NVPROF_LOG="nvprof_${exe_name}.csv"
    
    # Execute WITH nvprof to gather metrics. We discard the timing from this run.
    $SRUN_CMD nvprof --csv --log-file "$NVPROF_LOG" \
        --metrics gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput \
        ./$exe_name "$TESTCASE_PATH" "$TEMP_OUT_FILE" > /dev/null 2>&1
    
    # Parse nvprof CSV for Phase 3 metrics (assuming Phase 3 is the most representative)
    phase3_lines=$(grep "kernel_phase3" "$NVPROF_LOG")
    
    # Safely parse metrics, providing 0 as a default if not found
    gld=$(echo "$phase3_lines" | grep "gld_throughput" | awk -F',' '{print $(NF)}' | tr -d '"' | sed 's/GB\/s//' | head -n 1)
    gst=$(echo "$phase3_lines" | grep "gst_throughput" | awk -F',' '{print $(NF)}' | tr -d '"' | sed 's/GB\/s//' | head -n 1)
    shld=$(echo "$phase3_lines" | grep "shared_load_throughput" | awk -F',' '{print $(NF)}' | tr -d '"' | sed 's/GB\/s//' | head -n 1)
    shst=$(echo "$phase3_lines" | grep "shared_store_throughput" | awk -F',' '{print $(NF)}' | tr -d '"' | sed 's/GB\/s//' | head -n 1)

    gld=${gld:-0}; gst=${gst:-0}; shld=${shld:-0}; shst=${shst:-0}
    echo "     Shared Mem Load Throughput: ${shld} GB/s"

    # --- Combine and Save Results ---
    echo "$exe_name,$bf,$TC_NAME,$V,$total_ms,$compute_ms,$gops,$gld,$gst,$shld,$shst" >> "$OUTPUT_CSV"
    echo ""
done

# --- Cleanup ---
rm -f "$TEMP_OUT_FILE"
rm -f perf_*.log nvprof_*.csv
echo "=================================================="
echo "All experiments completed!"
echo "Final results saved to: $OUTPUT_CSV"
echo "=================================================="
