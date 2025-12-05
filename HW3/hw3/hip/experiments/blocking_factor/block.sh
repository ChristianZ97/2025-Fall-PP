#!/bin/bash

# ==============================================================================
#  Advanced Profiling Script for Blocked Floyd-Warshall (HIP/ROCm Version)
#  Updated: Removed rocprof stage, purely measures performance (GOPS)
# ==============================================================================

# --- Configuration ---
OUTPUT_CSV="bf_perf_results.csv"
TEMP_OUT_FILE="temp_bf_output.bin"
MAX_RETRIES=10  # 設定每個測試最多重試幾次

# --- Check for Testcase Directory Argument ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_testcase_directory>"
    echo "Example: $0 ../../../testcases-amd"
    exit 1
fi

TESTCASE_DIR=$1
if [ ! -d "$TESTCASE_DIR" ]; then
    echo "Error: Testcase directory not found at '$TESTCASE_DIR'"
    exit 1
fi

# --- Select a single, representative testcase for this experiment ---
TESTCASE_PATH="$TESTCASE_DIR/c21.1"
if [ ! -f "$TESTCASE_PATH" ]; then
    echo "Error: Testcase 'c21.1' not found in '$TESTCASE_DIR'."
    echo "Please use a directory containing the required testcases."
    exit 1
fi

TC_NAME=$(basename "$TESTCASE_PATH")
V=$(od -t d4 -N 4 "$TESTCASE_PATH" | head -n 1 | awk '{print $2}')
echo "Using testcase: $TC_NAME (V = $V)"

# --- Step 1: Compilation ---
echo "Step 1: Compiling all blocking factor versions (HIP)..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# --- Step 2: Initialize CSV Output File ---
echo "Step 2: Preparing results file..."
# Keeping headers compatible with previous scripts, but rocprof fields will be 0/NA
echo "Version,BF,Testcase,V,TotalTime_ms,ComputeTime_ms,GOPS,rocprof_KernelDuration_ns,gld_GBs_NA,gst_GBs_NA,shld_GBs_NA,shst_GBs_NA" > "$OUTPUT_CSV"

# --- Step 3: Run Experiments for Each Blocking Factor ---
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw3-2_bf*" | sort -V))
SRUN_CMD="srun -p amd -N1 -n1 --gres=gpu:1 --time=00:05:00" 

echo "Step 3: Starting experiment runs (Max Retries per BF: $MAX_RETRIES)..."
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    bf=$(echo "$exe_name" | grep -o 'bf[0-9]*' | sed 's/bf//')

    echo "=================================================="
    echo "Processing: $exe_name (Blocking Factor = $bf)"
    echo "=================================================="

    # Variables to store results across retries
    success=false
    total_ms=0
    compute_ms=0
    gops=0
    
    # --- RETRY LOOP ---
    for (( attempt=1; attempt<=MAX_RETRIES; attempt++ )); do
        if [ $attempt -gt 1 ]; then
            echo "   [WARNING] Attempt $attempt/$MAX_RETRIES..."
            sleep 2
        fi

        # --- PERFORMANCE RUN ---
        echo "  -> Running performance measurement (Attempt $attempt)..."
        PERF_LOG="perf_${exe_name}.log"
        
        $SRUN_CMD ./$exe_name "$TESTCASE_PATH" "$TEMP_OUT_FILE" > "$PERF_LOG" 2>&1

        perf_line=$(grep "\[PROF_RESULT\]" "$PERF_LOG")
        if [[ -z "$perf_line" ]]; then
            echo "     [FAIL] No [PROF_RESULT] found. Retrying..."
            continue # Try next attempt
        fi
        
        # Extract data if successful
        total_ms=$(echo "$perf_line" | cut -d',' -f2)
        compute_ms=$(echo "$perf_line" | cut -d',' -f3)
        gops=$(python3 -c "V=$V; t_ms=$compute_ms; print(f'{2 * (V**3) / (t_ms / 1000.0) / 1e9:.4f}')")
        echo "     > Compute Time: ${compute_ms} ms | GOPS: ${gops}"

        # If we got here, the run succeeded
        success=true
        break
    done

    # --- Write Results or Log Failure ---
    if [ "$success" = true ]; then
        # Fill dummy values for removed rocprof metrics
        kernel_duration="0"
        gld="0"; gst="0"; shld="0"; shst="0"
        
        echo "$exe_name,$bf,$TC_NAME,$V,$total_ms,$compute_ms,$gops,$kernel_duration,$gld,$gst,$shld,$shst" >> "$OUTPUT_CSV"
        echo "  [SUCCESS] Data recorded."
    else
        echo "  [ERROR] Failed to run $exe_name after $MAX_RETRIES attempts. Skipping."
        # Optionally write a failure line to CSV or just skip
    fi
    echo ""
done

# --- Cleanup ---
rm -f "$TEMP_OUT_FILE"
rm -f perf_*.log
echo "=================================================="
echo "All experiments completed!"
echo "Final results saved to: $OUTPUT_CSV"
echo "=================================================="
