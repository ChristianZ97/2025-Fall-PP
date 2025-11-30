#!/bin/bash

# --- Configuration ---
OUTPUT_CSV="bf_nvprof_results.csv"
TEMP_OUT_FILE="temp_bf_output.bin"

# --- Check for Testcase Directory Argument ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <testcases_directory>"
    echo "Example: $0 ../../testcases"
    exit 1
fi

TESTCASE_DIR=$1
if [ ! -d "$TESTCASE_DIR" ]; then
    echo "Error: Testcase directory not found at '$TESTCASE_DIR'"
    exit 1
fi

# --- Choose ONE testcase for BF experiment (avoid nvprof too slow) ---
# 建議用中等大小，例如 c21.1，如果不存在就 fallback 到 c04.1
if [ -f "$TESTCASE_DIR/c21.1" ]; then
    TESTCASE="$TESTCASE_DIR/c21.1"
elif [ -f "$TESTCASE_DIR/c15.1" ]; then
    TESTCASE="$TESTCASE_DIR/c15.1"
else
    echo "Error: neither c21.1 nor c15.1 found in $TESTCASE_DIR"
    exit 1
fi

TC_NAME=$(basename "$TESTCASE")

# 讀 V，用來算 GOPS
V=$(od -t d4 -N 4 "$TESTCASE" | head -n 1 | awk '{print $2}')
echo "Using testcase $TC_NAME, V=$V"

# --- Step 1: Compilation ---
echo "Step 1: Compiling all BF versions..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# --- Step 2: Initialize CSVs ---
echo "Step 2: Preparing results files..."
echo "Version,BF,Testcase,V,TotalTime_ms,ComputeTime_ms,GOPS,gld_GBs,gst_GBs,shld_GBs,shst_GBs,sm_eff_pct,achieved_occupancy" > "$OUTPUT_CSV"

# --- Step 3: Run Experiments ---
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw3-2_bf*" | sort))

echo "Step 3: Running nvprof experiments on $TC_NAME..."
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    # hw3-2_bf64 -> 64
    bf=$(echo "$exe_name" | grep -o "bf[0-9]*" | sed 's/bf//')

    echo "=========================================="
    echo "Testing Version: $exe_name (BF = $bf)"
    echo "=========================================="

    NVPROF_LOG="nvprof_${exe_name}.csv"
    RUN_LOG="run_${exe_name}.log"

    # 用 nvprof 跟你手動那條一樣，只是包起來
    SRUN_CMD="srun -p nvidia -N1 -n1 --gres=gpu:1 --time=00:05:00"

    $SRUN_CMD \
      nvprof --csv --log-file "$NVPROF_LOG" \
      --metrics gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput,sm_efficiency,achieved_occupancy \
      ./$exe_name "$TESTCASE" "$TEMP_OUT_FILE" 2>&1 | tee "$RUN_LOG"

    # --- Parse [PROF_RESULT] from your program ---
    prof_line=$(grep "\[PROF_RESULT\]" "$RUN_LOG")
    if [[ -z "$prof_line" ]]; then
        echo "  FAILED / NO PROF_RESULT"
        continue
    fi

    # [PROF_RESULT],Total,Compute,Comm,IO,Phase1,Phase2,Phase3
    total_ms=$(echo "$prof_line" | cut -d',' -f2)
    compute_ms=$(echo "$prof_line" | cut -d',' -f3)

    # Integer GOPS = 2 * V^3 / (compute_time_sec) / 1e9
    gops=$(python3 -c "V=$V; t_ms=$compute_ms; print( 2*(V**3) / (t_ms/1000.0) / 1e9 )")

    # --- Parse nvprof CSV for kernel_phase3 metrics ---
    phase3_lines=$(grep "kernel_phase3" "$NVPROF_LOG")

    gld=$(echo "$phase3_lines" | grep "gld_throughput" | awk -F',' '{print $NF}' | tr -d '"' | sed 's/GB\/s//')
    gst=$(echo "$phase3_lines" | grep "gst_throughput" | awk -F',' '{print $NF}' | tr -d '"' | sed 's/GB\/s//')
    shld=$(echo "$phase3_lines" | grep "shared_load_throughput" | awk -F',' '{print $NF}' | tr -d '"' | sed 's/GB\/s//')
    shst=$(echo "$phase3_lines" | grep "shared_store_throughput" | awk -F',' '{print $NF}' | tr -d '"' | sed 's/GB\/s//')
    sm_eff=$(echo "$phase3_lines" | grep "sm_efficiency" | awk -F',' '{print $NF}' | tr -d '"' | sed 's/%//')
    occ=$(echo "$phase3_lines" | grep "achieved_occupancy" | awk -F',' '{print $NF}' | tr -d '"')

    gld=${gld:-0}; gst=${gst:-0}; shld=${shld:-0}; shst=${shst:-0}
    sm_eff=${sm_eff:-0}; occ=${occ:-0}

    echo "  Total = ${total_ms} ms, Compute = ${compute_ms} ms, GOPS = ${gops}"
    echo "  gld = ${gld} GB/s, gst = ${gst} GB/s, shld = ${shld} GB/s, shst = ${shst} GB/s"
    echo "  sm_eff = ${sm_eff} %, occ = ${occ}"

    echo "$exe_name,$bf,$TC_NAME,$V,$total_ms,$compute_ms,$gops,$gld,$gst,$shld,$shst,$sm_eff,$occ" >> "$OUTPUT_CSV"

done

# Cleanup
rm -f "$TEMP_OUT_FILE"
echo "=========================================="
echo "All BF+nvprof experiments completed!"
echo "Details saved to: $OUTPUT_CSV"
