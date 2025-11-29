#!/bin/bash

# --- Configuration ---
OUTPUT_CSV="unroll_perf_results.csv"
SUMMARY_CSV="unroll_perf_summary.csv"
TEMP_OUT_FILE="temp_unroll_output.bin"

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

# --- Find Testcases (Target: p11k1 ~ p40k1) ---
# 我們使用 regex 來匹配 p11k1 到 p40k1
# Pattern: p[1-3][0-9]k1 (p10k1~p39k1) OR p40k1
# 或者更簡單，直接列出所有 pk1 檔案然後用 sort 過濾
echo "Searching for p11k1 to p40k1 in $TESTCASE_DIR ..."

TESTCASES=()
# Loop 11 to 40
for i in {11..40}; do
    target_file="$TESTCASE_DIR/p${i}k1"
    if [ -f "$target_file" ]; then
        TESTCASES+=("$target_file")
    else
        echo "Warning: $target_file not found, skipping."
    fi
done

NUM_TESTCASES=${#TESTCASES[@]}
if [ $NUM_TESTCASES -eq 0 ]; then
    echo "Error: No target testcases found in '$TESTCASE_DIR'"
    exit 1
fi

echo "Found $NUM_TESTCASES testcases to run (p11k1 ~ p40k1)."

# --- Step 1: Compilation ---
echo "Step 1: Compiling all unroll versions..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# --- Step 2: Initialize CSVs ---
echo "Step 2: Preparing results files..."
echo "Version,UnrollFactor,Testcase,TotalTime_ms,ComputeTime_ms" > "$OUTPUT_CSV"
echo "Version,UnrollFactor,GrandTotalTime_ms,TotalComputeTime_ms" > "$SUMMARY_CSV"

# --- Step 3: Run Experiments ---
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw3-2_unroll_*" | sort))

echo "Step 3: Running experiments..."
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    unroll_factor=$(echo "$exe_name" | sed 's/hw3-2_unroll_//')
    
    echo "=========================================="
    echo "Testing Version: $exe_name (Unroll Factor: $unroll_factor)"
    echo "=========================================="
    
    grand_total_time=0
    grand_compute_time=0
    
    for testcase in "${TESTCASES[@]}"; do
        tc_name=$(basename "$testcase")
        
        # Time limit set to 2 minutes just in case
        SRUN_CMD="srun -p nvidia -N1 -n1 --gres=gpu:1 --time=00:02:00"
        
        echo -n "  Running $tc_name ... "
        
        app_output=$($SRUN_CMD ./$exe_name "$testcase" "$TEMP_OUT_FILE" 2>&1)
        
        prof_line=$(echo "$app_output" | grep "\[PROF_RESULT\]")
        
        if [[ -z "$prof_line" ]]; then
            echo "FAILED / TIMEOUT"
            total_time=0; compute_time=0
        else
            total_time=$(echo "$prof_line" | cut -d',' -f2)
            compute_time=$(echo "$prof_line" | cut -d',' -f3)
            
            echo "${total_time} ms"
            
            grand_total_time=$(echo "$grand_total_time + $total_time" | bc)
            grand_compute_time=$(echo "$grand_compute_time + $compute_time" | bc)
            
            echo "$exe_name,$unroll_factor,$tc_name,$total_time,$compute_time" >> "$OUTPUT_CSV"
        fi
    done
    
    echo "------------------------------------------"
    echo "  Grand Total Time: $grand_total_time ms"
    echo "------------------------------------------"
    echo ""
    
    echo "$exe_name,$unroll_factor,$grand_total_time,$grand_compute_time" >> "$SUMMARY_CSV"

done

# Cleanup
rm -f "$TEMP_OUT_FILE"
echo "=========================================="
echo "All experiments completed!"
echo "Details saved to: $OUTPUT_CSV"
echo "Summary saved to: $SUMMARY_CSV"
