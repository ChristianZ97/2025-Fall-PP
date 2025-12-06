#!/bin/bash

# --- Configuration ---
OUTPUT_CSV="block_perf_results.csv"
SUMMARY_CSV="block_perf_summary.csv"
TEMP_OUT_FILE="temp_block_output.bin"

# --- Check for Testcase Directory Argument ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <testcases_directory>"
    echo "Example: $0 ../testcases-amd"
    exit 1
fi

TESTCASE_DIR=$1
if [ ! -d "$TESTCASE_DIR" ]; then
    echo "Error: Testcase directory not found at '$TESTCASE_DIR'"
    exit 1
fi

# --- Find Testcases (Target: t01 ~ t30) ---
echo "Searching for t01 to t30 in $TESTCASE_DIR ..."

TESTCASES=()
# Loop 01 to 30
for i in {01..30}; do
    # 使用 printf 確保數字是兩位數 (例如 1 -> 01)
    # 如果你的 shell 支援 {01..30}，變數 i 本身就會帶有前導零，
    # 但為了保險起見，也可以直接寫 t$i 如果 bash 版本夠新。
    # 這裡假設你的環境 bash 展開 {01..30} 會自動補零 (標準 bash 行為)。
    
    target_file="$TESTCASE_DIR/t${i}"
    
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

echo "Found $NUM_TESTCASES testcases to run (t01 ~ t30)."

# --- Step 1: Compilation ---
echo "Step 1: Compiling all block size versions..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# --- Step 2: Initialize CSVs ---
echo "Step 2: Preparing results files..."
# Updated header to match: total, io, commu, comp
echo "Version,BR,BC,Testcase,TotalTime_ms,IOTime_ms,CommuTime_ms,ComputeTime_ms" > "$OUTPUT_CSV"
# Summary keeps total and compute sum
echo "Version,BR,BC,GrandTotalTime_ms,GrandComputeTime_ms" > "$SUMMARY_CSV"

# --- Step 3: Run Experiments ---
# 搜尋並過濾出符合 hw4_XX_XX 格式的執行檔
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw4_*" | sort))

echo "Step 3: Running experiments..."
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    
    # Parse BR and BC from filename (e.g., hw4_64_64)
    br_val=$(echo "$exe_name" | cut -d'_' -f2)
    bc_val=$(echo "$exe_name" | cut -d'_' -f3)
    
    echo "=========================================="
    echo "Testing Version: $exe_name (BR: $br_val, BC: $bc_val)"
    echo "=========================================="
    
    grand_total_time=0
    grand_compute_time=0
    
    for testcase in "${TESTCASES[@]}"; do
        tc_name=$(basename "$testcase")
        
        # Time limit set to 2 minutes
        SRUN_CMD="srun -p amd -N1 -n1 --gres=gpu:1 --time=00:05:00"
        
        echo -n "  Running $tc_name ... "
        
        # Run executable, redirect stderr to stdout to capture PROF_RESULT
        app_output=$($SRUN_CMD ./$exe_name "$testcase" "$TEMP_OUT_FILE" 2>&1)
        
        # Format: [PROF_RESULT],total,io,commu,comp
        prof_line=$(echo "$app_output" | grep "\[PROF_RESULT\]")
        
        if [[ -z "$prof_line" ]]; then
            echo "FAILED / TIMEOUT"
            # Log zero or specific error code
            echo "$exe_name,$br_val,$bc_val,$tc_name,0,0,0,0" >> "$OUTPUT_CSV"
        else
            # Extract fields
            total_time=$(echo "$prof_line" | cut -d',' -f2)
            io_time=$(echo "$prof_line" | cut -d',' -f3)
            commu_time=$(echo "$prof_line" | cut -d',' -f4)
            comp_time=$(echo "$prof_line" | cut -d',' -f5)
            
            echo "${total_time} ms (Comp: ${comp_time} ms)"
            
            # Use bc for floating point arithmetic summation
            grand_total_time=$(echo "$grand_total_time + $total_time" | bc)
            grand_compute_time=$(echo "$grand_compute_time + $comp_time" | bc)
            
            echo "$exe_name,$br_val,$bc_val,$tc_name,$total_time,$io_time,$commu_time,$comp_time" >> "$OUTPUT_CSV"
        fi
    done
    
    echo "------------------------------------------"
    echo "  Grand Total Time:   $grand_total_time ms"
    echo "  Grand Compute Time: $grand_compute_time ms"
    echo "------------------------------------------"
    echo ""
    
    echo "$exe_name,$br_val,$bc_val,$grand_total_time,$grand_compute_time" >> "$SUMMARY_CSV"

done

# Cleanup
rm -f "$TEMP_OUT_FILE"
echo "=========================================="
echo "All experiments completed!"
echo "Details saved to: $OUTPUT_CSV"
echo "Summary saved to: $SUMMARY_CSV"
