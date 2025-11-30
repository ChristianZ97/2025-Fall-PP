#!/bin/bash

# --- Configuration ---
OUTPUT_CSV="matrix_perf_results.csv"
SUMMARY_CSV="matrix_perf_summary.csv"
TEMP_OUT_FILE="temp_matrix_output.bin"

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
echo "Searching for p11k1 to p40k1 in $TESTCASE_DIR ..."

TESTCASES=()
# Loop 11 to 40
for i in {11..40}; do
    target_file="$TESTCASE_DIR/p${i}k1"
    if [ -f "$target_file" ]; then
        TESTCASES+=("$target_file")
    else
        true
    fi
done

NUM_TESTCASES=${#TESTCASES[@]}
if [ $NUM_TESTCASES -eq 0 ]; then
    echo "Error: No target testcases found in '$TESTCASE_DIR'"
    exit 1
fi

echo "Found $NUM_TESTCASES testcases to run (p11k1 ~ p40k1)."

# --- Step 1: Compilation ---
echo "Step 1: Compiling all matrix versions..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# --- Step 2: Initialize CSVs ---
echo "Step 2: Preparing results files..."
# Header 包含四個維度
echo "Version,MemoryType,BlockFactor,Coarsening,Workload,Testcase,TotalTime_ms,ComputeTime_ms" > "$OUTPUT_CSV"
echo "Version,MemoryType,BlockFactor,Coarsening,Workload,GrandTotalTime_ms,TotalComputeTime_ms" > "$SUMMARY_CSV"

# --- Step 3: Run Experiments ---
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw3-2_*" ! -name "*.sh" ! -name "*.cu" | sort))

if [ ${#EXECUTABLES[@]} -eq 0 ]; then
    echo "Error: No executables found matching hw3-2_*"
    exit 1
fi

echo "Step 3: Running experiments..."
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    
    # --- Extract Parameters from Filename ---
    # Format: hw3-2_[global|sm]_bf[XX]_[coarsen|noncoarsen]_[XXXX]
    
    # 1. Memory Type
    if [[ "$exe_name" == *"_global_"* ]]; then
        mem_type="Global"
    elif [[ "$exe_name" == *"_sm_"* ]]; then
        mem_type="SharedMem"
    else
        mem_type="Unknown"
    fi
    
    # 2. Block Factor
    bf_val=$(echo "$exe_name" | grep -o "bf[0-9]*" | sed 's/bf//')
    
    # 3. Coarsening
    if [[ "$exe_name" == *"_noncoarsen"* ]]; then
        coarsen="No"
    elif [[ "$exe_name" == *"_coarsen"* ]]; then
        coarsen="Yes"
    else
        coarsen="Unknown"
    fi
    
    # 4. Workload
    workload=$(echo "$exe_name" | grep -o "[0-9]*$" )
    
    echo "=========================================="
    echo "Testing: $exe_name"
    echo "  -> Mem: $mem_type, BF: $bf_val, Coarsen: $coarsen, Threads: $workload"
    echo "=========================================="
    
    grand_total_time=0
    grand_compute_time=0
    
    for testcase in "${TESTCASES[@]}"; do
        tc_name=$(basename "$testcase")
        
        # Time limit set to 2 minutes
        SRUN_CMD="srun -p nvidia -N1 -n1 --gres=gpu:1 --time=00:02:00"
        
        echo -n "  Running $tc_name ... "
        
        app_output=$($SRUN_CMD ./$exe_name "$testcase" "$TEMP_OUT_FILE" 2>&1)
        
        prof_line=$(echo "$app_output" | grep "\[PROF_RESULT\]")
        
        if [[ -z "$prof_line" ]]; then
            echo "FAILED / TIMEOUT"
            # ★ 失敗時記錄為 -1，方便畫圖時過濾
            total_time="-1"
            compute_time="-1"
        else
            # [PROF_RESULT],TotalTime,ComputeTime,...
            total_time=$(echo "$prof_line" | cut -d',' -f2)
            compute_time=$(echo "$prof_line" | cut -d',' -f3)
            
            echo "${total_time} ms"
            
            # ★ 只有成功時才加入 Grand Total，避免 bc 報錯
            grand_total_time=$(echo "$grand_total_time + $total_time" | bc)
            grand_compute_time=$(echo "$grand_compute_time + $compute_time" | bc)
        fi
        
        # ★ 無論成功失敗，都寫入 CSV
        echo "$exe_name,$mem_type,$bf_val,$coarsen,$workload,$tc_name,$total_time,$compute_time" >> "$OUTPUT_CSV"
    done
    
    echo "------------------------------------------"
    echo "  Grand Total Time (Successful runs only): $grand_total_time ms"
    echo "------------------------------------------"
    echo ""
    
    # Append summary result
    echo "$exe_name,$mem_type,$bf_val,$coarsen,$workload,$grand_total_time,$grand_compute_time" >> "$SUMMARY_CSV"

done

# Cleanup
rm -f "$TEMP_OUT_FILE"
echo "=========================================="
echo "All experiments completed!"
echo "Details saved to: $OUTPUT_CSV"
echo "Summary saved to: $SUMMARY_CSV"
