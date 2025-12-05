#!/bin/bash

# ==============================================================================
#  Ablation Study Script (HIP/ROCm Version)
#  - Runs multiple versions of the executable against a range of testcases
#  - Includes RETRY mechanism for stability
# ==============================================================================

# --- Configuration ---
OUTPUT_CSV="ablation_perf_results.csv"
SUMMARY_CSV="ablation_perf_summary.csv"
TEMP_OUT_FILE="temp_ablation_output.bin"
MAX_RETRIES=10  # 每個 Testcase 最多重試 3 次

# --- Check for Testcase Directory Argument ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <testcases_directory>"
    echo "Example: $0 ../../../testcases-amd"
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
for i in {11..25}; do
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
echo "Step 1: Compiling all ablation versions (HIP)..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# --- Step 2: Initialize CSVs ---
echo "Step 2: Preparing results files..."
echo "Version,AblationTarget,Testcase,TotalTime_ms,ComputeTime_ms,CommTime_ms,IOTime_ms" > "$OUTPUT_CSV"
echo "Version,AblationTarget,GrandTotalTime_ms,TotalComputeTime_ms,TotalCommTime_ms" > "$SUMMARY_CSV"

# --- Step 3: Run Experiments ---
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw3-2_*" | sort))

if [ ${#EXECUTABLES[@]} -eq 0 ]; then
    echo "Error: No executables found matching hw3-2_*"
    exit 1
fi

# [MODIFIED] AMD Partition
SRUN_CMD="srun -p amd -N1 -n1 --gres=gpu:1 --time=00:02:00"

echo "Step 3: Running experiments (Max Retries: $MAX_RETRIES)..."
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    
    # Skip blocking factor experiments (e.g. hw3-2_bf64)
    if [[ "$exe_name" == *"bf"* ]]; then
        continue
    fi

    # Determine Ablation Target
    case "$exe_name" in
        "hw3-2_final")      target="Baseline(Final)" ;;
        "hw3-2_no_pin")     target="NoPinnedMemory" ;;
        "hw3-2_no_stream")  target="NoStreams" ;;
        "hw3-2_padding")    target="WithPadding" ;;
        "hw3-2_no_padding") target="NoPadding" ;;
        *) 
            if [[ "$exe_name" == "hw3-2" ]]; then target="Baseline(Default)"; else target="Unknown"; fi 
            ;;
    esac
    
    echo "=========================================="
    echo "Testing: $exe_name ($target)"
    echo "=========================================="
    
    grand_total_time=0
    grand_compute_time=0
    grand_comm_time=0
    
    for testcase in "${TESTCASES[@]}"; do
        tc_name=$(basename "$testcase")
        echo -n "  Running $tc_name ... "
        
        # --- RETRY LOOP FOR SINGLE TESTCASE ---
        success=false
        total_time=0; compute_time=0; comm_time=0; io_time=0
        
        for (( attempt=1; attempt<=MAX_RETRIES; attempt++ )); do
            
            app_output=$($SRUN_CMD ./$exe_name "$testcase" "$TEMP_OUT_FILE" 2>&1)
            prof_line=$(echo "$app_output" | grep "\[PROF_RESULT\]")
            
            if [[ ! -z "$prof_line" ]]; then
                # Success! Parse and break loop
                total_time=$(echo "$prof_line" | cut -d',' -f2)
                compute_time=$(echo "$prof_line" | cut -d',' -f3)
                comm_time=$(echo "$prof_line" | cut -d',' -f4)
                io_time=$(echo "$prof_line" | cut -d',' -f5)
                
                echo "${total_time} ms"
                success=true
                break
            else
                # Failure
                if [ $attempt -lt $MAX_RETRIES ]; then
                    echo -n "[Retry $attempt] "
                    sleep 1 # Brief pause
                fi
            fi
        done

        if [ "$success" = false ]; then
            echo "FAILED after $MAX_RETRIES attempts."
            # Mark as -1 in CSV to indicate failure
            total_time=-1; compute_time=-1; comm_time=-1; io_time=-1
        else
            # Only add to Grand Total if success
            grand_total_time=$(python3 -c "print($grand_total_time + $total_time)")
            grand_compute_time=$(python3 -c "print($grand_compute_time + $compute_time)")
            grand_comm_time=$(python3 -c "print($grand_comm_time + $comm_time)")
        fi

        # Append per-testcase result
        echo "$exe_name,$target,$tc_name,$total_time,$compute_time,$comm_time,$io_time" >> "$OUTPUT_CSV"
    done
    
    echo "------------------------------------------"
    echo "  Grand Total: $grand_total_time ms"
    echo "------------------------------------------"
    echo ""
    
    # Append summary result
    echo "$exe_name,$target,$grand_total_time,$grand_compute_time,$grand_comm_time" >> "$SUMMARY_CSV"

done

# Cleanup
rm -f "$TEMP_OUT_FILE"
echo "=========================================="
echo "All ablation experiments completed!"
echo "Details saved to: $OUTPUT_CSV"
echo "Summary saved to: $SUMMARY_CSV"
