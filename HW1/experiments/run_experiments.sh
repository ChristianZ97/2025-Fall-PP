#!/bin/bash

# ============================================================================
# Comprehensive Scaling Analysis Script for CS542200 HW1
# ============================================================================

EXECUTABLE="./hw1_prof"

# Clean up and compile once
echo "Compiling with 'make prof'..."
make prof
if [ $? -ne 0 ]; then
    echo "Make failed. Aborting."
    exit 1
fi
echo "Compilation successful."

# ============================================================================
# Loop through all testcases
# ============================================================================

for TESTCASE_NUM in {01..40}; do
    echo "========================================"
    echo "Processing Testcase $TESTCASE_NUM"
    echo "========================================"
    
    # --- Configuration for current testcase ---
    TESTCASE_IN="./testcases/${TESTCASE_NUM}.in"
    TESTCASE_OUT_PREFIX="my_${TESTCASE_NUM}"
    TESTCASE_SPEC="./testcases/${TESTCASE_NUM}.txt"
    
    # Check if files exist
    if [[ ! -f "$TESTCASE_IN" || ! -f "$TESTCASE_SPEC" ]]; then
        echo "Testcase $TESTCASE_NUM files not found. Skipping..."
        continue
    fi
    
    # Extract N parameter from spec file
    N=$(grep '"n":' "$TESTCASE_SPEC" | awk -F': ' '{print $2}' | tr -d ',')
    
    if [[ -z "$N" ]]; then
        echo "Failed to extract N for testcase $TESTCASE_NUM. Skipping..."
        continue
    fi
    
    echo "Extracted N=$N for testcase $TESTCASE_NUM"
    
    # Output files for this testcase
    RAW_LOG_FILE="experiments/scaling_raw_${TESTCASE_NUM}.log"
    CSV_RESULT_FILE="experiments/scaling_results_${TESTCASE_NUM}.csv"
    
    # Clean up previous results
    > "$RAW_LOG_FILE"
    echo "N,Nodes,Cores,Total_Time_srun,IO_Time,Comm_Time,CPU_Time" > "$CSV_RESULT_FILE"
    
    # Torture Test Configurations
    CONFIGS=()
    for p in {1..48}; do
        CONFIGS+=("4 $p")
    done
    
    # --- Main Experiment Loop for this testcase ---
    echo "Starting experiments for testcase $TESTCASE_NUM (N=$N)" | tee -a "$RAW_LOG_FILE"
    
    for config in "${CONFIGS[@]}"; do
        read -r nodes cores <<< "$config"
        output_file="${TESTCASE_OUT_PREFIX}_${cores}cores.out"
        
        echo "--- Running with $cores cores on $nodes node(s) ---" | tee -a "$RAW_LOG_FILE"
        
        CMD="srun -N$nodes -n$cores $EXECUTABLE $N $TESTCASE_IN $output_file"
        echo "Executing: $CMD" >> "$RAW_LOG_FILE"
        
        SRUN_TIME_OUTPUT=$( ( \time -f "%e" $CMD ) 2>&1 )
        
        SRUN_TIME=$(echo "$SRUN_TIME_OUTPUT" | tail -n 1)
        MPI_OUTPUT=$(echo "$SRUN_TIME_OUTPUT" | head -n -1)
        
        echo "$MPI_OUTPUT" >> "$RAW_LOG_FILE"
        echo "srun Wall Time: $SRUN_TIME seconds" | tee -a "$RAW_LOG_FILE"

        IO_TIME=$(echo "$MPI_OUTPUT" | grep 'IO Time:' | awk '{print $3}')
        COMM_TIME=$(echo "$MPI_OUTPUT" | grep 'Comm Time:' | awk '{print $3}')
        CPU_TIME=$(echo "$MPI_OUTPUT" | grep 'CPU Time:' | awk '{print $3}')
        
        if [[ -n "$SRUN_TIME" && -n "$IO_TIME" && -n "$COMM_TIME" && -n "$CPU_TIME" ]]; then
            echo "$N,$nodes,$cores,$SRUN_TIME,$IO_TIME,$COMM_TIME,$CPU_TIME" >> "$CSV_RESULT_FILE"
        else
            echo "Failed to parse output for $cores cores. Skipping CSV entry." | tee -a "$RAW_LOG_FILE"
        fi

        echo "Finished run for $cores cores."
        
        sleep 2
    done
    
    echo "Completed testcase $TESTCASE_NUM"
    echo "Results saved to: $CSV_RESULT_FILE"
    echo ""
done

echo "All testcases completed."
