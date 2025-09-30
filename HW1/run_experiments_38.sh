#!/bin/bash

# ============================================================================
# Comprehensive Scaling Analysis Script for CS542200 HW1
# ============================================================================

# --- Configuration ---
TESTCASE_IN="./testcases/38.in"
TESTCASE_OUT_PREFIX="my_38"
EXECUTABLE="./hw1_prof"
N=536831999 # N for testcase 38

# Output files
RAW_LOG_FILE="scaling_raw.log"
CSV_RESULT_FILE="scaling_results.csv"

# ============================================================================
# Torture Test Configurations for Advanced Analysis
# ============================================================================
# Format: "Nodes Cores"
# This array includes prime numbers, non-ideal distributions, and edge cases
# to rigorously test the algorithm's robustness and scalability limits.
CONFIGS=(
    # --- Baseline and Intra-Node Scaling ---
    "4 1"   # Serial baseline for speedup calculation
    "4 2"
    "4 6"   # Half a node
    "4 11"  # A prime number within a single node
    "4 12"  # Single node full load

    # --- Two-Node Scaling (Ideal and Non-Ideal) ---
    "4 13"  # A prime number just across the node boundary (12+1)
    "4 18"  # Non-ideal, 1.5 nodes (12+6)
    "4 23"  # A prime number filling almost two nodes
    "4 24"  # Ideal, two full nodes

    # --- Multi-Node Scaling (3 Nodes) ---
    "4 25"  # Just over two nodes (12+12+1)
    "4 30"  # 2.5 nodes
    "4 35"  # Almost three full nodes
    "4 36"  # Ideal, three full nodes

    # --- Maximum Scale and Boundary Tests (4 Nodes) ---
    "4 37"  # A prime number just across the three-node boundary
    "4 42"  # 3.5 nodes
    "4 47"  # A prime number at the upper limit
    "4 48"  # Maximum allowed resources
)


# --- Script Execution ---

# Clean up previous results
> "$RAW_LOG_FILE"
echo "Nodes,Cores,Total_Time_srun,IO_Time,Comm_Time,CPU_Time" > "$CSV_RESULT_FILE"

# Compile the profiling version of the executable
echo "Compiling with 'make prof'..."
make prof
if [ $? -ne 0 ]; then
    echo "Make failed. Aborting."
    exit 1
fi
echo "Compilation successful."

# --- Main Experiment Loop ---


# --- Main Experiment Loop ---
echo "=================================================" | tee -a "$RAW_LOG_FILE"
echo "Starting Comprehensive Strong Scaling Experiment" | tee -a "$RAW_LOG_FILE"
echo "Problem Size (N): $N, Testcase: $TESTCASE_IN" | tee -a "$RAW_LOG_FILE"
echo "=================================================" | tee -a "$RAW_LOG_FILE"

# The T1 run is now part of the main loop, not a separate step.
# We will extract T1 later from the CSV for speedup calculation.

# Loop through ALL configurations
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
        echo "$nodes,$cores,$SRUN_TIME,$IO_TIME,$COMM_TIME,$CPU_TIME" >> "$CSV_RESULT_FILE"
    else
        echo "Failed to parse output for $cores cores. Skipping CSV entry." | tee -a "$RAW_LOG_FILE"
    fi

    echo "Finished run for $cores cores."
    echo "-------------------------------------------------"
    
    sleep 5
done

echo "All experiments completed."
echo "Raw logs are in: $RAW_LOG_FILE"
echo "Formatted results are in: $CSV_RESULT_FILE"
