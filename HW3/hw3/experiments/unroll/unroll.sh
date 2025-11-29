#!/bin/bash

# --- Configuration ---
OUTPUT_CSV="unroll_results.csv"
TEMP_OUT_FILE="temp_unroll_output.bin" # Dummy output file for the program

# --- Check for Testcase Argument ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_testcase>"
    echo "Example: $0 ../../testcases/p30k1"
    exit 1
fi
TESTCASE=$1
if [ ! -f "$TESTCASE" ]; then
    echo "Error: Testcase file not found at '$TESTCASE'"
    exit 1
fi

# --- Step 1: Compilation ---
echo "Step 1: Compiling all unroll versions..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# --- Step 2: Initialize CSV ---
echo "Step 2: Preparing results file: $OUTPUT_CSV"
echo "Version,UnrollFactor,TotalTime_ms,ComputeTime_ms,gld_throughput_GBs,gst_throughput_GBs,sm_efficiency_pct,achieved_occupancy" > "$OUTPUT_CSV"

# --- Step 3: Run Experiments ---
# Get the list of executables from make
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw3-2_unroll_*" | sort))

echo "Step 3: Running experiments on testcase: $(basename $TESTCASE)"
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    unroll_factor=$(echo "$exe_name" | sed 's/hw3-2_unroll_//')
    
    echo "------------------------------------------"
    echo "Testing: $exe_name (Unroll Factor: $unroll_factor)"

    # The command to run with srun and nvprof
    SRUN_CMD="srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics gld_throughput,gst_throughput,sm_efficiency,achieved_occupancy"
    
    # Run and capture all output (stdout and stderr)
    output_log=$($SRUN_CMD $exe "$TESTCASE" "$TEMP_OUT_FILE" 2>&1)
    
    # --- Parse Your Program's Profiling Output ---
    total_time=$(echo "$output_log" | grep "Total time" | awk -F': ' '{print $2}' | awk '{print $1}')
    compute_time=$(echo "$output_log" | grep "Compute-all" | awk -F': ' '{print $2}' | awk '{print $1}')
    
    # --- Parse nvprof's Metric Output ---
    # We focus on kernel_phase3 as it's the most time-consuming
    # Note: awk '{print $NF}' gets the last column (the 'Avg' value)
    gld_throughput=$(echo "$output_log" | grep -A 3 "kernel_phase3" | grep "gld_throughput" | awk '{print $NF}' | sed 's/GB\/s//')
    gst_throughput=$(echo "$output_log" | grep -A 3 "kernel_phase3" | grep "gst_throughput" | awk '{print $NF}' | sed 's/GB\/s//')
    sm_efficiency=$(echo "$output_log" | grep -A 3 "kernel_phase3" | grep "sm_efficiency" | awk '{print $NF}' | sed 's/%//')
    achieved_occupancy=$(echo "$output_log" | grep -A 3 "kernel_phase3" | grep "achieved_occupancy" | awk '{print $NF}')

    # Default to 0 if parsing fails
    total_time=${total_time:-0}; compute_time=${compute_time:-0}
    gld_throughput=${gld_throughput:-0}; gst_throughput=${gst_throughput:-0}
    sm_efficiency=${sm_efficiency:-0}; achieved_occupancy=${achieved_occupancy:-0}

    echo "  -> Total Time: ${total_time} ms | Compute Time: ${compute_time} ms"
    echo "  -> GLD Throughput: ${gld_throughput} GB/s | SM Efficiency: ${sm_efficiency}%"
    
    # --- Append to CSV ---
    echo "$exe_name,$unroll_factor,$total_time,$compute_time,$gld_throughput,$gst_throughput,$sm_efficiency,$achieved_occupancy" >> "$OUTPUT_CSV"

done

# --- Cleanup ---
rm -f "$TEMP_OUT_FILE"
echo "------------------------------------------"
echo ""
echo "All unroll experiments completed!"
echo "Results saved to $OUTPUT_CSV"
