#!/bin/bash

#================================================================
# Unroll Experiments - Using Makefile with Testcase Parameters
# usage: ./run_unroll_experiments.sh 2>&1 | tee experiment_$(date +%Y%m%d_%H%M%S).log
#================================================================

# Configuration
TESTCASE_FILE="../../testcases/strict34.txt"
OUTPUT_DIR="unroll_results"
RESULT_FILE="${OUTPUT_DIR}/detailed_results.csv"

# Check if testcase file exists
if [[ ! -f "$TESTCASE_FILE" ]]; then
    echo "Error: Testcase file $TESTCASE_FILE not found!"
    exit 1
fi

# Read and display testcase parameters
echo "=========================================="
echo " Reading Testcase Parameters"
echo "=========================================="
cat "$TESTCASE_FILE"
echo ""

# Extract parameters from testcase file
TESTCASE=$(cat "$TESTCASE_FILE")

if [[ -z "$TESTCASE" ]]; then
    echo "Error: Failed to read testcase parameters!"
    exit 1
fi

echo "Testcase parameters: $TESTCASE"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Initialize result file with detailed headers
echo "Program,Unroll,Total_Time,IO_Time,IO_Pct,Compute_Time,Compute_Pct,Sync_Comm_Time,Sync_Comm_Pct,Imbalance_Pct,Type" > $RESULT_FILE

echo "=========================================="
echo " Step 1: Compilation (via make)"
echo "=========================================="

# Clean and compile all targets
make clean
make all

if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

echo ""
echo "=========================================="
echo " Step 2: Running Experiments"
echo "=========================================="

# Get list of compiled hw2a executables
hw2a_targets=$(make list 2>/dev/null | grep "hw2a targets:" | cut -d':' -f2)

for exe in $hw2a_targets; do
    if [ -x "$exe" ]; then
        echo ""
        echo "Testing: $exe"
        echo "---"
        
        # Extract unroll number (e.g., hw2a_unroll8 -> 8)
        unroll=$(echo $exe | grep -oP 'unroll\K\d+')
        
        # Define output PNG filename
        OUTPUT_PNG="${OUTPUT_DIR}/${exe}_output.png"
        LOG_FILE="${OUTPUT_DIR}/${exe}.log"
        
        # Run and capture output
        srun -n1 -c12 ./$exe $OUTPUT_PNG $TESTCASE > $LOG_FILE 2>&1
        
        # Parse profiling output
        total_time=$(grep "Total Time:" $LOG_FILE | awk '{print $4}')
        io_time=$(grep "IO Time:" $LOG_FILE | awk '{print $3}')
        io_pct=$(grep "IO Time:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        compute_time=$(grep "Avg Thread Compute Time:" $LOG_FILE | awk '{print $5}')
        compute_pct=$(grep "Avg Thread Compute Time:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        sync_time=$(grep "Avg Thread Sync Time:" $LOG_FILE | awk '{print $5}')
        sync_pct=$(grep "Avg Thread Sync Time:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        imbalance=$(grep "Imbalance:" $LOG_FILE | grep -oP '\d+\.\d+(?=%)')
        
        echo "Total: ${total_time}s, Compute: ${compute_time}s, Imbalance: ${imbalance}%"
        
        # Write to CSV
        echo "$exe,$unroll,$total_time,$io_time,$io_pct,$compute_time,$compute_pct,$sync_time,$sync_pct,$imbalance,hw2a" >> $RESULT_FILE
    fi
done

# Get list of compiled hw2b executables
hw2b_targets=$(make list 2>/dev/null | grep "hw2b targets:" | cut -d':' -f2)

for exe in $hw2b_targets; do
    if [ -x "$exe" ]; then
        echo ""
        echo "Testing: $exe (MPI: 3 processes)"
        echo "---"
        
        # Extract unroll number
        unroll=$(echo $exe | grep -oP 'unroll\K\d+')
        
        # Define output PNG filename
        OUTPUT_PNG="${OUTPUT_DIR}/${exe}_output.png"
        LOG_FILE="${OUTPUT_DIR}/${exe}.log"
        
        # Run with MPI
        srun -n3 -c4 ./$exe $OUTPUT_PNG $TESTCASE > $LOG_FILE 2>&1
        
        # Parse profiling output (hw2b format)
        total_time=$(grep "Avg Total Time:" $LOG_FILE | awk '{print $5}')
        io_time=$(grep "Avg IO Time:" $LOG_FILE | awk '{print $4}')
        io_pct=$(grep "Avg IO Time:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        cpu_time=$(grep "Avg CPU Time:" $LOG_FILE | awk '{print $4}')
        cpu_pct=$(grep "Avg CPU Time:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        comm_time=$(grep "Avg Comm Time:" $LOG_FILE | awk '{print $4}')
        comm_pct=$(grep "Avg Comm Time:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        imbalance=$(grep "Imbalance:" $LOG_FILE | grep -oP '\d+\.\d+(?=%)')
        
        echo "Total: ${total_time}s, CPU: ${cpu_time}s, Comm: ${comm_time}s, Imbalance: ${imbalance}%"
        
        # Write to CSV (use CPU time as compute, Comm as sync)
        echo "$exe,$unroll,$total_time,$io_time,$io_pct,$cpu_time,$cpu_pct,$comm_time,$comm_pct,$imbalance,hw2b" >> $RESULT_FILE
    fi
done

echo ""
echo "=========================================="
echo " Results Summary"
echo "=========================================="

# Display results
column -t -s',' $RESULT_FILE

# Find best performers
echo ""
echo "Best hw2a: $(tail -n +2 $RESULT_FILE | grep hw2a | sort -t',' -k3 -n | head -1 | cut -d',' -f1,3)"
echo "Best hw2b: $(tail -n +2 $RESULT_FILE | grep hw2b | sort -t',' -k3 -n | head -1 | cut -d',' -f1,3)"

echo ""
echo "✓ Done! Detailed results saved to: $RESULT_FILE"
