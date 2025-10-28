#!/bin/bash

#================================================================
# Unroll Experiments - Updated for New Output Format
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
echo "Program,Unroll,Total_Time,Compute_Time,Compute_Pct,Sync_Comm_Time,Sync_Comm_Pct,IO_Time,IO_Pct,Imbalance_Pct,Parallel_Eff,Type" > $RESULT_FILE

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
        
        # Parse NEW profiling output format for hw2a
        total_time=$(grep "Total Time:" $LOG_FILE | awk '{print $3}')
        compute_time=$(grep "Compute (avg):" $LOG_FILE | awk '{print $3}')
        compute_pct=$(grep "Compute (avg):" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        sync_time=$(grep "Sync (avg):" $LOG_FILE | awk '{print $3}')
        sync_pct=$(grep "Sync (avg):" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        io_time=$(grep "IO:" $LOG_FILE | awk '{print $2}')
        io_pct=$(grep "IO:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        parallel_eff=$(grep "Parallel Efficiency:" $LOG_FILE | awk '{print $3}' | tr -d '%')
        imbalance=$(grep "Imbalance:" $LOG_FILE | awk '{print $2}' | tr -d '%')
        
        echo "Total: ${total_time}s, Compute: ${compute_time}s (${compute_pct}%), Imbalance: ${imbalance}%"
        
        # Write to CSV
        echo "$exe,$unroll,$total_time,$compute_time,$compute_pct,$sync_time,$sync_pct,$io_time,$io_pct,$imbalance,$parallel_eff,hw2a" >> $RESULT_FILE
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
        
        # Parse NEW profiling output format for hw2b
        total_time=$(grep "Total Time:" $LOG_FILE | grep "Timing Breakdown" -A5 | grep "Total Time:" | awk '{print $3}')
        compute_time=$(grep "Compute:" $LOG_FILE | grep "Timing Breakdown" -A5 | grep "Compute:" | awk '{print $2}')
        compute_pct=$(grep "Compute:" $LOG_FILE | grep "Timing Breakdown" -A5 | grep "Compute:" | grep -oP '\(\K[0-9.]+')
        comm_time=$(grep "Communication:" $LOG_FILE | awk '{print $2}')
        comm_pct=$(grep "Communication:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        io_time=$(grep "IO:" $LOG_FILE | grep "Timing Breakdown" -A5 | grep "IO:" | awk '{print $2}')
        io_pct=$(grep "IO:" $LOG_FILE | grep "Timing Breakdown" -A5 | grep "IO:" | grep -oP '\(\K[0-9.]+')
        parallel_eff=$(grep "Parallel Efficiency:" $LOG_FILE | awk '{print $3}' | tr -d '%')
        imbalance=$(grep "Total Imbalance:" $LOG_FILE | awk '{print $3}' | tr -d '%')
        
        echo "Total: ${total_time}s, Compute: ${compute_time}s (${compute_pct}%), Comm: ${comm_time}s, Imbalance: ${imbalance}%"
        
        # Write to CSV (use Compute as compute, Communication as sync)
        echo "$exe,$unroll,$total_time,$compute_time,$compute_pct,$comm_time,$comm_pct,$io_time,$io_pct,$imbalance,$parallel_eff,hw2b" >> $RESULT_FILE
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
