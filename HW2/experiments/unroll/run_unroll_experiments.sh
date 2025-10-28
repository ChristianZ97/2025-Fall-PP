#!/bin/bash

#================================================================
# Unroll Experiments - Using Makefile with Testcase Parameters
#================================================================

# Configuration
TESTCASE_FILE="./testcase/strict34.txt"
OUTPUT_DIR="unroll_results"
RESULT_FILE="${OUTPUT_DIR}/results.csv"

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
# Expected format: 10000 -2 2 -2 2 800 800
TESTCASE=$(cat "$TESTCASE_FILE")

if [[ -z "$TESTCASE" ]]; then
    echo "Error: Failed to read testcase parameters!"
    exit 1
fi

echo "Testcase parameters: $TESTCASE"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Initialize result file
echo "Program,Time(s),Type" > $RESULT_FILE

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
        
        # Define output PNG filename
        OUTPUT_PNG="${OUTPUT_DIR}/${exe}_output.png"
        
        # Measure execution time
        start=$(date +%s.%N)
        srun -n1 -c12 ./$exe $OUTPUT_PNG $TESTCASE > ${OUTPUT_DIR}/${exe}.log 2>&1
        end=$(date +%s.%N)
        
        time_val=$(echo "$end - $start" | bc)
        echo "Time: $time_val seconds"
        
        # Record to CSV
        echo "$exe,$time_val,hw2a" >> $RESULT_FILE
    fi
done

# Get list of compiled hw2b executables
hw2b_targets=$(make list 2>/dev/null | grep "hw2b targets:" | cut -d':' -f2)

for exe in $hw2b_targets; do
    if [ -x "$exe" ]; then
        echo ""
        echo "Testing: $exe (MPI: ${MPI_PROCS} processes, ${MPI_CORES} cores)"
        echo "---"
        
        # Define output PNG filename
        OUTPUT_PNG="${OUTPUT_DIR}/${exe}_output.png"
        
        # Measure execution time
        start=$(date +%s.%N)
        srun -n3 -c4 ./$exe $OUTPUT_PNG $TESTCASE > ${OUTPUT_DIR}/${exe}.log 2>&1
        end=$(date +%s.%N)
        
        time_val=$(echo "$end - $start" | bc)
        echo "Time: $time_val seconds"
        
        # Record to CSV
        echo "$exe,$time_val,hw2b" >> $RESULT_FILE
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
echo "Best hw2a: $(tail -n +2 $RESULT_FILE | grep hw2a | sort -t',' -k2 -n | head -1 | cut -d',' -f1,2)"
echo "Best hw2b: $(tail -n +2 $RESULT_FILE | grep hw2b | sort -t',' -k2 -n | head -1 | cut -d',' -f1,2)"

echo ""
echo "✓ Done! Results saved to: $RESULT_FILE"
