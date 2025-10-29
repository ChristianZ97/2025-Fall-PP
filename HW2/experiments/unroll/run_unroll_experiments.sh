#!/bin/bash
#================================================================
# Unroll Experiments (All Testcases & New Output Format) - FIX 2
# usage: ./run_unroll_experiments.sh 2>&1 | tee unroll_experiment_$(date +%Y%m%d_%H%M%S).log
#================================================================

# Configuration
TESTCASE_DIR="../../testcases"
OUTPUT_DIR="unroll_results_all_cases"
RESULT_FILE="${OUTPUT_DIR}/summary_results.csv"
TEMP_LOG="temp_run.log"

# Check if testcase directory exists
if [[ ! -d "$TESTCASE_DIR" ]]; then
    echo "Error: Testcase directory $TESTCASE_DIR not found!"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Initialize CSV file header
echo "Program,Unroll_Factor,Total_Execution_Time,Avg_Thread_Imbalance_Pct,Type" > $RESULT_FILE

echo "=========================================="
echo " Step 1: Compilation (via make)"
echo "=========================================="
make clean
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi
echo ""

TESTCASE_FILES=($(find "$TESTCASE_DIR" -type f -name "*.txt" | sort))
TESTCASE_COUNT=${#TESTCASE_FILES[@]}

if [[ $TESTCASE_COUNT -eq 0 ]]; then
    echo "Error: No testcases found in $TESTCASE_DIR"
    exit 1
fi

echo "Found $TESTCASE_COUNT testcases to run for unroll experiments."
echo ""

# --- Function to run and parse results ---
run_and_parse() {
    local exe=$1
    local testcase_file=$2
    local srun_args=$3

    local testcase_name=$(basename "$testcase_file" .txt)
    # FIX: Redirect progress message to stderr (>&2)
    echo -n "  Running on $testcase_name... " >&2
    
    local TESTCASE_ARGS=$(cat "$testcase_file")
    if [[ -z "$TESTCASE_ARGS" ]]; then
        # FIX: Redirect skip message to stderr
        echo "SKIPPED (empty testcase file)" >&2
        return 1
    fi

    # Execute command
    eval "$srun_args ./$exe \"${OUTPUT_DIR}/${exe}_${testcase_name}.png\" $TESTCASE_ARGS" > "$TEMP_LOG" 2>&1
    
    # Use awk '{print $3}' to get the 3rd field, which is the numeric value.
    time=$(grep "Total Time:" "$TEMP_LOG" | head -1 | awk '{print $3}')
    imbalance=$(grep "Thread Imbalance:" "$TEMP_LOG" | head -1 | awk '{print $3}' | tr -d '%')
    
    # Check to ensure variables are numeric
    if ! [[ "$time" =~ ^[0-9.]+$ && "$imbalance" =~ ^[0-9.]+$ ]]; then
        # FIX: Redirect all error-related messages to stderr
        echo "FAILED (Could not parse numeric results from $TEMP_LOG)" >&2
        echo "Parsed Time: '$time', Parsed Imbalance: '$imbalance'" >&2
        cat "$TEMP_LOG" >&2
        return 1
    fi

    # This is the ONLY output to stdout, which gets captured by 'results'.
    echo "$time $imbalance"
    
    # This final status message goes to stderr.
    echo "Done (${time}s, ${imbalance}%)" >&2
}


# --- hw2a Unroll Experiments ---
echo "=========================================="
echo " Step 2: Running Unroll Experiments for hw2a"
echo "=========================================="
hw2a_targets=($(make list 2>/dev/null | grep "hw2a targets:" | cut -d':' -f2))

for exe in "${hw2a_targets[@]}"; do
    if [ ! -x "$exe" ]; then continue; fi

    echo ""
    echo "Testing Config: $exe"
    unroll_factor=$(echo $exe | grep -oP 'unroll\K\d+')
    if [[ -z "$unroll_factor" ]]; then unroll_factor=0; fi
    
    total_exec_time=0
    total_imbalance=0
    
    for testcase_file in "${TESTCASE_FILES[@]}"; do
        results=$(run_and_parse "$exe" "$testcase_file" "srun -n1 -c12")
        if [[ $? -eq 0 ]]; then
            time=$(echo $results | awk '{print $1}')
            imbalance=$(echo $results | awk '{print $2}')
            total_exec_time=$(echo "$total_exec_time + $time" | bc)
            total_imbalance=$(echo "$total_imbalance + $imbalance" | bc)
        fi
    done

    avg_imbalance=$(echo "scale=4; $total_imbalance / $TESTCASE_COUNT" | bc)
    
    echo "-----------------------------------------------------"
    echo "  Summary for $exe:"
    echo "  Total Execution Time (all cases): ${total_exec_time}s"
    echo "  Average Thread Imbalance: ${avg_imbalance}%"
    echo "-----------------------------------------------------"
    
    echo "$exe,$unroll_factor,$total_exec_time,$avg_imbalance,hw2a" >> $RESULT_FILE
done

# --- hw2b Unroll Experiments ---
echo ""
echo "=========================================="
echo " Step 3: Running Unroll Experiments for hw2b"
echo "=========================================="
hw2b_targets=($(make list 2>/dev/null | grep "hw2b targets:" | cut -d':' -f2))

for exe in "${hw2b_targets[@]}"; do
    if [ ! -x "$exe" ]; then continue; fi
    
    echo ""
    echo "Testing Config: $exe (MPI: 4 processes, 12 threads/proc)"
    unroll_factor=$(echo $exe | grep -oP 'unroll\K\d+')
    if [[ -z "$unroll_factor" ]]; then unroll_factor=0; fi
    
    total_exec_time=0
    total_imbalance=0
    
    for testcase_file in "${TESTCASE_FILES[@]}"; do
        results=$(run_and_parse "$exe" "$testcase_file" "srun -n4 -c12")
        if [[ $? -eq 0 ]]; then
            time=$(echo $results | awk '{print $1}')
            imbalance=$(echo $results | awk '{print $2}')
            total_exec_time=$(echo "$total_exec_time + $time" | bc)
            total_imbalance=$(echo "$total_imbalance + $imbalance" | bc)
        fi
    done
    
    avg_imbalance=$(echo "scale=4; $total_imbalance / $TESTCASE_COUNT" | bc)

    echo "-----------------------------------------------------"
    echo "  Summary for $exe:"
    echo "  Total Execution Time (all cases): ${total_exec_time}s"
    echo "  Average Thread Imbalance: ${avg_imbalance}%"
    echo "-----------------------------------------------------"
    
    echo "$exe,$unroll_factor,$total_exec_time,$avg_imbalance,hw2b" >> $RESULT_FILE
done

# Clean up temp file
rm -f "$TEMP_LOG"

echo ""
echo "=========================================="
echo " Results Summary"
echo "=========================================="

# Display results table
column -t -s',' $RESULT_FILE

# Find best performers
echo ""
echo "Best hw2a unroll config: $(tail -n +2 $RESULT_FILE | grep ",hw2a" | sort -t',' -k3 -n | head -1 | cut -d',' -f1,2,3)"
echo "Best hw2b unroll config: $(tail -n +2 $RESULT_FILE | grep ",hw2b" | sort -t',' -k3 -n | head -1 | cut -d',' -f1,2,3)"

echo ""
echo "✓ Done! Summary results saved to: $RESULT_FILE"

