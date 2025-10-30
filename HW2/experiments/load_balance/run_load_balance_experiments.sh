#!/bin/bash

TESTCASE_DIR="../../testcases"
OUTPUT_DIR="load_balance_results_all_cases"
RESULT_FILE="${OUTPUT_DIR}/summary_results.csv"
TEMP_LOG="temp_run.log"
SCRIPT_DIR=$(pwd)
COMPUTE_DIR=${SCRIPT_DIR/\/beegfs/}
RETRY_DELAY=2

if [[ ! -d "$TESTCASE_DIR" ]]; then
    echo "Error: Testcase directory $TESTCASE_DIR not found!"
    exit 1
fi

mkdir -p $OUTPUT_DIR

echo "Program,Strategy,Total_Execution_Time,Avg_Thread_Imbalance_Pct,Type" > $RESULT_FILE

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

echo "Found $TESTCASE_COUNT testcases to run for load balance experiments."
echo ""

run_and_parse() {
    local exe=$1
    local testcase_file=$2
    local srun_args=$3

    local testcase_name=$(basename "$testcase_file" .txt)
    
    local TESTCASE_ARGS=$(cat "$testcase_file")
    if [[ -z "$TESTCASE_ARGS" ]]; then
        echo -n "  Running on $testcase_name... " >&2
        echo "SKIPPED (empty testcase file)" >&2
        return 1
    fi

    while true; do
        echo -n "  Running on $testcase_name... " >&2

        eval "$srun_args \"$COMPUTE_DIR/$exe\" \"${OUTPUT_DIR}/${exe}_${testcase_name}.png\" $TESTCASE_ARGS" > "$TEMP_LOG" 2>&1
        
        time=$(grep "Total Time:" "$TEMP_LOG" | head -1 | awk '{print $3}')
        imbalance=$(grep "Thread Imbalance:" "$TEMP_LOG" | head -1 | awk '{print $3}' | tr -d '%')
        
        if [[ "$time" =~ ^[0-9.]+$ && "$imbalance" =~ ^[0-9.]+$ ]]; then
            echo "$time $imbalance"
            echo "Done (${time}s, ${imbalance}%)" >&2
            return 0
        fi

        echo "FAILED. Retrying in $RETRY_DELAY seconds..." >&2
        sleep $RETRY_DELAY
    done
}

echo "=========================================="
echo " Step 2: Running Experiments for hw2a"
echo "=========================================="
hw2a_targets=($(make list 2>/dev/null | grep "hw2a targets:" | cut -d':' -f2))

for exe in "${hw2a_targets[@]}"; do
    if [ ! -x "$exe" ]; then continue; fi

    echo ""
    echo "Testing Config: $exe"
    strategy=$(echo $exe | sed 's/hw2a_//')
    
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
    echo "  Total Execution Time: ${total_exec_time}s"
    echo "  Average Thread Imbalance: ${avg_imbalance}%"
    echo "-----------------------------------------------------"
    
    echo "$exe,$strategy,$total_exec_time,$avg_imbalance,hw2a" >> $RESULT_FILE
done


echo ""
echo "=========================================="
echo " Step 3: Running Experiments for hw2b"
echo "=========================================="
hw2b_targets=($(make list 2>/dev/null | grep "hw2b targets:" | cut -d':' -f2))

for exe in "${hw2b_targets[@]}"; do
    if [ ! -x "$exe" ]; then continue; fi
    
    echo ""
    echo "Testing Config: $exe (MPI: 4 processes, 12 threads/proc)"
    strategy=$(echo $exe | sed 's/hw2b_//')
    
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
    echo "  Total Execution Time: ${total_exec_time}s"
    echo "  Average Thread Imbalance: ${avg_imbalance}%"
    echo "-----------------------------------------------------"
    
    echo "$exe,$strategy,$total_exec_time,$avg_imbalance,hw2b" >> $RESULT_FILE
