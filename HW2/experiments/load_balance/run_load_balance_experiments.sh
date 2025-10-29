#!/bin/bash
#================================================================
# Load Balance Experiments (All Testcases) - FIX 2
# 使用方法: ./run_load_balance_experiments.sh 2>&1 | tee lb_experiment_$(date +%Y%m%d_%H%M%S).log
#================================================================

# 配置
TESTCASE_DIR="../../testcases"
OUTPUT_DIR="load_balance_results_all_cases"
RESULT_FILE="${OUTPUT_DIR}/summary_results.csv"
TEMP_LOG="temp_run.log"
SCRIPT_DIR=$(pwd)
COMPUTE_DIR=${SCRIPT_DIR/\/beegfs/}
RETRY_DELAY=2

# 檢查測試案例目錄
if [[ ! -d "$TESTCASE_DIR" ]]; then
    echo "Error: Testcase directory $TESTCASE_DIR not found!"
    exit 1
fi

# 建立輸出目錄
mkdir -p $OUTPUT_DIR

# 初始化 CSV 檔案標頭
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

# --- Function to run and parse results with infinite retry ---
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

    # Use an infinite loop to retry until success
    while true; do
        echo -n "  Running on $testcase_name... " >&2

        # Execute command using compute-node-valid absolute path
        eval "$srun_args \"$COMPUTE_DIR/$exe\" \"${OUTPUT_DIR}/${exe}_${testcase_name}.png\" $TESTCASE_ARGS" > "$TEMP_LOG" 2>&1
        
        time=$(grep "Total Time:" "$TEMP_LOG" | head -1 | awk '{print $3}')
        imbalance=$(grep "Thread Imbalance:" "$TEMP_LOG" | head -1 | awk '{print $3}' | tr -d '%')
        
        # If parsing is successful, we are done. Return success.
        if [[ "$time" =~ ^[0-9.]+$ && "$imbalance" =~ ^[0-9.]+$ ]]; then
            echo "$time $imbalance" # This is the ONLY output to stdout
            echo "Done (${time}s, ${imbalance}%)" >&2
            return 0 # Exit function with success, breaking the loop
        fi

        # If we reach here, it failed. Print a message and wait before retrying.
        echo "FAILED. Retrying in $RETRY_DELAY seconds..." >&2
        sleep $RETRY_DELAY
    done
}

# --- hw2a 實驗 ---
echo "=========================================="
echo " Step 2: Running Experiments for hw2a"
echo "=========================================="
# FIX: Use parentheses to create an array
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


# --- hw2b 實驗 ---
echo ""
echo "=========================================="
echo " Step 3: Running Experiments for hw2b"
echo "=========================================="
# FIX: Use parentheses to create an array
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
done

# 清理臨時log
rm -f "$TEMP_LOG"

echo ""
echo "=========================================="
echo " Results Summary"
echo "=========================================="

# 顯示結果表格
column -t -s',' $RESULT_FILE

# 找出最佳策略
echo ""
echo "Best hw2a on all testcases: $(tail -n +2 $RESULT_FILE | grep ",hw2a" | sort -t',' -k3 -n | head -1 | cut -d',' -f1,2,3)"
echo "Best hw2b on all testcases: $(tail -n +2 $RESULT_FILE | grep ",hw2b" | sort -t',' -k3 -n | head -1 | cut -d',' -f1,2,3)"

echo ""
echo "✓ Done! Summary results saved to: $RESULT_FILE"
