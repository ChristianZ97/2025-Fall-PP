#!/bin/bash
#================================================================
# Scalability Experiments (All Testcases) - Strong Scaling
# 使用方法: ./run_scalability_experiments.sh 2>&1 | tee scalability_$(date +%Y%m%d_%H%M%S).log
#================================================================

# 配置
TESTCASE_DIR="../../testcases"
OUTPUT_DIR="scalability_results"
RESULT_FILE="${OUTPUT_DIR}/scalability_summary.csv"
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
echo "Program,Num_Processes,Num_Threads,Total_Cores,Total_Execution_Time,Avg_Imbalance_Pct,Speedup,Efficiency,Type" > $RESULT_FILE

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

echo "Found $TESTCASE_COUNT testcases to run for scalability experiments."
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

# --- hw2a Scalability 實驗 ---
echo "=========================================="
echo " Step 2: hw2a Strong Scaling (1-12 threads)"
echo "=========================================="

HW2A_EXE="hw2a_best"
HW2A_THREADS=(1 2 3 4 5 6 7 8 9 10 11 12)

# Baseline for speedup calculation (1 thread)
hw2a_baseline_time=0

for num_threads in "${HW2A_THREADS[@]}"; do
    echo ""
    echo "Testing hw2a with $num_threads threads"
    
    total_exec_time=0
    total_imbalance=0
    
    for testcase_file in "${TESTCASE_FILES[@]}"; do
        results=$(run_and_parse "$HW2A_EXE" "$testcase_file" "srun -n1 -c${num_threads}")
        if [[ $? -eq 0 ]]; then
            time=$(echo $results | awk '{print $1}')
            imbalance=$(echo $results | awk '{print $2}')
            total_exec_time=$(echo "$total_exec_time + $time" | bc)
            total_imbalance=$(echo "$total_imbalance + $imbalance" | bc)
        fi
    done

    avg_imbalance=$(echo "scale=4; $total_imbalance / $TESTCASE_COUNT" | bc)

    if [[ $avg_imbalance == .* ]]; then
        avg_imbalance="0${avg_imbalance}"
    fi
    
    # Calculate speedup and efficiency
    if [ $num_threads -eq 1 ]; then
        hw2a_baseline_time=$total_exec_time
        speedup="1.0000"
        efficiency="100.00"
    else
        speedup=$(echo "scale=4; $hw2a_baseline_time / $total_exec_time" | bc)
        efficiency=$(echo "scale=2; $speedup / $num_threads * 100" | bc)
    fi
    
    echo "-----------------------------------------------------"
    echo "  Summary for hw2a with $num_threads threads:"
    echo "  Total Execution Time: ${total_exec_time}s"
    echo "  Average Thread Imbalance: ${avg_imbalance}%"
    echo "  Speedup: ${speedup}x"
    echo "  Efficiency: ${efficiency}%"
    echo "-----------------------------------------------------"
    
    echo "$HW2A_EXE,1,$num_threads,$num_threads,$total_exec_time,$avg_imbalance,$speedup,$efficiency,hw2a" >> $RESULT_FILE
done


# --- hw2b Scalability 實驗 ---
echo ""
echo "=========================================="
echo " Step 3: hw2b Strong Scaling"
echo "=========================================="

HW2B_EXE="hw2b_best"

HW2B_CONFIGS=(
    # 格式: "processes threads"
    # 限制: n × c ≤ 48 且 c ≤ min(12, n)
    
    "1 12"   # 1×12 = 12 CPUs ✓
    "2 12"   # 2×12 = 24 CPUs ✓
    "3 12"   # 3×12 = 36 CPUs ✓
    "4 12"   # 4×12 = 48 CPUs ✓
    
    "48 1"
    "24 2"
    "16 3"
    "12 4"
    "8 6"
)

# Baseline for speedup calculation
hw2b_baseline_time=0
hw2b_baseline_cores=0

for config in "${HW2B_CONFIGS[@]}"; do
    num_procs=$(echo $config | awk '{print $1}')
    threads_per_proc=$(echo $config | awk '{print $2}')
    total_cores=$((num_procs * threads_per_proc))
    
    echo ""
    echo "Testing hw2b with $num_procs processes × $threads_per_proc threads = $total_cores cores"
    
    total_exec_time=0
    total_imbalance=0
    
    for testcase_file in "${TESTCASE_FILES[@]}"; do
        results=$(run_and_parse "$HW2B_EXE" "$testcase_file" "srun -n${num_procs} -c${threads_per_proc}")
        if [[ $? -eq 0 ]]; then
            time=$(echo $results | awk '{print $1}')
            imbalance=$(echo $results | awk '{print $2}')
            total_exec_time=$(echo "$total_exec_time + $time" | bc)
            total_imbalance=$(echo "$total_imbalance + $imbalance" | bc)
        fi
    done

    avg_imbalance=$(echo "scale=4; $total_imbalance / $TESTCASE_COUNT" | bc)

    if [[ $avg_imbalance == .* ]]; then
        avg_imbalance="0${avg_imbalance}"
    fi
    
    # Calculate speedup and efficiency (relative to first config)
    if [ $hw2b_baseline_cores -eq 0 ]; then
        hw2b_baseline_time=$total_exec_time
        hw2b_baseline_cores=$total_cores
        speedup="1.0000"
        efficiency="100.00"
    else
        speedup=$(echo "scale=4; $hw2b_baseline_time / $total_exec_time" | bc)
        # Efficiency relative to baseline cores
        cores_ratio=$(echo "scale=4; $total_cores / $hw2b_baseline_cores" | bc)
        efficiency=$(echo "scale=2; $speedup / $cores_ratio * 100" | bc)
    fi
    
    echo "-----------------------------------------------------"
    echo "  Summary for hw2b ($num_procs × $threads_per_proc = $total_cores cores):"
    echo "  Total Execution Time: ${total_exec_time}s"
    echo "  Average Process Imbalance: ${avg_imbalance}%"
    echo "  Speedup: ${speedup}x"
    echo "  Efficiency: ${efficiency}%"
    echo "-----------------------------------------------------"
    
    echo "$HW2B_EXE,$num_procs,$threads_per_proc,$total_cores,$total_exec_time,$avg_imbalance,$speedup,$efficiency,hw2b" >> $RESULT_FILE
done

# 清理臨時log
rm -f "$TEMP_LOG"

echo ""
echo "=========================================="
echo " Scalability Results Summary"
echo "=========================================="

# 顯示結果表格
column -t -s',' $RESULT_FILE

echo ""
echo "✓ Done! Scalability results saved to: $RESULT_FILE"
echo ""
echo "Analysis:"
echo "---------"
echo "hw2a:"
tail -n +2 $RESULT_FILE | grep ",hw2a" | awk -F',' '{printf "  %2d threads: %8.2fs (Speedup: %6.2fx, Efficiency: %5.1f%%)\n", $3, $5, $7, $8}'

echo ""
echo "hw2b:"
tail -n +2 $RESULT_FILE | grep ",hw2b" | awk -F',' '{printf "  %d×%2d=%2d cores: %8.2fs (Speedup: %6.2fx, Efficiency: %5.1f%%)\n", $2, $3, $4, $5, $7, $8}'
