#!/bin/bash

# --- Configuration ---
# 輸出檔名，紀錄每一筆測資的詳細分段時間
OUTPUT_CSV="detailed_perf_results.csv"
# 執行時的暫存輸出檔
TEMP_OUT_FILE="temp_app_output.bin"

# --- Check for Testcase Directory Argument ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <testcases_directory>"
    echo "Example: $0 ../../testcases"
    exit 1
fi

TESTCASE_DIR=$1
if [ ! -d "$TESTCASE_DIR" ]; then
    echo "Error: Testcase directory not found at '$TESTCASE_DIR'"
    exit 1
fi

TESTCASES=()

# --- Find Testcases (c01.1 ~ c21.1) ---
echo "Searching for c01.1 to c21.1 in $TESTCASE_DIR ..."
for i in {1..21}; do
    # 格式化數字為兩位數 (例如 01, 02 ... 21)
    printf -v num "%02d" $i
    target_file="$TESTCASE_DIR/c${num}.1"
    if [ -f "$target_file" ]; then
        TESTCASES+=("$target_file")
    fi
done

# --- Find Testcases (p11k1 ~ p40k1) ---
echo "Searching for p11k1 to p40k1 in $TESTCASE_DIR ..."
for i in {11..40}; do
    target_file="$TESTCASE_DIR/p${i}k1"
    if [ -f "$target_file" ]; then
        TESTCASES+=("$target_file")
    fi
done

NUM_TESTCASES=${#TESTCASES[@]}
if [ $NUM_TESTCASES -eq 0 ]; then
    echo "Error: No target testcases found in '$TESTCASE_DIR'"
    exit 1
fi

echo "Found $NUM_TESTCASES testcases in total."

# --- Step 1: Compilation ---
echo "Step 1: Compiling all versions (if needed)..."
# 建議在正式執行前先編譯，這裡只做檢查
make clean > /dev/null
make all
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# --- Step 2: Initialize CSV ---
echo "Step 2: Preparing results file..."
# Header 對應你的 .cu fprintf 順序:
# [PROF_RESULT], workload, total, compute, comm, io, p1, p2, p3
echo "Executable,Testcase,Workload,TotalTime,ComputeTime,CommTime,IOTime,Phase1,Phase2,Phase3" > "$OUTPUT_CSV"

# --- Step 3: Run Experiments ---
# 搜尋所有 hw3-2_ 開頭的可執行檔
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw3-2*" | sort))

if [ ${#EXECUTABLES[@]} -eq 0 ]; then
    echo "Error: No executables found matching hw3-2*"
    exit 1
fi

echo "Step 3: Running experiments..."
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    
    echo "=========================================="
    echo "Testing Executable: $exe_name"
    echo "=========================================="
    
    for testcase in "${TESTCASES[@]}"; do
        tc_name=$(basename "$testcase")
        
        # 設定資源管理系統 srun 的指令 (例如使用 Slurm)
        SRUN_CMD="srun -p nvidia -N1 -n1 --gres=gpu:1 --time=00:05:00"
        
        echo -n "  Running $tc_name ... "
        
        # 執行程式並捕捉輸出 (包含 stderr)
        # 注意: 你的 PROF_RESULT 是輸出到 stderr (2>&1)
        app_output=$($SRUN_CMD ./$exe_name "$testcase" "$TEMP_OUT_FILE" 2>&1)
        
        # 抓取包含 [PROF_RESULT] 的那一行
        prof_line=$(echo "$app_output" | grep "\[PROF_RESULT\]")
        
        if [[ -z "$prof_line" ]]; then
            echo "FAILED / TIMEOUT / NO OUTPUT"
            # 寫入錯誤標記，保持 CSV 格式不亂 (填入 -1)
            # 寫入格式: 執行檔, 測資, Workload, Total, Compute, Comm, IO, P1, P2, P3
            echo "$exe_name,$tc_name,-1,-1,-1,-1,-1,-1,-1,-1" >> "$OUTPUT_CSV"
        else
            # 移除 tag "[PROF_RESULT]," 剩下的就是數據: val,val,val...
            # 你的輸出格式範例: [PROF_RESULT],123,1.1,2.2,...
            metrics=$(echo "$prof_line" | sed 's/\[PROF_RESULT\],//')
            
            echo "Done."
            
            # 寫入 CSV: 執行檔名, 測資名, 數據 (Workload, Time...)
            echo "$exe_name,$tc_name,$metrics" >> "$OUTPUT_CSV"
        fi
    done
    echo ""
done

# Cleanup
rm -f "$TEMP_OUT_FILE"
echo "=========================================="
echo "All experiments completed!"
echo "Details saved to: $OUTPUT_CSV"