#!/bin/bash
#================================================================
# Load Balance Experiments
# 使用方法: ./run_load_balance_experiments.sh 2>&1 | tee experiment_$(date +%Y%m%d_%H%M%S).log
#================================================================

# 配置
TESTCASE_FILE="../../testcases/strict34.txt"
OUTPUT_DIR="load_balance_results"
RESULT_FILE="${OUTPUT_DIR}/detailed_results.csv"

# 檢查測試案例檔案
if [[ ! -f "$TESTCASE_FILE" ]]; then
    echo "Error: Testcase file $TESTCASE_FILE not found!"
    exit 1
fi

# 讀取並顯示測試參數
echo "=========================================="
echo " Reading Testcase Parameters"
echo "=========================================="
cat "$TESTCASE_FILE"
echo ""

TESTCASE=$(cat "$TESTCASE_FILE")
if [[ -z "$TESTCASE" ]]; then
    echo "Error: Failed to read testcase parameters!"
    exit 1
fi

echo "Testcase parameters: $TESTCASE"
echo ""

# 建立輸出目錄
mkdir -p $OUTPUT_DIR

# 初始化 CSV 檔案標頭
echo "Program,Strategy,Total_Time,Compute_Time,Compute_Pct,Sync_Comm_Time,Sync_Comm_Pct,IO_Time,IO_Pct,Imbalance_Pct,Parallel_Eff,Type" > $RESULT_FILE

echo "=========================================="
echo " Step 1: Compilation (via make)"
echo "=========================================="

# 清理並編譯所有目標
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

# 取得已編譯的 hw2a 執行檔列表
hw2a_targets=$(make list 2>/dev/null | grep "hw2a targets:" | cut -d':' -f2)

for exe in $hw2a_targets; do
    if [ -x "$exe" ]; then
        echo ""
        echo "Testing: $exe"
        echo "---"
        
        # 提取策略名稱 (e.g., hw2a_pthread_dynamic_chunk1 -> pthread_dynamic_chunk1)
        strategy=$(echo $exe | sed 's/hw2a_//')
        
        OUTPUT_PNG="${OUTPUT_DIR}/${exe}_output.png"
        LOG_FILE="${OUTPUT_DIR}/${exe}.log"
        
        # 執行程式並記錄輸出
        srun -n1 -c12 ./$exe $OUTPUT_PNG $TESTCASE > $LOG_FILE 2>&1
        
        # 解析效能指標
        total_time=$(grep "Total Time:" $LOG_FILE | awk '{print $3}')
        compute_time=$(grep "Compute:" $LOG_FILE | awk '{print $2}')
        compute_pct=$(grep "Compute:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        comm_time=$(grep "Communication:" $LOG_FILE | awk '{print $2}')
        comm_pct=$(grep "Communication:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        io_time=$(grep "IO:" $LOG_FILE | awk '{print $2}')
        io_pct=$(grep "IO:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        parallel_eff=$(grep "Parallel Efficiency:" $LOG_FILE | awk '{print $3}' | tr -d '%')
        imbalance=$(grep "Thread Imbalance:" $LOG_FILE | awk '{print $3}' | tr -d '%')
        
        echo "Total: ${total_time}s, Compute: ${compute_time}s (${compute_pct}%), Comm: ${comm_time}s, Imbalance: ${imbalance}%"
        
        # 寫入 CSV
        echo "$exe,$strategy,$total_time,$compute_time,$compute_pct,$comm_time,$comm_pct,$io_time,$io_pct,$imbalance,$parallel_eff,hw2a" >> $RESULT_FILE
    fi
done

# 取得已編譯的 hw2b 執行檔列表
hw2b_targets=$(make list 2>/dev/null | grep "hw2b targets:" | cut -d':' -f2)

for exe in $hw2b_targets; do
    if [ -x "$exe" ]; then
        echo ""
        echo "Testing: $exe (MPI: 3 processes)"
        echo "---"
        
        # 提取策略名稱
        strategy=$(echo $exe | sed 's/hw2b_//')
        
        OUTPUT_PNG="${OUTPUT_DIR}/${exe}_output.png"
        LOG_FILE="${OUTPUT_DIR}/${exe}.log"
        
        # 執行 MPI 程式
        srun -n3 -c4 ./$exe $OUTPUT_PNG $TESTCASE > $LOG_FILE 2>&1
        
        # 解析效能指標
        total_time=$(grep "Total Time:" $LOG_FILE | awk '{print $3}')
        compute_time=$(grep "Compute:" $LOG_FILE | awk '{print $2}')
        compute_pct=$(grep "Compute:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        comm_time=$(grep "Communication:" $LOG_FILE | awk '{print $2}')
        comm_pct=$(grep "Communication:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        io_time=$(grep "IO:" $LOG_FILE | awk '{print $2}')
        io_pct=$(grep "IO:" $LOG_FILE | grep -oP '\(\K[0-9.]+')
        parallel_eff=$(grep "Parallel Efficiency:" $LOG_FILE | awk '{print $3}' | tr -d '%')
        imbalance=$(grep "Thread Imbalance:" $LOG_FILE | awk '{print $3}' | tr -d '%')
        
        echo "Total: ${total_time}s, Compute: ${compute_time}s (${compute_pct}%), Comm: ${comm_time}s, Imbalance: ${imbalance}%"
        
        # 寫入 CSV
        echo "$exe,$strategy,$total_time,$compute_time,$compute_pct,$comm_time,$comm_pct,$io_time,$io_pct,$imbalance,$parallel_eff,hw2b" >> $RESULT_FILE
    fi
done

echo ""
echo "=========================================="
echo " Results Summary"
echo "=========================================="

# 顯示結果表格
column -t -s',' $RESULT_FILE

# 找出最佳策略
echo ""
echo "Best hw2a: $(tail -n +2 $RESULT_FILE | grep hw2a | sort -t',' -k3 -n | head -1 | cut -d',' -f1,2,3)"
echo "Best hw2b: $(tail -n +2 $RESULT_FILE | grep hw2b | sort -t',' -k3 -n | head -1 | cut -d',' -f1,2,3)"

echo ""
echo "✓ Done! Detailed results saved to: $RESULT_FILE"
