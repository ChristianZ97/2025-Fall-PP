#!/bin/bash

# --- Configuration ---
OUTPUT_CSV="unroll_results.csv"
TEMP_OUT_FILE="temp_unroll_output.bin"
NVPROF_LOG="temp_nvprof.csv" # 暫存 nvprof 的輸出

# --- Check for Testcase Argument ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <testcase_path>"
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
echo "Version,UnrollFactor,TotalTime_ms,ComputeTime_ms,Phase3_ms,gld_throughput_GBs,gst_throughput_GBs,sm_efficiency_pct,achieved_occupancy" > "$OUTPUT_CSV"

# --- Step 3: Run Experiments ---
EXECUTABLES=($(find . -maxdepth 1 -type f -executable -name "hw3-2_unroll_*" | sort))

echo "Step 3: Running experiments on testcase: $(basename $TESTCASE)"
echo ""

for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    unroll_factor=$(echo "$exe_name" | sed 's/hw3-2_unroll_//')
    
    echo "------------------------------------------"
    echo "Testing: $exe_name (Unroll Factor: $unroll_factor)"
    
    # 關鍵修改 1: 使用 --csv 並將 log 導向到獨立檔案，避免混淆
    # 關鍵修改 2: --log-file 強制 nvprof 寫入檔案，不干擾 stderr
    SRUN_CMD="srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --csv --log-file $NVPROF_LOG --metrics gld_throughput,gst_throughput,sm_efficiency,achieved_occupancy"
    
    # 執行程式，並抓取 stderr (現在裡面會有我們的 [PROF_RESULT])
    # 這裡只抓程式本身的輸出
    app_output=$($SRUN_CMD ./$exe_name "$TESTCASE" "$TEMP_OUT_FILE" 2>&1)
    
    # --- Parse Program Output (Time) ---
    # 直接找 [PROF_RESULT] 這一行，用逗號分隔
    prof_line=$(echo "$app_output" | grep "\[PROF_RESULT\]")
    
    if [[ -z "$prof_line" ]]; then
        echo "Warning: Could not find [PROF_RESULT] tag in output. Did the program run?"
        total_time=0
        compute_time=0
        phase3_time=0
    else
        # cut -d',' -f2 對應 TotalTime, f3 對應 ComputeTime, f7 對應 Phase3
        total_time=$(echo "$prof_line" | cut -d',' -f2)
        compute_time=$(echo "$prof_line" | cut -d',' -f3)
        phase3_time=$(echo "$prof_line" | cut -d',' -f7)
    fi

    # --- Parse nvprof Output (Metrics) ---
    # nvprof 的 CSV 格式通常是:
    # "Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"
    # 我們只需要 "kernel_phase3" 的那幾行
    
    if [ -f "$NVPROF_LOG" ]; then
        # 抓取 kernel_phase3 相關的 metrics
        # awk -F',' '{print $NF}' 抓取最後一欄 (Avg)
        # tr -d '"' 去除引號
        
        gld=$(grep "kernel_phase3" "$NVPROF_LOG" | grep "gld_throughput" | awk -F',' '{print $NF}' | tr -d '"' | sed 's/GB\/s//')
        gst=$(grep "kernel_phase3" "$NVPROF_LOG" | grep "gst_throughput" | awk -F',' '{print $NF}' | tr -d '"' | sed 's/GB\/s//')
        eff=$(grep "kernel_phase3" "$NVPROF_LOG" | grep "sm_efficiency" | awk -F',' '{print $NF}' | tr -d '"' | sed 's/%//')
        occ=$(grep "kernel_phase3" "$NVPROF_LOG" | grep "achieved_occupancy" | awk -F',' '{print $NF}' | tr -d '"')
    else
        gld=0; gst=0; eff=0; occ=0
    fi
    
    # Sanitize defaults
    gld=${gld:-0}; gst=${gst:-0}; eff=${eff:-0}; occ=${occ:-0}
    
    echo " -> Time: Total=${total_time}ms, Compute=${compute_time}ms, Phase3=${phase3_time}ms"
    echo " -> Metrics: GLD=${gld} GB/s, Eff=${eff}%"
    
    # Append to CSV
    echo "$exe_name,$unroll_factor,$total_time,$compute_time,$phase3_time,$gld,$gst,$eff,$occ" >> "$OUTPUT_CSV"

done

# Cleanup
rm -f "$TEMP_OUT_FILE" "$NVPROF_LOG"
echo "------------------------------------------"
echo "All unroll experiments completed! Results saved to $OUTPUT_CSV"
