#!/bin/bash

# ==============================================================================
#  Weak Scaling Experiment Script (HIP/ROCm - OpenMP Version)
#  - Optimized for Single Process (OpenMP) + Multi-GPU
#  - Uses -n1 + --cpus-per-task strategy
# ==============================================================================

# Output File
CSV_FILE="weak_scaling.csv"
TEMP_OUT_FILE="temp_weak_scaling.bin"
MAX_RETRIES=10

# Header
echo "PairID,GPU1_Case,GPU1_V,GPU1_Time,GPU2_Case,GPU2_V,GPU2_Time,WorkloadRatio,ActualTimeRatio,Efficiency" > $CSV_FILE

# Define Pairs
PAIRS=(
"p11k1 p14k1"
"p12k1 p15k1"
"p15k1 p19k1"
"p24k1 p30k1"
"p34k1 p43k1"
"p35k1 p44k1"
"p36k1 p45k1"
)

TESTCASE_DIR="../../../testcases-amd"

if [ ! -d "$TESTCASE_DIR" ]; then
    echo "Error: Testcase directory not found at '$TESTCASE_DIR'"
    exit 1
fi

echo "Compiling (HIP)..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then echo "Make failed"; exit 1; fi

# --- Function to Run Case ---
run_case() {
    local exe=$1
    local case_file=$2
    local gpus=$3
    
    # [OpenMP Strategy for HIP]
    # Always use -n 1 (Single Process)
    # Allocate 1 CPU core per GPU thread to avoid context switching
    local ntasks=1
    local cpus=$gpus 
    
    local final_time="-1"

    for (( attempt=1; attempt<=MAX_RETRIES; attempt++ )); do
        
        # OMP_NUM_THREADS=$gpus is crucial for HIP runtime to behave correctly with OpenMP
        cmd="OMP_NUM_THREADS=$gpus srun -p amd -N1 -n $ntasks --cpus-per-task=$cpus --gres=gpu:$gpus ./$exe \"$case_file\" \"$TEMP_OUT_FILE\" 2>&1"
        
        out=$(eval "$cmd")
        
        time_ms=$(echo "$out" | grep "\[PROF_RESULT\]" | cut -d',' -f2)
        
        if [ -z "$time_ms" ]; then
            time_ms=$(echo "$out" | grep "Total time" | awk -F': ' '{print $2}' | awk '{print $1}')
        fi

        if [[ ! -z "$time_ms" ]]; then
            final_time=$time_ms
            break
        else
            if [ $attempt -lt $MAX_RETRIES ]; then
                >&2 echo "   [Retry $attempt] Failed..."
                sleep 1
            fi
        fi
    done

    echo "$final_time"
}

# --- Main Loop ---
for pair in "${PAIRS[@]}"; do
    read c1 c2 <<< "$pair"
    f1="$TESTCASE_DIR/$c1"
    f2="$TESTCASE_DIR/$c2"
    
    if [[ ! -f "$f1" || ! -f "$f2" ]]; then
        echo "Skipping $c1 vs $c2"
        continue
    fi
    
    v1=$(od -t d4 -N 4 "$f1" | head -n 1 | awk '{print $2}')
    v2=$(od -t d4 -N 4 "$f2" | head -n 1 | awk '{print $2}')
    
    echo "------------------------------------------------"
    echo "Weak Scaling: $c1 (1 GPU) vs $c2 (2 GPUs)"
    
    # Run 1 GPU
    echo -n "  Running 1 GPU ($c1)... "
    t1=$(run_case "hw3-2" "$f1" 1)
    echo "${t1} ms"
    
    # Run 2 GPUs
    echo -n "  Running 2 GPUs ($c2)... "
    t2=$(run_case "hw3-3" "$f2" 2)
    echo "${t2} ms"
    
    if [[ "$t1" != "-1" && "$t2" != "-1" ]]; then
        calc_res=$(python3 -c "
v1=$v1; v2=$v2; t1=$t1; t2=$t2
workload_ratio = (v2/v1)**3
efficiency = t1 / t2 
time_ratio = t2 / t1
print(f'{workload_ratio:.4f},{time_ratio:.4f},{efficiency:.4f}')
")
        IFS=',' read w_ratio a_ratio eff <<< "$calc_res"
    else
        w_ratio="ERR"; a_ratio="ERR"; eff="ERR"
    fi
    
    echo "  Efficiency: $eff"
    echo "Pair_$c1_$c2,$c1,$v1,$t1,$c2,$v2,$t2,$w_ratio,$a_ratio,$eff" >> $CSV_FILE
done

rm -f "$TEMP_OUT_FILE"
echo "Done. Results in $CSV_FILE"
