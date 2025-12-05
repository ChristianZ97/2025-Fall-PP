#!/bin/bash

# ==============================================================================
#  Weak Scaling Script (CUDA - OpenMP Version)
#  - Optimized for Single Process (OpenMP) + Multi-GPU
#  - Ensures distinct CPU cores are allocated for each GPU thread
# ==============================================================================

# --- Configuration ---
CSV_FILE="weak_scaling.csv"
TEMP_OUT_FILE="temp_weak_scaling.bin"
MAX_RETRIES=3

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

TESTCASE_DIR="../../testcases" # 請確認這是正確的 CUDA testcase 路徑

if [ ! -d "$TESTCASE_DIR" ]; then
    echo "Error: Testcase directory not found at $TESTCASE_DIR"
    exit 1
fi

# --- Compilation ---
echo "Compiling (CUDA)..."
make clean > /dev/null
make all
if [ $? -ne 0 ]; then echo "Make failed"; exit 1; fi

# --- Function to Run Case ---
run_case() {
    local exe=$1
    local case_file=$2
    local gpus=$3
    
    # [OpenMP Configuration Strategy]
    # 1. ntasks=1 : We always want ONE process (the ./hw3 executable).
    # 2. cpus-per-task=$gpus : 
    #    - If 1 GPU, request 1 CPU core.
    #    - If 2 GPUs, request 2 CPU cores.
    #    This ensures threads don't context switch on a single core.
    
    local ntasks=1
    local cpus=$gpus 
    
    local final_time="-1"

    for (( attempt=1; attempt<=MAX_RETRIES; attempt++ )); do
        
        # Execution Command
        # OMP_NUM_THREADS=$gpus : Explicitly tell OpenMP how many threads to spawn
        # OMP_PROC_BIND=true    : (Optional) Helps keep threads pinned to cores
        # OMP_PLACES=cores      : (Optional) Tells OpenMP to use physical cores
        
        cmd="OMP_NUM_THREADS=$gpus srun -p nvidia -N1 -n $ntasks --cpus-per-task=$cpus --gres=gpu:$gpus ./$exe \"$case_file\" \"$TEMP_OUT_FILE\" 2>&1"
        
        out=$(eval "$cmd")
        
        # Parse Result
        time_ms=$(echo "$out" | grep "\[PROF_RESULT\]" | cut -d',' -f2)
        
        # Fallback parsing
        if [ -z "$time_ms" ]; then
            time_ms=$(echo "$out" | grep "Total time" | awk -F': ' '{print $2}' | awk '{print $1}')
        fi

        if [[ ! -z "$time_ms" ]]; then
            final_time=$time_ms
            break
        else
            if [ $attempt -lt $MAX_RETRIES ]; then
                >&2 echo "   [Retry $attempt] Failed to get time... (Output: ${out:0:50}...)"
                sleep 1
            fi
        fi
    done

    echo "$final_time"
}

# --- Main Experiment Loop ---
for pair in "${PAIRS[@]}"; do
    read c1 c2 <<< "$pair"
    
    f1="$TESTCASE_DIR/$c1"
    f2="$TESTCASE_DIR/$c2"
    
    if [[ ! -f "$f1" || ! -f "$f2" ]]; then
        echo "Skipping pair $c1 vs $c2 (File not found)"
        continue
    fi
    
    # Get V size
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
    
    # Calculate Metrics
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

# Cleanup
rm -f "$TEMP_OUT_FILE"
echo "============================================================"
echo "Done. Results saved to $CSV_FILE"
