#!/bin/bash

# Output File
CSV_FILE="weak_scaling.csv"
# Header: 增加 WorkloadRatio, ExpectedT2 (相對於T1), ActualT2/T1
echo "PairID,GPU1_Case,GPU1_V,GPU1_Time,GPU2_Case,GPU2_V,GPU2_Time,WorkloadRatio,ActualTimeRatio,Efficiency" > $CSV_FILE

# Define Pairs
PAIRS=(
"p11k1 p14k1"
"p12k1 p15k1"
"p15k1 p19k1"
"p24k1 p30k1"
)

TESTCASE_DIR="../../testcases"

# Compile
echo "Compiling..."
make all
if [ $? -ne 0 ]; then exit 1; fi

run_case() {
    exe=$1
    case_file=$2
    gpus=$3
    
    out=$(srun -p nvidia -N1 -n1 --gres=gpu:$gpus ./$exe "$case_file" /dev/null 2>&1)
    time_ms=$(echo "$out" | grep "\[PROF_RESULT\]" | cut -d',' -f2)
    
    if [ -z "$time_ms" ]; then
        time_ms=$(echo "$out" | grep "Total time" | awk -F': ' '{print $2}' | awk '{print $1}')
    fi
    
    # Fallback if failed
    if [ -z "$time_ms" ]; then echo "-1"; else echo $time_ms; fi
}

for pair in "${PAIRS[@]}"; do
    read c1 c2 <<< "$pair"
    
    f1="$TESTCASE_DIR/$c1"
    f2="$TESTCASE_DIR/$c2"
    
    echo "------------------------------------------------"
    echo "Testing Weak Scaling Pair: $c1 (1 GPU) vs $c2 (2 GPUs)"
    
    # Get V size
    v1=$(od -t d4 -N 4 "$f1" | head -n 1 | awk '{print $2}')
    v2=$(od -t d4 -N 4 "$f2" | head -n 1 | awk '{print $2}')
    
    # Run 1 GPU
    echo -n "Running 1 GPU ($c1, V=$v1)... "
    t1=$(run_case "hw3-2" "$f1" 1)
    echo "${t1} ms"
    
    # Run 2 GPUs
    echo -n "Running 2 GPUs ($c2, V=$v2)... "
    t2=$(run_case "hw3-3" "$f2" 2)
    echo "${t2} ms"
    
    if [[ "$t1" != "-1" && "$t2" != "-1" ]]; then
        # Use Python for precise calculation
        # WorkloadRatio = (V2/V1)^3
        # IdealTimeRatio = WorkloadRatio / 2
        # ActualTimeRatio = T2 / T1
        # Efficiency = IdealTimeRatio / ActualTimeRatio (Ideal = 1.0)
        
        calc_res=$(python3 -c "
v1=$v1; v2=$v2; t1=$t1; t2=$t2
workload_ratio = (v2/v1)**3
expected_time_ratio = workload_ratio / 2.0
actual_time_ratio = t2 / t1
efficiency = expected_time_ratio / actual_time_ratio
print(f'{workload_ratio:.4f},{actual_time_ratio:.4f},{efficiency:.4f}')
")
        IFS=',' read w_ratio a_ratio eff <<< "$calc_res"
    else
        w_ratio="ERR"; a_ratio="ERR"; eff="ERR"
    fi
    
    echo "Workload Ratio (V^3): $w_ratio"
    echo "Time Ratio (T2/T1): $a_ratio"
    echo "Weak Scaling Efficiency: $eff"
    
    echo "Pair_$c1_$c2,$c1,$v1,$t1,$c2,$v2,$t2,$w_ratio,$a_ratio,$eff" >> $CSV_FILE
done

echo "Done. Results in $CSV_FILE"
