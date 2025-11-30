#!/bin/bash
# Usage: ./scan_v.sh ../../testcases

TC_DIR=$1

echo "Testcase,V,V_cubed"

for i in {1..7}; do
    file="$TC_DIR/c0${i}.1"
    if [ -f "$file" ]; then
        # Read first 4 bytes as integer (Little Endian)
        # od -t d4 -N 4 reads 4 bytes as decimal int
        # awk '{print $2}' gets the value
        V=$(od -t d4 -N 4 "$file" | head -n 1 | awk '{print $2}')
        
        # Calculate V^3 for workload comparison (using python for large numbers)
        V3=$(python3 -c "print($V**3)")
        
        echo "c0${i}.1,$V,$V3"
    fi
done

# Scan p11k1 to p40k1
for i in {11..50}; do
    file="$TC_DIR/p${i}k1"
    if [ -f "$file" ]; then
        # Read V (4 bytes int)
        V=$(od -t d4 -N 4 "$file" | head -n 1 | awk '{print $2}')
        
        # Calculate V^3
        V3=$(python3 -c "print($V**3)")
        
        echo "p${i}k1,$V,$V3"
    fi
done
