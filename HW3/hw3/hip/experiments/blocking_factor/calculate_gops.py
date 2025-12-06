import pandas as pd
import sys
import math

def calculate_gops(row):
    """根據 V 和 ComputeTime_ms (或 TotalTime_ms) 計算 GOPS。"""
    V = row['V']
    
    # 優先嘗試讀取 ComputeTime_ms
    compute_time_ms = row['ComputeTime_ms']
    
    # 如果 ComputeTime_ms 是 NaN (空值)，則改用 TotalTime_ms
    if pd.isna(compute_time_ms):
        compute_time_ms = row['TotalTime_ms']
    
    # 如果還是 NaN 或 0，無法計算
    if pd.isna(compute_time_ms) or compute_time_ms == 0:
        return 0.0

    # 換算成秒
    compute_time_s = compute_time_ms / 1000.0

    # Floyd-Warshall 的總運算量 (Integer Ops)
    total_ops = 2 * (V ** 3)

    # GOPS = (總運算量 / 秒數) / 10^9
    gops = (total_ops / compute_time_s) / 1e9

    return gops

def fill_compute_time(row):
    """如果 ComputeTime_ms 是空的，就填入 TotalTime_ms 的值。"""
    if pd.isna(row['ComputeTime_ms']):
        return row['TotalTime_ms']
    return row['ComputeTime_ms']

def main():
    default_filename = "bf_perf_results.csv"
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
        csv_filename = default_filename
        print(f"Using default filename: '{csv_filename}'")

    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        return

    print("Processing...")
    
    # 1. 補齊 ComputeTime_ms
    df['ComputeTime_ms'] = df.apply(fill_compute_time, axis=1)

    # 2. 計算 GOPS
    df['GOPS'] = df.apply(calculate_gops, axis=1)

    # 3. 格式化 (小數點後 2 位)
    df['GOPS'] = df['GOPS'].round(2)

    # 4. 寫回檔案
    df.to_csv(csv_filename, index=False)

    print("\n--- CSV Updated Successfully! ---")
    print(df.to_string())

if __name__ == "__main__":
    main()
