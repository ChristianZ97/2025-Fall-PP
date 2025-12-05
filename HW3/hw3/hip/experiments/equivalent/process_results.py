import pandas as pd
import glob
import os

# --- AMD Logic (Unchanged) ---
def process_amd_log(df):
    """ Parses rocprof CSV (AMD) - Wide Format """
    results = {}
    df.columns = df.columns.str.strip()
    
    if 'KernelName' not in df.columns:
        return {}
        
    kernels = df.groupby('KernelName')
    
    for name, group in kernels:
        simple_name = name.split('<')[0]
        
        # 1. Duration
        if 'DurationNs' in group.columns:
            total_duration_ns = group['DurationNs'].sum()
            kernel_duration_s = total_duration_ns / 1e9
        elif 'GRBM_GUI_ACTIVE' in group.columns:
            total_cycles = group['GRBM_GUI_ACTIVE'].sum()
            kernel_duration_s = total_cycles / (1.5 * 1e9)
        else:
            kernel_duration_s = 0

        # 2. Data
        fetch_kb = group['FETCH_SIZE'].sum() if 'FETCH_SIZE' in group.columns else 0
        write_kb = group['WRITE_SIZE'].sum() if 'WRITE_SIZE' in group.columns else 0
        fetch_bytes = fetch_kb * 1024
        write_bytes = write_kb * 1024
        
        # 3. Throughput
        gld_gbs = (fetch_bytes / kernel_duration_s) / 1e9 if kernel_duration_s > 0 else 0
        gst_gbs = (write_bytes / kernel_duration_s) / 1e9 if kernel_duration_s > 0 else 0
        
        # 4. Waves
        waves = group['SQ_WAVES'].sum() if 'SQ_WAVES' in group.columns else 0
        
        results[simple_name] = {
            'Duration (ms)': f"{kernel_duration_s * 1000:.2f}",
            'Global Load (GB/s)': f"{gld_gbs:.2f}",
            'Global Store (GB/s)': f"{gst_gbs:.2f}",
            'Shared Load (GB/s)': "N/A",
            'Shared Store (GB/s)': "N/A",
            'Occupancy': f"{int(waves)} Waves"
        }
    return results

# --- NVIDIA Logic (Updated for correct parsing) ---
def process_nvidia_log(filepath):
    """ Parses nvprof CSV (NVIDIA) - Long Format with Units """
    try:
        # Read checking for the correct header line
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        header_row = 0
        for i, line in enumerate(lines):
            if '"Device","Kernel"' in line or '"Metric Name"' in line:
                header_row = i
                break
        
        df = pd.read_csv(filepath, skiprows=header_row)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Dictionary to hold aggregated results per kernel
    kernel_metrics = {}

    # Iterate through rows
    for _, row in df.iterrows():
        kernel_full = row['Kernel']
        metric_name = row['Metric Name']
        avg_val_str = str(row['Avg']) # e.g., "121.34GB/s"
        
        # Simplify Kernel Name
        simple_name = kernel_full.split('(')[0]
        
        if simple_name not in kernel_metrics:
            kernel_metrics[simple_name] = {}

        # Parse Value (Remove Units)
        # Units can be GB/s, %, etc.
        val_clean = avg_val_str.replace('GB/s', '').replace('%', '').strip()
        try:
            val = float(val_clean)
        except ValueError:
            val = 0.0
            
        kernel_metrics[simple_name][metric_name] = val

    # Format results for display
    results = {}
    for kname, metrics in kernel_metrics.items():
        results[kname] = {
            'Duration (ms)': "See Log",
            'Global Load (GB/s)': f"{metrics.get('gld_throughput', 0):.2f}",
            'Global Store (GB/s)': f"{metrics.get('gst_throughput', 0):.2f}",
            'Shared Load (GB/s)': f"{metrics.get('shared_load_throughput', 0):.2f}",
            'Shared Store (GB/s)': f"{metrics.get('shared_store_throughput', 0):.2f}",
            'Occupancy': f"{metrics.get('achieved_occupancy', 0):.4f}"
        }
        
    return results

def main():
    # 1. Process AMD Logs
    print("========== AMD ROCm Analysis ==========")
    amd_logs = glob.glob("rocprof_*.csv")
    for log in amd_logs:
        print(f"\n--- {log} ---")
        try:
            df = pd.read_csv(log)
            data = process_amd_log(df)
            if data:
                print(pd.DataFrame.from_dict(data, orient='index').to_string())
        except Exception as e:
            print(f"Skipping {log}: {e}")

    # 2. Process NVIDIA Logs
    print("\n\n========== NVIDIA CUDA Analysis ==========")
    nv_logs = glob.glob("nvprof_*.csv")
    for log in nv_logs:
        print(f"\n--- {log} ---")
        data = process_nvidia_log(log)
        if data:
            print(pd.DataFrame.from_dict(data, orient='index').to_string())
        else:
            print("No data found or parse error.")

if __name__ == "__main__":
    main()
