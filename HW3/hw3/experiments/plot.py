import pandas as pd

# 讀取數據
df = pd.read_csv('hw3-2_final_prof.csv', skiprows=5)
df.columns = [c.strip().replace('"', '') for c in df.columns]

# 清理並簡化 Kernel 名稱
def clean_kernel(k):
    if 'phase1' in k: return 'Phase 1'
    if 'phase2_row' in k: return 'Phase 2 (Row)'
    if 'phase2_col' in k: return 'Phase 2 (Col)'
    if 'phase3' in k: return 'Phase 3'
    return k

df['Kernel'] = df['Kernel'].str.replace('"', '').apply(clean_kernel)
df['Metric Name'] = df['Metric Name'].str.replace('"', '')

# 需要的指標
metrics = {
    'achieved_occupancy': 'Achieved Occupancy',
    'sm_efficiency': 'SM Efficiency',
    'shared_load_throughput': 'Shared Mem Load',
    'shared_store_throughput': 'Shared Mem Store',
    'gld_throughput': 'Global Mem Load',
    'gst_throughput': 'Global Mem Store'
}

# 提取數據
for metric_key, metric_name in metrics.items():
    print(f"% --- {metric_name} ---")
    sub_df = df[df['Metric Name'] == metric_key]
    
    for stat in ['Min', 'Max', 'Avg']:
        row_str = f" & {stat}"
        for phase in ['Phase 1', 'Phase 2 (Row)', 'Phase 2 (Col)', 'Phase 3']:
            val = sub_df[sub_df['Kernel'] == phase][stat].values
            if len(val) > 0:
                row_str += f" & {val[0].strip().replace('\"', '')}"
            else:
                row_str += " & -"
        print(row_str + " \\\\")
    print("\\addlinespace")