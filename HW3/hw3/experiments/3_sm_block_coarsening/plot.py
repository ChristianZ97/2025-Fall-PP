import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 1. 讀取數據
file_path = './matrix_perf_results.csv'
df = pd.read_csv(file_path)
df = df[df['TotalTime_ms'] > 0]

# 建立 Config Label
df['Configuration'] = df.apply(lambda x: f"{x['MemoryType']}-BF{x['BlockFactor']}-{'Coarse' if x['Coarsening']=='Yes' else 'NoCoarse'}", axis=1)
df_clean = df.groupby(['Testcase', 'Configuration'])['TotalTime_ms'].mean().reset_index()

# 排序 Testcase
def extract_k(testcase_name):
    match = re.search(r'p(\d+)k1', testcase_name)
    return int(match.group(1)) if match else 0

df_clean['SortKey'] = df_clean['Testcase'].apply(extract_k)
df_clean = df_clean.sort_values('SortKey')

# 2. ★★★ 定義顏色與樣式 ★★★

configs = df_clean['Configuration'].unique()
palette = {}
markers = {}

# 定義色系
# Global -> 紅色
# Shared BF64 -> 藍色
# Shared BF32 -> 綠色
# Shared BF16 -> 紫色

# 這裡列出每個色系的具體顏色 (深淺)
# [Main Color, Lighter, Darker]
colors_global = ['#d62728', '#ff9896'] # Red
colors_sm64   = ['#1f77b4', '#aec7e8'] # Blue
colors_sm32   = ['#2ca02c', '#98df8a'] # Green
colors_sm16   = ['#9467bd', '#c5b0d5'] # Purple

# 計數器，用來輪替顏色
idx_global = 0
idx_sm64 = 0
idx_sm32 = 0
idx_sm16 = 0

for conf in sorted(configs):
    if 'Global' in conf:
        # Global Group
        color = colors_global[idx_global % len(colors_global)]
        marker = 'X' # Cross
        idx_global += 1
    else:
        # Shared Memory Groups
        if 'BF64' in conf:
            color = colors_sm64[idx_sm64 % len(colors_sm64)]
            marker = 'o' # Circle
            idx_sm64 += 1
        elif 'BF32' in conf:
            color = colors_sm32[idx_sm32 % len(colors_sm32)]
            marker = 's' # Square
            idx_sm32 += 1
        elif 'BF16' in conf:
            color = colors_sm16[idx_sm16 % len(colors_sm16)]
            marker = '^' # Triangle
            idx_sm16 += 1
        else:
            color = 'black' # Fallback
            marker = '.'
    
    palette[conf] = color
    markers[conf] = marker

# 3. 畫圖
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

sns.lineplot(
    data=df_clean,
    x='Testcase',
    y='TotalTime_ms',
    hue='Configuration',
    style='Configuration',
    palette=palette,
    markers=markers,
    dashes=False,
    linewidth=2.5,
    markersize=9
)

plt.title('Total Execution Time: Global vs. Shared Memory (grouped by BF)', fontsize=16)
plt.xlabel('Testcase (Input Size)', fontsize=14)
plt.ylabel('Total Time (ms) - Log Scale', fontsize=14)
plt.yscale('log')
plt.xticks(rotation=45)

# 優化 Legend
plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

plt.savefig('matrix_perf_grouped_bf.png', dpi=300)
plt.show()
