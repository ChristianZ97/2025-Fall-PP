import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 1. 讀取數據
file_path = './register_perf_results.csv'
df = pd.read_csv(file_path)
df = df[df['TotalTime_ms'] > 0]

# 排序 Testcase
def extract_k(testcase_name):
    match = re.search(r'p(\d+)k1', testcase_name)
    return int(match.group(1)) if match else 0

df['SortKey'] = df['Testcase'].apply(extract_k)
df = df.sort_values('SortKey')

# 2. 畫圖
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# 定義顏色
palette = {'off': '#7f7f7f', 'on': '#1f77b4'} # 灰 vs 藍
markers = {'off': 'X', 'on': 'o'}

sns.lineplot(
    data=df,
    x='Testcase',
    y='TotalTime_ms',
    hue='RegisterTiling',
    style='RegisterTiling',
    palette=palette,
    markers=markers,
    dashes=False,
    linewidth=2.5,
    markersize=9
)

# 3. 計算並標註提升幅度 (Speedup = Time_Off / Time_On)
df_pivot = df.pivot(index='Testcase', columns='RegisterTiling', values='TotalTime_ms')

if 'off' in df_pivot.columns and 'on' in df_pivot.columns:
    # ★ 修正：Speedup = Baseline(Off) / Optimized(On)
    df_pivot['Speedup'] = df_pivot['off'] / df_pivot['on']
    avg_speedup = df_pivot['Speedup'].mean()
    
    # 找出最大 Speedup 發生在哪個 Testcase
    max_speedup_val = df_pivot['Speedup'].max()
    max_speedup_case = df_pivot['Speedup'].idxmax()
    
    title_text = f'Register Tiling Impact: On vs Off\n(Avg Speedup: {avg_speedup:.3f}x, Max: {max_speedup_val:.3f}x @ {max_speedup_case})'
else:
    title_text = 'Register Tiling Impact: On vs Off'

plt.title(title_text, fontsize=16)
plt.xlabel('Testcase (Input Size)', fontsize=14)
plt.ylabel('Total Time (ms) - Log Scale', fontsize=14)
plt.yscale('log')
plt.xticks(rotation=45)
plt.legend(title='Register Tiling', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

plt.savefig('register_perf_lineplot.png', dpi=300)
plt.show()
