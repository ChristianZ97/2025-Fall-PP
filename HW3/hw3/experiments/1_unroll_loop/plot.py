import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 讀取數據
file_path = './unroll_perf_summary.csv'
df = pd.read_csv(file_path)

# 2. 資料處理
# 分離數值型 Factor (4, 8, 16, 32, 64) 和非數值型 (default, none)
df_numeric = df[pd.to_numeric(df['UnrollFactor'], errors='coerce').notnull()].copy()
df_numeric['UnrollFactor'] = df_numeric['UnrollFactor'].astype(int)
df_numeric = df_numeric.sort_values('UnrollFactor')

# 取得 Default 和 None 的時間作為基準線
time_default = df[df['UnrollFactor'] == 'default']['GrandTotalTime_ms'].values[0]
time_none = df[df['UnrollFactor'] == 'none']['GrandTotalTime_ms'].values[0]
time_best = df_numeric['GrandTotalTime_ms'].min()

# 3. 畫圖
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 畫數值型 Factor 的折線圖
plt.plot(df_numeric['UnrollFactor'], df_numeric['GrandTotalTime_ms'], 
         marker='o', markersize=10, linewidth=3, color='#1f77b4', label='Unroll Factor')

# 畫基準線
plt.axhline(y=time_default, color='green', linestyle='--', label=f'Default (#pragma unroll) : {time_default:,.0f} ms')
plt.axhline(y=time_none, color='red', linestyle='--', label=f'No Unroll : {time_none:,.0f} ms')

# 標出最佳點
best_factor = df_numeric.loc[df_numeric['GrandTotalTime_ms'].idxmin()]['UnrollFactor']
plt.plot(best_factor, time_best, marker='*', markersize=20, color='gold', markeredgecolor='black', label=f'Best (#pragma unroll {best_factor})')

# 標註每個點的數值
for x, y in zip(df_numeric['UnrollFactor'], df_numeric['GrandTotalTime_ms']):
    plt.text(x, y + 5000, f'{y:,.0f}', ha='center', fontsize=10)

plt.title('Unroll Factor Parameter Sweep\n(Lower is Better)', fontsize=16)
plt.xlabel('Unroll Factor', fontsize=14)
plt.ylabel('Grand Total Time (ms)', fontsize=14)
plt.xscale('log', base=2) # 因為 Factor 是 4, 8, 16... 用 Log2 scale 比較好看
plt.xticks([4, 8, 16, 32, 64], [4, 8, 16, 32, 64], fontsize=12)

# 調整 Y 軸範圍讓圖好看一點
y_min = df['GrandTotalTime_ms'].min()
y_max = df['GrandTotalTime_ms'].max()
plt.ylim(y_min * 0.98, y_max * 1.02)

plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig('unroll_perf_sweep.png', dpi=300)
plt.show()
