import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 讀取數據
file_path = './ablation_perf_summary.csv'
df = pd.read_csv(file_path)

# 設定 Final 為基準
baseline_time = df[df['Version'] == 'hw3-2_final']['GrandTotalTime_ms'].values[0]

# 計算 Slowdown (相對於 Final)
df['Slowdown'] = df['GrandTotalTime_ms'] / baseline_time
df['Label'] = df['AblationTarget'] + '\n(' + df['Version'] + ')'

# 排序，讓最慢的在最上面
df = df.sort_values('GrandTotalTime_ms', ascending=False)

# 2. 畫圖
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 畫水平條形圖
colors = ['#d62728' if 'padding' in v else '#1f77b4' for v in df['Version']] # Padding 用紅色凸顯
ax = sns.barplot(x='GrandTotalTime_ms', y='Label', data=df, palette=colors)

# 標註數值
for i, p in enumerate(ax.patches):
    width = p.get_width()
    slowdown = df.iloc[i]['Slowdown']
    
    # 在條形圖右側標註時間和倍率
    ax.text(width + 10000, p.get_y() + p.get_height() / 2,
            f'{width:,.0f} ms\n({slowdown:.2f}x Time)',
            va='center', fontsize=12, fontweight='bold')

plt.title('Ablation Study: Impact of Optimizations', fontsize=16)
plt.xlabel('Grand Total Time (ms)', fontsize=14)
plt.ylabel('Configuration', fontsize=14)
plt.xlim(0, df['GrandTotalTime_ms'].max() * 1.2) # 留點空間給文字

plt.tight_layout()
plt.savefig('ablation_perf_bar.png', dpi=300)
plt.show()
