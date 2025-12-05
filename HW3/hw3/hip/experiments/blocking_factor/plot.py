import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 讀取數據
file_path = './bf_perf_results.csv'
df = pd.read_csv(file_path)
df['BF'] = df['BF'].astype(int)
df = df.sort_values('BF')

# 2. 設定畫布
sns.set_context("notebook", font_scale=1.2)
sns.set_style("ticks")
fig, ax1 = plt.subplots(figsize=(10, 6)) # 稍微調整寬高比，單一圖表不需要那麼寬

# 顏色定義 (保留你喜歡的配色)
color_gops = '#546E7A'

# X 軸 Log2 Scale 設定 (讓 32, 64 等間距)
x_pos = np.log2(df['BF']) 
bar_width = 0.5 # 稍微加寬柱狀，因為沒有折線干擾了

# --- 左軸：Integer GOPS (柱狀圖) ---
bars = ax1.bar(x_pos, df['GOPS'], color=color_gops, alpha=0.7, width=bar_width, label='Integer GOPS')

# 設定 X 軸
ax1.set_xlabel('Blocking Factor', fontsize=14, labelpad=10, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df['BF'].values)
ax1.set_xlim(x_pos.min() - 0.6, x_pos.max() + 0.6)

# 設定 Y 軸 (GOPS)
ax1.set_ylabel('Integer GOPS', fontsize=14, color=color_gops, fontweight='bold', labelpad=15)
ax1.tick_params(axis='y', labelcolor=color_gops, width=2)
ax1.spines['left'].set_color(color_gops)
ax1.spines['left'].set_linewidth(2)
ax1.spines['right'].set_visible(False) # 移除右軸邊框
ax1.spines['top'].set_visible(False)   # 移除上軸邊框

# 設定 Y 軸 Range (留一點空間給上面的文字)
gops_max = df['GOPS'].max()
ax1.set_ylim(0, gops_max * 1.15)

# 標註 GOPS 數值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + (gops_max * 0.01),
             f'{height:.2f}', ha='center', va='bottom', color=color_gops, fontweight='bold', fontsize=12)

# 設定標題與網格
plt.title('Performance Analysis (Blocking Factor vs. GOPS)', fontsize=16, pad=20, fontweight='bold')
plt.grid(True, axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('bf_perf_gops_only.png', dpi=300)
plt.show()
