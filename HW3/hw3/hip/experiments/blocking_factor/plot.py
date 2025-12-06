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
fig, ax1 = plt.subplots(figsize=(12, 7))

# 顏色定義
color_gops = '#546E7A'
color_sh_ld = '#1565C0'
color_sh_st = '#64B5F6'
color_gl_ld = '#2E7D32'
color_gl_st = '#81C784'

# X 軸 Log2 Scale
x_pos = np.log2(df['BF']) 
bar_width = 0.3 

# --- 左軸：Integer GOPS (柱狀圖) ---
bars = ax1.bar(x_pos, df['GOPS'], color=color_gops, alpha=0.6, width=bar_width, label='Integer GOPS')

ax1.set_xlabel('Blocking Factor', fontsize=14, labelpad=10, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df['BF'].values)
ax1.set_xlim(x_pos.min() - 0.5, x_pos.max() + 0.5)

ax1.set_ylabel('Integer GOPS', fontsize=14, color=color_gops, fontweight='bold', labelpad=15)
ax1.tick_params(axis='y', labelcolor=color_gops, width=2)
ax1.spines['left'].set_color(color_gops)
ax1.spines['left'].set_linewidth(2)

# ★★★ 設定左軸 Range：拉高上限，讓 Bar 變矮，留出上方空間給折線 ★★★
gops_max = df['GOPS'].max()
ax1.set_ylim(0, gops_max * 1.7) # 設為最大值的 1.6 倍

# 標註 GOPS 數值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}', ha='center', va='bottom', color=color_gops, fontweight='bold', fontsize=12)

# --- 右軸：Bandwidth (折線圖) ---
ax2 = ax1.twinx()

l1, = ax2.plot(x_pos, df['shld_GBs'], marker='o', color=color_sh_ld, linewidth=3, markersize=9, label='Shared Load')
l2, = ax2.plot(x_pos, df['shst_GBs'], marker='o', color=color_sh_st, linewidth=2, markersize=7, linestyle='--', label='Shared Store')
l3, = ax2.plot(x_pos, df['gld_GBs'], marker='s', color=color_gl_ld, linewidth=3, markersize=9, label='Global Load')
l4, = ax2.plot(x_pos, df['gst_GBs'], marker='s', color=color_gl_st, linewidth=2, markersize=7, linestyle='--', label='Global Store')

ax2.set_ylabel('Throughput (GB/s)', fontsize=14, color=color_sh_ld, fontweight='bold', labelpad=15)
ax2.tick_params(axis='y', labelcolor=color_sh_ld, width=2)
ax2.spines['right'].set_color(color_sh_ld)
ax2.spines['right'].set_linewidth(2)
ax2.spines['left'].set_visible(False)

# ★★★ 設定右軸 Range：如果需要也可以微調，通常 0 開始即可 ★★★
bandwidth_max = df['shld_GBs'].max()
ax2.set_ylim(0, bandwidth_max * 1.1)

# 合併 Legend
lines = [bars, l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
legend = ax1.legend(lines, labels, loc='upper left', fontsize=11, frameon=True, fancybox=True, framealpha=0.9)
legend.get_frame().set_edgecolor('#CCCCCC')

plt.title('Performance vs. Bandwidth Analysis', fontsize=18, pad=20, fontweight='bold')
plt.grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('bf_perf_bandwidth_final.png', dpi=300)
plt.show()
