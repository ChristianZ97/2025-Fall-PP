#!/usr/bin/env python3
"""
plot_unroll_results.py (所有標註都放左右側)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== 配置 ====================
# CSV 檔案的路徑
CSV_FILE = 'summary_results.csv'
# 儲存圖表的資料夾名稱
OUTPUT_DIR = 'unroll_plots'
# 建立輸出資料夾，如果已存在則不做任何事
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 設定 hw2a 和 hw2b 在圖表中的代表顏色
COLOR_HW2A = '#2E86AB'  # hw2a 藍色
COLOR_HW2B = '#A23B72'  # hw2b 紫紅色

# ==================== 讀取資料 ====================
print("讀取資料中...")
# 使用 pandas 讀取 CSV 檔案
df = pd.read_csv(CSV_FILE)

# FIX: 使用 CSV 檔案中實際的欄位名稱來建立新欄位
# 將 'Unroll_Factor' 欄位轉為整數型別，並存入新的 'Unroll' 欄位
df['Unroll'] = df['Unroll_Factor'].astype(int)
# 將 'Total_Execution_Time' 欄位轉為數值型別，並存入新的 'Total_Time' 欄位
df['Total_Time'] = pd.to_numeric(df['Total_Execution_Time'], errors='coerce')

# 根據 'Type' 欄位篩選出 hw2a 和 hw2b 的資料，並按 'Unroll' 排序
df_hw2a = df[df['Type'] == 'hw2a'].sort_values('Unroll')
df_hw2b = df[df['Type'] == 'hw2b'].sort_values('Unroll')

print(f"hw2a: {len(df_hw2a)} 點, hw2b: {len(df_hw2b)} 點")

# ==================== 單圖疊加 (雙 Y 軸) ====================
# 建立一個圖表 (figure) 和一個座標軸 (axes)，設定圖表大小
fig, ax1 = plt.subplots(figsize=(14, 8))

# --- 左側 Y 軸: Total Time (柱狀圖) ---
# 產生 hw2a 柱狀圖的 x 軸位置 (0, 1, 2, ...)
x_pos_a = np.arange(len(df_hw2a))
# 產生 hw2b 柱狀圖的 x 軸位置，稍微向右偏移以便並排顯示
x_pos_b = x_pos_a + 0.35
# 設定每個柱子的寬度
bar_width = 0.35

# 繪製 hw2a 的柱狀圖
# alpha: 透明度 (0.0-1.0)，數值越小越透明
bars_a = ax1.bar(x_pos_a, df_hw2a['Total_Time'], bar_width,
                 label='hw2a Total Time', color=COLOR_HW2A, alpha=0.6)
# 繪製 hw2b 的柱狀圖
bars_b = ax1.bar(x_pos_b, df_hw2b['Total_Time'], bar_width,
                 label='hw2b Total Time', color=COLOR_HW2B, alpha=0.6)

# 標註執行時間 (在每個柱子上方顯示數值)
for bar, val in zip(bars_a, df_hw2a['Total_Time']):
    # ax1.text(x, y, text) 在指定座標 (x, y) 放置文字
    # bar.get_x() + bar.get_width()/2: 柱子中心點的 x 座標
    # val + 2: 柱子頂端再往上 2 個單位，避免文字與柱子重疊
    # f'{val:.1f}s': 將數值格式化到小數點後一位，並加上 's'
    # ha='center': 水平對齊方式 (center, left, right)
    # va='bottom': 垂直對齊方式 (bottom, top, center)
    ax1.text(bar.get_x() + bar.get_width()/2, val + 2,
             f'{val:.1f}s', ha='center', va='bottom', fontsize=8, color=COLOR_HW2A, fontweight='bold')
for bar, val in zip(bars_b, df_hw2b['Total_Time']):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 2,
             f'{val:.1f}s', ha='center', va='bottom', fontsize=8, color=COLOR_HW2B, fontweight='bold')

# 設定 X 軸與左側 Y 軸的標籤
ax1.set_xlabel('Loop Unroll Factor', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total Execution Time (seconds)', fontsize=13, fontweight='bold', 
               color='black')

# 設定左側 Y 軸的範圍
# max_time * 1.1: 找到最大時間，並將 Y 軸上限設為其 1.1 倍，以留出空間
max_time = df['Total_Time'].max()
ax1.set_ylim(0, max_time * 1.2)

# 設定 Y 軸刻度標籤的顏色
ax1.tick_params(axis='y', labelcolor='black')
# 設定 X 軸刻度的位置 (在兩組柱子的中間)
ax1.set_xticks(x_pos_a + bar_width / 2)
# 設定 X 軸刻度顯示的文字 (使用 Unroll Factor 的值)
ax1.set_xticklabels(df_hw2a['Unroll'])
# 畫出 Y 軸的水平格線，方便對齊
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# --- 右側 Y 軸: Speedup (折線圖) ---
# 建立一個共享 X 軸但有獨立 Y 軸的新座標軸
ax2 = ax1.twinx()

# 取得 Unroll=1 時的執行時間作為計算 Speedup 的基準
baseline_hw2a = df_hw2a[df_hw2a['Unroll'] == 1]['Total_Time'].values[0]
baseline_hw2b = df_hw2b[df_hw2b['Unroll'] == 1]['Total_Time'].values[0]

# 計算 Speedup (公式: Baseline Time / Current Time)
speedup_hw2a = baseline_hw2a / df_hw2a['Total_Time']
speedup_hw2b = baseline_hw2b / df_hw2b['Total_Time']

# 折線繪製
# 設定折線圖數據點的 X 軸位置 (與 X 軸刻度對齊)
x_centers = x_pos_a + bar_width / 2

# 繪製 hw2a 的折線圖
# marker: 數據點的標記樣式 ('o' 是圓形, 's' 是方形, 'x', '^', ...)
# markersize: 標記的大小
# linewidth: 線條的寬度
# alpha: 透明度 (0.0-1.0)
# linestyle: 線條樣式 ('-' 是實線, '--' 是虛線, ':' 是點線)
line_a = ax2.plot(x_centers, speedup_hw2a, marker='o', markersize=5,
                  linewidth=2.0, color=COLOR_HW2A, label='hw2a Speedup',
                  linestyle='-', alpha=0.6)
# 繪製 hw2b 的折線圖
line_b = ax2.plot(x_centers, speedup_hw2b, marker='s', markersize=5,
                  linewidth=2.0, color=COLOR_HW2B, label='hw2b Speedup',
                  linestyle='-', alpha=0.6)

# **改進：hw2a 標註在上方，hw2b 在下方**
# hw2a 標註：放在點的上方
for x, y in zip(x_centers, speedup_hw2a):
    # 將 va='center' 改為 va='bottom'
    ax2.text(x, y + 0.05, f'{y:.2f}×', ha='center', va='bottom',
             fontsize=8, color=COLOR_HW2A, fontweight='bold')

# hw2b 標註：放在點的下方
for x, y in zip(x_centers, speedup_hw2b):
    # 將 va='center' 改為 va='top'
    ax2.text(x, y - 0.05, f'{y:.2f}×', ha='center', va='top',
             fontsize=8, color=COLOR_HW2B, fontweight='bold')

# 設定右側 Y 軸的標籤
ax2.set_ylabel('Speedup (T₁ / Tₙ)', fontsize=13, fontweight='bold', color='black')

# 設定右側 Y 軸的範圍
# max_speedup * 1.1: 找到最大 Speedup，並將 Y 軸上限設為其 1.1 倍
max_speedup = max(speedup_hw2a.max(), speedup_hw2b.max())
# 將 Y 軸下限設為 1，因為 Speedup 的基準是 1
ax2.set_ylim(0.9, max_speedup * 1.1)

# 設定右側 Y 軸刻度標籤的顏色
ax2.tick_params(axis='y', labelcolor='black')
# 在 Y=1.0 的位置畫一條水平參考線
ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# --- 合併圖例 ---
# 取得柱狀圖和折線圖的圖例控制代碼
bars_handles = [bars_a, bars_b]
lines_handles = [line_a[0], line_b[0]]
# 合併所有圖例控制代碼和對應的標籤
all_handles = bars_handles + lines_handles
all_labels = ['hw2a Time', 'hw2b Time', 'hw2a Speedup', 'hw2b Speedup']

# 顯示合併後的圖例
# loc: 圖例的位置 ('upper right', 'upper left', 'lower left', 'lower right', 'best')
ax1.legend(all_handles, all_labels, loc='upper right', fontsize=10, framealpha=0.95)

# --- 標題 ---
# 設定整個圖表的標題
plt.title('Loop Unrolling: Execution Time vs Speedup', 
          fontsize=15, fontweight='bold', pad=20)

# 自動調整佈局以避免標籤重疊
plt.tight_layout()
# 儲存圖表為 PNG 檔案
# dpi: 解析度, bbox_inches='tight': 盡可能減少白邊
plt.savefig(f'{OUTPUT_DIR}/unroll_combined.png', dpi=300, bbox_inches='tight')
print(f"\n✓ 已儲存: {OUTPUT_DIR}/unroll_combined.png")
# 關閉圖表以釋放記憶體
plt.close()

# ==================== 統計摘要 ====================
print("\n" + "="*70)
print("統計摘要")
print("="*70)

print("\nhw2a (pthread):")
print(f"  Baseline (unroll=1): {baseline_hw2a:.2f}s")
# .loc[df_hw2a['Total_Time'].idxmin()] 找到時間最短的那一列
print(f"  最佳 (unroll={df_hw2a.loc[df_hw2a['Total_Time'].idxmin(), 'Unroll']:.0f}):     "
      f"{df_hw2a['Total_Time'].min():.2f}s  →  {speedup_hw2a.max():.2f}x speedup")

print("\nhw2b (MPI+OpenMP):")
print(f"  Baseline (unroll=1): {baseline_hw2b:.2f}s")
print(f"  最佳 (unroll={df_hw2b.loc[df_hw2b['Total_Time'].idxmin(), 'Unroll']:.0f}):     "
      f"{df_hw2b['Total_Time'].min():.2f}s  →  {speedup_hw2b.max():.2f}x speedup")

print("="*70)
