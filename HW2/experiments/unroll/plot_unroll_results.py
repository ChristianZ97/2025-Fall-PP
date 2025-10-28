#!/usr/bin/env python3
"""
plot_unroll_results.py (所有標註都放左右側)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== 配置 ====================
CSV_FILE = 'unroll_results/detailed_results.csv'
OUTPUT_DIR = 'unroll_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_HW2A = '#2E86AB'  # hw2a 藍色
COLOR_HW2B = '#A23B72'  # hw2b 紫紅色

# ==================== 讀取資料 ====================
print("讀取資料中...")
df = pd.read_csv(CSV_FILE)

df['Unroll'] = df['Unroll'].astype(int)
df['Total_Time'] = pd.to_numeric(df['Total_Time'], errors='coerce')

df_hw2a = df[df['Type'] == 'hw2a'].sort_values('Unroll')
df_hw2b = df[df['Type'] == 'hw2b'].sort_values('Unroll')

print(f"hw2a: {len(df_hw2a)} 點, hw2b: {len(df_hw2b)} 點")

# ==================== 單圖疊加 (雙 Y 軸) ====================
fig, ax1 = plt.subplots(figsize=(14, 8))

# --- 左側 Y 軸: Total Time (柱狀圖) ---
x_pos_a = np.arange(len(df_hw2a))
x_pos_b = x_pos_a + 0.35
bar_width = 0.35

bars_a = ax1.bar(x_pos_a, df_hw2a['Total_Time'], bar_width,
                 label='hw2a Total Time', color=COLOR_HW2A, alpha=0.6)
bars_b = ax1.bar(x_pos_b, df_hw2b['Total_Time'], bar_width,
                 label='hw2b Total Time', color=COLOR_HW2B, alpha=0.6)

# 標註執行時間
for bar, val in zip(bars_a, df_hw2a['Total_Time']):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 2,
             f'{val:.1f}s', ha='center', va='bottom', fontsize=8, color=COLOR_HW2A)
for bar, val in zip(bars_b, df_hw2b['Total_Time']):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 2,
             f'{val:.1f}s', ha='center', va='bottom', fontsize=8, color=COLOR_HW2B)

ax1.set_xlabel('Loop Unroll Factor', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total Execution Time (seconds)', fontsize=13, fontweight='bold', 
               color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(x_pos_a + bar_width / 2)
ax1.set_xticklabels(df_hw2a['Unroll'])
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# --- 右側 Y 軸: Speedup (折線圖) ---
ax2 = ax1.twinx()

baseline_hw2a = df_hw2a[df_hw2a['Unroll'] == 1]['Total_Time'].values[0]
baseline_hw2b = df_hw2b[df_hw2b['Unroll'] == 1]['Total_Time'].values[0]

speedup_hw2a = baseline_hw2a / df_hw2a['Total_Time']
speedup_hw2b = baseline_hw2b / df_hw2b['Total_Time']

# 折線繪製
x_centers = x_pos_a + bar_width / 2

line_a = ax2.plot(x_centers, speedup_hw2a, marker='o', markersize=8,
                  linewidth=2.5, color=COLOR_HW2A, label='hw2a Speedup', 
                  linestyle='-', alpha=1.0)
line_b = ax2.plot(x_centers, speedup_hw2b, marker='s', markersize=8,
                  linewidth=2.5, color=COLOR_HW2B, label='hw2b Speedup',
                  linestyle='-', alpha=1.0)

# **改進：所有標註都放在左右側**
# hw2a 標註：全部放在點的右側
for x, y in zip(x_centers, speedup_hw2a):
    ax2.text(x + 0.12, y, f'{y:.2f}×', ha='left', va='center',
             fontsize=9, color=COLOR_HW2A, fontweight='bold')

# hw2b 標註：全部放在點的左側
for x, y in zip(x_centers, speedup_hw2b):
    ax2.text(x - 0.12, y, f'{y:.2f}×', ha='right', va='center',
             fontsize=9, color=COLOR_HW2B, fontweight='bold')

ax2.set_ylabel('Speedup (T₁ / Tₙ)', fontsize=13, fontweight='bold', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# --- 合併圖例 ---
bars_handles = [bars_a, bars_b]
lines_handles = [line_a[0], line_b[0]]
all_handles = bars_handles + lines_handles
all_labels = ['hw2a Time', 'hw2b Time', 'hw2a Speedup', 'hw2b Speedup']

ax1.legend(all_handles, all_labels, loc='upper right', fontsize=10, framealpha=0.95)

# --- 標題 ---
plt.title('Loop Unrolling: Execution Time vs Speedup', 
          fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/unroll_combined.png', dpi=300, bbox_inches='tight')
print(f"\n✓ 已儲存: {OUTPUT_DIR}/unroll_combined.png")
plt.close()

# ==================== 統計摘要 ====================
print("\n" + "="*70)
print("統計摘要")
print("="*70)

print("\nhw2a (pthread):")
print(f"  Baseline (unroll=1): {baseline_hw2a:.2f}s")
print(f"  最佳 (unroll={df_hw2a.loc[df_hw2a['Total_Time'].idxmin(), 'Unroll']:.0f}):     "
      f"{df_hw2a['Total_Time'].min():.2f}s  →  {speedup_hw2a.max():.2f}x speedup")

print("\nhw2b (MPI+OpenMP):")
print(f"  Baseline (unroll=1): {baseline_hw2b:.2f}s")
print(f"  最佳 (unroll={df_hw2b.loc[df_hw2b['Total_Time'].idxmin(), 'Unroll']:.0f}):     "
      f"{df_hw2b['Total_Time'].min():.2f}s  →  {speedup_hw2b.max():.2f}x speedup")

print("="*70)
