#!/usr/bin/env python3
"""
plot_unroll_results.py
為 loop unrolling 實驗產生視覺化圖表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== 配置 ====================
CSV_FILE = 'unroll_results/detailed_results.csv'
OUTPUT_DIR = 'unroll_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 色彩配置
COLOR_COMPUTE = '#479A40'  # 綠色 - 計算時間
COLOR_COMM = '#E68A00'     # 橘色 - 通訊時間
COLOR_IO = '#4C83C9'       # 藍色 - I/O 時間
COLOR_HW2A = '#2E86AB'     # hw2a 專用色
COLOR_HW2B = '#A23B72'     # hw2b 專用色

# ==================== 讀取資料 ====================
print("讀取資料中...")
df = pd.read_csv(CSV_FILE, sep=',')

# 資料清理
df['Unroll'] = df['Unroll'].astype(int)
df['Total_Time'] = pd.to_numeric(df['Total_Time'], errors='coerce')
df['Compute_Time'] = pd.to_numeric(df['Compute_Time'], errors='coerce')
df['IO_Time'] = pd.to_numeric(df['IO_Time'], errors='coerce')
df['Sync_Comm_Time'] = pd.to_numeric(df['Sync_Comm_Time'], errors='coerce')

# 分離 hw2a 和 hw2b
df_hw2a = df[df['Type'] == 'hw2a'].sort_values('Unroll')
df_hw2b = df[df['Type'] == 'hw2b'].sort_values('Unroll')

print(f"hw2a 資料點: {len(df_hw2a)}")
print(f"hw2b 資料點: {len(df_hw2b)}")

# ==================== 圖表 1: Total Time 比較 ====================
print("\n生成圖表 1: Total Time 比較...")
fig, ax = plt.subplots(figsize=(12, 6))

x_pos_a = np.arange(len(df_hw2a))
x_pos_b = x_pos_a + 0.35
bar_width = 0.35

bars_a = ax.bar(x_pos_a, df_hw2a['Total_Time'], bar_width,
                label='hw2a (pthread)', color=COLOR_HW2A, alpha=0.85)
bars_b = ax.bar(x_pos_b, df_hw2b['Total_Time'], bar_width,
                label='hw2b (MPI+OpenMP)', color=COLOR_HW2B, alpha=0.85)

# 標註數值
for i, (bar, val) in enumerate(zip(bars_a, df_hw2a['Total_Time'])):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.1f}s', ha='center', va='bottom', fontsize=8)
for i, (bar, val) in enumerate(zip(bars_b, df_hw2b['Total_Time'])):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.1f}s', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Loop Unroll Factor', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Execution Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Effect of Loop Unrolling on Total Execution Time', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos_a + bar_width / 2)
ax.set_xticklabels(df_hw2a['Unroll'])
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_total_time_comparison.png', dpi=300)
print(f"已儲存: {OUTPUT_DIR}/1_total_time_comparison.png")
plt.close()

# ==================== 圖表 2: Time Breakdown (堆疊圖) ====================
print("生成圖表 2: Time Breakdown 堆疊圖...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# hw2a
x_pos = np.arange(len(df_hw2a))
bars_compute = ax1.bar(x_pos, df_hw2a['Compute_Time'], 
                       label='Compute', color=COLOR_COMPUTE)
bars_io = ax1.bar(x_pos, df_hw2a['IO_Time'], 
                  bottom=df_hw2a['Compute_Time'],
                  label='I/O', color=COLOR_IO)

ax1.set_title('hw2a (pthread) - Time Breakdown', fontsize=13, fontweight='bold')
ax1.set_xlabel('Loop Unroll Factor', fontsize=11)
ax1.set_ylabel('Time (seconds)', fontsize=11)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df_hw2a['Unroll'])
ax1.legend(fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# hw2b
x_pos = np.arange(len(df_hw2b))
bars_compute = ax2.bar(x_pos, df_hw2b['Compute_Time'], 
                       label='Compute', color=COLOR_COMPUTE)
bars_comm = ax2.bar(x_pos, df_hw2b['Sync_Comm_Time'],
                    bottom=df_hw2b['Compute_Time'],
                    label='Communication', color=COLOR_COMM)
bars_io = ax2.bar(x_pos, df_hw2b['IO_Time'],
                  bottom=df_hw2b['Compute_Time'] + df_hw2b['Sync_Comm_Time'],
                  label='I/O', color=COLOR_IO)

ax2.set_title('hw2b (MPI+OpenMP) - Time Breakdown', fontsize=13, fontweight='bold')
ax2.set_xlabel('Loop Unroll Factor', fontsize=11)
ax2.set_ylabel('Time (seconds)', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df_hw2b['Unroll'])
ax2.legend(fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2_time_breakdown.png', dpi=300)
print(f"已儲存: {OUTPUT_DIR}/2_time_breakdown.png")
plt.close()

# ==================== 圖表 3: Speedup Curves ====================
print("生成圖表 3: Speedup 曲線...")
fig, ax = plt.subplots(figsize=(12, 7))

# 計算 speedup (以 unroll=1 為基準)
baseline_hw2a = df_hw2a[df_hw2a['Unroll'] == 1]['Total_Time'].values[0]
baseline_hw2b = df_hw2b[df_hw2b['Unroll'] == 1]['Total_Time'].values[0]

speedup_hw2a = baseline_hw2a / df_hw2a['Total_Time']
speedup_hw2b = baseline_hw2b / df_hw2b['Total_Time']

# 繪製曲線
ax.plot(df_hw2a['Unroll'], speedup_hw2a, marker='o', markersize=10,
        linewidth=2.5, color=COLOR_HW2A, label='hw2a (pthread)', alpha=0.85)
ax.plot(df_hw2b['Unroll'], speedup_hw2b, marker='s', markersize=10,
        linewidth=2.5, color=COLOR_HW2B, label='hw2b (MPI+OpenMP)', alpha=0.85)

# 標註數值
for x, y in zip(df_hw2a['Unroll'], speedup_hw2a):
    ax.text(x, y + 0.05, f'{y:.2f}x', ha='center', fontsize=9, color=COLOR_HW2A)
for x, y in zip(df_hw2b['Unroll'], speedup_hw2b):
    ax.text(x, y - 0.12, f'{y:.2f}x', ha='center', fontsize=9, color=COLOR_HW2B)

# 參考線 (baseline)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (unroll=1)')

ax.set_xlabel('Loop Unroll Factor', fontsize=12, fontweight='bold')
ax.set_ylabel('Speedup (T₁ / Tₙ)', fontsize=12, fontweight='bold')
ax.set_title('Speedup vs Loop Unroll Factor', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(df_hw2a['Unroll'])
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim(bottom=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3_speedup_curves.png', dpi=300)
print(f"已儲存: {OUTPUT_DIR}/3_speedup_curves.png")
plt.close()

# ==================== 圖表 4: Compute Percentage ====================
print("生成圖表 4: Compute Percentage 比較...")
fig, ax = plt.subplots(figsize=(12, 6))

x_pos_a = np.arange(len(df_hw2a))
x_pos_b = x_pos_a + 0.35

bars_a = ax.bar(x_pos_a, df_hw2a['Compute_Pct'], bar_width,
                label='hw2a (pthread)', color=COLOR_HW2A, alpha=0.85)
bars_b = ax.bar(x_pos_b, df_hw2b['Compute_Pct'], bar_width,
                label='hw2b (MPI+OpenMP)', color=COLOR_HW2B, alpha=0.85)

# 標註數值
for bar in bars_a:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
for bar in bars_b:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Loop Unroll Factor', fontsize=12, fontweight='bold')
ax.set_ylabel('Compute Time Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Compute Efficiency: Percentage of Time Spent in Computation',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos_a + bar_width / 2)
ax.set_xticklabels(df_hw2a['Unroll'])
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_ylim([85, 100])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/4_compute_percentage.png', dpi=300)
print(f"已儲存: {OUTPUT_DIR}/4_compute_percentage.png")
plt.close()

# ==================== 新圖表 5: Parallel Efficiency ====================
print("生成圖表 5: Parallel Efficiency...")
fig, ax = plt.subplots(figsize=(12, 6))

x_pos_a = np.arange(len(df_hw2a))
x_pos_b = x_pos_a + 0.35

bars_a = ax.bar(x_pos_a, df_hw2a['Parallel_Eff'], bar_width,
                label='hw2a (pthread)', color=COLOR_HW2A, alpha=0.85)
bars_b = ax.bar(x_pos_b, df_hw2b['Parallel_Eff'], bar_width,
                label='hw2b (MPI+OpenMP)', color=COLOR_HW2B, alpha=0.85)

# 標註數值
for bar in bars_a:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
for bar in bars_b:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Loop Unroll Factor', fontsize=12, fontweight='bold')
ax.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
ax.set_title('Parallel Efficiency: Compute Time / Total Time',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos_a + bar_width / 2)
ax.set_xticklabels(df_hw2a['Unroll'])
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_ylim([95, 100])  # 聚焦高效區間
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/5_parallel_efficiency.png', dpi=300)
print(f"已儲存: {OUTPUT_DIR}/5_parallel_efficiency.png")
plt.close()

# ==================== 統計摘要 ====================
print("\n" + "="*70)
print("統計摘要")
print("="*70)

print("\nhw2a (pthread) 效能:")
print(f"  Unroll=1 (baseline): {df_hw2a[df_hw2a['Unroll']==1]['Total_Time'].values[0]:.2f}s")
print(f"  最佳效能 (Unroll={df_hw2a.loc[df_hw2a['Total_Time'].idxmin(), 'Unroll']:.0f}): "
      f"{df_hw2a['Total_Time'].min():.2f}s")
print(f"  最大加速比: {speedup_hw2a.max():.2f}x")

print("\nhw2b (MPI+OpenMP) 效能:")
print(f"  Unroll=1 (baseline): {df_hw2b[df_hw2b['Unroll']==1]['Total_Time'].values[0]:.2f}s")
print(f"  最佳效能 (Unroll={df_hw2b.loc[df_hw2b['Total_Time'].idxmin(), 'Unroll']:.0f}): "
      f"{df_hw2b['Total_Time'].min():.2f}s")
print(f"  最大加速比: {speedup_hw2b.max():.2f}x")

print("\n✓ 所有圖表已生成完畢!")
print(f"✓ 輸出目錄: {OUTPUT_DIR}/")
print("="*70)
