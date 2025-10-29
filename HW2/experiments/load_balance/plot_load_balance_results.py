#!/usr/bin/env python3
"""
plot_load_balance_results.py
繪製負載平衡策略比較圖
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== 配置 ====================
CSV_FILE = 'load_balance_results/detailed_results.csv'
OUTPUT_DIR = 'load_balance_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_HW2A = '#2E86AB'  # hw2a 藍色
COLOR_HW2B = '#A23B72'  # hw2b 紫紅色

# ==================== 讀取資料 ====================
print("讀取資料中...")
df = pd.read_csv(CSV_FILE)
df['Total_Time'] = pd.to_numeric(df['Total_Time'], errors='coerce')
df['Imbalance_Pct'] = pd.to_numeric(df['Imbalance_Pct'], errors='coerce')

df_hw2a = df[df['Type'] == 'hw2a'].sort_values('Total_Time')
df_hw2b = df[df['Type'] == 'hw2b'].sort_values('Total_Time')

print(f"hw2a: {len(df_hw2a)} 個策略, hw2b: {len(df_hw2b)} 個策略")

# ==================== 圖 1: 執行時間比較 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# hw2a 執行時間
if not df_hw2a.empty:
    bars_a = ax1.barh(range(len(df_hw2a)), df_hw2a['Total_Time'], 
                       color=COLOR_HW2A, alpha=0.7)
    ax1.set_yticks(range(len(df_hw2a)))
    ax1.set_yticklabels(df_hw2a['Strategy'], fontsize=9)
    ax1.set_xlabel('Total Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('hw2a (pthread) Load Balance Strategies', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 標註時間
    for bar, val in zip(bars_a, df_hw2a['Total_Time']):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}s', va='center', fontsize=8, color=COLOR_HW2A, fontweight='bold')

# hw2b 執行時間
if not df_hw2b.empty:
    bars_b = ax2.barh(range(len(df_hw2b)), df_hw2b['Total_Time'], 
                       color=COLOR_HW2B, alpha=0.7)
    ax2.set_yticks(range(len(df_hw2b)))
    ax2.set_yticklabels(df_hw2b['Strategy'], fontsize=9)
    ax2.set_xlabel('Total Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('hw2b (MPI+OpenMP) Load Balance Strategies', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 標註時間
    for bar, val in zip(bars_b, df_hw2b['Total_Time']):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}s', va='center', fontsize=8, color=COLOR_HW2B, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/load_balance_time.png', dpi=300, bbox_inches='tight')
print(f"✓ 已儲存: {OUTPUT_DIR}/load_balance_time.png")
plt.close()

# ==================== 圖 2: 負載不平衡度比較 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# hw2a 不平衡度
if not df_hw2a.empty:
    df_hw2a_sorted = df_hw2a.sort_values('Imbalance_Pct')
    bars_a = ax1.barh(range(len(df_hw2a_sorted)), df_hw2a_sorted['Imbalance_Pct'], 
                       color=COLOR_HW2A, alpha=0.7)
    ax1.set_yticks(range(len(df_hw2a_sorted)))
    ax1.set_yticklabels(df_hw2a_sorted['Strategy'], fontsize=9)
    ax1.set_xlabel('Thread Imbalance (%)', fontsize=11, fontweight='bold')
    ax1.set_title('hw2a Load Imbalance Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 標註不平衡度
    for bar, val in zip(bars_a, df_hw2a_sorted['Imbalance_Pct']):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=8, color=COLOR_HW2A, fontweight='bold')

# hw2b 不平衡度
if not df_hw2b.empty:
    df_hw2b_sorted = df_hw2b.sort_values('Imbalance_Pct')
    bars_b = ax2.barh(range(len(df_hw2b_sorted)), df_hw2b_sorted['Imbalance_Pct'], 
                       color=COLOR_HW2B, alpha=0.7)
    ax2.set_yticks(range(len(df_hw2b_sorted)))
    ax2.set_yticklabels(df_hw2b_sorted['Strategy'], fontsize=9)
    ax2.set_xlabel('Thread Imbalance (%)', fontsize=11, fontweight='bold')
    ax2.set_title('hw2b Load Imbalance Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 標註不平衡度
    for bar, val in zip(bars_b, df_hw2b_sorted['Imbalance_Pct']):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=8, color=COLOR_HW2B, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/load_balance_imbalance.png', dpi=300, bbox_inches='tight')
print(f"✓ 已儲存: {OUTPUT_DIR}/load_balance_imbalance.png")
plt.close()

# ==================== 圖 3: 時間分解比較 (最佳策略) ====================
if not df_hw2a.empty and not df_hw2b.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    best_hw2a = df_hw2a.iloc[0]
    best_hw2b = df_hw2b.iloc[0]
    
    strategies = [best_hw2a['Strategy'], best_hw2b['Strategy']]
    compute = [best_hw2a['Compute_Pct'], best_hw2b['Compute_Pct']]
    comm = [best_hw2a['Sync_Comm_Pct'], best_hw2b['Sync_Comm_Pct']]
    io = [best_hw2a['IO_Pct'], best_hw2b['IO_Pct']]
    
    x = np.arange(len(strategies))
    width = 0.6
    
    p1 = ax.bar(x, compute, width, label='Compute', color='#3A86FF')
    p2 = ax.bar(x, comm, width, bottom=compute, label='Sync/Comm', color='#FF006E')
    p3 = ax.bar(x, io, width, bottom=np.array(compute)+np.array(comm), label='IO', color='#FFBE0B')
    
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Time Breakdown: Best Load Balance Strategies', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['hw2a: ' + strategies[0], 'hw2b: ' + strategies[1]], fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/load_balance_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已儲存: {OUTPUT_DIR}/load_balance_breakdown.png")
    plt.close()

# ==================== 統計摘要 ====================
print("\n" + "="*70)
print("統計摘要")
print("="*70)

if not df_hw2a.empty:
    print("\nhw2a (pthread) 最佳策略:")
    best = df_hw2a.iloc[0]
    print(f"  策略: {best['Strategy']}")
    print(f"  執行時間: {best['Total_Time']:.2f}s")
    print(f"  負載不平衡: {best['Imbalance_Pct']:.2f}%")
    print(f"  平行效率: {best['Parallel_Eff']:.2f}%")

if not df_hw2b.empty:
    print("\nhw2b (MPI+OpenMP) 最佳策略:")
    best = df_hw2b.iloc[0]
    print(f"  策略: {best['Strategy']}")
    print(f"  執行時間: {best['Total_Time']:.2f}s")
    print(f"  負載不平衡: {best['Imbalance_Pct']:.2f}%")
    print(f"  平行效率: {best['Parallel_Eff']:.2f}%")

print("="*70)
