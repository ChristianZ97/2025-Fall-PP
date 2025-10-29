#!/usr/bin/env python3
"""
plot_load_balance_results.py
改進版:解決標籤重疊問題
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    from adjustText import adjust_text
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'adjustText'])
    from adjustText import adjust_text

# ==================== 配置 ====================
CSV_FILE = 'load_balance_results/detailed_results.csv'
OUTPUT_DIR = 'load_balance_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_HW2A = '#2E86AB'
COLOR_HW2B = '#A23B72'

# ==================== 讀取資料 ====================
print("讀取資料中...")
df = pd.read_csv(CSV_FILE)
df['Total_Time'] = pd.to_numeric(df['Total_Time'], errors='coerce')
df['Imbalance_Pct'] = pd.to_numeric(df['Imbalance_Pct'], errors='coerce')

df_hw2a = df[df['Type'] == 'hw2a'].sort_values('Imbalance_Pct').reset_index(drop=True)
df_hw2b = df[df['Type'] == 'hw2b'].sort_values('Imbalance_Pct').reset_index(drop=True)

print(f"hw2a: {len(df_hw2a)} 個策略, hw2b: {len(df_hw2b)} 個策略")

# ==================== 圖 1: hw2a ====================
if not df_hw2a.empty:
    fig, ax = plt.subplots(figsize=(22, 14))
    
    # 繪製折線
    ax.plot(df_hw2a['Imbalance_Pct'], df_hw2a['Total_Time'],
            linewidth=2.5, color=COLOR_HW2A, alpha=0.4, zorder=1)
    
    # 繪製資料點
    ax.scatter(df_hw2a['Imbalance_Pct'], df_hw2a['Total_Time'],
               s=100, color=COLOR_HW2A, edgecolor='white', linewidth=2.5,
               zorder=3, alpha=0.9)
    
    # 建立文字標註(使用箭頭連接)
    texts = []
    for idx, row in df_hw2a.iterrows():
        label = f"{row['Strategy']} ({row['Total_Time']:.2f}s, {row['Imbalance_Pct']:.2f}%)"
        
        txt = ax.text(row['Imbalance_Pct'], row['Total_Time'], label,
                     fontsize=8,
                     fontweight='normal',
                     color='#333333',
                     ha='center',
                     va='center',
                     bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='white',
                              alpha=0.92,
                              edgecolor=COLOR_HW2A,
                              linewidth=1.2),
                     zorder=10)
        texts.append(txt)
    
    # 智慧調整標籤位置(使用箭頭)
    adjust_text(texts,
                ax=ax,
                arrowprops=dict(arrowstyle='->', color=COLOR_HW2A, lw=1, alpha=0.6),
                expand_points=(1.8, 2.5),
                expand_text=(1.5, 2.0),
                force_text=(0.8, 1.2),
                force_points=(0.4, 0.7),
                lim=3000)
    
    ax.set_xlabel('Thread Imbalance (%)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Total Execution Time (seconds)', fontsize=20, fontweight='bold')
    ax.set_title('hw2a (pthread) Load Balancing Strategies',
                fontsize=30, fontweight='bold', pad=25)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hw2a_load_balance.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已儲存: {OUTPUT_DIR}/hw2a_load_balance.png")
    plt.close()

# ==================== 圖 2: hw2b ====================
if not df_hw2b.empty:
    fig, ax = plt.subplots(figsize=(26, 18))
    
    # 繪製折線
    ax.plot(df_hw2b['Imbalance_Pct'], df_hw2b['Total_Time'],
            linewidth=2.5, color=COLOR_HW2B, alpha=0.4, zorder=1)
    
    # 繪製資料點
    ax.scatter(df_hw2b['Imbalance_Pct'], df_hw2b['Total_Time'],
               s=100, color=COLOR_HW2B, edgecolor='white', linewidth=2.5,
               zorder=3, alpha=0.9, marker='s')
    
    # 建立文字標註(使用箭頭連接)
    texts = []
    for idx, row in df_hw2b.iterrows():
        label = f"{row['Strategy']} ({row['Total_Time']:.2f}s, {row['Imbalance_Pct']:.2f}%)"
        
        txt = ax.text(row['Imbalance_Pct'], row['Total_Time'], label,
                     fontsize=7.5,
                     fontweight='normal',
                     color='#333333',
                     ha='center',
                     va='center',
                     bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='white',
                              alpha=0.92,
                              edgecolor=COLOR_HW2B,
                              linewidth=1.2),
                     zorder=10)
        texts.append(txt)
    
    # 智慧調整標籤位置(使用箭頭,加強分散)
    adjust_text(texts,
                ax=ax,
                arrowprops=dict(arrowstyle='->', color=COLOR_HW2B, lw=0.8, alpha=0.6),
                expand_points=(2.2, 3.5),
                expand_text=(2.0, 3.0),
                force_text=(1.2, 1.8),
                force_points=(0.6, 1.0),
                lim=4000)
    
    ax.set_xlabel('Thread Imbalance (%)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Total Execution Time (seconds)', fontsize=20, fontweight='bold')
    ax.set_title('hw2b (MPI+OpenMP) Load Balancing Strategies',
                fontsize=30, fontweight='bold', pad=25)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hw2b_load_balance.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已儲存: {OUTPUT_DIR}/hw2b_load_balance.png")
    plt.close()

# ==================== 統計摘要 ====================
print("\n" + "="*90)
print("統計摘要")
print("="*90)

if not df_hw2a.empty:
    best = df_hw2a.loc[df_hw2a['Total_Time'].idxmin()]
    worst = df_hw2a.loc[df_hw2a['Total_Time'].idxmax()]
    print(f"\nhw2a 最佳: {best['Strategy']} ({best['Total_Time']:.2f}s, {best['Imbalance_Pct']:.2f}%)")
    print(f"hw2a 最差: {worst['Strategy']} ({worst['Total_Time']:.2f}s, {worst['Imbalance_Pct']:.2f}%)")

if not df_hw2b.empty:
    best = df_hw2b.loc[df_hw2b['Total_Time'].idxmin()]
    worst = df_hw2b.loc[df_hw2b['Total_Time'].idxmax()]
    print(f"\nhw2b 最佳: {best['Strategy']} ({best['Total_Time']:.2f}s, {best['Imbalance_Pct']:.2f}%)")
    print(f"hw2b 最差: {worst['Strategy']} ({worst['Total_Time']:.2f}s, {worst['Imbalance_Pct']:.2f}%)")

print("="*90)
