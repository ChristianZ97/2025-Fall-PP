#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CSV_FILE = "summary_results.csv"
OUTPUT_DIR = "load_balance_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_HW2A = "#2E86AB"
COLOR_HW2B = "#A23B72"
FIGURE_SIZE = (26, 18)

print("Loading data...")
df = pd.read_csv(CSV_FILE)

df["Total_Execution_Time"] = pd.to_numeric(df["Total_Execution_Time"], errors="coerce")
df["Avg_Thread_Imbalance_Pct"] = pd.to_numeric(df["Avg_Thread_Imbalance_Pct"], errors="coerce")

df["Total_Time"] = df["Total_Execution_Time"]
df["Imbalance_Pct"] = df["Avg_Thread_Imbalance_Pct"]
df["Strategy"] = df["Program"]

df_hw2a = df[df["Type"] == "hw2a"].sort_values("Imbalance_Pct").reset_index(drop=True)
df_hw2b = df[df["Type"] == "hw2b"].sort_values("Imbalance_Pct").reset_index(drop=True)

def apply_kmeans_coloring(df_data, base_color):
    """K-means 分群並產生相近色系（色差更明顯）"""
    features = df_data[['Imbalance_Pct', 'Total_Time']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_data['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # 相近色系但色差更大（同色系內不同色調）
    if base_color == COLOR_HW2A:  # 藍色系：深→中→淺→亮
        color_palette = ['#003D82', '#0066CC', '#2E86AB', '#4DB8FF']
    else:  # 紫紅系：深→中→淺→亮
        color_palette = ['#6B0C47', '#A23B72', '#D47FA6', '#F0A5D4']
    
    df_data['Color'] = df_data['Cluster'].map(lambda x: color_palette[x])
    return df_data, color_palette


print(f"hw2a: {len(df_hw2a)} strategies, hw2b: {len(df_hw2b)} strategies")

if not df_hw2a.empty:
    df_hw2a, palette_hw2a = apply_kmeans_coloring(df_hw2a, COLOR_HW2A)
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    ax.plot(df_hw2a["Imbalance_Pct"], df_hw2a["Total_Time"],
            linewidth=2.5, color=COLOR_HW2A, alpha=0.2, zorder=1)
    
    ax.scatter(df_hw2a["Imbalance_Pct"], df_hw2a["Total_Time"],
               s=150, c=df_hw2a['Color'], edgecolor="white",
               linewidth=2.5, zorder=3, alpha=1)
    
    for idx, row in df_hw2a.iterrows():
        ax.annotate(
            str(idx + 1),
            xy=(row["Imbalance_Pct"], row["Total_Time"]),
            xytext=(30, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=row['Color'],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=row['Color'], linewidth=1.5, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=row['Color'], 
                            lw=0.8, connectionstyle="arc3,rad=0.3"),
            zorder=4
        )
    
    # 表格列表加入顏色標示
    table_lines = []
    for idx, row in df_hw2a.iterrows():
        color_circle = "●"
        line = f"{color_circle} [{idx + 1:1d}] {row['Strategy']:30s} ({row['Total_Time']:6.1f}s, {row['Imbalance_Pct']:5.1f}%)"
        table_lines.append((line, row['Color']))
    
    y_pos = 0.96
    for line, color in table_lines:
        ax.text(0.02, y_pos, line, 
               fontsize=16, ha="left", va="top", family="monospace",
               color=color, fontweight="bold",
               transform=ax.transAxes, zorder=10)
        y_pos -= 0.032
    
    ax.set_xlabel("Thread Imbalance (%)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Total Execution Time (seconds)", fontsize=18, fontweight="bold")
    ax.set_title("hw2a (pthread) Load Balancing Strategies", fontsize=24,
                 fontweight="bold", pad=20)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.tick_params(labelsize=14)
    
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.savefig(f"{OUTPUT_DIR}/hw2a_load_balance.png", dpi=300,
                bbox_inches="tight", pad_inches=0.3)
    print(f"✓ Saved: {OUTPUT_DIR}/hw2a_load_balance.png")
    plt.close()

if not df_hw2b.empty:
    df_hw2b, palette_hw2b = apply_kmeans_coloring(df_hw2b, COLOR_HW2B)
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    ax.plot(df_hw2b["Imbalance_Pct"], df_hw2b["Total_Time"],
            linewidth=2.5, color=COLOR_HW2B, alpha=0.2, zorder=1)
    
    ax.scatter(df_hw2b["Imbalance_Pct"], df_hw2b["Total_Time"],
               s=150, c=df_hw2b['Color'], marker="s", edgecolor="white",
               linewidth=2.5, zorder=3, alpha=1)
    
    for idx, row in df_hw2b.iterrows():
        ax.annotate(
            str(idx + 1),
            xy=(row["Imbalance_Pct"], row["Total_Time"]),
            xytext=(30, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=row['Color'],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=row['Color'], linewidth=1.5, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=row['Color'], 
                            lw=0.8, connectionstyle="arc3,rad=0.3"),
            zorder=4
        )
    
    table_lines = []
    for idx, row in df_hw2b.iterrows():
        color_circle = "●"
        line = f"{color_circle} [{idx + 1:2d}] {row['Strategy']:48s} ({row['Total_Time']:6.1f}s, {row['Imbalance_Pct']:5.1f}%)"
        table_lines.append((line, row['Color']))
    
    y_pos = 0.96
    for line, color in table_lines:
        ax.text(0.32, y_pos, line,
               fontsize=16, ha="left", va="top", family="monospace",
               color=color, fontweight="bold",
               transform=ax.transAxes, zorder=10)
        y_pos -= 0.024
    
    ax.set_xlabel("Thread Imbalance (%)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Total Execution Time (seconds)", fontsize=18, fontweight="bold")
    ax.set_title("hw2b (MPI+OpenMP) Load Balancing Strategies", fontsize=24,
                 fontweight="bold", pad=20)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.tick_params(labelsize=14)
    
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.savefig(f"{OUTPUT_DIR}/hw2b_load_balance.png", dpi=300,
                bbox_inches="tight", pad_inches=0.3)
    print(f"✓ Saved: {OUTPUT_DIR}/hw2b_load_balance.png")
    plt.close()

print("\n" + "=" * 90)
print("Summary Statistics")
print("=" * 90)

if not df_hw2a.empty:
    best = df_hw2a.loc[df_hw2a["Total_Time"].idxmin()]
    worst = df_hw2a.loc[df_hw2a["Total_Time"].idxmax()]
    print(f"\nhw2a Best: {best['Strategy']} ({best['Total_Time']:.2f}s, {best['Imbalance_Pct']:.2f}%)")
    print(f"hw2a Worst: {worst['Strategy']} ({worst['Total_Time']:.2f}s, {worst['Imbalance_Pct']:.2f}%)")

if not df_hw2b.empty:
    best = df_hw2b.loc[df_hw2b["Total_Time"].idxmin()]
    worst = df_hw2b.loc[df_hw2b["Total_Time"].idxmax()]
    print(f"\nhw2b Best: {best['Strategy']} ({best['Total_Time']:.2f}s, {best['Imbalance_Pct']:.2f}%)")
    print(f"hw2b Worst: {worst['Strategy']} ({worst['Total_Time']:.2f}s, {worst['Imbalance_Pct']:.2f}%)")

print("=" * 90)
