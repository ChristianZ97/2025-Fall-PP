#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    from adjustText import adjust_text
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "adjustText"])
    from adjustText import adjust_text

CSV_FILE = "summary_results.csv"
OUTPUT_DIR = "load_balance_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_HW2A = "#2E86AB"
COLOR_HW2B = "#A23B72"
FIGURE_SIZE = (26, 18)

print("Loading data...")

df = pd.read_csv(CSV_FILE)

df["Total_Execution_Time"] = pd.to_numeric(df["Total_Execution_Time"], errors="coerce")
df["Avg_Thread_Imbalance_Pct"] = pd.to_numeric(
    df["Avg_Thread_Imbalance_Pct"], errors="coerce"
)

df["Total_Time"] = df["Total_Execution_Time"]
df["Imbalance_Pct"] = df["Avg_Thread_Imbalance_Pct"]
df["Strategy"] = df["Program"]

df_hw2a = df[df["Type"] == "hw2a"].sort_values("Imbalance_Pct").reset_index(drop=True)
df_hw2b = df[df["Type"] == "hw2b"].sort_values("Imbalance_Pct").reset_index(drop=True)

print(f"hw2a: {len(df_hw2a)} strategies, hw2b: {len(df_hw2b)} strategies")

if not df_hw2a.empty:

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(
        df_hw2a["Imbalance_Pct"],
        df_hw2a["Total_Time"],
        linewidth=2.5,
        color=COLOR_HW2A,
        alpha=0.4,
        zorder=1,
    )

    ax.scatter(
        df_hw2a["Imbalance_Pct"],
        df_hw2a["Total_Time"],
        s=100,
        color=COLOR_HW2A,
        edgecolor="white",
        linewidth=2.5,
        zorder=3,
        alpha=0.9,
    )

    number_texts = []

    for idx, row in df_hw2a.iterrows():

        txt = ax.text(
            row["Imbalance_Pct"],
            row["Total_Time"],
            str(idx + 1),
            fontsize=12,
            fontweight="bold",
            color=COLOR_HW2A,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=COLOR_HW2A,
                linewidth=1.5,
                alpha=0.95,
            ),
            zorder=4,
        )

        number_texts.append(txt)

    adjust_text(
        number_texts,
        ax=ax,
        arrowprops=dict(arrowstyle="->", color=COLOR_HW2A, lw=0.8, alpha=0.6),
        expand_points=(3.0, 4.0),
        expand_text=(8.0, 10.0),
        force_text=(0.8, 1.0),
        force_points=(0.5, 0.6),
        lim=30000,
        only_move={"text": "xy"},
    )

    table_lines = []

    for idx, row in df_hw2a.iterrows():

        line = f"[{idx + 1:1d}] {row['Strategy']:30s} ({row['Total_Time']:6.1f}s, {row['Imbalance_Pct']:5.1f}%)"

        table_lines.append(line)

    table_text = "\n".join(table_lines)

    ax.text(
        0.02,
        0.98,
        table_text,
        fontsize=18,
        ha="left",
        va="top",
        family="monospace",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            alpha=0.95,
            edgecolor=COLOR_HW2A,
            linewidth=2.0,
        ),
        transform=ax.transAxes,
        zorder=10,
    )

    ax.set_xlabel("Thread Imbalance (%)", fontsize=18, fontweight="bold")

    ax.set_ylabel("Total Execution Time (seconds)", fontsize=18, fontweight="bold")

    ax.set_title(
        "hw2a (pthread) Load Balancing Strategies",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )

    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

    ax.tick_params(labelsize=14)

    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    plt.savefig(
        f"{OUTPUT_DIR}/hw2a_load_balance.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.3,
    )

    print(f"✓ Saved: {OUTPUT_DIR}/hw2a_load_balance.png")

    plt.close()

if not df_hw2b.empty:

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(
        df_hw2b["Imbalance_Pct"],
        df_hw2b["Total_Time"],
        linewidth=2.5,
        color=COLOR_HW2B,
        alpha=0.4,
        zorder=1,
    )

    ax.scatter(
        df_hw2b["Imbalance_Pct"],
        df_hw2b["Total_Time"],
        s=100,
        color=COLOR_HW2B,
        edgecolor="white",
        linewidth=2.5,
        zorder=3,
        alpha=0.9,
        marker="s",
    )

    number_texts = []

    for idx, row in df_hw2b.iterrows():

        txt = ax.text(
            row["Imbalance_Pct"],
            row["Total_Time"],
            str(idx + 1),
            fontsize=12,
            fontweight="bold",
            color=COLOR_HW2B,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=COLOR_HW2B,
                linewidth=1.5,
                alpha=0.95,
            ),
            zorder=4,
        )

        number_texts.append(txt)

    adjust_text(
        number_texts,
        ax=ax,
        arrowprops=dict(arrowstyle="->", color=COLOR_HW2B, lw=0.8, alpha=0.6),
        expand_points=(3.0, 4.0),
        expand_text=(8.0, 10.0),
        force_text=(0.8, 1.0),
        force_points=(0.5, 0.6),
        lim=30000,
        only_move={"text": "xy"},
    )

    table_lines = []

    for idx, row in df_hw2b.iterrows():

        line = f"[{idx + 1:2d}] {row['Strategy']:50s} ({row['Total_Time']:6.1f}s, {row['Imbalance_Pct']:5.1f}%)"

        table_lines.append(line)

    table_text = "\n".join(table_lines)

    ax.text(
        0.2,
        0.98,
        table_text,
        fontsize=18,
        ha="left",
        va="top",
        family="monospace",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            alpha=0.95,
            edgecolor=COLOR_HW2B,
            linewidth=2.0,
        ),
        transform=ax.transAxes,
        zorder=10,
    )

    ax.set_xlabel("Thread Imbalance (%)", fontsize=18, fontweight="bold")

    ax.set_ylabel("Total Execution Time (seconds)", fontsize=18, fontweight="bold")

    ax.set_title(
        "hw2b (MPI+OpenMP) Load Balancing Strategies",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )

    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

    ax.tick_params(labelsize=14)

    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    plt.savefig(
        f"{OUTPUT_DIR}/hw2b_load_balance.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.3,
    )

    print(f"✓ Saved: {OUTPUT_DIR}/hw2b_load_balance.png")

    plt.close()

print("\n" + "=" * 90)
print("Summary Statistics")
print("=" * 90)

if not df_hw2a.empty:

    best = df_hw2a.loc[df_hw2a["Total_Time"].idxmin()]

    worst = df_hw2a.loc[df_hw2a["Total_Time"].idxmax()]

    print(
        f"\nhw2a Best: {best['Strategy']} ({best['Total_Time']:.2f}s, {best['Imbalance_Pct']:.2f}%)"
    )

    print(
        f"hw2a Worst: {worst['Strategy']} ({worst['Total_Time']:.2f}s, {worst['Imbalance_Pct']:.2f}%)"
    )

if not df_hw2b.empty:

    best = df_hw2b.loc[df_hw2b["Total_Time"].idxmin()]

    worst = df_hw2b.loc[df_hw2b["Total_Time"].idxmax()]

    print(
        f"\nhw2b Best: {best['Strategy']} ({best['Total_Time']:.2f}s, {best['Imbalance_Pct']:.2f}%)"
    )

    print(
        f"hw2b Worst: {worst['Strategy']} ({worst['Total_Time']:.2f}s, {worst['Imbalance_Pct']:.2f}%)"
    )

print("=" * 90)
