#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

CSV_FILE = "summary_results.csv"
OUTPUT_DIR = "unroll_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_HW2A = "#2E86AB"
COLOR_HW2B = "#A23B72"

print("Loading data...")

df = pd.read_csv(CSV_FILE)

df["Unroll"] = df["Unroll_Factor"].astype(int)

df["Total_Time"] = pd.to_numeric(df["Total_Execution_Time"], errors="coerce")

df_hw2a = df[df["Type"] == "hw2a"].sort_values("Unroll")

df_hw2b = df[df["Type"] == "hw2b"].sort_values("Unroll")

print(f"hw2a: {len(df_hw2a)} points, hw2b: {len(df_hw2b)} points")

fig, ax1 = plt.subplots(figsize=(14, 8))

x_pos_a = np.arange(len(df_hw2a))

x_pos_b = x_pos_a + 0.35

bar_width = 0.35

bars_a = ax1.bar(
    x_pos_a,
    df_hw2a["Total_Time"],
    bar_width,
    label="hw2a Total Time",
    color=COLOR_HW2A,
    alpha=0.6,
)

bars_b = ax1.bar(
    x_pos_b,
    df_hw2b["Total_Time"],
    bar_width,
    label="hw2b Total Time",
    color=COLOR_HW2B,
    alpha=0.6,
)

for bar, val in zip(bars_a, df_hw2a["Total_Time"]):

    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 2,
        f"{val:.1f}s",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLOR_HW2A,
    )

for bar, val in zip(bars_b, df_hw2b["Total_Time"]):

    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 2,
        f"{val:.1f}s",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLOR_HW2B,
    )

ax1.set_xlabel("Loop Unroll Factor", fontsize=12, fontweight="bold")

ax1.set_ylabel(
    "Total Execution Time (seconds)", fontsize=12, fontweight="bold", color="black"
)

max_time = df["Total_Time"].max()

ax1.set_ylim(0, max_time * 1.2)

ax1.tick_params(axis="y", labelcolor="black", labelsize=8)

ax1.tick_params(axis="x", labelsize=8)

ax1.set_xticks(x_pos_a + bar_width / 2)

ax1.set_xticklabels(df_hw2a["Unroll"])

ax1.grid(axis="y", linestyle="--", alpha=0.3)

ax2 = ax1.twinx()

baseline_hw2a = df_hw2a[df_hw2a["Unroll"] == 1]["Total_Time"].values[0]

baseline_hw2b = df_hw2b[df_hw2b["Unroll"] == 1]["Total_Time"].values[0]

speedup_hw2a = baseline_hw2a / df_hw2a["Total_Time"]

speedup_hw2b = baseline_hw2b / df_hw2b["Total_Time"]

x_centers = x_pos_a + bar_width / 2

line_a = ax2.plot(
    x_centers,
    speedup_hw2a,
    marker="o",
    markersize=5,
    linewidth=2.0,
    color=COLOR_HW2A,
    label="hw2a Speedup",
    linestyle="-",
    alpha=0.6,
)

line_b = ax2.plot(
    x_centers,
    speedup_hw2b,
    marker="s",
    markersize=5,
    linewidth=2.0,
    color=COLOR_HW2B,
    label="hw2b Speedup",
    linestyle="-",
    alpha=0.6,
)

for x, y in zip(x_centers, speedup_hw2a):

    ax2.text(
        x,
        y + 0.05,
        f"{y:.2f}×",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLOR_HW2A,
        fontweight="bold",
    )

for x, y in zip(x_centers, speedup_hw2b):

    ax2.text(
        x,
        y - 0.05,
        f"{y:.2f}×",
        ha="center",
        va="top",
        fontsize=10,
        color=COLOR_HW2B,
        fontweight="bold",
    )

ax2.set_ylabel("Speedup (T₁ / Tₙ)", fontsize=12, fontweight="bold", color="black")

max_speedup = max(speedup_hw2a.max(), speedup_hw2b.max())

ax2.set_ylim(0.9, max_speedup * 1.1)

ax2.tick_params(axis="y", labelcolor="black", labelsize=8)

ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

bars_handles = [bars_a, bars_b]

lines_handles = [line_a[0], line_b[0]]

all_handles = bars_handles + lines_handles

all_labels = ["hw2a (Pthread) Time", "hw2b (MPI+OpenMP) Time", "hw2a (Pthread) Speedup", "hw2b (MPI+OpenMP) Speedup"]

ax1.legend(all_handles, all_labels, loc="upper right", fontsize=12, framealpha=0.95)

plt.title(
    "Loop Unrolling: Execution Time vs Speedup", fontsize=20, fontweight="bold", pad=20
)

plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

plt.savefig(
    f"{OUTPUT_DIR}/unroll_combined.png", dpi=300, bbox_inches="tight", pad_inches=0.3
)

print(f"\n✓ Saved: {OUTPUT_DIR}/unroll_combined.png")

plt.close()

print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)

print("\nhw2a (Pthread):")

print(f" Baseline (unroll=1): {baseline_hw2a:.2f}s")

print(
    f" Best (unroll={df_hw2a.loc[df_hw2a['Total_Time'].idxmin(), 'Unroll']:.0f}): "
    f"{df_hw2a['Total_Time'].min():.2f}s → {speedup_hw2a.max():.2f}× speedup"
)

print("\nhw2b (MPI+OpenMP):")

print(f" Baseline (unroll=1): {baseline_hw2b:.2f}s")

print(
    f" Best (unroll={df_hw2b.loc[df_hw2b['Total_Time'].idxmin(), 'Unroll']:.0f}): "
    f"{df_hw2b['Total_Time'].min():.2f}s → {speedup_hw2b.max():.2f}× speedup"
)

print("=" * 70)
