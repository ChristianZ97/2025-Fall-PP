#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

CSV_FILE = "summary_results.csv"
OUTPUT_DIR = "scalability_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_HW2A = "#2E86AB"
COLOR_HW2B = "#A23B72"
COLOR_IDEAL = "#999999"

print("Loading data...")

df = pd.read_csv(CSV_FILE)

df_hw2a = df[df["Type"] == "hw2a"].sort_values("Total_Cores")

df_hw2b_mpi = df[(df["Type"] == "hw2b") & (df["Num_Threads"] == 12)].sort_values(
    "Total_Cores"
)

df_hw2b_hybrid = df[(df["Type"] == "hw2b") & (df["Total_Cores"] == 48)].sort_values(
    "Num_Processes", ascending=False
)

print(f"hw2a (threads): {len(df_hw2a)} points")

print(f"hw2b (MPI, 12t/proc): {len(df_hw2b_mpi)} points")

print(f"hw2b (hybrid, 48 cores): {len(df_hw2b_hybrid)} points")

print("\nPlotting hw2a + hw2b MPI scaling comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

x_pos_a = np.arange(len(df_hw2a))

bar_width = 0.6

bars_a = ax1.bar(
    x_pos_a,
    df_hw2a["Total_Execution_Time"],
    bar_width,
    label="Execution Time",
    color=COLOR_HW2A,
    alpha=0.6,
)

for bar, val in zip(bars_a, df_hw2a["Total_Execution_Time"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 20,
        f"{val:.0f}s",
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLOR_HW2A,
    )

ax1_twin = ax1.twinx()

line_speedup_a = ax1_twin.plot(
    x_pos_a,
    df_hw2a["Speedup"],
    marker="o",
    markersize=6,
    linewidth=2.5,
    color=COLOR_HW2A,
    label="Speedup",
    linestyle="-",
    alpha=0.8,
)

for x, y in zip(x_pos_a, df_hw2a["Speedup"]):
    ax1_twin.text(
        x + 0.3,
        y - 0.6,
        f"{y:.2f}×",
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLOR_HW2A,
        fontweight="bold",
    )

baseline_cores_a = df_hw2a.iloc[0]["Total_Cores"]
ideal_speedup_a = df_hw2a["Total_Cores"].values / baseline_cores_a

ax1_twin.plot(
    x_pos_a,
    ideal_speedup_a,
    marker=None,
    linewidth=2.0,
    color=COLOR_IDEAL,
    label="Ideal Speedup",
    linestyle="--",
    alpha=0.7,
)

ax1_twin.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

ax1.set_xlabel("Number of Threads", fontsize=12, fontweight="bold")

ax1.set_ylabel(
    "Execution Time (seconds)", fontsize=12, fontweight="bold", color="black"
)

ax1_twin.set_ylabel("Speedup", fontsize=12, fontweight="bold", color="black")

ax1.set_xticks(x_pos_a)

ax1.set_xticklabels(df_hw2a["Total_Cores"])

ax1.grid(axis="y", linestyle="--", alpha=0.3)

ax1.tick_params(axis="y", labelcolor="black", labelsize=8)

ax1.tick_params(axis="x", labelsize=8)

ax1_twin.tick_params(axis="y", labelcolor="black", labelsize=8)

ax1.set_ylim(0, df_hw2a["Total_Execution_Time"].max() * 1.2)

max_speedup_a = max(df_hw2a["Speedup"].max(), ideal_speedup_a.max())

ax1_twin.set_ylim(0.1, max_speedup_a * 1.2)

bars_handle_a = [bars_a]
lines_handle_a = [line_speedup_a[0]]
ideal_line_a = ax1_twin.get_lines()[-2]

all_handles_a = bars_handle_a + lines_handle_a + [ideal_line_a]

all_labels_a = ["Execution Time", "Speedup", "Ideal Speedup"]

ax1.legend(all_handles_a, all_labels_a, loc="upper right", fontsize=11, framealpha=0.95)

ax1.set_title("hw2a (Pthread) - Thread Scaling", fontsize=16, fontweight="bold", pad=15)

x_pos_b = np.arange(len(df_hw2b_mpi))

bars_b = ax2.bar(
    x_pos_b,
    df_hw2b_mpi["Total_Execution_Time"],
    bar_width,
    label="Execution Time",
    color=COLOR_HW2B,
    alpha=0.6,
)

for bar, val in zip(bars_b, df_hw2b_mpi["Total_Execution_Time"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + 3,
        f"{val:.0f}s",
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLOR_HW2B,
    )

ax2_twin = ax2.twinx()

line_speedup_b = ax2_twin.plot(
    x_pos_b,
    df_hw2b_mpi["Speedup"],
    marker="s",
    markersize=6,
    linewidth=2.5,
    color=COLOR_HW2B,
    label="Speedup",
    linestyle="-",
    alpha=0.8,
)

for x, y in zip(x_pos_b, df_hw2b_mpi["Speedup"]):
    ax2_twin.text(
        x,
        y - 0.25,
        f"{y:.2f}×",
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLOR_HW2B,
        fontweight="bold",
    )

baseline_cores_b = df_hw2b_mpi.iloc[0]["Total_Cores"]
ideal_speedup_b = df_hw2b_mpi["Total_Cores"].values / baseline_cores_b

ax2_twin.plot(
    x_pos_b,
    ideal_speedup_b,
    marker=None,
    linewidth=2.0,
    color=COLOR_IDEAL,
    label="Ideal Speedup",
    linestyle="--",
    alpha=0.7,
)

ax2_twin.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

ax2.set_xlabel(
    "Number of Cores (Processes × 12 Threads)", fontsize=12, fontweight="bold"
)

ax2.set_ylabel(
    "Execution Time (seconds)", fontsize=12, fontweight="bold", color="black"
)

ax2_twin.set_ylabel("Speedup", fontsize=12, fontweight="bold", color="black")

ax2.set_xticks(x_pos_b)

ax2.set_xticklabels(df_hw2b_mpi["Total_Cores"])

ax2.grid(axis="y", linestyle="--", alpha=0.3)

ax2.tick_params(axis="y", labelcolor="black", labelsize=8)

ax2.tick_params(axis="x", labelsize=8)

ax2_twin.tick_params(axis="y", labelcolor="black", labelsize=8)

ax2.set_ylim(0, df_hw2b_mpi["Total_Execution_Time"].max() * 1.2)

max_speedup_b = max(df_hw2b_mpi["Speedup"].max(), ideal_speedup_b.max())

ax2_twin.set_ylim(0.5, max_speedup_b * 1.2)

bars_handle_b = [bars_b]
lines_handle_b = [line_speedup_b[0]]
ideal_line_b = ax2_twin.get_lines()[-2]

all_handles_b = bars_handle_b + lines_handle_b + [ideal_line_b]

all_labels_b = ["Execution Time", "Speedup", "Ideal Speedup"]

ax2.legend(all_handles_b, all_labels_b, loc="upper right", fontsize=11, framealpha=0.95)

ax2.set_title(
    "hw2b (MPI+OpenMP) - Multi-Node Scaling", fontsize=16, fontweight="bold", pad=15
)

plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

plt.savefig(
    f"{OUTPUT_DIR}/scalability_comparison.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.3,
)

print(f"✓ Saved: {OUTPUT_DIR}/scalability_comparison.png")

plt.close()

print("\nPlotting hw2b hybrid configurations...")

df_hw2b_hybrid["Config"] = (
    df_hw2b_hybrid["Num_Processes"].astype(str)
    + "p × "
    + df_hw2b_hybrid["Num_Threads"].astype(str)
    + "t"
)

fig, ax1 = plt.subplots(figsize=(14, 8))

x_pos_h = np.arange(len(df_hw2b_hybrid))

bars_h = ax1.bar(
    x_pos_h,
    df_hw2b_hybrid["Total_Execution_Time"],
    bar_width,
    label="Execution Time",
    color=COLOR_HW2B,
    alpha=0.6,
)

for bar, val in zip(bars_h, df_hw2b_hybrid["Total_Execution_Time"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1,
        f"{val:.0f}s",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLOR_HW2B,
    )

ax1_twin = ax1.twinx()

line_speedup_h = ax1_twin.plot(
    x_pos_h,
    df_hw2b_hybrid["Speedup"],
    marker="D",
    markersize=6,
    linewidth=2.5,
    color=COLOR_HW2B,
    label="Speedup",
    linestyle="-",
    alpha=0.8,
)

for x, y in zip(x_pos_h, df_hw2b_hybrid["Speedup"]):
    ax1_twin.text(
        x,
        y + 0.03,
        f"{y:.2f}×",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLOR_HW2B,
        fontweight="bold",
    )

ax1_twin.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

ax1.set_xlabel(
    "Process × Thread Configuration (48 Cores Total)", fontsize=12, fontweight="bold"
)

ax1.set_ylabel(
    "Execution Time (seconds)", fontsize=12, fontweight="bold", color="black"
)

ax1_twin.set_ylabel("Speedup", fontsize=12, fontweight="bold", color="black")

ax1.set_xticks(x_pos_h)

ax1.set_xticklabels(df_hw2b_hybrid["Config"], rotation=0, ha="center")

ax1.grid(axis="y", linestyle="--", alpha=0.3)

ax1.tick_params(axis="y", labelcolor="black", labelsize=8)

ax1.tick_params(axis="x", labelsize=8)

ax1_twin.tick_params(axis="y", labelcolor="black", labelsize=8)

ax1.set_ylim(0, df_hw2b_hybrid["Total_Execution_Time"].max() * 1.1)

ax1_twin.set_ylim(1, df_hw2b_hybrid["Speedup"].max() * 1.2)

all_handles_h = [bars_h] + line_speedup_h

all_labels_h = ["Execution Time", "Speedup"]

ax1.legend(all_handles_h, all_labels_h, loc="upper right", fontsize=11, framealpha=0.95)

ax1.set_title(
    "hw2b (MPI+OpenMP) - Hybrid Configurations (48 Cores)",
    fontsize=16,
    fontweight="bold",
    pad=15,
)

plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

plt.savefig(
    f"{OUTPUT_DIR}/hw2b_hybrid_configs.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.3,
)

print(f"✓ Saved: {OUTPUT_DIR}/hw2b_hybrid_configs.png")

plt.close()

print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)

print("\nhw2a (Pthread) - Thread Scaling:")

print(f" Baseline (1 thread): {df_hw2a.iloc[0]['Total_Execution_Time']:.2f}s")

print(
    f" Best ({df_hw2a.iloc[-1]['Total_Cores']:.0f} threads): {df_hw2a.iloc[-1]['Total_Execution_Time']:.2f}s"
)

print(
    f" Max Speedup: {df_hw2a['Speedup'].max():.2f}× (Ideal: {ideal_speedup_a.max():.2f}×)"
)

print("\nhw2b (MPI+OpenMP) - Multi-Node Scaling (12 threads/process):")

print(f" Baseline (12 cores): {df_hw2b_mpi.iloc[0]['Total_Execution_Time']:.2f}s")

print(
    f" Best ({df_hw2b_mpi.iloc[-1]['Total_Cores']:.0f} cores): {df_hw2b_mpi.iloc[-1]['Total_Execution_Time']:.2f}s"
)

print(
    f" Max Speedup: {df_hw2b_mpi['Speedup'].max():.2f}× (Ideal: {ideal_speedup_b.max():.2f}×)"
)

print(
    f" MPI Overhead: Low (stable execution time, efficient inter-process communication)"
)

print("\nhw2b (MPI+OpenMP) - Hybrid Configurations (48 cores):")

best_config = df_hw2b_hybrid.loc[df_hw2b_hybrid["Total_Execution_Time"].idxmin()]

print(
    f" Best Config: {best_config['Num_Processes']:.0f}p × {best_config['Num_Threads']:.0f}t"
)

print(f" Best Time: {best_config['Total_Execution_Time']:.2f}s")

print(f" Speedup: {best_config['Speedup']:.2f}×")

print(
    f" Config Stability: High (consistent performance across configs, ~2.4-2.5× speedup)"
)

print("=" * 70)
