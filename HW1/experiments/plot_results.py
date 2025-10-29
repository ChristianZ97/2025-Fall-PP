import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# --- Configuration ---
CSV_PATTERN = "results/scaling_results_*.csv"  # 匹配所有測資的 CSV
OUTPUT_DIR = "plots"  # 輸出目錄

CPU_COLOR = "#479A40"
COMM_COLOR = "#E68A00"
IO_COLOR = "#4C83C9"

# 創建輸出目錄
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 找到所有匹配的 CSV 檔案
csv_files = sorted(glob.glob(CSV_PATTERN))

if not csv_files:
    print(f"Error: No CSV files found matching pattern '{CSV_PATTERN}'")
    exit()

print(f"Found {len(csv_files)} CSV files to process:")
for f in csv_files:
    print(f"  - {f}")

# --- Process each CSV file ---
for csv_file in csv_files:
    # 從檔名提取測資編號 (例如 scaling_results_01.csv -> 01)
    testcase_num = csv_file.split("_")[-1].replace(".csv", "")
    print(f"\n{'='*60}")
    print(f"Processing Testcase {testcase_num}")
    print(f"{'='*60}")

    # 定義輸出檔名
    time_profile_img = os.path.join(OUTPUT_DIR, f"time_profile_bar_{testcase_num}.png")
    speedup_factor_img = os.path.join(OUTPUT_DIR, f"speedup_factor_{testcase_num}.png")

    # --- Data Loading and Cleaning ---
    try:
        df = pd.read_csv(csv_file)

        # 檢查是否有資料
        if df.empty:
            print(f"Warning: {csv_file} is empty. Skipping...")
            continue

        df.loc[df["Cores"] == 1, "Comm_Time"] = 0.0

        for col in ["Total_Time_srun", "IO_Time", "Comm_Time", "CPU_Time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["CPU_Time"] = df.apply(
            lambda row: max(
                0, row["Total_Time_srun"] - row["IO_Time"] - row["Comm_Time"]
            ),
            axis=1,
        )

        df.dropna(inplace=True)

        if df.empty:
            print(f"Warning: No valid data in {csv_file} after cleaning. Skipping...")
            continue

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        continue

    print(f"Loaded {len(df)} data points")

    # --- Plot 1: Time Profile ---
    print("Generating Time Profile plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    x_labels = df["Cores"].astype(str)
    x_pos = np.arange(len(x_labels))
    y_cpu = df["CPU_Time"]
    y_comm = df["Comm_Time"]
    y_io = df["IO_Time"]

    bar_width = 0.6

    ax.bar(x_pos, y_cpu, bar_width, label="CPU Time", color=CPU_COLOR)
    ax.bar(
        x_pos,
        y_comm,
        bar_width,
        bottom=y_cpu,
        label="Communication Time",
        color=COMM_COLOR,
    )
    ax.bar(
        x_pos, y_io, bar_width, bottom=y_cpu + y_comm, label="I/O Time", color=IO_COLOR
    )

    for i, (idx, row) in enumerate(df.iterrows()):
        total_time = row["Total_Time_srun"]
        ax.text(
            x_pos[i],
            total_time,
            f"{total_time:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_title(
        f"Testcase {testcase_num}: Execution Time with Increasing Parallelism",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Processes (p)", fontsize=10)
    ax.set_ylabel("Runtime (seconds)", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(time_profile_img, dpi=300)
    print(f"Saved: {time_profile_img}")
    plt.close()

    # --- Plot 2: Speedup Factor ---
    print("Generating Speedup Factor plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # 檢查是否有 1 core 的資料
    if 1 not in df["Cores"].values:
        print(
            f"Warning: No 1-core baseline found for {csv_file}. Using minimum cores as baseline."
        )
        t1 = df.loc[df["Cores"] == df["Cores"].min(), "Total_Time_srun"].iloc[0]
    else:
        t1 = df.loc[df["Cores"] == 1, "Total_Time_srun"].iloc[0]

    df["Speedup"] = t1 / df["Total_Time_srun"]

    ax.plot(
        df["Cores"],
        df["Speedup"],
        marker="o",
        linestyle="-",
        color="#4c72b0",
        linewidth=2,
        markersize=8,
        label="Measured Speedup",
    )

    ax.set_title(
        f"Testcase {testcase_num}: Speedup with Increasing Parallelism",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Processes (p)", fontsize=10)
    ax.set_ylabel("Speedup (T₁ / Tₚ)", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xticks(df["Cores"])
    ax.set_xticklabels(df["Cores"].astype(str), fontsize=8)
    ax.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig(speedup_factor_img, dpi=300)
    print(f"Saved: {speedup_factor_img}")
    plt.close()

print(f"\n{'='*60}")
print(f"All plots generated successfully!")
print(f"Output directory: {OUTPUT_DIR}/")
print(f"{'='*60}")
