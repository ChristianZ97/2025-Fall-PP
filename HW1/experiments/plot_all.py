import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob
import os

# --- Configuration ---
CSV_PATTERN = "results/scaling_results_*.csv"
OUTPUT_IMG = "combined_speedup_subplots.png"
OUTPUT_SINGLE = "combined_speedup_single.png"

# 找到所有 CSV 檔案
csv_files = sorted(glob.glob(CSV_PATTERN))

if not csv_files:
    print(f"Error: No CSV files found matching pattern '{CSV_PATTERN}'")
    exit()

print(f"Found {len(csv_files)} CSV files to process")

# --- Process data first ---
all_data = []

for csv_file in csv_files:
    testcase_num = csv_file.split("_")[-1].replace(".csv", "")

    try:
        df = pd.read_csv(csv_file)

        if df.empty or "N" not in df.columns:
            print(f"Skipping {csv_file}: empty or missing N column")
            continue

        # 數值轉換
        for col in ["Total_Time_srun", "IO_Time", "Comm_Time", "CPU_Time"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(inplace=True)

        if df.empty or 1 not in df["Cores"].values:
            print(
                f"Warning: Testcase {testcase_num} has no valid data or 1-core baseline"
            )
            continue

        N = df["N"].iloc[0]
        t1 = df.loc[df["Cores"] == 1, "Total_Time_srun"].iloc[0]
        df["Speedup"] = t1 / df["Total_Time_srun"]
        df["Efficiency"] = df["Speedup"] / df["Cores"]

        all_data.append(
            {
                "testcase": testcase_num,
                "N": N,
                "log_N": np.log10(N) if N > 0 else 0,
                "df": df,
                "max_speedup": df["Speedup"].max(),
                "max_cores": df["Cores"].max(),
                "avg_efficiency": df["Efficiency"].mean(),
            }
        )

        print(
            f"Processed TC{testcase_num}: N={N:,}, max speedup={df['Speedup'].max():.2f}x"
        )

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        continue

if not all_data:
    print("No valid data to plot")
    exit()

# --- 根據數量級分組（固定邊界）---
# 根據你的數據分佈，建議分成 5 組：
# Group 1: N ≤ 1,000 (small)
# Group 2: 1,000 < N ≤ 20,000 (medium-small)
# Group 3: 20,000 < N ≤ 200,000 (medium)
# Group 4: 200,000 < N ≤ 10,000,000 (large)
# Group 5: N > 10,000,000 (very large)

bin_boundaries = [0, 1000, 20000, 200000, 10000000, float("inf")]
bin_labels = [
    "Small (N ≤ 1K)",
    "Medium-Small (1K < N ≤ 20K)",
    "Medium (20K < N ≤ 200K)",
    "Large (200K < N ≤ 10M)",
    "Very Large (N > 10M)",
]

# 分配組別
for data in all_data:
    N = data["N"]
    for i in range(len(bin_boundaries) - 1):
        if bin_boundaries[i] < N <= bin_boundaries[i + 1]:
            data["bin"] = i
            data["bin_label"] = bin_labels[i]
            break

# 排序數據
all_data.sort(key=lambda x: (x["bin"], x["N"]))

# 計算實際的組數
num_bins = len(set(d["bin"] for d in all_data))

# --- 定義色系 ---
colormaps = ["Blues", "Greens", "Oranges", "Reds", "Purples", "YlOrBr"]

# ============================================
# 方案 1: 子圖版本（推薦）
# ============================================
# 只創建有數據的子圖
bins_with_data = sorted(set(d["bin"] for d in all_data))
num_subplots = len(bins_with_data)

fig, axes = plt.subplots(num_subplots, 1, figsize=(16, 4 * num_subplots), sharex=True)
if num_subplots == 1:
    axes = [axes]

subplot_idx = 0
for bin_idx in bins_with_data:
    ax = axes[subplot_idx]
    bin_data = [d for d in all_data if d["bin"] == bin_idx]

    if not bin_data:
        continue

    bin_data.sort(key=lambda x: x["N"])
    cmap = cm.get_cmap(colormaps[bin_idx % len(colormaps)])
    n_items = len(bin_data)

    min_N = min(d["N"] for d in bin_data)
    max_N = max(d["N"] for d in bin_data)
    bin_label = bin_data[0]["bin_label"]

    # 繪製每條曲線
    for i, data in enumerate(bin_data):
        color = cmap(0.4 + 0.6 * i / max(n_items - 1, 1))
        df = data["df"]

        ax.plot(
            df["Cores"],
            df["Speedup"],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=5,
            color=color,
            label=f"TC{data['testcase']} (N={data['N']:,})",
            alpha=0.85,
        )

    # 設定軸標籤和標題
    ax.set_ylabel("Speedup (T₁ / Tₚ)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Group {subplot_idx+1}: {bin_label} — [{min_N:,} to {max_N:,}] — {len(bin_data)} testcases",
        fontsize=13,
        fontweight="bold",
        pad=10,
        loc="left",
    )
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    # Legend 設定
    ncol = min(4, (len(bin_data) + 3) // 10 + 1)
    ax.legend(loc="upper left", fontsize=9, ncol=ncol, framealpha=0.95, shadow=True)

    # Y 軸從 0 開始
    ax.set_ylim(bottom=0)

    subplot_idx += 1

# 設定 X 軸標籤
axes[-1].set_xlabel("Number of Processes (p)", fontsize=13, fontweight="bold")

# 總標題
fig.suptitle(
    "Strong Scaling: Speedup Comparison (Stratified by Problem Size)",
    fontsize=16,
    fontweight="bold",
    y=0.9995,
)

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches="tight")
print(f"\n{'='*60}")
print(f"Subplot version saved: {OUTPUT_IMG}")
print(f"{'='*60}\n")
plt.close()

# ============================================
# 方案 2: 單圖版本（改進配色）
# ============================================
fig, ax = plt.subplots(figsize=(16, 10))

for bin_idx in bins_with_data:
    bin_data = [d for d in all_data if d["bin"] == bin_idx]

    if not bin_data:
        continue

    bin_data.sort(key=lambda x: x["N"])
    cmap = cm.get_cmap(colormaps[bin_idx % len(colormaps)])
    n_items = len(bin_data)

    for i, data in enumerate(bin_data):
        color = cmap(0.4 + 0.6 * i / max(n_items - 1, 1))
        df = data["df"]

        ax.plot(
            df["Cores"],
            df["Speedup"],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"TC{data['testcase']} (N={data['N']:,})",
            alpha=0.75,
        )

ax.set_title(
    "Strong Scaling: Speedup Comparison (Grouped by Problem Size)",
    fontsize=16,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Number of Processes (p)", fontsize=13, fontweight="bold")
ax.set_ylabel("Speedup (T₁ / Tₚ)", fontsize=13, fontweight="bold")
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_ylim(bottom=0)

# Legend 放在圖外
ax.legend(
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    fontsize=8,
    ncol=1,
    framealpha=0.95,
    shadow=True,
)

plt.tight_layout()
plt.savefig(OUTPUT_SINGLE, dpi=300, bbox_inches="tight")
print(f"Single plot version saved: {OUTPUT_SINGLE}\n")
plt.close()

# ============================================
# 生成摘要統計
# ============================================
summary_data = []
for data in all_data:
    summary_data.append(
        {
            "Testcase": f"TC{data['testcase']}",
            "N": data["N"],
            "log10(N)": f"{data['log_N']:.2f}",
            "Group": data["bin"] + 1,
            "Group_Label": data["bin_label"],
            "Max_Speedup": f"{data['max_speedup']:.2f}",
            "Max_Cores": data["max_cores"],
            "Avg_Efficiency": f"{data['avg_efficiency']:.2%}",
        }
    )

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(["Group", "N"])

print("=" * 100)
print("TESTCASE SUMMARY (Grouped by Problem Size)")
print("=" * 100)
print(summary_df.to_string(index=False))
print("=" * 100)

summary_df.to_csv("testcase_summary.csv", index=False)
print(f"\nSummary saved to: testcase_summary.csv")

# 組別統計
print("\n" + "=" * 100)
print("GROUP STATISTICS")
print("=" * 100)
for bin_idx in bins_with_data:
    bin_data = [d for d in all_data if d["bin"] == bin_idx]
    if bin_data:
        min_N = min(d["N"] for d in bin_data)
        max_N = max(d["N"] for d in bin_data)
        avg_speedup = np.mean([d["max_speedup"] for d in bin_data])
        avg_efficiency = np.mean([d["avg_efficiency"] for d in bin_data])
        bin_label = bin_data[0]["bin_label"]

        print(
            f"Group {bin_idx+1} ({bin_label:30s}): {len(bin_data):2d} testcases | "
            f"N range: [{min_N:>12,}, {max_N:>12,}] | "
            f"Avg max speedup: {avg_speedup:5.2f}x | "
            f"Avg efficiency: {avg_efficiency:5.1%}"
        )
print("=" * 100)
