#!/usr/bin/env python3
"""
plot_load_balance_results.py
使用 adjustText 處理編號標籤（帶箭頭），上方用單一表格顯示完整資訊（左上角）
"""

import pandas as pd  # 用於讀取和處理 CSV 資料
import matplotlib.pyplot as plt  # 繪圖主要套件
import numpy as np  # 數值計算套件
import os  # 檔案系統操作

# 嘗試匯入 adjustText，若不存在則自動安裝
try:
    from adjustText import adjust_text  # 自動調整文字標籤位置以避免重疊
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "adjustText"])
    from adjustText import adjust_text

# ==================== 配置 ====================
CSV_FILE = "summary_results.csv"  # 輸入資料檔案
OUTPUT_DIR = "load_balance_plots"  # 輸出圖片的目錄
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 建立輸出目錄（若已存在則不報錯）

# 定義圖表配色
COLOR_HW2A = "#2E86AB"  # hw2a 使用藍色系
COLOR_HW2B = "#A23B72"  # hw2b 使用紫紅色系

# 統一圖表大小（寬26英寸 x 高18英寸）
FIGURE_SIZE = (26, 18)

# ==================== 讀取資料 ====================
print("讀取資料中...")
df = pd.read_csv(CSV_FILE)  # 從 CSV 讀取所有資料

# 確保數值欄位為數字類型（若有非數字會轉為 NaN）
df["Total_Execution_Time"] = pd.to_numeric(df["Total_Execution_Time"], errors="coerce")
df["Avg_Thread_Imbalance_Pct"] = pd.to_numeric(
    df["Avg_Thread_Imbalance_Pct"], errors="coerce"
)

# 建立簡短的欄位別名，方便後續使用
df["Total_Time"] = df["Total_Execution_Time"]
df["Imbalance_Pct"] = df["Avg_Thread_Imbalance_Pct"]
df["Strategy"] = df["Program"]

# 根據 Type 欄位分離 hw2a 和 hw2b 的資料，並按 Imbalance_Pct 排序
df_hw2a = df[df["Type"] == "hw2a"].sort_values("Imbalance_Pct").reset_index(drop=True)
df_hw2b = df[df["Type"] == "hw2b"].sort_values("Imbalance_Pct").reset_index(drop=True)

print(f"hw2a: {len(df_hw2a)} 個策略, hw2b: {len(df_hw2b)} 個策略")

# ==================== 圖 1: hw2a ====================
if not df_hw2a.empty:  # 確保有資料才繪圖
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)  # 建立圖表和座標軸

    # 繪製折線（連接所有資料點）
    ax.plot(
        df_hw2a["Imbalance_Pct"],
        df_hw2a["Total_Time"],
        linewidth=2.5,  # 線條寬度
        color=COLOR_HW2A,  # 線條顏色
        alpha=0.4,  # 透明度40%（讓線條較淡）
        zorder=1,
    )  # 圖層順序（1最底層）

    # 繪製散點圖（資料點）
    ax.scatter(
        df_hw2a["Imbalance_Pct"],
        df_hw2a["Total_Time"],
        s=100,  # 點的大小
        color=COLOR_HW2A,  # 點的顏色
        edgecolor="white",  # 點的邊框顏色
        linewidth=2.5,  # 邊框寬度
        zorder=3,  # 圖層順序（3在折線之上）
        alpha=0.9,
    )  # 透明度90%

    # ===== 建立編號標籤 =====
    # 在每個資料點旁邊標註編號（1, 2, 3...）
    number_texts = []  # 儲存所有文字物件
    for idx, row in df_hw2a.iterrows():
        txt = ax.text(
            row["Imbalance_Pct"],  # X 座標（初始位置在資料點上）
            row["Total_Time"],  # Y 座標
            str(idx + 1),  # 顯示的文字（編號從1開始）
            fontsize=12,
            fontweight="bold",
            color=COLOR_HW2A,
            ha="center",  # 水平置中
            va="center",  # 垂直置中
            bbox=dict(  # 文字外框設定
                boxstyle="round,pad=0.35",  # 圓角矩形，內距0.35
                facecolor="white",  # 背景白色
                edgecolor=COLOR_HW2A,  # 邊框顏色
                linewidth=1.5,
                alpha=0.95,
            ),  # 幾乎不透明
            zorder=4,
        )  # 圖層順序（4在資料點之上）
        number_texts.append(txt)

    # ===== 使用 adjustText 自動調整編號位置 =====
    adjust_text(
        number_texts,  # 要調整的文字物件列表
        ax=ax,
        arrowprops=dict(  # 箭頭設定（從標籤指向資料點）
            arrowstyle="->", color=COLOR_HW2A, lw=0.8, alpha=0.6  # 箭頭樣式  # 線寬
        ),  # 透明度
        expand_points=(3.0, 4.0),  # 資料點周圍的排斥範圍（X方向3倍，Y方向4倍）
        expand_text=(8.0, 10.0),  # 文字框之間的排斥範圍
        force_text=(0.8, 1.0),  # 文字移動的力道
        force_points=(0.5, 0.6),  # 遠離資料點的力道
        lim=30000,  # 最大迭代次數
        only_move={"text": "xy"},
    )  # 只移動文字，資料點位置不變

    # ===== 建立上方的策略對照表 =====
    table_lines = []
    for idx, row in df_hw2a.iterrows():
        # 格式化每一行：[編號] 策略名稱 (時間, 不平衡度)
        # {idx + 1:1d} = 編號，至少1位數
        # {row['Strategy']:30s} = 策略名稱，左對齊30字元寬
        # {row['Total_Time']:6.1f} = 時間，6字元寬含1位小數
        # {row['Imbalance_Pct']:5.1f} = 不平衡度，5字元寬含1位小數
        line = f"[{idx + 1:1d}] {row['Strategy']:30s} ({row['Total_Time']:6.1f}s, {row['Imbalance_Pct']:5.1f}%)"
        table_lines.append(line)

    table_text = "\n".join(table_lines)  # 用換行符號連接所有行

    # ===== 在圖表左上角放置表格 =====
    ax.text(
        0.02,
        0.98,  # 相對座標 (X=2%, Y=98%)，即左上角
        table_text,
        fontsize=18,
        ha="left",  # 文字從左側開始
        va="top",  # 文字從頂部向下延伸
        family="monospace",  # 等寬字體確保對齊整齊
        color="#333333",  # 深灰色文字
        bbox=dict(  # 表格外框
            boxstyle="round,pad=0.6",
            facecolor="white",
            alpha=0.95,
            edgecolor=COLOR_HW2A,
            linewidth=2.0,
        ),
        transform=ax.transAxes,  # 使用相對座標系統（0-1範圍）
        zorder=10,
    )  # 最上層

    # ===== 圖表美化設定 =====
    ax.set_xlabel("Thread Imbalance (%)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Total Execution Time (seconds)", fontsize=20, fontweight="bold")
    ax.set_title(
        "hw2a (pthread) Load Balancing Strategies",
        fontsize=30,
        fontweight="bold",
        pad=25,
    )
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)  # 背景網格
    ax.tick_params(labelsize=8)  # 刻度標籤字體大小

    # 自動調整佈局並儲存
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hw2a_load_balance.png", dpi=300, bbox_inches="tight")
    print(f"✓ 已儲存: {OUTPUT_DIR}/hw2a_load_balance.png")
    plt.close()  # 關閉圖表釋放記憶體

# ==================== 圖 2: hw2b ====================
# hw2b 的處理邏輯與 hw2a 幾乎相同，主要差異：
if not df_hw2b.empty:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # 繪製折線和散點
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
    )  # 使用方形標記（區別於 hw2a 的圓形）

    # 建立編號標籤
    number_texts = []
    for idx, row in df_hw2b.iterrows():
        txt = ax.text(
            row["Imbalance_Pct"],
            row["Total_Time"],
            str(idx + 1),
            fontsize=12,  # hw2b 的編號字體稍大
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

    # 自動調整編號位置
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

    # 建立表格文字
    table_lines = []
    for idx, row in df_hw2b.iterrows():
        # hw2b 的策略名稱較長，使用50字元寬度
        line = f"[{idx + 1:2d}] {row['Strategy']:50s} ({row['Total_Time']:6.1f}s, {row['Imbalance_Pct']:5.1f}%)"
        table_lines.append(line)

    table_text = "\n".join(table_lines)

    # 表格位置：X=0.2（右移20%），Y=0.98（頂部）
    ax.text(
        0.2,
        0.98,
        table_text,  # hw2b 的表格右移以避免遮擋資料
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

    # 圖表美化
    ax.set_xlabel("Thread Imbalance (%)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Total Execution Time (seconds)", fontsize=20, fontweight="bold")
    ax.set_title(
        "hw2b (MPI+OpenMP) Load Balancing Strategies",
        fontsize=30,
        fontweight="bold",
        pad=25,
    )
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hw2b_load_balance.png", dpi=300, bbox_inches="tight")
    print(f"✓ 已儲存: {OUTPUT_DIR}/hw2b_load_balance.png")
    plt.close()

# ==================== 統計摘要 ====================
# 在終端機輸出最佳和最差的策略
print("\n" + "=" * 90)
print("統計摘要")
print("=" * 90)

if not df_hw2a.empty:
    # 找出執行時間最短和最長的策略
    best = df_hw2a.loc[df_hw2a["Total_Time"].idxmin()]  # idxmin() 返回最小值的索引
    worst = df_hw2a.loc[df_hw2a["Total_Time"].idxmax()]
    print(
        f"\nhw2a 最佳: {best['Strategy']} ({best['Total_Time']:.2f}s, {best['Imbalance_Pct']:.2f}%)"
    )
    print(
        f"hw2a 最差: {worst['Strategy']} ({worst['Total_Time']:.2f}s, {worst['Imbalance_Pct']:.2f}%)"
    )

if not df_hw2b.empty:
    best = df_hw2b.loc[df_hw2b["Total_Time"].idxmin()]
    worst = df_hw2b.loc[df_hw2b["Total_Time"].idxmax()]
    print(
        f"\nhw2b 最佳: {best['Strategy']} ({best['Total_Time']:.2f}s, {best['Imbalance_Pct']:.2f}%)"
    )
    print(
        f"hw2b 最差: {worst['Strategy']} ({worst['Total_Time']:.2f}s, {worst['Imbalance_Pct']:.2f}%)"
    )

print("=" * 90)
