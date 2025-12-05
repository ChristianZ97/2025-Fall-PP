import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

# 1. 讀取與排序
file_path = './detailed_perf_results.csv'
df = pd.read_csv(file_path)

def sort_key(testcase):
    match = re.match(r'([cp])(\d+)', testcase)
    if match:
        prefix = match.group(1)
        num = int(match.group(2))
        priority = 0 if prefix == 'c' else 1
        return (priority, num)
    return (2, 0)

df['sort_val'] = df['Testcase'].apply(sort_key)
df = df.sort_values('sort_val').drop(columns=['sort_val'])

# 統一用數值 x 軸，便於控制 bar 寬度
x = np.arange(len(df))
xtick_labels = df['Testcase']

# -------------------------------------------------------------------
# 圖一：絕對時間趨勢 (折線圖 + Log Scale)
# -------------------------------------------------------------------
plt.figure(figsize=(14, 7))
sns.set_style("ticks")
ax = plt.gca()
ax.set_facecolor('#f5f5f5')  # 淡灰背景

plt.plot(x, df['ComputeTime'], marker='o', label='Compute Time',
         color='#4F81BD', linewidth=2.2, markersize=5)
plt.plot(x, df['CommTime'], marker='^', label='Comm Time',
         color='#9BBB59', linewidth=2.2, markersize=5)
plt.plot(x, df['IOTime'], marker='s', label='I/O Time',
         color='#F2C37E', linewidth=2.2, markersize=5)

plt.title('Execution Time Trend (Log Scale)', fontsize=16, fontweight='bold')
plt.xlabel('Testcase', fontsize=14)
plt.ylabel('Time (seconds) - Log Scale', fontsize=14)
plt.yscale('log')
plt.legend(fontsize=12, frameon=True)
plt.xticks(x, xtick_labels, rotation=90)
plt.grid(True, which="both", ls="--", alpha=0.35)
plt.tight_layout()
plt.savefig('time_trend_log.png', dpi=300)
plt.show()

# -------------------------------------------------------------------
# 圖二：時間佔比分析 (Compute / Comm / IO 百分比堆疊)
# -------------------------------------------------------------------
totals = df['ComputeTime'] + df['CommTime'] + df['IOTime']
pct_compute = df['ComputeTime'] / totals * 100
pct_comm    = df['CommTime']   / totals * 100
pct_io      = df['IOTime']     / totals * 100

plt.figure(figsize=(14, 7))
sns.set_style("ticks")
ax = plt.gca()
ax.set_facecolor('#f5f5f5')

bar_width = 0.9  # 幾乎貼在一起，減少白縫

plt.bar(x, pct_compute, label='Compute',
        color='#4F81BD', alpha=0.9, width=bar_width)
plt.bar(x, pct_comm, bottom=pct_compute, label='Communication',
        color='#9BBB59', alpha=0.9, width=bar_width)
plt.bar(x, pct_io, bottom=pct_compute + pct_comm, label='I/O',
        color='#F2C37E', alpha=0.9, width=bar_width)

plt.title('Execution Time Breakdown (Percentage)', fontsize=16, fontweight='bold')
plt.xlabel('Testcase', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.xticks(x, xtick_labels, rotation=90)
plt.margins(x=0.005)
plt.tight_layout()
plt.savefig('time_breakdown_pct.png', dpi=300)
plt.show()

# -------------------------------------------------------------------
# 圖三：Compute Time Internal Breakdown (Phase 1 / 2 / 3 百分比)
# -------------------------------------------------------------------
compute_total = df['Phase1'] + df['Phase2'] + df['Phase3']
pct_p3 = df['Phase3'] / compute_total * 100
pct_p2 = df['Phase2'] / compute_total * 100
pct_p1 = df['Phase1'] / compute_total * 100

plt.figure(figsize=(14, 7))
sns.set_style("ticks")
ax = plt.gca()
ax.set_facecolor('#f5f5f5')

# 藍色系漸層，去掉白邊，bar 幾乎連在一起
plt.bar(x, pct_p3, label='Phase 3 (Inner)',
        color='#2E5F8B', alpha=0.95, width=bar_width)
plt.bar(x, pct_p2, bottom=pct_p3, label='Phase 2 (Panel)',
        color='#6BAED6', alpha=0.95, width=bar_width)
plt.bar(x, pct_p1, bottom=pct_p3 + pct_p2, label='Phase 1 (Pivot)',
        color='#C6DBEF', alpha=0.95, width=bar_width)

plt.title('Compute Time Breakdown: Kernel Phases', fontsize=16, fontweight='bold')
plt.xlabel('Testcase', fontsize=14)
plt.ylabel('Percentage of Compute Time (%)', fontsize=14)
plt.ylim(0, 100)
plt.legend(loc='lower right', fontsize=12, frameon=True)
plt.xticks(x, xtick_labels, rotation=90)
plt.margins(x=0.005)
plt.tight_layout()
plt.savefig('kernel_breakdown_pct.png', dpi=300)
plt.show()
