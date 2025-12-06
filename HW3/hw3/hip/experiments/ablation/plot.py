import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 讀取數據
file_path = './ablation_perf_summary.csv'
df = pd.read_csv(file_path)

# --- 修改重點 1: 自動產生漂亮的標籤 ---
# 移除 'hw3-2_' 前綴，讓標籤更乾淨 (e.g., "no_pad", "baseline")
df['CleanName'] = df['Version'].str.replace('hw3-2_', '', regex=False)
df['Label'] = df['CleanName']

# --- 修改重點 2: 設定基準點 (Reference) ---
# 通常 Ablation 是跟 "Optimized" 版本比，看拿掉某個功能會變慢多少
# 如果你還是想跟 Baseline 比，把下面這行改成 'hw3-2_baseline' 即可
ref_version = 'hw3-2_opt' 

try:
    ref_time = df[df['Version'] == ref_version]['GrandTotalTime_ms'].values[0]
    ref_label = "Optimized"
except IndexError:
    # 萬一找不到 opt，就預設用第一筆當基準
    ref_time = df['GrandTotalTime_ms'].iloc[0]
    ref_version = df['Version'].iloc[0]
    ref_label = "Ref"

# 計算相對於基準的倍率 (Relative Time)
# > 1.0 代表比基準慢，< 1.0 代表比基準快
df['RelativeTime'] = df['GrandTotalTime_ms'] / ref_time

# 排序：依照時間長短排序
df = df.sort_values('GrandTotalTime_ms', ascending=False)

# 2. 畫圖
plt.figure(figsize=(12, 7)) # 稍微加寬一點
sns.set_style("whitegrid")

# --- 修改重點 3: 動態設定顏色 ---
# 基準版本用紅色，其他用藍色，baseline 用灰色
def get_color(name):
    if name == ref_version: return '#d62728'  # Red for Reference (Opt)
    if 'baseline' in name: return '#7f7f7f'   # Grey for Baseline
    return '#1f77b4'                          # Blue for others

colors = [get_color(v) for v in df['Version']]

ax = sns.barplot(x='GrandTotalTime_ms', y='Label', data=df, palette=colors)

# 標註數值
for i, p in enumerate(ax.patches):
    width = p.get_width()
    rel_time = df.iloc[i]['RelativeTime']
    
    # 格式化文字
    time_str = f'{width:,.0f} ms'
    ratio_str = f'{rel_time:.2f}x'
    
    # 在條形圖右側標註
    ax.text(width + (df['GrandTotalTime_ms'].max() * 0.02), 
            p.get_y() + p.get_height() / 2,
            f'{time_str} ({ratio_str})',
            va='center', fontsize=11, fontweight='bold', color='black')

plt.title(f'Ablation Study (Reference: {ref_version})', fontsize=16, pad=20)
plt.xlabel('Grand Total Time (ms)', fontsize=14)
plt.ylabel('Configuration', fontsize=14)

# 自動調整 X 軸範圍，留空間給文字
plt.xlim(0, df['GrandTotalTime_ms'].max() * 1.25)

plt.tight_layout()
plt.savefig('ablation_perf_bar.png', dpi=300)
plt.show()

print(f"Plot generated! Reference version: {ref_version} ({ref_time:.2f} ms)")
