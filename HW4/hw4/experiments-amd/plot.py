import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 設定全域繪圖風格
# ==========================================
sns.set_context("talk")
sns.set_style("whitegrid", {
    "grid.linestyle": ":", 
    "axes.edgecolor": ".3",
    "grid.color": ".85"
})
plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 2. 資料處理
# ==========================================
file_path = 'block_perf_results.csv'
df = pd.read_csv(file_path)

# 簡化 X 軸標籤
df['ConfigLabel'] = df.apply(lambda x: f"({x['BR']}, {x['BC']})", axis=1)

# 聚合數據
df_agg = df.groupby(['ConfigLabel', 'BR']).agg({
    'TotalTime_ms': 'sum',
    'ComputeTime_ms': 'sum',
    'IOTime_ms': 'sum',
    'CommuTime_ms': 'sum'
}).reset_index()

# 依據 BR 排序
df_agg = df_agg.sort_values('BR')

# 轉為長格式 (只包含 Compute, IO, Comm)
df_melt_sub = df_agg.melt(
    id_vars=['ConfigLabel', 'BR'], 
    value_vars=['ComputeTime_ms', 'IOTime_ms', 'CommuTime_ms'], 
    var_name='Metric', 
    value_name='Time'
)

metric_map = {
    'ComputeTime_ms': 'Compute',
    'IOTime_ms': 'I/O',
    'CommuTime_ms': 'Communication'
}
df_melt_sub['Metric'] = df_melt_sub['Metric'].map(metric_map)

# ==========================================
# 3. 繪圖核心
# ==========================================
plt.figure(figsize=(14, 7))

# 定義配色 (Compute/IO/Comm)
palette_sub = {
    'Compute':       '#4F81BD',
    'Communication': '#9BBB59',
    'I/O':           '#F2C37E'
}

# 第一層：畫 Compute, Comm, IO (實線，不透明)
ax = sns.lineplot(
    data=df_melt_sub,
    x='ConfigLabel',
    y='Time',
    hue='Metric',
    style='Metric',
    palette=palette_sub,
    markers=True,
    dashes=False,
    linewidth=3,
    markersize=11,
    markeredgecolor='white',
    markeredgewidth=1.5,
    zorder=2 
)

# 第二層：單獨畫 Total Time (虛線，半透明)
plt.plot(
    df_agg['ConfigLabel'], 
    df_agg['TotalTime_ms'], 
    color='#C0504D',      # 深紅
    marker='o',           # 圓點
    linestyle='--',       # 虛線
    linewidth=4,          # 稍粗一點
    markersize=12,
    alpha=0.35,           # 透明度 35%
    label='Total',        
    zorder=1              
)

# ==========================================
# 4. 視覺優化
# ==========================================
plt.title('Performance Trend: (BR, BC) Configuration', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Block Size Configuration (BR, BC)', fontsize=14, fontweight='bold')
plt.ylabel('Accumulated Time (ms)', fontsize=14, fontweight='bold')

plt.xticks(rotation=0, fontsize=11) 
plt.yticks(fontsize=12)

# Legend 設定
handles, labels = ax.get_legend_handles_labels()
plt.legend(title=None, fontsize=12, loc='upper right', frameon=True, fancybox=True, framealpha=0.9)

# 數值標註 (Total Time)
# 找出最低點
min_total_pos = df_agg['TotalTime_ms'].values.argmin()

for i in range(df_agg.shape[0]):
    total_val = df_agg.iloc[i]['TotalTime_ms']
    
    if i == min_total_pos:
        # 最低點：使用空心橢圓圈起來 (facecolor='none', edgecolor='#C0504D')
        plt.text(i, total_val * 1.05, f'{int(total_val)}', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold', 
                 color='#C0504D',
                 bbox=dict(boxstyle="ellipse,pad=0.3", facecolor='none', edgecolor='#C0504D', linewidth=2))
    else:
        plt.text(i, total_val * 1.05, f'{int(total_val)}', 
                 ha='center', va='bottom', fontsize=11, color='#C0504D')

sns.despine()
plt.tight_layout()

plt.savefig('config_trend_transparent_total.png', dpi=300)
print("圖表已生成：config_trend_transparent_total.png")
plt.show()
