import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 讀取數據
file_path = './weak_scaling.csv'
df = pd.read_csv(file_path)

# ★ 修改 X 軸標籤格式為 (Case1, Case2)
df['PairLabel'] = df.apply(lambda x: f"({x['GPU1_Case']}, {x['GPU2_Case']})", axis=1)

# 2. 畫圖
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# 柱狀圖
ax = sns.barplot(x='PairLabel', y='Efficiency', data=df, color='#1f77b4', alpha=0.85)

# 畫理想線
plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Ideal Efficiency (1.0)')

# 遍歷每個柱子標註數值
for i, p in enumerate(ax.patches):
    height = p.get_height()
    
    # 頂部標註 Efficiency
    ax.text(p.get_x() + p.get_width() / 2., height + 0.02,
            f'{height:.2f}',
            ha="center", va='bottom', fontsize=12, fontweight='bold', color='#333333')
    
    # 中間標註 Workload Ratio
    workload_ratio = df.iloc[i]['WorkloadRatio']
    ax.text(p.get_x() + p.get_width() / 2., height / 2.,
            f'W-Load:\n{workload_ratio:.2f}x',
            ha="center", va='center', fontsize=10, color='white', fontweight='bold')

plt.title('Weak Scaling Efficiency on Multi-GPU', fontsize=16)
plt.xlabel('Testcase Pair (1 GPU, 2 GPUs)', fontsize=14)
plt.ylabel('Efficiency (Ideal = 1.0)', fontsize=14)
plt.ylim(0, 1.15)
plt.legend(fontsize=12, loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('weak_scaling_bar_tuple.png', dpi=300)
plt.show()
