import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
csv_file = 'optimization_perf_summary.csv'
output_image = 'performance_optimization_chart_labels_top.png'

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

try:
    # --- Step 1: Read Data ---
    df = pd.read_csv(csv_file)
    df['VersionID'] = pd.to_numeric(df['VersionID'], errors='coerce')
    df = df.sort_values('VersionID').reset_index(drop=True)

    # --- Step 2: Calculate Percentage Change ---
    df['Time_s'] = df['GrandTotalTime_ms'] / 1000.0
    
    # 計算變化率: (新 - 舊) / 舊
    previous_time = df['Time_s'].shift(1)
    df['Change_pct'] = ((df['Time_s'] - previous_time) / previous_time) * 100
    
    df['Label'] = df['Description'].str.replace('_', ' ').str.title()

    # --- Step 3: Plotting ---
    plt.figure(figsize=(12, 8)) # 稍微加高一點，留空間給上面的字
    # 使用深綠色
    ax = sns.barplot(x='Label', y='Time_s', data=df, color='#4d7773')

    # --- Step 4: Formatting ---
    plt.title('Performance Optimization Steps', fontsize=18, pad=30, fontweight='bold')
    plt.xlabel('Optimization Method', fontsize=14, labelpad=15)
    plt.ylabel('Elapsed Time (s)', fontsize=14, labelpad=15)
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    sns.despine(top=True, right=True)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
    ax.set_axisbelow(True)

    # 設定 Y 軸上限，留出空間給 Bar 上面的文字，避免被切掉
    max_height = df['Time_s'].max()
    ax.set_ylim(0, max_height * 1.2) 

    # --- Step 5: Add Labels ---
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        x_center = p.get_x() + p.get_width() / 2

        # 1. [修改處] 顯示主要時間：在 Bar 上方 (Outside)
        time_str = f'{height:.3f}s'
        # y 座標設為 height + 一點點偏移
        ax.text(x_center, height + (max_height * 0.01), time_str, 
                ha='center', va='bottom', color='black', fontsize=11, fontweight='bold')

        # 2. 顯示變化百分比：保留在 Bar 內部 (Inside)
        # 如果 Bar 太矮 (例如 < 5% max_height)，把百分比也移到外面，或是隱藏
        change = df['Change_pct'].iloc[i]
        
        if pd.notna(change):
            if change > 0: # Worsen
                change_str = f'(+{change:.1f}%)'
                text_color = '#ffcccb' # 淺紅警告色
                font_weight = 'bold'
            elif change < 0: # Improve
                change_str = f'({change:.1f}%)'
                text_color = 'white' # 白色
                font_weight = 'normal'
            else:
                change_str = '(0.0%)'
                text_color = 'white'
                font_weight = 'normal'

            # 只有當 Bar 夠高時才印在裡面，不然會很醜
            if height > max_height * 0.05:
                ax.text(x_center, height - (max_height * 0.02), change_str,
                        ha='center', va='top', color=text_color, 
                        fontsize=10, fontweight=font_weight)
            else:
                # Bar 太矮，印在時間上面 (雖然這種情況很少見)
                pass

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Chart saved to {output_image}")
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
