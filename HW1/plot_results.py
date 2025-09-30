import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CSV_FILE = 'scaling_results.csv'
TIME_PROFILE_IMG = 'time_profile_bar.png'
SPEEDUP_FACTOR_IMG = 'speedup_factor.png'

# --- Data Loading and Cleaning ---
try:
    df = pd.read_csv(CSV_FILE)
    # Force Comm_Time to 0 for single core case for logical correctness
    df.loc[df['Cores'] == 1, 'Comm_Time'] = 0.0
    # Convert all relevant columns to numeric
    for col in ['Total_Time_srun', 'IO_Time', 'Comm_Time', 'CPU_Time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Recalculate CPU time to ensure no negative values
    df['CPU_Time'] = df.apply(lambda row: max(0, row['Total_Time_srun'] - row['IO_Time'] - row['Comm_Time']), axis=1)
    df.dropna(inplace=True)
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while processing the CSV file: {e}")
    exit()

print("CSV data loaded and cleaned successfully.")
print(df)

# --- Plot 1: Time Profile (Stacked Bar Chart) with Total Time Labels ---
print("Generating Time Profile plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))

x_labels = df['Cores'].astype(str)
x_pos = np.arange(len(x_labels))
y_cpu = df['CPU_Time']
y_comm = df['Comm_Time']
y_io = df['IO_Time']
bar_width = 0.6

ax.bar(x_pos, y_cpu, bar_width, label='CPU Time', color='#4c72b0')
ax.bar(x_pos, y_comm, bar_width, bottom=y_cpu, label='Communication Time', color='#dd8452')
ax.bar(x_pos, y_io, bar_width, bottom=y_cpu + y_comm, label='I/O Time', color='#55a868')

# Add total time labels on top of each bar
for i, (idx, row) in enumerate(df.iterrows()):
    total_time = row['Total_Time_srun']
    ax.text(x_pos[i], total_time + 0.3, f'{total_time:.2f}', 
            ha='center', va='bottom', fontsize=9)

ax.set_title('Time Profile vs. Number of Cores (Strong Scaling)', fontsize=18, fontweight='bold')
ax.set_xlabel('Number of Cores (p)', fontsize=14)
ax.set_ylabel('Execution Time (seconds)', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.grid(axis='y', linestyle='--', linewidth=0.5)
ax.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig(TIME_PROFILE_IMG, dpi=300)
print(f"'{TIME_PROFILE_IMG}' saved successfully.")

# --- Plot 2: Speedup Factor ---
print("Generating Speedup Factor plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax_speedup = plt.subplots(figsize=(12, 7))

# Get the correct T1 from the dataframe
t1 = df.loc[df['Cores'] == 1, 'Total_Time_srun'].iloc[0]
df['Speedup'] = t1 / df['Total_Time_srun']

ax_speedup.plot(df['Cores'], df['Speedup'], marker='o', linestyle='-', color='#4c72b0', label='Actual Speedup')
ax_speedup.plot(df['Cores'], df['Cores'], linestyle='--', color='#c44e52', label='Ideal Linear Speedup')

ax_speedup.set_title('Speedup Factor vs. Number of Cores', fontsize=16, fontweight='bold')
ax_speedup.set_xlabel('Number of Cores (p)', fontsize=12)
ax_speedup.set_ylabel('Speedup Factor (T1 / Tp)', fontsize=12)
ax_speedup.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_speedup.set_xticks(df['Cores'])
ax_speedup.set_xticklabels(df['Cores'].astype(str), rotation=45, ha='right')
ax_speedup.set_xlim(min(df['Cores']), max(df['Cores']))
ax_speedup.set_ylim(0)
ax_speedup.legend(loc='upper left')
plt.tight_layout()
plt.savefig(SPEEDUP_FACTOR_IMG, dpi=300)
print(f"'{SPEEDUP_FACTOR_IMG}' saved successfully.")
