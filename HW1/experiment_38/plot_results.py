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
    df.loc[df['Cores'] == 1, 'Comm_Time'] = 0.0
    for col in ['Total_Time_srun', 'IO_Time', 'Comm_Time', 'CPU_Time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
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

# --- Plot 1: Time Profile ---
print("Generating Time Profile plot...")
fig, ax = plt.subplots(figsize=(12, 7))

x_labels = df['Cores'].astype(str)
x_pos = np.arange(len(x_labels))
y_cpu = df['CPU_Time']
y_comm = df['Comm_Time']
y_io = df['IO_Time']
bar_width = 0.6

ax.bar(x_pos, y_cpu, bar_width, label='CPU Time', color='#4c72b0')
ax.bar(x_pos, y_comm, bar_width, bottom=y_cpu, label='Communication Time', color='#dd8452')
ax.bar(x_pos, y_io, bar_width, bottom=y_cpu + y_comm, label='I/O Time', color='#55a868')

for i, (idx, row) in enumerate(df.iterrows()):
    total_time = row['Total_Time_srun']
    ax.text(x_pos[i], total_time + 0.3, f'{total_time:.1f}', 
            ha='center', va='bottom', fontsize=7)

ax.set_title('Measured Execution Time with Increasing Parallelism', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Processes (p)', fontsize=10)
ax.set_ylabel('Runtime (seconds)', fontsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=8)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig(TIME_PROFILE_IMG, dpi=300)
print(f"'{TIME_PROFILE_IMG}' saved successfully.")
plt.close()

# --- Plot 2: Speedup Factor (Log Scale with Ideal) ---
print("Generating Speedup Factor plot...")
fig, ax = plt.subplots(figsize=(12, 7))

t1 = df.loc[df['Cores'] == 1, 'Total_Time_srun'].iloc[0]
df['Speedup'] = t1 / df['Total_Time_srun']

ax.plot(df['Cores'], df['Speedup'], marker='o', linestyle='-', 
        color='#4c72b0', linewidth=2, markersize=8, label='Measured Speedup')

# Ideal speedup line
#ideal_speedup = df['Cores'].values
#ax.plot(df['Cores'], ideal_speedup, linestyle='--', linewidth=1.5, 
#        color='gray', alpha=0.7, label='Ideal Speedup')

#ax.set_yscale('log')
#ax.set_xscale('log')

ax.set_title('Measured Speedup with Increasing Parallelism', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Processes (p)', fontsize=10)
ax.set_ylabel('Speedup (T₁ / Tₚ)', fontsize=10)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xticks(df['Cores'])
ax.set_xticklabels(df['Cores'].astype(str), fontsize=8)
ax.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig(SPEEDUP_FACTOR_IMG, dpi=300)
print(f"'{SPEEDUP_FACTOR_IMG}' saved successfully.")
plt.close()
