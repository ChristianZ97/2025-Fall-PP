import pandas as pd
import numpy as np
import sys
import os
import argparse


def compare_results(file_c, file_cu, tolerance=1e-5):
    """
    Compares two N-body simulation output CSV files for numerical consistency.
    """

    # 1. Check if files exist
    if not os.path.exists(file_c):
        print(f"Error: Base file '{file_c}' not found.")
        sys.exit(1)
    if not os.path.exists(file_cu):
        print(f"Error: Comparison file '{file_cu}' not found.")
        sys.exit(1)

    # 2. Load CSV files
    try:
        df_c = pd.read_csv(file_c)
        df_cu = pd.read_csv(file_cu)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        sys.exit(1)

    # 3. Standardize column names
    # Ensure these match the fprintf format in your C/CUDA code
    columns = ["step", "t", "id", "x", "y", "z", "vx", "vy", "vz", "m"]

    # Validation: Check if column count matches
    if len(df_c.columns) != len(columns) or len(df_cu.columns) != len(columns):
        print(
            f"Warning: Column count in CSV ({len(df_c.columns)}) does not match expected ({len(columns)})."
        )
        # Proceeding by assuming standard columns, but this might fail if format is very different.

    df_c.columns = columns
    df_cu.columns = columns

    # 4. Structure Check: Verify 'step' and 'id' alignment
    # If the rows are not in the same order, value comparison is meaningless.
    if not np.array_equal(df_c[["step", "id"]], df_cu[["step", "id"]]):
        print("Error: File structure mismatch.")
        print(
            "The 'step' or 'id' columns do not align perfectly between the two files."
        )
        sys.exit(1)

    # 5. Numerical Comparison
    diff_cols = ["x", "y", "z", "vx", "vy", "vz"]

    # Use numpy's allclose for efficient comparison with tolerance
    is_close = np.allclose(df_c[diff_cols], df_cu[diff_cols], atol=tolerance)

    if is_close:
        print(f"PASSED: Results match within tolerance ({tolerance}).")
        return True
    else:
        print("FAILED: Significant differences detected.")

        # Calculate absolute differences
        diff = np.abs(df_c[diff_cols] - df_cu[diff_cols])

        # Report maximum error per column
        print("\nMaximum Absolute Difference per Column:")
        print(diff.max())

        # Identify the row with the worst error
        max_diff_idx = diff.max(axis=1).idxmax()
        step_num = df_c.iloc[max_diff_idx]["step"]
        particle_id = df_c.iloc[max_diff_idx]["id"]

        print(
            f"\nWorst discrepancy found at Row {max_diff_idx} (Step {step_num}, Particle {particle_id}):"
        )
        print("-" * 30)
        print(f"{'Column':<5} | {'Base (C)':<12} | {'Target (Cu)':<12} | {'Diff':<12}")
        print("-" * 30)
        for col in diff_cols:
            val_c = df_c.iloc[max_diff_idx][col]
            val_cu = df_cu.iloc[max_diff_idx][col]
            val_diff = diff.iloc[max_diff_idx][col]
            mark = "(*)" if val_diff > tolerance else ""
            print(
                f"{col:<5} | {val_c:12.6f} | {val_cu:12.6f} | {val_diff:12.6f} {mark}"
            )
        print("-" * 30)

        sys.exit(1)


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Compare N-Body simulation results.")

    # Add arguments for the two file paths
    parser.add_argument(
        "file_c",
        nargs="?",
        default="traj_10.csv",
        help="Path to the base CSV file (e.g., C output)",
    )
    parser.add_argument(
        "file_cu",
        nargs="?",
        default="traj_10_cu.csv",
        help="Path to the comparison CSV file (e.g., CUDA output)",
    )

    args = parser.parse_args()

    # Run comparison
    compare_results(args.file_c, args.file_cu)
