# run_tests.py

import os
import glob
import subprocess
import sys
import re

TEST_DIR = "./testcases"
EXECUTABLE = "./nbody_cu"
COMPARE_SCRIPT = "compare_nbody.py"
SRUN_CMD = ["srun", "-p", "nvidia", "-N1", "-n1", "--gres=gpu:1"]

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def run_tests():
    input_files = glob.glob(os.path.join(TEST_DIR, "*_in.txt"))
    input_files.sort()

    if not input_files:
        print(f"Error: No input files found in {TEST_DIR}")
        sys.exit(1)

    passed_count = 0

    for input_file in input_files:
        base_name = os.path.basename(input_file).replace("_in.txt", "")
        ground_truth_file = os.path.join(TEST_DIR, f"{base_name}_out.csv")
        user_output_file = f"{base_name}_run_out.csv"

        cmd = SRUN_CMD + [EXECUTABLE, input_file, user_output_file]

        try:
            result = subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stderr_output = result.stderr.decode()
            time_match = re.search(r"ELAPSED_TIME: ([\d.]+)", stderr_output)
            exec_time = float(time_match.group(1)) if time_match else 0.0
        except subprocess.CalledProcessError:
            print(f"{base_name:<8} {'N/A':>8}   {RED}failed{RESET}")
            continue

        compare_cmd = [
            "uv",
            "run",
            "--with",
            "pandas,numpy",
            COMPARE_SCRIPT,
            ground_truth_file,
            user_output_file,
        ]

        try:
            subprocess.run(compare_cmd, check=True, capture_output=True)
            status = f"{GREEN}accepted{RESET}"
            passed_count += 1
        except subprocess.CalledProcessError:
            status = f"{RED}failed{RESET}"

        print(f"{base_name:<8} {exec_time:>8.2f}s   {status}")

    print(f"\n{passed_count}/{len(input_files)} passed")

    if passed_count != len(input_files):
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
