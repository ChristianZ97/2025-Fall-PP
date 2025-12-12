# N-Body Simulation (CPU & GPU)
This project simulates the N-body problem using both a standard C implementation (CPU) and a CUDA implementation (GPU).

## How to Run
Follow these steps to generate data, run simulations, and verify the results. In the examples, `<N>` is a placeholder for the number of bodies (e.g., 1000).

### Step 1: Generate Initial Conditions
This script generates an input file with initial positions, velocities, and masses for `<N>` bodies.
```
python gen_input.py <number_of_bodies> <output_filename>
python gen_input.py <N> input_<N>.txt
```

### Step 2: Compile and Run the Simulation
First, compile the code. Then, run the simulation using `srun`.

#### Compile the C version (ground truth) and the CUDA version
```
make
make debug # for debug version
make prof # for profiling version
```

#### Run the simulation
```
srun ./nbody_cpu <input_file> <output_trajectory_file>
srun -N1 -n1 ./nbody_c input_<N>.txt traj_<N>.csv
```

#### Run the simulation on a GPU node
```
srun ./nbody_gpu <input_file> <output_trajectory_file>
srun -p nvidia -N1 -n1 --gres=gpu:1 ./nbody_cu input_<N>.txt traj_<N>_cu.csv
```

### Step 3: Verify Correctness
This Python script compares the CPU and GPU outputs to ensure they are numerically consistent within a given tolerance.

**Option A (Recommended, using `uv`):**
If you have `uv` installed, this command will automatically handle dependencies.
```
uv run ... <cpu_output> <gpu_output>
uv run --with pandas,numpy compare_nbody.py traj_<N>.csv traj_<N>_cu.csv
```

**Option B (Using a Python virtual environment):**
If you have `pandas` and `numpy` installed in your environment.
```
python compare_nbody.py <cpu_output> <gpu_output>
python compare_nbody.py traj_<N>.csv traj_<N>_cu.csv
```

### Step 4: Visualize the Results (Optional)
To create an animation of the trajectory, first copy the output file (`traj_<N>.csv`) to your local machine.

#### Run the animation script locally
```
python animate.py traj_<N>.csv result_<N>.gif
```

### Step 5: Profile the Results (Optional)

#### Profile with Nsight Systems
```
srun -p nvidia -N1 -n1 --gres=gpu:1 nsys profile -o nbody_cu --stats=true ./nbody_cu input_<N>.txt traj_<N>_cu.csv
```
#### Profile with Nsight Compute (if supported)
```
srun -p nvidia -N1 -n1 --gres=gpu:1 ncu -o nbody_cu --set full ./nbody_cu input_<N>.txt traj_<N>_cu.csv
```
#### Profile with nvprof
```
srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof
--query-metrics # check all metrics
--csv --log-file <output>.csv # store the profiled results

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof   --kernels "compute_acceleration_kernel"   --metrics stall_exec_dependency,stall_memory_dependency,stall_sync   ./nbody_cu ./testcases/c1_in.txt temp.csv

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof   --kernels "compute_acceleration_kernel"   --metrics ipc,eligible_warps_per_cycle,issue_slot_utilization   ./nbody_cu ./testcases/c1_in.txt temp.csv

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof   --kernels "compute_acceleration_kernel"   --metrics achieved_occupancy,sm_efficiency,l2_tex_read_hit_rate,l2_tex_write_hit_rate   ./nbody_cu ./testcases/c1_in.txt temp.csv

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof   --kernels "compute_acceleration_kernel"   --metrics inst_fp_64,inst_fp_32,inst_integer   ./nbody_cu ./testcases/c1_in.txt temp.csv
```
#####

- stall_exec_dependency: Percentage of stalls occurring because an input required by the instruction is not yet available
- stall_memory_dependency: Percentage of stalls occurring because a memory operation cannot be performed due to the required resources not being available or fully utilized, or because too many requests of a given type are outstanding
- stall_sync: Percentage of stalls occurring because the warp is blocked at a __syncthreads() call

- ipc: Instructions executed per cycle
- eligible_warps_per_cycle: Average number of warps that are eligible to issue per active cycle
- issue_slot_utilization: Percentage of issue slots that issued at least one instruction, averaged across all cycles

- achieved_occupancy: Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
- sm_efficiency: The percentage of time at least one warp is active on a specific multiprocessor
- l2_tex_read_hit_rate: Hit rate at L2 cache for all read requests from texture cache
- l2_tex_write_hit_rate: Hit Rate at L2 cache for all write requests from texture cache

- inst_fp_64: Number of double-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)
- inst_fp_32: Number of single-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)
- inst_integer: Number of integer instructions executed by non-predicated threads
