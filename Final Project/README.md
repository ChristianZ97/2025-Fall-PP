# N-Body Simulation (CPU & GPU)
This project simulates the N-body problem using both a standard C implementation (CPU) and a CUDA implementation (GPU).

## How to Run
Follow these steps to generate data, run simulations, and verify the results. In the examples, `<N>` is a placeholder for the number of bodies (e.g., 1000).

### Step 1: Generate Initial Conditions
This script generates an input file with initial positions, velocities, and masses for `<N>` bodies.
```
python gen_input.py <N> <output>.csv
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
srun -N1 -n1 ./nbody_c <input>.txt <output>.csv
```

#### Run the simulation on a GPU node
```
srun -p nvidia -N1 -n1 --gres=gpu:1 /usr/bin/time -p ./nbody_cu <input>.txt <output>.csv
srun -p amd -N1 -n1 --gres=gpu:1 /usr/bin/time -p ./nbody_hip <input>.txt <output>.csv
```

### Step 3: Verify Correctness
This Python script compares the CPU and GPU outputs to ensure they are numerically consistent within a given tolerance.

**Option A (Recommended, using `uv`):**
If you have `uv` installed, this command will automatically handle dependencies.
```
uv run ... <cpu_output> <gpu_output>
uv run --with pandas,numpy compare_nbody.py <ground_truth>.csv <predict>.csv
```

**Option B (Using a Python virtual environment):**
If you have `pandas` and `numpy` installed in your environment.
```
python compare_nbody.py <ground_truth>.csv <predict>.csv
```

### Step 4: Visualize the Results (Optional)
To create an animation of the trajectory, first copy the output file (`<output>.csv`) to your local machine.

#### Run the animation script locally
```
python animate.py <output>.csv <animate>.gif
```

### Step 5: Profile the Results (Optional)

#### Profile with Nsight Systems
```
srun -p nvidia -N1 -n1 --gres=gpu:1 nsys profile -o nbody_cu --stats=true ./nbody_cu <input>.txt <output>.csv
```
#### Profile with Nsight Compute (if supported)
```
srun -p nvidia -N1 -n1 --gres=gpu:1 ncu -o nbody_cu --set full ./nbody_cu <input>.txt <output>.csv
```
#### Profile with nvprof
```
srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof
--query-metrics # check all metrics
--csv --log-file <output>.csv # store the profiled results

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof \
--kernels "<your kernel function>" \
--metrics <metric1>,<metric2>,<metric3> \
./nbody_cu <input>.txt <output>.csv


srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof \
  --kernels "compute_acceleration_kernel" \
  --metrics achieved_occupancy,sm_efficiency,warp_execution_efficiency,issue_slot_utilization \
  ./nbody_cu ./testcases/c1_in.txt temp.csv

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof \
  --kernels "compute_acceleration_kernel" \
  --metrics stall_exec_dependency,stall_memory_dependency,stall_sync \
  ./nbody_cu ./testcases/c1_in.txt temp.csv

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof \
  --kernels "compute_acceleration_kernel" \
  --metrics gld_efficiency,gst_efficiency,shared_load_transactions_per_request,shared_store_transactions_per_request \
  ./nbody_cu ./testcases/c1_in.txt temp.csv
```

- achieved_occupancy: Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor.
- sm_efficiency: The percentage of time at least one warp is active on a specific multiprocessor.
- warp_execution_efficiency: Ratio of the average active threads per warp to the maximum number of threads per warp supported on a multiprocessor.
- issue_slot_utilization: Percentage of issue slots that issued at least one instruction, averaged across all cycles.
- stall_exec_dependency: Percentage of stalls occurring because an input required by the instruction is not yet available.
- stall_memory_dependency: Percentage of stalls occurring because a memory operation cannot be performed due to the required resources not being available or fully utilized, or because too many requests of a given type are outstanding.
- stall_sync: Percentage of stalls occurring because the warp is blocked at a __syncthreads() call.
- gld_efficiency: Ratio of requested global memory load throughput to required global memory load throughput expressed as percentage.
- gst_efficiency: Ratio of requested global memory store throughput to required global memory store throughput expressed as percentage.
- shared_load_transactions_per_request: Average number of shared memory load transactions performed for each shared memory load.
- shared_store_transactions_per_request: Average number of shared memory store transactions performed for each shared memory store.

### Step 6: Demo

```
srun -p nvidia -N1 -n1 --gres=gpu:1 /usr/bin/time -p ./nbody_cu ./testcases/c1_in.txt ./temp.csv
srun -p nvidia -N1 -n1 --gres=gpu:1 /usr/bin/time -p ./nbody_cu ./testcases/c4_in.txt ./temp.csv
srun -p amd -N1 -n1 --gres=gpu:1 /usr/bin/time -p ./nbody_hip ./testcases/c4_in.txt ./temp.csv

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof \
  --kernels "compute_acceleration_kernel" \
  --metrics gld_efficiency,gst_efficiency,shared_store_transactions_per_request,achieved_occupancy,sm_efficiency,stall_exec_dependency,stall_memory_dependency,stall_sync \
  ./nbody_cu ./testcases/c1_in.txt temp.csv
```
