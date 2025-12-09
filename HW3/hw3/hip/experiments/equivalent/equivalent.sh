#!/bin/bash


source /home/pp25/share/venv/bin/activate


srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute profile -n group0_hip -- ./hw3-2-group0-hip ./testcases-amd/p11k1 temp.out
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute analyze -p workloads/group0_hip/MI100/

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof ./hw3-2-group0-cuda ./testcases/p11k1 temp.out
srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput ./hw3-2-group0-cuda ./testcases/p11k1 temp.out


srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute profile -n group1_hip -- ./hw3-2-group1-hip ./testcases-amd/p11k1 temp.out
srun -p amd -N1 -n1 --gres=gpu:1 rocprof-compute analyze -p workloads/group1_hip/MI100/

srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof ./hw3-2-group0-cuda ./testcases/p11k1 temp.out
srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput ./hw3-2-group1-cuda ./testcases/p11k1 temp.out