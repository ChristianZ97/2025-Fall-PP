---
title: PP HW 5 Report Template

---

# PP HW 5 Report Template
> - Please include both brief and detailed answers.
> - The report should be based on the UCX code.
> - Describe the code using the 'permalink' from [GitHub repository](https://github.com/NTHU-LSALAB/UCX-lsalab).

## 1. Overview
> In conjunction with the UCP architecture mentioned in the lecture, please read [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/examples/ucp_hello_world.c)
1. Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API, including at least the following functions:
    - `ucp_init`
    - `ucp_worker_create`
    - `ucp_ep_create`

Context represents the global scope and resource management for the UCP library. In the code, `status = ucp_init(&ucp_params, config, &ucp_context);` initializes the UCP context. This function uses the configuration read earlier via `ucp_config_read` and the parameters specified in `ucp_params` (which includes features like OOB communication support) to set up the shared resources and capabilities available for the application.

Following context initialization, `status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);` is called to create a Worker. The Worker is created within the scope of a specific Context and utilizes the resources managed by that Context. It serves as the engine for progress, responsible for processing communication requests, managing operational queues, and polling hardware resources for event completions.

Finally, Endpoints are created to establish connections. The calls `status = ucp_ep_create(ucp_worker, &ep_params, &server_ep);` and `status = ucp_ep_create(ucp_worker, &ep_params, &client_ep);` instantiate Endpoints attached to the specific `ucp_worker`. An Endpoint represents a logical, persistent connection to a specific remote peer (whose details are provided in `ep_params`). An Endpoint cannot exist independently of a Worker, as it relies on the Worker to drive the underlying transport operations for all point-to-point communications.

2. UCX abstracts communication into three layers as below. Please provide a diagram illustrating the architectural design of UCX.
    - `ucp_context`
    - `ucp_worker`
    - `ucp_ep`
> Please provide detailed example information in the diagram corresponding to the execution of the command `srun -N 2 ./send_recv.out` or `mpiucx --host HostA:1,HostB:1 ./send_recv.out`

<img width="2160" height="1215" alt="ucx_arch" src="https://github.com/user-attachments/assets/562b726b-e5a2-4d9c-be1e-a07fe8815282" />

The diagram illustrates the UCX architecture during the execution of `srun -N 2 ./send_recv.out`, where two MPI processes (Rank 0 and Rank 1) communicate across two separate nodes (Host A and Host B). Host A (Rank 0) and Host B (Rank 1) each initialize a `ucp_context`. Both contexts detect and utilize the InfiniBand device `ibp3s0:1` as the underlying resource, configured to use the `ud_verbs` transport layer as specified by the environment (`UCX_TLS=ud_verbs`). A `ucp_worker` is created on each host (Addresses: `0x557...` on Host A, `0x556...` on Host B). These workers manage the progress of communication and interface with the network hardware. On Host A, the worker manages two endpoints: 1) `ucp_ep (self cfg#0)`: A loopback connection for intra-process communication; 2) `ucp_ep (inter-node cfg#1)`: The critical connection established to transmit data to the remote peer (Host B). On Host B, the worker manages its own `ucp_ep (self cfg#0)` for local operations. The dotted arrow represents the transmission of the message "Hello from rank 0". The message originates from Host A, travels through the `inter-node` endpoint using the `ud_verbs` protocol over the physical network, and is received by Host B's worker.

3. Based on the description in HW5, where do you think the following information is loaded/created?
    - `UCX_TLS`
    - TLS selected by UCX

## 2. Implementation
> Please complete the implementation according to the [spec](https://docs.google.com/document/d/1fmm0TFpLxbDP7neNcbLDn8nhZpqUBi9NGRzWjgxZaPE/edit?usp=sharing)
> Describe how you implemented the two special features of HW5.
1. Which files did you modify, and where did you choose to print Line 1 and Line 2?
2. How do the functions in these files call each other? Why is it designed this way?
3. Observe when Line 1 and 2 are printed during the call of which UCP API?
4. Does it match your expectations for questions **1-3**? Why?
5. In implementing the features, we see variables like lanes, tl_rsc, tl_name, tl_device, bitmap, iface, etc., used to store different Layer's protocol information. Please explain what information each of them stores.

## 3. Optimize System 
1. Below are the current configurations for OpenMPI and UCX in the system. Based on your learning, what methods can you use to optimize single-node performance by setting UCX environment variables?

```
-------------------------------------------------------------------
/opt/modulefiles/openmpi/ucx-pp:


module load     openmpi/5.0.8
prepend-path    PATH $ucx_prefix/bin
prepend-path    LD_LIBRARY_PATH $ucx_prefix/lib
setenv          UCX_TLS ud_verbs
setenv          UCX_NET_DEVICES ibp3s0:1
-------------------------------------------------------------------
```

1. Please use the following commands to test different data sizes for latency and bandwidth, to verify your ideas:
```bash
module load openmpi/ucx-pp
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_latency
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_bw
```
2. Please create a chart to illustrate the impact of different parameter options on various data sizes and the effects of different testsuite.
3. Based on the chart, explain the impact of different TLS implementations and hypothesize the possible reasons (references required).


### Advanced Challenge: Multi-Node Testing

This challenge involves testing the performance across multiple nodes. You can accomplish this by utilizing the sbatch script provided below. The task includes creating tables and providing explanations based on your findings. Notably, Writing a comprehensive report on this exercise can earn you up to 5 additional points.

- For information on sbatch, refer to the documentation at [Slurm's sbatch page](https://slurm.schedmd.com/sbatch.html).
- To conduct multi-node testing, use the following command:
```
cd ~/UCX-lsalab/test/
sbatch run.batch
```


## 4. Experience & Conclusion
1. What have you learned from this homework?
2. How long did you spend on the assignment?
3. Feedback (optional)
