
// hw3-3.hip with profiling

/* Headers */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/* Constants & Global Variables */

#define BLOCKING_FACTOR 64
#define HALF_BLOCK BLOCKING_FACTOR / 2
#define INF ((1 << 30) - 1)

static int *D; // Host pointer
static int *d_D[2]; // Device pointers for 2 GPUs
static int V, E; // Original vertices, edges
static int V_padded; // Padded vertices (multiple of 64)

hipStream_t stream_main[2], stream_row[2], stream_col[2];
hipEvent_t event_p1_done[2], event_p2_row_done[2], event_p2_col_done[2];

/* Helper Functions for int4 Optimization (From hw3-2) */

__device__ __forceinline__ void load_block_int4(const int *__restrict__ src, const int stride, int dst[BLOCKING_FACTOR][BLOCKING_FACTOR + 1]) {
    const int tid = threadIdx.y * HALF_BLOCK + threadIdx.x;
    const int iterations = (BLOCKING_FACTOR * BLOCKING_FACTOR) / (HALF_BLOCK * HALF_BLOCK * 4);
#pragma unroll
    for (int i = 0; i < iterations; ++i) {
        const int offset = tid + i * (HALF_BLOCK * HALF_BLOCK);
        const int r = offset / (BLOCKING_FACTOR / 4);
        const int c = (offset % (BLOCKING_FACTOR / 4)) * 4;
        int4 tmp = *((int4 *)(src + r * stride + c));
        dst[r][c] = tmp.x;
        dst[r][c + 1] = tmp.y;
        dst[r][c + 2] = tmp.z;
        dst[r][c + 3] = tmp.w;
    }
}

__device__ __forceinline__ void store_block_int4(int *__restrict__ dst, const int stride, int src[BLOCKING_FACTOR][BLOCKING_FACTOR + 1]) {
    const int tid = threadIdx.y * HALF_BLOCK + threadIdx.x;
    const int iterations = (BLOCKING_FACTOR * BLOCKING_FACTOR) / (HALF_BLOCK * HALF_BLOCK * 4);
#pragma unroll
    for (int i = 0; i < iterations; ++i) {
        const int offset = tid + i * (HALF_BLOCK * HALF_BLOCK);
        const int r = offset / (BLOCKING_FACTOR / 4);
        const int c = (offset % (BLOCKING_FACTOR / 4)) * 4;
        int4 tmp;
        tmp.x = src[r][c];
        tmp.y = src[r][c + 1];
        tmp.z = src[r][c + 2];
        tmp.w = src[r][c + 3];
        *((int4 *)(dst + r * stride + c)) = tmp;
    }
}

/* Function Prototypes */

__global__ void __launch_bounds__(1024) kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase2_col(int *d_D, const int r, const int V_padded, const int row_offset, const int row_limit);
__global__ void __launch_bounds__(1024) kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset);

void input(char *infile);
void output(char *outfile);

void block_FW(const int dev_id
#ifdef PROFILING
, double *time_phase1_ms,
double *time_phase2_row_ms,
double *time_phase2_col_ms,
double *time_phase3_ms
#endif
);

/* Main */

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

#ifdef PROFILING
    // -------------------------
    // Host-side HIP events for H2D / D2H
    // -------------------------
    hipEvent_t event_h2d_start, event_h2d_stop;
    hipEvent_t event_d2h_start, event_d2h_stop;
    hipEventCreate(&event_h2d_start);
    hipEventCreate(&event_h2d_stop);
    hipEventCreate(&event_d2h_start);
    hipEventCreate(&event_d2h_stop);

    // Phase-wise timing aggregates (device kernels)
    double time_phase1_ms = 0.0;
    double time_phase2_row_ms = 0.0;
    double time_phase2_col_ms = 0.0;
    double time_phase3_ms = 0.0;

    // CPU-side I/O timing
    double time_io_read_ms = 0.0;
    double time_io_write_ms = 0.0;
#endif

#ifdef PROFILING
    clock_t io_read_start = clock();
    input(argv[1]);
    clock_t io_read_end = clock();
    time_io_read_ms = 1000.0 * (io_read_end - io_read_start) / CLOCKS_PER_SEC;
#else
    input(argv[1]);
#endif

    // Use size_t for total size calculation
    size_t size = (size_t)V_padded * V_padded * sizeof(int);

    // Enable peer access between 2 GPUs
    hipSetDevice(0);
    hipDeviceEnablePeerAccess(1, 0);
    hipSetDevice(1);
    hipDeviceEnablePeerAccess(0, 0);

#ifdef PROFILING
    hipEventRecord(event_h2d_start, 0);
#endif

#pragma omp parallel num_threads(2)
    {
        const int dev_id = omp_get_thread_num();
        hipSetDevice(dev_id);

        hipStreamCreate(&stream_main[dev_id]);
        hipStreamCreate(&stream_row[dev_id]);
        hipStreamCreate(&stream_col[dev_id]);

        hipEventCreate(&event_p1_done[dev_id]);
        hipEventCreate(&event_p2_row_done[dev_id]);
        hipEventCreate(&event_p2_col_done[dev_id]);

        hipMalloc(&d_D[dev_id], size);

        // H2D: copy full matrix to each GPU
        hipMemcpy(d_D[dev_id], D, size, hipMemcpyHostToDevice);

#pragma omp barrier
#ifdef PROFILING
        if (dev_id == 0) {
            hipEventRecord(event_h2d_stop, 0);
            hipEventSynchronize(event_h2d_stop);
        }
#endif

        // Aggregation arrays for per-device phase timing
#ifdef PROFILING
        double time_phase1_ms_local = 0.0;
        double time_phase2_row_ms_local = 0.0;
        double time_phase2_col_ms_local = 0.0;
        double time_phase3_ms_local = 0.0;
        block_FW(dev_id, &time_phase1_ms_local, &time_phase2_row_ms_local, &time_phase2_col_ms_local, &time_phase3_ms_local);
#else
        block_FW(dev_id);
#endif

        hipDeviceSynchronize();

#ifdef PROFILING
        if (dev_id == 0) {
            time_phase1_ms += time_phase1_ms_local;
            time_phase2_row_ms += time_phase2_row_ms_local;
            time_phase2_col_ms += time_phase2_col_ms_local;
            time_phase3_ms += time_phase3_ms_local;
        } else {
            time_phase1_ms += time_phase1_ms_local;
            time_phase2_row_ms += time_phase2_row_ms_local;
            time_phase2_col_ms += time_phase2_col_ms_local;
            time_phase3_ms += time_phase3_ms_local;
        }
        if (dev_id == 0) {
            hipEventRecord(event_d2h_start, 0);
        }
#pragma omp barrier
#endif

        const int round = V_padded / BLOCKING_FACTOR;
        const int row_mid = round / 2;
        const int row_block_start = (dev_id == 0) ? 0 : row_mid;
        const int row_block_count = (dev_id == 0) ? row_mid : (round - row_mid);
        const int row_start = row_block_start * BLOCKING_FACTOR;
        const int row_num = row_block_count * BLOCKING_FACTOR;

        // Copy back only the owned rows from each GPU
        if (row_num > 0) {
            hipMemcpy(D + (size_t)row_start * V_padded, d_D[dev_id] + (size_t)row_start * V_padded, (size_t)row_num * V_padded * sizeof(int), hipMemcpyDeviceToHost);
        }

        hipFree(d_D[dev_id]);
        hipStreamDestroy(stream_main[dev_id]);
        hipStreamDestroy(stream_row[dev_id]);
        hipStreamDestroy(stream_col[dev_id]);
        hipEventDestroy(event_p1_done[dev_id]);
        hipEventDestroy(event_p2_row_done[dev_id]);
        hipEventDestroy(event_p2_col_done[dev_id]);

#ifdef PROFILING
        if (dev_id == 0) {
            hipEventRecord(event_d2h_stop, 0);
            hipEventSynchronize(event_d2h_stop);
        }
#pragma omp barrier
#endif
    } // end parallel region

#ifdef PROFILING
    clock_t io_write_start = clock();
    output(argv[2]);
    clock_t io_write_end = clock();
    time_io_write_ms = 1000.0 * (io_write_end - io_write_start) / CLOCKS_PER_SEC;
#else
    output(argv[2]);
#endif

#ifdef PROFILING
    // ============================================================
    // Profiling result: only compact summary line
    // ============================================================
    float t_h2d_f = 0.0f, t_d2h_f = 0.0f;
    hipEventElapsedTime(&t_h2d_f, event_h2d_start, event_h2d_stop);
    hipEventElapsedTime(&t_d2h_f, event_d2h_start, event_d2h_stop);

    double time_h2d_ms = (double)t_h2d_f;
    double time_d2h_ms = (double)t_d2h_f;
    double time_compute_total_ms = time_phase1_ms + time_phase2_row_ms + time_phase2_col_ms + time_phase3_ms;
    double time_comm_total_ms = time_h2d_ms + time_d2h_ms;
    double time_io_total_ms = time_io_read_ms + time_io_write_ms;
    double total_time_ms = time_compute_total_ms + time_comm_total_ms + time_io_total_ms;

    fprintf(stderr, "[PROF_RESULT],%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            total_time_ms,
            time_compute_total_ms,
            time_comm_total_ms,
            time_io_total_ms,
            time_phase1_ms,
            time_phase2_row_ms + time_phase2_col_ms,
            time_phase3_ms);

    hipEventDestroy(event_h2d_start);
    hipEventDestroy(event_h2d_stop);
    hipEventDestroy(event_d2h_start);
    hipEventDestroy(event_d2h_stop);
#endif // PROFILING

    return 0;
}

/* Function Definitions */

void block_FW(const int dev_id
#ifdef PROFILING
, double *time_phase1_ms,
double *time_phase2_row_ms,
double *time_phase2_col_ms,
double *time_phase3_ms
#endif
) {
    const int round = V_padded / BLOCKING_FACTOR;
    dim3 threads_per_block(HALF_BLOCK, HALF_BLOCK);
    const int row_mid = round / 2;
    const int start_row_block = (dev_id == 0) ? 0 : row_mid;
    const int num_row_blocks = (dev_id == 0) ? row_mid : (round - row_mid);

    // Grids
    dim3 grid_p2_row(round, 1);
    dim3 grid_p2_col(num_row_blocks, 1);
    dim3 grid_p3(round, num_row_blocks);

#ifdef PROFILING
    // Per-round timing aggregation via HIP events
    hipEvent_t e_p1_start, e_p1_stop;
    hipEvent_t e_p2_row_start, e_p2_row_stop;
    hipEvent_t e_p2_col_start, e_p2_col_stop;
    hipEvent_t e_p3_start, e_p3_stop;
    hipEventCreate(&e_p1_start);
    hipEventCreate(&e_p1_stop);
    hipEventCreate(&e_p2_row_start);
    hipEventCreate(&e_p2_row_stop);
    hipEventCreate(&e_p2_col_start);
    hipEventCreate(&e_p2_col_stop);
    hipEventCreate(&e_p3_start);
    hipEventCreate(&e_p3_stop);
#endif

    for (int r = 0; r < round; ++r) {
        const int owner_id = (r < row_mid) ? 0 : 1;

        // Phase 1 + Phase 2 row on owner device
        if (dev_id == owner_id) {
#ifdef PROFILING
            hipEventRecord(e_p1_start, stream_main[dev_id]);
#endif
            hipLaunchKernelGGL(kernel_phase1, 1, threads_per_block, 0, stream_main[dev_id], d_D[dev_id], r, V_padded);
#ifdef PROFILING
            hipEventRecord(e_p1_stop, stream_main[dev_id]);
#endif
            hipEventRecord(event_p1_done[dev_id], stream_main[dev_id]);
            hipStreamWaitEvent(stream_row[dev_id], event_p1_done[dev_id], 0);

#ifdef PROFILING
            hipEventRecord(e_p2_row_start, stream_row[dev_id]);
#endif
            hipLaunchKernelGGL(kernel_phase2_row, grid_p2_row, threads_per_block, 0, stream_row[dev_id], d_D[dev_id], r, V_padded);
#ifdef PROFILING
            hipEventRecord(e_p2_row_stop, stream_row[dev_id]);
#endif
            hipEventRecord(event_p2_row_done[dev_id], stream_row[dev_id]);

            // Send updated pivot row to peer device
            size_t copy_size = (size_t)BLOCKING_FACTOR * V_padded * sizeof(int);
            size_t offset = (size_t)r * BLOCKING_FACTOR * V_padded;
            const int peer_id = 1 - dev_id;
            hipMemcpyPeerAsync(d_D[peer_id] + offset, peer_id, d_D[dev_id] + offset, dev_id, copy_size, stream_main[dev_id]);
        }

#pragma omp barrier

        // Phase 2 col + Phase 3 on both devices, using updated pivot row
        if (dev_id == owner_id) {
            hipStreamWaitEvent(stream_main[dev_id], event_p2_row_done[dev_id], 0);
        }

        if (num_row_blocks > 0) {
#ifdef PROFILING
            hipEventRecord(e_p2_col_start, stream_col[dev_id]);
#endif
            hipLaunchKernelGGL(kernel_phase2_col, grid_p2_col, threads_per_block, 0, stream_col[dev_id], d_D[dev_id], r, V_padded, start_row_block, num_row_blocks);
#ifdef PROFILING
            hipEventRecord(e_p2_col_stop, stream_col[dev_id]);
#endif
            hipEventRecord(event_p2_col_done[dev_id], stream_col[dev_id]);
            hipStreamWaitEvent(stream_main[dev_id], event_p2_col_done[dev_id], 0);

#ifdef PROFILING
            hipEventRecord(e_p3_start, stream_main[dev_id]);
#endif
            hipLaunchKernelGGL(kernel_phase3, grid_p3, threads_per_block, 0, stream_main[dev_id], d_D[dev_id], r, V_padded, start_row_block);
#ifdef PROFILING
            hipEventRecord(e_p3_stop, stream_main[dev_id]);
#endif
        }

#ifdef PROFILING
        float t_p1 = 0, t_p2_row = 0, t_p2_col = 0, t_p3 = 0;

        if (dev_id == owner_id) {
            hipEventSynchronize(e_p1_stop);
            hipEventElapsedTime(&t_p1, e_p1_start, e_p1_stop);
            *time_phase1_ms += t_p1;

            hipEventSynchronize(e_p2_row_stop);
            hipEventElapsedTime(&t_p2_row, e_p2_row_start, e_p2_row_stop);
            *time_phase2_row_ms += t_p2_row;
        }

        if (num_row_blocks > 0) {
            hipEventSynchronize(e_p2_col_stop);
            hipEventElapsedTime(&t_p2_col, e_p2_col_start, e_p2_col_stop);
            *time_phase2_col_ms += t_p2_col;

            hipEventSynchronize(e_p3_stop);
            hipEventElapsedTime(&t_p3, e_p3_start, e_p3_stop);
            *time_phase3_ms += t_p3;
        }
#endif
    }

#ifdef PROFILING
    hipEventDestroy(e_p1_start);
    hipEventDestroy(e_p1_stop);
    hipEventDestroy(e_p2_row_start);
    hipEventDestroy(e_p2_row_stop);
    hipEventDestroy(e_p2_col_start);
    hipEventDestroy(e_p2_col_stop);
    hipEventDestroy(e_p3_start);
    hipEventDestroy(e_p3_stop);
#endif
}

void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);
    V_padded = (V + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR * BLOCKING_FACTOR;
    hipHostMalloc(&D, (size_t)V_padded * V_padded * sizeof(int));
    for (int i = 0; i < V_padded; ++i)
        for (int j = 0; j < V_padded; ++j)
            D[(size_t)i * V_padded + j] = (i == j) ? 0 : INF;
    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        D[(size_t)pair[0] * V_padded + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outfile) {
    FILE *f = fopen(outfile, "wb");
    for (int i = 0; i < V; ++i)
        fwrite(&D[(size_t)i * V_padded], sizeof(int), V, f);
    fclose(f);
    hipHostFree(D);
}

/* Kernels - Logic from hw3-2, Adapted for Multi-GPU (offsets) */

__global__ void __launch_bounds__(1024) kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Shared Memory with padding (+1) from hw3-2
    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    const int b_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);

    // Use vectorized load
    load_block_int4(d_D + b_start, V_padded, sm_pivot);
    __syncthreads();

#pragma unroll 32
    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int r0 = sm_pivot[ty][k];
        const int r1 = sm_pivot[ty + HALF_BLOCK][k];
        const int c0 = sm_pivot[k][tx];
        const int c1 = sm_pivot[k][tx + HALF_BLOCK];

        sm_pivot[ty][tx] = min(sm_pivot[ty][tx], r0 + c0);
        sm_pivot[ty][tx + HALF_BLOCK] = min(sm_pivot[ty][tx + HALF_BLOCK], r0 + c1);
        sm_pivot[ty + HALF_BLOCK][tx] = min(sm_pivot[ty + HALF_BLOCK][tx], r1 + c0);
        sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = min(sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK], r1 + c1);
        __syncthreads();
    }

    store_block_int4(d_D + b_start, V_padded, sm_pivot);
}

__global__ void __launch_bounds__(1024) kernel_phase2_row(int *d_D, const int r, const int V_padded) {
    const int b_idx_x = blockIdx.x;
    if (b_idx_x == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_y = r; // Fixed row for this phase

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    const int pivot_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const int self_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    load_block_int4(d_D + pivot_start, V_padded, sm_pivot);
    load_block_int4(d_D + self_start, V_padded, sm_self);
    __syncthreads();

    int reg_self[2][2];
    reg_self[0][0] = sm_self[ty][tx];
    reg_self[0][1] = sm_self[ty][tx + HALF_BLOCK];
    reg_self[1][0] = sm_self[ty + HALF_BLOCK][tx];
    reg_self[1][1] = sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK];

#pragma unroll 32
    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int r0 = sm_pivot[ty][k];
        const int r1 = sm_pivot[ty + HALF_BLOCK][k];
        const int c0 = sm_self[k][tx];
        const int c1 = sm_self[k][tx + HALF_BLOCK];
        reg_self[0][0] = min(reg_self[0][0], r0 + c0);
        reg_self[0][1] = min(reg_self[0][1], r0 + c1);
        reg_self[1][0] = min(reg_self[1][0], r1 + c0);
        reg_self[1][1] = min(reg_self[1][1], r1 + c1);
    }

    sm_self[ty][tx] = reg_self[0][0];
    sm_self[ty][tx + HALF_BLOCK] = reg_self[0][1];
    sm_self[ty + HALF_BLOCK][tx] = reg_self[1][0];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = reg_self[1][1];
    __syncthreads();

    store_block_int4(d_D + self_start, V_padded, sm_self);
}

__global__ void __launch_bounds__(1024) kernel_phase2_col(int *d_D, const int r, const int V_padded, const int row_offset, const int row_limit) {
    const int b_idx_y = blockIdx.x + row_offset; // Apply Offset
    if (b_idx_y == r) return;

    // Safety check for multi-gpu bounds (though launch grid should be exact)
    if (b_idx_y >= row_offset + row_limit) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_x = r; // Fixed col for this phase

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    const int pivot_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const int self_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    load_block_int4(d_D + pivot_start, V_padded, sm_pivot);
    load_block_int4(d_D + self_start, V_padded, sm_self);
    __syncthreads();

    int reg_self[2][2];
    reg_self[0][0] = sm_self[ty][tx];
    reg_self[0][1] = sm_self[ty][tx + HALF_BLOCK];
    reg_self[1][0] = sm_self[ty + HALF_BLOCK][tx];
    reg_self[1][1] = sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK];

#pragma unroll 32
    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int r0 = sm_self[ty][k];
        const int r1 = sm_self[ty + HALF_BLOCK][k];
        const int c0 = sm_pivot[k][tx];
        const int c1 = sm_pivot[k][tx + HALF_BLOCK];
        reg_self[0][0] = min(reg_self[0][0], r0 + c0);
        reg_self[0][1] = min(reg_self[0][1], r0 + c1);
        reg_self[1][0] = min(reg_self[1][0], r1 + c0);
        reg_self[1][1] = min(reg_self[1][1], r1 + c1);
    }

    sm_self[ty][tx] = reg_self[0][0];
    sm_self[ty][tx + HALF_BLOCK] = reg_self[0][1];
    sm_self[ty + HALF_BLOCK][tx] = reg_self[1][0];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = reg_self[1][1];
    __syncthreads();

    store_block_int4(d_D + self_start, V_padded, sm_self);
}

__global__ void __launch_bounds__(1024) kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset) {
    const int b_idx_x = blockIdx.x;
    const int b_idx_y = blockIdx.y + row_offset; // Apply Offset
    if (b_idx_x == r || b_idx_y == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ int sm_row[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];
    __shared__ int sm_col[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    const int row_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const int col_start = (r * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);
    const int self_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    load_block_int4(d_D + row_start, V_padded, sm_row);
    load_block_int4(d_D + col_start, V_padded, sm_col);
    __syncthreads();

    int reg_self[2][2];
    reg_self[0][0] = d_D[self_start + ty * V_padded + tx];
    reg_self[0][1] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    reg_self[1][0] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    reg_self[1][1] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

#pragma unroll 32
    for (int k = 0; k < BLOCKING_FACTOR; k++) {
        const int r0 = sm_row[ty][k];
        const int r1 = sm_row[ty + HALF_BLOCK][k];
        const int c0 = sm_col[k][tx];
        const int c1 = sm_col[k][tx + HALF_BLOCK];
        reg_self[0][0] = min(reg_self[0][0], r0 + c0);
        reg_self[0][1] = min(reg_self[0][1], r0 + c1);
        reg_self[1][0] = min(reg_self[1][0], r1 + c0);
        reg_self[1][1] = min(reg_self[1][1], r1 + c1);
    }

    d_D[self_start + ty * V_padded + tx] = reg_self[0][0];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = reg_self[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = reg_self[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = reg_self[1][1];
}
