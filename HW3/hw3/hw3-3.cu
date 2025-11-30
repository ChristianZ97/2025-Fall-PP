// hw3-3.cu

/* Headers */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Constants & Global Variables */

#define BLOCKING_FACTOR 64
#define HALF_BLOCK 32
#define INF ((1 << 30) - 1)

static int *D;        // Host pointer
static int *d_D[2];   // Device pointers for 2 GPUs
static int V, E;      // Original vertices, edges
static int V_padded;  // Padded vertices (multiple of 64)

cudaStream_t stream_main[2], stream_row[2], stream_col[2];
cudaEvent_t event_p1_done[2], event_p2_row_done[2], event_p2_col_done[2];

/* Function Prototypes */

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded, const int row_offset, const int row_limit);
__global__ void kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset);

void input(char *infile);
void output(char *outfile);

void block_FW(const int dev_id
#ifdef PROFILING
              ,
              double *time_phase1_ms,
              double *time_phase2_row_ms,
              double *time_phase2_col_ms,
              double *time_phase3_ms
#endif
);

/* Main */

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

#ifdef PROFILING
    double time_io_read_ms = 0.0;
    double time_io_write_ms = 0.0;

    // Phase-wise timing on both devices
    double time_phase1_ms[2] = {0.0, 0.0};
    double time_phase2_row_ms[2] = {0.0, 0.0};
    double time_phase2_col_ms[2] = {0.0, 0.0};
    double time_phase3_ms[2] = {0.0, 0.0};

    // H2D / D2H per device
    double time_h2d_ms[2] = {0.0, 0.0};
    double time_d2h_ms[2] = {0.0, 0.0};

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
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

#pragma omp parallel num_threads(2)
    {
        const int dev_id = omp_get_thread_num();

        cudaSetDevice(dev_id);

        cudaStreamCreate(&stream_main[dev_id]);
        cudaStreamCreate(&stream_row[dev_id]);
        cudaStreamCreate(&stream_col[dev_id]);

        cudaEventCreate(&event_p1_done[dev_id]);
        cudaEventCreate(&event_p2_row_done[dev_id]);
        cudaEventCreate(&event_p2_col_done[dev_id]);

        cudaMalloc(&d_D[dev_id], size);

#ifdef PROFILING
        // Host-device transfer timing per device
        cudaEvent_t event_h2d_start, event_h2d_stop;
        cudaEvent_t event_d2h_start, event_d2h_stop;

        cudaEventCreate(&event_h2d_start);
        cudaEventCreate(&event_h2d_stop);
        cudaEventCreate(&event_d2h_start);
        cudaEventCreate(&event_d2h_stop);

        cudaEventRecord(event_h2d_start);
#endif
        // H2D: copy full matrix to each GPU
        cudaMemcpy(d_D[dev_id], D, size, cudaMemcpyHostToDevice);
#ifdef PROFILING
        cudaEventRecord(event_h2d_stop);
        cudaEventSynchronize(event_h2d_stop);

        float t_h2d_f = 0.0f;
        cudaEventElapsedTime(&t_h2d_f, event_h2d_start, event_h2d_stop);
        time_h2d_ms[dev_id] = (double)t_h2d_f;
#endif

#pragma omp barrier

#ifdef PROFILING
        double t_p1 = 0.0;
        double t_p2r = 0.0;
        double t_p2c = 0.0;
        double t_p3 = 0.0;

        block_FW(dev_id, &t_p1, &t_p2r, &t_p2c, &t_p3);

        time_phase1_ms[dev_id] = t_p1;
        time_phase2_row_ms[dev_id] = t_p2r;
        time_phase2_col_ms[dev_id] = t_p2c;
        time_phase3_ms[dev_id] = t_p3;
#else
        block_FW(dev_id);
#endif

        cudaDeviceSynchronize();

        const int round = V_padded / BLOCKING_FACTOR;
        const int row_mid = round / 2;
        const int row_block_start = (dev_id == 0) ? 0 : row_mid;
        const int row_block_count = (dev_id == 0) ? row_mid : (round - row_mid);
        const int row_start = row_block_start * BLOCKING_FACTOR;
        const int row_num = row_block_count * BLOCKING_FACTOR;

#ifdef PROFILING
        cudaEventRecord(event_d2h_start);
#endif
        // Copy back only the owned rows from each GPU
        cudaMemcpy(D + (size_t)row_start * V_padded, d_D[dev_id] + (size_t)row_start * V_padded, (size_t)row_num * V_padded * sizeof(int), cudaMemcpyDeviceToHost);
#ifdef PROFILING
        cudaEventRecord(event_d2h_stop);
        cudaEventSynchronize(event_d2h_stop);

        float t_d2h_f = 0.0f;
        cudaEventElapsedTime(&t_d2h_f, event_d2h_start, event_d2h_stop);
        time_d2h_ms[dev_id] = (double)t_d2h_f;

        cudaEventDestroy(event_h2d_start);
        cudaEventDestroy(event_h2d_stop);
        cudaEventDestroy(event_d2h_start);
        cudaEventDestroy(event_d2h_stop);
#endif

        cudaFree(d_D[dev_id]);

        cudaStreamDestroy(stream_main[dev_id]);
        cudaStreamDestroy(stream_row[dev_id]);
        cudaStreamDestroy(stream_col[dev_id]);

        cudaEventDestroy(event_p1_done[dev_id]);
        cudaEventDestroy(event_p2_row_done[dev_id]);
        cudaEventDestroy(event_p2_col_done[dev_id]);
    }  // end parallel region

#ifdef PROFILING
    clock_t io_write_start = clock();
    output(argv[2]);
    clock_t io_write_end = clock();
    time_io_write_ms = 1000.0 * (io_write_end - io_write_start) / CLOCKS_PER_SEC;
#else
    output(argv[2]);
#endif

#ifdef PROFILING
    // Aggregate results over 2 devices
    double time_phase1_total = time_phase1_ms[0] + time_phase1_ms[1];
    double time_phase2_row_total = time_phase2_row_ms[0] + time_phase2_row_ms[1];
    double time_phase2_col_total = time_phase2_col_ms[0] + time_phase2_col_ms[1];
    double time_phase3_total = time_phase3_ms[0] + time_phase3_ms[1];

    double time_h2d_total = time_h2d_ms[0] + time_h2d_ms[1];
    double time_d2h_total = time_d2h_ms[0] + time_d2h_ms[1];

    double time_compute_total_ms = time_phase1_total + time_phase2_row_total + time_phase2_col_total + time_phase3_total;
    double time_comm_total_ms = time_h2d_total + time_d2h_total;
    double time_io_total_ms = time_io_read_ms + time_io_write_ms;
    double total_time_ms = time_compute_total_ms + time_comm_total_ms + time_io_total_ms;

    // Same compact CSV-style line as hw3-2.cu
    fprintf(stderr, "[PROF_RESULT],%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", total_time_ms, time_compute_total_ms, time_comm_total_ms, time_io_total_ms, time_phase1_total,
            time_phase2_row_total + time_phase2_col_total, time_phase3_total);
#endif

    return 0;
}

/* Function Definitions */

void block_FW(const int dev_id
#ifdef PROFILING
              ,
              double *time_phase1_ms,
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

    dim3 grid_p2_row(round, 1);
    dim3 grid_p2_col(num_row_blocks, 1);
    dim3 grid_p3(round, num_row_blocks);

#ifdef PROFILING
    // Per-round timing aggregation via CUDA events
    cudaEvent_t e_p1_start, e_p1_stop;
    cudaEvent_t e_p2_row_start, e_p2_row_stop;
    cudaEvent_t e_p2_col_start, e_p2_col_stop;
    cudaEvent_t e_p3_start, e_p3_stop;

    cudaEventCreate(&e_p1_start);
    cudaEventCreate(&e_p1_stop);
    cudaEventCreate(&e_p2_row_start);
    cudaEventCreate(&e_p2_row_stop);
    cudaEventCreate(&e_p2_col_start);
    cudaEventCreate(&e_p2_col_stop);
    cudaEventCreate(&e_p3_start);
    cudaEventCreate(&e_p3_stop);

    float acc_p1 = 0.0f;
    float acc_p2_row = 0.0f;
    float acc_p2_col = 0.0f;
    float acc_p3 = 0.0f;

    *time_phase1_ms = 0.0;
    *time_phase2_row_ms = 0.0;
    *time_phase2_col_ms = 0.0;
    *time_phase3_ms = 0.0;
#endif

    for (int r = 0; r < round; ++r) {
        const int owner_id = (r < row_mid) ? 0 : 1;

        // Phase 1 + Phase 2 row on owner device
        if (dev_id == owner_id) {
#ifdef PROFILING
            cudaEventRecord(e_p1_start, stream_main[dev_id]);
#endif
            // Phase 1: Pivot block
            kernel_phase1<<<1, threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded);
            cudaEventRecord(event_p1_done[dev_id], stream_main[dev_id]);
#ifdef PROFILING
            cudaEventRecord(e_p1_stop, stream_main[dev_id]);
            cudaEventSynchronize(e_p1_stop);
            float t_p1 = 0.0f;
            cudaEventElapsedTime(&t_p1, e_p1_start, e_p1_stop);
            acc_p1 += t_p1;
#endif

            cudaStreamWaitEvent(stream_row[dev_id], event_p1_done[dev_id], 0);

#ifdef PROFILING
            cudaEventRecord(e_p2_row_start, stream_row[dev_id]);
#endif
            // Phase 2: Pivot row
            kernel_phase2_row<<<grid_p2_row, threads_per_block, 0, stream_row[dev_id]>>>(d_D[dev_id], r, V_padded);
            cudaEventRecord(event_p2_row_done[dev_id], stream_row[dev_id]);
#ifdef PROFILING
            cudaEventRecord(e_p2_row_stop, stream_row[dev_id]);
            cudaEventSynchronize(e_p2_row_stop);
            float t_p2_row = 0.0f;
            cudaEventElapsedTime(&t_p2_row, e_p2_row_start, e_p2_row_stop);
            acc_p2_row += t_p2_row;
#endif

            // Send updated pivot row to peer device
            size_t copy_size = (size_t)BLOCKING_FACTOR * V_padded * sizeof(int);
            size_t offset = (size_t)r * BLOCKING_FACTOR * V_padded;
            const int peer_id = 1 - dev_id;

            cudaMemcpyPeer(d_D[peer_id] + offset, peer_id, d_D[dev_id] + offset, dev_id, copy_size);
        }

#pragma omp barrier

        // Phase 2 col + Phase 3 on both devices, using updated pivot row
        cudaStreamWaitEvent(stream_main[dev_id], event_p2_row_done[dev_id], 0);

#ifdef PROFILING
        cudaEventRecord(e_p2_col_start, stream_col[dev_id]);
#endif
        kernel_phase2_col<<<grid_p2_col, threads_per_block, 0, stream_col[dev_id]>>>(d_D[dev_id], r, V_padded, start_row_block, num_row_blocks);
        cudaEventRecord(event_p2_col_done[dev_id], stream_col[dev_id]);
#ifdef PROFILING
        cudaEventRecord(e_p2_col_stop, stream_col[dev_id]);
        cudaEventSynchronize(e_p2_col_stop);
        float t_p2_col = 0.0f;
        cudaEventElapsedTime(&t_p2_col, e_p2_col_start, e_p2_col_stop);
        acc_p2_col += t_p2_col;
#endif

        cudaStreamWaitEvent(stream_main[dev_id], event_p2_col_done[dev_id], 0);

#ifdef PROFILING
        cudaEventRecord(e_p3_start, stream_main[dev_id]);
#endif
        kernel_phase3<<<grid_p3, threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded, start_row_block);
#ifdef PROFILING
        cudaEventRecord(e_p3_stop, stream_main[dev_id]);
        cudaEventSynchronize(e_p3_stop);
        float t_p3 = 0.0f;
        cudaEventElapsedTime(&t_p3, e_p3_start, e_p3_stop);
        acc_p3 += t_p3;
#endif

        cudaDeviceSynchronize();

#pragma omp barrier
    }

#ifdef PROFILING
    *time_phase1_ms = acc_p1;
    *time_phase2_row_ms = acc_p2_row;
    *time_phase2_col_ms = acc_p2_col;
    *time_phase3_ms = acc_p3;

    cudaEventDestroy(e_p1_start);
    cudaEventDestroy(e_p1_stop);
    cudaEventDestroy(e_p2_row_start);
    cudaEventDestroy(e_p2_row_stop);
    cudaEventDestroy(e_p2_col_start);
    cudaEventDestroy(e_p2_col_stop);
    cudaEventDestroy(e_p3_start);
    cudaEventDestroy(e_p3_stop);
#endif
}

void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    // Calculate padded size (round up to multiple of 64)
    V_padded = (V + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR * BLOCKING_FACTOR;

    // Use pinned memory for faster host-device transfer
    cudaHostAlloc(&D, (size_t)V_padded * V_padded * sizeof(int), cudaHostAllocDefault);

    // Initialize with INF (and 0 diagonal), including padding
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
    // Write only the valid part (V x V), skipping padding
    for (int i = 0; i < V; ++i)
        fwrite(&D[(size_t)i * V_padded], sizeof(int), V, f);
    fclose(f);

    cudaFreeHost(D);  // Free pinned memory
}

/* Kernels */

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;  // 0..31
    const int ty = threadIdx.y;  // 0..31

    // Shared Memory for the 64x64 block (+1 to mitigate bank conflict if desired)
    __shared__ int sm[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    // Global memory offset for the pivot block (r, r)
    const size_t b_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);

    // 1. Load Global -> Shared (each thread loads 4 ints)
    sm[ty][tx] = d_D[b_start + ty * V_padded + tx];
    sm[ty][tx + HALF_BLOCK] = d_D[b_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm[ty + HALF_BLOCK][tx] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    __syncthreads();

    // 2. Floyd–Warshall computation within the block

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int pivot_row_val1 = sm[ty][k];
        const int pivot_row_val2 = sm[ty + HALF_BLOCK][k];
        const int pivot_col_val1 = sm[k][tx];
        const int pivot_col_val2 = sm[k][tx + HALF_BLOCK];

        sm[ty][tx] = min(sm[ty][tx], pivot_row_val1 + pivot_col_val1);
        sm[ty][tx + HALF_BLOCK] = min(sm[ty][tx + HALF_BLOCK], pivot_row_val1 + pivot_col_val2);
        sm[ty + HALF_BLOCK][tx] = min(sm[ty + HALF_BLOCK][tx], pivot_row_val2 + pivot_col_val1);
        sm[ty + HALF_BLOCK][tx + HALF_BLOCK] = min(sm[ty + HALF_BLOCK][tx + HALF_BLOCK], pivot_row_val2 + pivot_col_val2);

        __syncthreads();
    }

    // 3. Write Shared -> Global
    d_D[b_start + ty * V_padded + tx] = sm[ty][tx];
    d_D[b_start + ty * V_padded + (tx + HALF_BLOCK)] = sm[ty][tx + HALF_BLOCK];
    d_D[b_start + (ty + HALF_BLOCK) * V_padded + tx] = sm[ty + HALF_BLOCK][tx];
    d_D[b_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = sm[ty + HALF_BLOCK][tx + HALF_BLOCK];
}

__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded) {
    const int b_idx_x = blockIdx.x;
    if (b_idx_x == r) return;  // Skip pivot block itself

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int b_idx_y = r;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    const size_t pivot_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const size_t self_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // Load pivot
    sm_pivot[ty][tx] = d_D[pivot_start + ty * V_padded + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[pivot_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    // Load self
    sm_self[ty][tx] = d_D[self_start + ty * V_padded + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    __syncthreads();

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        int pivot_val1 = sm_pivot[ty][k];
        int pivot_val2 = sm_pivot[ty + HALF_BLOCK][k];
        int self_val1 = sm_self[k][tx];
        int self_val2 = sm_self[k][tx + HALF_BLOCK];

        sm_self[ty][tx] = min(sm_self[ty][tx], pivot_val1 + self_val1);
        sm_self[ty][tx + HALF_BLOCK] = min(sm_self[ty][tx + HALF_BLOCK], pivot_val1 + self_val2);
        sm_self[ty + HALF_BLOCK][tx] = min(sm_self[ty + HALF_BLOCK][tx], pivot_val2 + self_val1);
        sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = min(sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK], pivot_val2 + self_val2);

        __syncthreads();
    }

    d_D[self_start + ty * V_padded + tx] = sm_self[ty][tx];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = sm_self[ty][tx + HALF_BLOCK];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = sm_self[ty + HALF_BLOCK][tx];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK];
}

__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded, const int row_offset, const int row_limit) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int b_idx_y = blockIdx.x + row_offset;
    if (b_idx_y == r || b_idx_y >= row_offset + row_limit) return;

    const int b_idx_x = r;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    const size_t pivot_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const size_t self_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // Load pivot (r, r)
    sm_pivot[ty][tx] = d_D[pivot_start + ty * V_padded + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[pivot_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    // Load self (block (y, r))
    sm_self[ty][tx] = d_D[self_start + ty * V_padded + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    __syncthreads();

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        int pivot_val1 = sm_pivot[k][tx];
        int pivot_val2 = sm_pivot[k][tx + HALF_BLOCK];
        int self_val1 = sm_self[ty][k];
        int self_val2 = sm_self[ty + HALF_BLOCK][k];

        sm_self[ty][tx] = min(sm_self[ty][tx], self_val1 + pivot_val1);
        sm_self[ty][tx + HALF_BLOCK] = min(sm_self[ty][tx + HALF_BLOCK], self_val1 + pivot_val2);
        sm_self[ty + HALF_BLOCK][tx] = min(sm_self[ty + HALF_BLOCK][tx], self_val2 + pivot_val1);
        sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = min(sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK], self_val2 + pivot_val2);
    }

    // Write back
    d_D[self_start + ty * V_padded + tx] = sm_self[ty][tx];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = sm_self[ty][tx + HALF_BLOCK];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = sm_self[ty + HALF_BLOCK][tx];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK];
}

__global__ void kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset) {
    const int b_idx_x = blockIdx.x;
    const int b_idx_y = blockIdx.y + row_offset;

    if (b_idx_x == r || b_idx_y == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ int sm_row[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Row block (y, r)
    __shared__ int sm_col[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Col block (r, x)

    const size_t row_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const size_t col_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);
    const size_t self_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // Load row block
    sm_row[ty][tx] = d_D[row_start + ty * V_padded + tx];
    sm_row[ty][tx + HALF_BLOCK] = d_D[row_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_row[ty + HALF_BLOCK][tx] = d_D[row_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_row[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[row_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    // Load col block
    sm_col[ty][tx] = d_D[col_start + ty * V_padded + tx];
    sm_col[ty][tx + HALF_BLOCK] = d_D[col_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_col[ty + HALF_BLOCK][tx] = d_D[col_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_col[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[col_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    __syncthreads();

    // Keep self in registers
    int val[2][2];
    val[0][0] = d_D[self_start + ty * V_padded + tx];
    val[0][1] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    val[1][0] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    val[1][1] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int r1 = sm_row[ty][k];
        const int r2 = sm_row[ty + HALF_BLOCK][k];
        const int c1 = sm_col[k][tx];
        const int c2 = sm_col[k][tx + HALF_BLOCK];

        val[0][0] = min(val[0][0], r1 + c1);
        val[0][1] = min(val[0][1], r1 + c2);
        val[1][0] = min(val[1][0], r2 + c1);
        val[1][1] = min(val[1][1], r2 + c2);
    }

    // Write back
    d_D[self_start + ty * V_padded + tx] = val[0][0];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = val[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = val[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = val[1][1];
}
