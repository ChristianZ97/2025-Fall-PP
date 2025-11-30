// hw3-2.cu

/* Headers*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Blocked Floyd–Warshall with CUDA
 * Optional high-resolution profiling (enable with -DPROFILING).
 * When PROFILING is not defined, the core algorithm and behavior are unchanged.
 */

/* Constants & Global Variables */

// [OPT] Blocking factor (tile size). Tune this (e.g., 32/64/128) to study "large blocking factor" vs occupancy & bandwidth.
#define BLOCKING_FACTOR 8  // Matches v2 (64x64 data block)

// [OPT] Thread block dimension (32x32 = 1024 threads). This controls CUDA 2D alignment and occupancy.
#define HALF_BLOCK BLOCKING_FACTOR / 2  // Thread block dimension (32x32 threads)

#define INF ((1 << 30) - 1)

static int *D;        // Host pointer
static int *d_D;      // Device pointer
static int V, E;      // Original vertices, edges
static int V_padded;  // Padded vertices (multiple of 64)

// [OPT] Multiple streams & events are used to overlap different phases (streaming / reduce idle time).
cudaStream_t stream_main, stream_row, stream_col;
cudaEvent_t event_p1_done, event_p2_row_done, event_p2_col_done;

/* Function Prototypes */

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase3(int *d_D, const int r, const int V_padded);

void input(char *infile);
void output(char *outfile);
void block_FW(
#ifdef PROFILING
    double *time_phase1_ms, double *time_phase2_row_ms, double *time_phase2_col_ms, double *time_phase3_ms
#endif
);

/* Main */
int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: %s \n", argv[0]);
        return 1;
    }

#ifdef PROFILING
    // -------------------------
    // Host-side CUDA events for H2D / D2H
    // -------------------------
    cudaEvent_t event_h2d_start, event_h2d_stop;
    cudaEvent_t event_d2h_start, event_d2h_stop;
    cudaEventCreate(&event_h2d_start);
    cudaEventCreate(&event_h2d_stop);
    cudaEventCreate(&event_d2h_start);
    cudaEventCreate(&event_d2h_stop);

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

    cudaStreamCreate(&stream_main);
    cudaStreamCreate(&stream_row);
    cudaStreamCreate(&stream_col);
    cudaEventCreate(&event_p1_done);
    cudaEventCreate(&event_p2_row_done);
    cudaEventCreate(&event_p2_col_done);

    size_t size = V_padded * V_padded * sizeof(int);
    cudaMalloc(&d_D, size);

    // -------------------------
    // H2D transfer
    // -------------------------
#ifdef PROFILING
    cudaEventRecord(event_h2d_start);
#endif
    cudaMemcpy(d_D, D, size, cudaMemcpyHostToDevice);
#ifdef PROFILING
    cudaEventRecord(event_h2d_stop);
    cudaEventSynchronize(event_h2d_stop);
#endif

    // -------------------------
    // Kernel execution
    // -------------------------
#ifdef PROFILING
    block_FW(&time_phase1_ms, &time_phase2_row_ms, &time_phase2_col_ms, &time_phase3_ms);
#else
    block_FW();
#endif

    // -------------------------
    // D2H transfer
    // -------------------------
#ifdef PROFILING
    cudaEventRecord(event_d2h_start);
#endif
    cudaMemcpy(D, d_D, size, cudaMemcpyDeviceToHost);
#ifdef PROFILING
    cudaEventRecord(event_d2h_stop);
    cudaEventSynchronize(event_d2h_stop);
#endif

#ifdef PROFILING
    clock_t io_write_start = clock();
    output(argv[2]);
    clock_t io_write_end = clock();
    time_io_write_ms = 1000.0 * (io_write_end - io_write_start) / CLOCKS_PER_SEC;
#else
    output(argv[2]);
#endif

    cudaFree(d_D);
    cudaStreamDestroy(stream_main);
    cudaStreamDestroy(stream_row);
    cudaStreamDestroy(stream_col);
    cudaEventDestroy(event_p1_done);
    cudaEventDestroy(event_p2_row_done);
    cudaEventDestroy(event_p2_col_done);

#ifdef PROFILING
    // ============================================================
    // Profiling result: only compact summary line
    // ============================================================
    float t_h2d_f = 0.0f, t_d2h_f = 0.0f;
    cudaEventElapsedTime(&t_h2d_f, event_h2d_start, event_h2d_stop);
    cudaEventElapsedTime(&t_d2h_f, event_d2h_start, event_d2h_stop);
    double time_h2d_ms = (double)t_h2d_f;
    double time_d2h_ms = (double)t_d2h_f;

    double time_compute_total_ms = time_phase1_ms + time_phase2_row_ms + time_phase2_col_ms + time_phase3_ms;
    double time_comm_total_ms = time_h2d_ms + time_d2h_ms;
    double time_io_total_ms = time_io_read_ms + time_io_write_ms;
    double total_time_ms = time_compute_total_ms + time_comm_total_ms + time_io_total_ms;

    fprintf(stderr, "[PROF_RESULT],%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", total_time_ms, time_compute_total_ms, time_comm_total_ms, time_io_total_ms, time_phase1_ms,
            time_phase2_row_ms + time_phase2_col_ms, time_phase3_ms);

    cudaEventDestroy(event_h2d_start);
    cudaEventDestroy(event_h2d_stop);
    cudaEventDestroy(event_d2h_start);
    cudaEventDestroy(event_d2h_stop);
#endif  // PROFILING

    return 0;
}

/* Function Definitions */
void block_FW(
#ifdef PROFILING
    double *time_phase1_ms, double *time_phase2_row_ms, double *time_phase2_col_ms, double *time_phase3_ms
#endif
) {
    const int round = V_padded / BLOCKING_FACTOR;

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

    // [OPT] 2D thread block (HALF_BLOCK x HALF_BLOCK) for good CUDA 2D alignment and coalesced accesses.
    dim3 threads_per_block(HALF_BLOCK, HALF_BLOCK);  // 32x32 threads

    for (int r = 0; r < round; ++r) {
        // 1. Phase 1: Pivot Block
#ifdef PROFILING
        cudaEventRecord(e_p1_start, stream_main);
#endif
        kernel_phase1<<<1, threads_per_block, 0, stream_main>>>(d_D, r, V_padded);
        cudaEventRecord(event_p1_done, stream_main);
#ifdef PROFILING
        cudaEventRecord(e_p1_stop, stream_main);
        cudaEventSynchronize(e_p1_stop);
        float t_p1 = 0.0f;
        cudaEventElapsedTime(&t_p1, e_p1_start, e_p1_stop);
        acc_p1 += t_p1;
#endif

        // 2. Phase 2: Pivot Row (row stream)
        cudaStreamWaitEvent(stream_row, event_p1_done, 0);
#ifdef PROFILING
        cudaEventRecord(e_p2_row_start, stream_row);
#endif
        kernel_phase2_row<<<round, threads_per_block, 0, stream_row>>>(d_D, r, V_padded);
        cudaEventRecord(event_p2_row_done, stream_row);
#ifdef PROFILING
        cudaEventRecord(e_p2_row_stop, stream_row);
        cudaEventSynchronize(e_p2_row_stop);
        float t_p2_row = 0.0f;
        cudaEventElapsedTime(&t_p2_row, e_p2_row_start, e_p2_row_stop);
        acc_p2_row += t_p2_row;
#endif

        // 2. Phase 2: Pivot Col (col stream)
        cudaStreamWaitEvent(stream_col, event_p1_done, 0);
#ifdef PROFILING
        cudaEventRecord(e_p2_col_start, stream_col);
#endif
        kernel_phase2_col<<<round, threads_per_block, 0, stream_col>>>(d_D, r, V_padded);
        cudaEventRecord(event_p2_col_done, stream_col);
#ifdef PROFILING
        cudaEventRecord(e_p2_col_stop, stream_col);
        cudaEventSynchronize(e_p2_col_stop);
        float t_p2_col = 0.0f;
        cudaEventElapsedTime(&t_p2_col, e_p2_col_start, e_p2_col_stop);
        acc_p2_col += t_p2_col;
#endif

        cudaStreamWaitEvent(stream_main, event_p2_row_done, 0);
        cudaStreamWaitEvent(stream_main, event_p2_col_done, 0);

        // 3. Phase 3: Independent Blocks
        // (round, round) blocks per grid
#ifdef PROFILING
        cudaEventRecord(e_p3_start, stream_main);
#endif
        kernel_phase3<<<dim3(round, round), threads_per_block, 0, stream_main>>>(d_D, r, V_padded);
#ifdef PROFILING
        cudaEventRecord(e_p3_stop, stream_main);
        cudaEventSynchronize(e_p3_stop);
        float t_p3 = 0.0f;
        cudaEventElapsedTime(&t_p3, e_p3_start, e_p3_stop);
        acc_p3 += t_p3;
#endif
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

    // Calculate Padded Size (Round up to multiple of 64)
    // [OPT] Padding V up to a multiple of BLOCKING_FACTOR simplifies index math and improves coalesced access.
    V_padded = (V + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR * BLOCKING_FACTOR;

    // Use Pinned Memory for faster host-device transfer
    // [OPT] Pinned host memory (cudaHostAlloc) increases H2D/D2H bandwidth; affects "memory copy" time in profiling.
    cudaHostAlloc(&D, V_padded * V_padded * sizeof(int), cudaHostAllocDefault);

    // Initialize with INF (and 0 diagonal)
    // Note: Padding areas are also initialized to avoid side effects

#pragma unroll 32

    for (int i = 0; i < V_padded; ++i)
        for (int j = 0; j < V_padded; ++j)
            D[i * V_padded + j] = (i == j) ? 0 : INF;

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        D[pair[0] * V_padded + pair[1]] = pair[2];
    }

    fclose(file);
}

void output(char *outfile) {
    FILE *f = fopen(outfile, "w");

    // Write only the valid part (V x V), skipping padding
    for (int i = 0; i < V; ++i)
        fwrite(&D[i * V_padded], sizeof(int), V, f);

    fclose(f);
    cudaFreeHost(D);  // Free Pinned Memory
}

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;  // 0..31
    const int ty = threadIdx.y;  // 0..31

    // Shared Memory for the 64x64 block
    // [OPT] Shared memory tile + extra column (+1) to reduce global memory traffic and avoid shared memory bank conflicts.
    __shared__ int sm[BLOCKING_FACTOR][BLOCKING_FACTOR];

    // Global Memory Offset for the Pivot Block (r, r)
    const int b_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);

    // 1. Load Global -> Shared (Each thread loads 4 ints)
    // [OPT] Coalesced global memory loads: threads in a warp read contiguous elements of each row in the 64x64 tile.
    // Access pattern: Top-Left, Top-Right, Bottom-Left, Bottom-Right relative to thread
    sm[ty][tx] = d_D[b_start + ty * V_padded + tx];
    sm[ty][tx + HALF_BLOCK] = d_D[b_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm[ty + HALF_BLOCK][tx] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    __syncthreads();

    // 2. Floyd-Warshall Computation within the block

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
    if (b_idx_x == r) return;  // Skip the pivot block itself (handled in Phase 1)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_y = r;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    const int pivot_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const int self_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // Pivot
    sm_pivot[ty][tx] = d_D[pivot_start + ty * V_padded + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[pivot_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    // Self
    sm_self[ty][tx] = d_D[self_start + ty * V_padded + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    __syncthreads();

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        int pivot_val1, pivot_val2, self_val1, self_val2;
        pivot_val1 = sm_pivot[ty][k];
        pivot_val2 = sm_pivot[ty + HALF_BLOCK][k];
        self_val1 = sm_self[k][tx];
        self_val2 = sm_self[k][tx + HALF_BLOCK];

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

__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded) {

    const int b_idx_y = blockIdx.x;
    if (b_idx_y == r) return;  // Skip the pivot block itself (handled in Phase 1)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_x = r;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    const int pivot_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const int self_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // Pivot
    sm_pivot[ty][tx] = d_D[pivot_start + ty * V_padded + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[pivot_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    // Self
    sm_self[ty][tx] = d_D[self_start + ty * V_padded + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    __syncthreads();

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        int pivot_val1, pivot_val2, self_val1, self_val2;
        self_val1 = sm_self[ty][k];
        self_val2 = sm_self[ty + HALF_BLOCK][k];
        pivot_val1 = sm_pivot[k][tx];
        pivot_val2 = sm_pivot[k][tx + HALF_BLOCK];

        sm_self[ty][tx] = min(sm_self[ty][tx], self_val1 + pivot_val1);
        sm_self[ty][tx + HALF_BLOCK] = min(sm_self[ty][tx + HALF_BLOCK], self_val1 + pivot_val2);
        sm_self[ty + HALF_BLOCK][tx] = min(sm_self[ty + HALF_BLOCK][tx], self_val2 + pivot_val1);
        sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = min(sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK], self_val2 + pivot_val2);
        __syncthreads();
    }

    d_D[self_start + ty * V_padded + tx] = sm_self[ty][tx];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = sm_self[ty][tx + HALF_BLOCK];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = sm_self[ty + HALF_BLOCK][tx];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK];
}

__global__ void kernel_phase3(int *d_D, const int r, const int V_padded) {

    const int b_idx_x = blockIdx.x;
    const int b_idx_y = blockIdx.y;
    if (b_idx_x == r || b_idx_y == r) return;  // Skip Phase 1 & 2 blocks

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ int sm_row[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Row Block (y, r)
    __shared__ int sm_col[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Col Block (r, x)

    const int row_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const int col_start = (r * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);
    const int self_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // Row Block
    sm_row[ty][tx] = d_D[row_start + ty * V_padded + tx];
    sm_row[ty][tx + HALF_BLOCK] = d_D[row_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_row[ty + HALF_BLOCK][tx] = d_D[row_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_row[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[row_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    // Col Block
    sm_col[ty][tx] = d_D[col_start + ty * V_padded + tx];
    sm_col[ty][tx + HALF_BLOCK] = d_D[col_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_col[ty + HALF_BLOCK][tx] = d_D[col_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_col[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[col_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    __syncthreads();

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

    d_D[self_start + ty * V_padded + tx] = val[0][0];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = val[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = val[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = val[1][1];
}
