// hw3-2_prof.hip

/* Headers */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Constants */
#define BLOCKING_FACTOR 64
#define HALF_BLOCK 32
#define INF ((1 << 30) - 1)

/* Global Variables */
static int *D;
static int *d_D;
static int V, E;
static int V_padded;

/* Streams and Events */
hipStream_t stream_main, stream_row, stream_col;
hipEvent_t event_p1_done, event_p2_row_done, event_p2_col_done;

/* Function Prototypes */
__global__ void __launch_bounds__(1024) kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase2_col(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase3(int *d_D, const int r, const int V_padded);

void input(char *infile);
void output(char *outfile);

/* Modified block_FW signature for profiling */
void block_FW(
#ifdef PROFILING
    double *time_phase1_ms, double *time_phase2_row_ms, double *time_phase2_col_ms, double *time_phase3_ms
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
    // Host-side Events & Timers
    // -------------------------
    hipEvent_t event_h2d_start, event_h2d_stop;
    hipEvent_t event_d2h_start, event_d2h_stop;
    hipEventCreate(&event_h2d_start);
    hipEventCreate(&event_h2d_stop);
    hipEventCreate(&event_d2h_start);
    hipEventCreate(&event_d2h_stop);

    double time_phase1_ms = 0.0;
    double time_phase2_row_ms = 0.0;
    double time_phase2_col_ms = 0.0;
    double time_phase3_ms = 0.0;

    double time_io_read_ms = 0.0;
    double time_io_write_ms = 0.0;

    clock_t io_read_start = clock();
#endif

    input(argv[1]);

#ifdef PROFILING
    clock_t io_read_end = clock();
    time_io_read_ms = 1000.0 * (io_read_end - io_read_start) / CLOCKS_PER_SEC;
#endif

    hipStreamCreate(&stream_main);
    hipStreamCreate(&stream_row);
    hipStreamCreate(&stream_col);

    hipEventCreate(&event_p1_done);
    hipEventCreate(&event_p2_row_done);
    hipEventCreate(&event_p2_col_done);

    size_t size = (size_t)V_padded * V_padded * sizeof(int);
    hipMalloc(&d_D, size);

    // -------------------------
    // H2D Transfer
    // -------------------------
#ifdef PROFILING
    hipEventRecord(event_h2d_start, 0);
#endif

    hipMemcpy(d_D, D, size, hipMemcpyHostToDevice);

#ifdef PROFILING
    hipEventRecord(event_h2d_stop, 0);
    hipEventSynchronize(event_h2d_stop);
#endif

    // -------------------------
    // Kernel Execution
    // -------------------------
#ifdef PROFILING
    block_FW(&time_phase1_ms, &time_phase2_row_ms, &time_phase2_col_ms, &time_phase3_ms);
#else
    block_FW();
#endif

    // -------------------------
    // D2H Transfer
    // -------------------------
#ifdef PROFILING
    hipEventRecord(event_d2h_start, 0);
#endif

    hipMemcpy(D, d_D, size, hipMemcpyDeviceToHost);

#ifdef PROFILING
    hipEventRecord(event_d2h_stop, 0);
    hipEventSynchronize(event_d2h_stop);
#endif

#ifdef PROFILING
    clock_t io_write_start = clock();
#endif

    output(argv[2]);

#ifdef PROFILING
    clock_t io_write_end = clock();
    time_io_write_ms = 1000.0 * (io_write_end - io_write_start) / CLOCKS_PER_SEC;
#endif

    // Cleanup
    hipFree(d_D);
    hipHostFree(D);  // Changed from hipFreeHost to match standard HIP API (or hipHostFree)

    hipStreamDestroy(stream_main);
    hipStreamDestroy(stream_row);
    hipStreamDestroy(stream_col);

    hipEventDestroy(event_p1_done);
    hipEventDestroy(event_p2_row_done);
    hipEventDestroy(event_p2_col_done);

#ifdef PROFILING
    // -------------------------
    // Profiling Result Output
    // -------------------------
    float t_h2d_f = 0.0f, t_d2h_f = 0.0f;
    hipEventElapsedTime(&t_h2d_f, event_h2d_start, event_h2d_stop);
    hipEventElapsedTime(&t_d2h_f, event_d2h_start, event_d2h_stop);

    double time_h2d_ms = (double)t_h2d_f;
    double time_d2h_ms = (double)t_d2h_f;
    double time_compute_total_ms = time_phase1_ms + time_phase2_row_ms + time_phase2_col_ms + time_phase3_ms;
    double time_comm_total_ms = time_h2d_ms + time_d2h_ms;
    double time_io_total_ms = time_io_read_ms + time_io_write_ms;
    double total_time_ms = time_compute_total_ms + time_comm_total_ms + time_io_total_ms;

    fprintf(stderr, "[PROF_RESULT],%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", total_time_ms, time_compute_total_ms, time_comm_total_ms, time_io_total_ms, time_phase1_ms,
            time_phase2_row_ms + time_phase2_col_ms, time_phase3_ms);

    hipEventDestroy(event_h2d_start);
    hipEventDestroy(event_h2d_stop);
    hipEventDestroy(event_d2h_start);
    hipEventDestroy(event_d2h_stop);
#endif

    return 0;
}

void block_FW(
#ifdef PROFILING
    double *time_phase1_ms, double *time_phase2_row_ms, double *time_phase2_col_ms, double *time_phase3_ms
#endif
) {
    const int round = V_padded / BLOCKING_FACTOR;
    dim3 block_dim(HALF_BLOCK, HALF_BLOCK);

#ifdef PROFILING
    // Local accumulators and events
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

    double acc_p1 = 0.0;
    double acc_p2_row = 0.0;
    double acc_p2_col = 0.0;
    double acc_p3 = 0.0;
    float t_ms = 0.0f;
#endif

    for (int r = 0; r < round; ++r) {
        // --------------------------------------------
        // 1. Phase 1: Pivot Block (Main Stream)
        // --------------------------------------------
#ifdef PROFILING
        hipEventRecord(e_p1_start, stream_main);
#endif
        kernel_phase1<<<1, block_dim, 0, stream_main>>>(d_D, r, V_padded);
        hipEventRecord(event_p1_done, stream_main);
#ifdef PROFILING
        hipEventRecord(e_p1_stop, stream_main);
        hipEventSynchronize(e_p1_stop);
        hipEventElapsedTime(&t_ms, e_p1_start, e_p1_stop);
        acc_p1 += t_ms;
#endif

        // --------------------------------------------
        // 2. Phase 2: Row & Col (Separate Streams)
        // --------------------------------------------

        // --- Row Stream ---
        hipStreamWaitEvent(stream_row, event_p1_done, 0);
#ifdef PROFILING
        hipEventRecord(e_p2_row_start, stream_row);
#endif
        kernel_phase2_row<<<round, block_dim, 0, stream_row>>>(d_D, r, V_padded);
        hipEventRecord(event_p2_row_done, stream_row);
#ifdef PROFILING
        hipEventRecord(e_p2_row_stop, stream_row);
        hipEventSynchronize(e_p2_row_stop);
        hipEventElapsedTime(&t_ms, e_p2_row_start, e_p2_row_stop);
        acc_p2_row += t_ms;
#endif

        // --- Col Stream ---
        hipStreamWaitEvent(stream_col, event_p1_done, 0);
#ifdef PROFILING
        hipEventRecord(e_p2_col_start, stream_col);
#endif
        kernel_phase2_col<<<round, block_dim, 0, stream_col>>>(d_D, r, V_padded);
        hipEventRecord(event_p2_col_done, stream_col);
#ifdef PROFILING
        hipEventRecord(e_p2_col_stop, stream_col);
        hipEventSynchronize(e_p2_col_stop);
        hipEventElapsedTime(&t_ms, e_p2_col_start, e_p2_col_stop);
        acc_p2_col += t_ms;
#endif

        // --------------------------------------------
        // 3. Phase 3: Remaining Blocks (Main Stream)
        // --------------------------------------------
        hipStreamWaitEvent(stream_main, event_p2_row_done, 0);
        hipStreamWaitEvent(stream_main, event_p2_col_done, 0);

        dim3 grid_phase3(round, round);
#ifdef PROFILING
        hipEventRecord(e_p3_start, stream_main);
#endif
        kernel_phase3<<<grid_phase3, block_dim, 0, stream_main>>>(d_D, r, V_padded);
#ifdef PROFILING
        hipEventRecord(e_p3_stop, stream_main);
        hipEventSynchronize(e_p3_stop);
        hipEventElapsedTime(&t_ms, e_p3_start, e_p3_stop);
        acc_p3 += t_ms;
#endif
    }

#ifdef PROFILING
    *time_phase1_ms = acc_p1;
    *time_phase2_row_ms = acc_p2_row;
    *time_phase2_col_ms = acc_p2_col;
    *time_phase3_ms = acc_p3;

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
    if (!file) {
        perror("fopen");
        exit(1);
    }
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    V_padded = (V + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR * BLOCKING_FACTOR;

    hipHostMalloc(&D, (size_t)V_padded * V_padded * sizeof(int), hipHostMallocDefault);

    for (int i = 0; i < V_padded; ++i) {
        for (int j = 0; j < V_padded; ++j) {
            D[i * V_padded + j] = (i == j) ? 0 : INF;
        }
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        D[pair[0] * V_padded + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outfile) {
    FILE *f = fopen(outfile, "w");
    if (!f) {
        perror("fopen output");
        exit(1);
    }
    for (int i = 0; i < V; ++i)
        fwrite(&D[i * V_padded], sizeof(int), V, f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Helper Functions and Kernels (Preserved from your original HIP code)
// ---------------------------------------------------------------------------

// int4 Vectorized Load
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

// int4 Vectorized Store
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

__global__ void __launch_bounds__(1024) kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    const int b_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
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
    const int b_idx_y = r;

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

__global__ void __launch_bounds__(1024) kernel_phase2_col(int *d_D, const int r, const int V_padded) {
    const int b_idx_y = blockIdx.x;
    if (b_idx_y == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_x = r;

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

__global__ void __launch_bounds__(1024) kernel_phase3(int *d_D, const int r, const int V_padded) {
    const int b_idx_x = blockIdx.x;
    const int b_idx_y = blockIdx.y;
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
    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
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
