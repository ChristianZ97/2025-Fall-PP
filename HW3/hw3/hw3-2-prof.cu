/**
 * Blocked Floyd–Warshall with CUDA
 * High-resolution profiling (latency vs bandwidth, occupancy, shared memory)
 *
 * NOTE: Profiling code is conditionally compiled with -DPROFILING.
 * When PROFILING is not defined, the core algorithm and behavior are unchanged.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKING_FACTOR 256
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 32

/*
BLOCK_DIM_X * BLOCK_DIM_Y <= 1024 (kernel launch limit)
(BLOCK_DIM_X + BLOCK_DIM_Y) * BLOCKING_FACTOR * 4 <= 49152（shared mem limit）
*/


const int INF = ((1 << 30) - 1);
static int *Dist;
static int *d_Dist;
static int n, m;

void input(char *inFileName);
void output(char *outFileName);

void block_FW(
#ifdef PROFILING
    double *time_phase1_ms, double *time_phase23_ms, long long *single_calls, long long *multiple_calls, double *logical_updates_single, double *logical_updates_multiple
#endif
);
void cal(const int current_round,
         const int block_start_row,
         const int block_start_col,
         const int block_height,
         const int block_width,
         const int dependent
#ifdef PROFILING
         ,
         long long *single_calls,
         long long *multiple_calls,
         double *logical_updates_single,
         double *logical_updates_multiple
#endif
);

__global__ void kernel_single(int *d_Dist, const int n, const int start_row, const int start_col, const int end_row, const int end_col, const int k);

__global__ void kernel_multiple(int *d_Dist, const int n, const int start_row, const int start_col, const int end_row, const int end_col, const int start_k, const int end_k);

int main(int argc, char *argv[]) {

    input(argv[1]);

    cudaMalloc(&d_Dist, (size_t)n * (size_t)n * sizeof(int));

#ifdef PROFILING
    // -------------------------
    // Host-side CUDA events
    // -------------------------
    cudaEvent_t event_h2d_start, event_h2d_stop;
    cudaEvent_t event_d2h_start, event_d2h_stop;
    cudaEventCreate(&event_h2d_start);
    cudaEventCreate(&event_h2d_stop);
    cudaEventCreate(&event_d2h_start);
    cudaEventCreate(&event_d2h_stop);

    // Phase-wise timing aggregates (device kernels)
    double time_phase1_ms = 0.0;
    double time_phase23_ms = 0.0;

    long long single_calls = 0;
    long long multiple_calls = 0;
    double logical_updates_single = 0.0;
    double logical_updates_multiple = 0.0;
#endif

    // -------------------------
    // H2D transfer
    // -------------------------
#ifdef PROFILING
    cudaEventRecord(event_h2d_start);
#endif
    cudaMemcpy(d_Dist, Dist, (size_t)n * (size_t)n * sizeof(int), cudaMemcpyHostToDevice);
#ifdef PROFILING
    cudaEventRecord(event_h2d_stop);
    cudaEventSynchronize(event_h2d_stop);
#endif

    // -------------------------
    // Kernel execution
    // -------------------------
#ifdef PROFILING
    block_FW(&time_phase1_ms, &time_phase23_ms, &single_calls, &multiple_calls, &logical_updates_single, &logical_updates_multiple);
#else
    block_FW();
#endif

    // -------------------------
    // D2H transfer
    // -------------------------
#ifdef PROFILING
    cudaEventRecord(event_d2h_start);
#endif
    cudaMemcpy(Dist, d_Dist, (size_t)n * (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);
#ifdef PROFILING
    cudaEventRecord(event_d2h_stop);
    cudaEventSynchronize(event_d2h_stop);
#endif

    cudaFree(d_Dist);

#ifdef PROFILING
    // ============================================================
    // Profiling report (latency vs bandwidth, occupancy, etc.)
    // ============================================================
    float time_h2d = 0.0f, time_d2h = 0.0f;
    cudaEventElapsedTime(&time_h2d, event_h2d_start, event_h2d_stop);
    cudaEventElapsedTime(&time_d2h, event_d2h_start, event_d2h_stop);

    float time_kernel = (float)(time_phase1_ms + time_phase23_ms);
    float total_time = time_h2d + time_kernel + time_d2h;
    size_t total_bytes = (size_t)n * (size_t)n * sizeof(int);

    fprintf(stderr, "\n=== CUDA PROFILING (Blocked Floyd–Warshall) ===\n");
    fprintf(stderr, "n = %d, m = %d\n", n, m);
    fprintf(stderr, "Matrix size: %d x %d, total bytes = %zu (%.2f GiB)\n", n, n, total_bytes, (double)total_bytes / (1024.0 * 1024.0 * 1024.0));

    // -------------------------
    // Time & bandwidth
    // -------------------------
    fprintf(stderr, "\n[Host–Device Transfers]\n");
    fprintf(stderr, "  H2D: %.3f ms, %.2f GB/s\n", time_h2d, (total_bytes / 1e9) / (time_h2d / 1000.0 + 1e-12));
    fprintf(stderr, "  D2H: %.3f ms, %.2f GB/s\n", time_d2h, (total_bytes / 1e9) / (time_d2h / 1000.0 + 1e-12));

    fprintf(stderr, "\n[Kernel Time Breakdown]\n");
    fprintf(stderr, "  Phase 1 (dependent, kernel_single): %.3f ms\n", time_phase1_ms);
    fprintf(stderr, "  Phase 2+3 (independent, kernel_multiple): %.3f ms\n", time_phase23_ms);
    fprintf(stderr, "  Kernel total: %.3f ms\n", time_kernel);
    fprintf(stderr, "  Total wall time: %.3f ms\n", total_time);

    // -------------------------
    // Latency vs bandwidth heuristics
    // -------------------------
    fprintf(stderr, "\n[Latency vs Bandwidth]\n");
    float frac_h2d = 100.0f * time_h2d / (total_time + 1e-9f);
    float frac_d2h = 100.0f * time_d2h / (total_time + 1e-9f);
    float frac_kernel = 100.0f * time_kernel / (total_time + 1e-9f);
    fprintf(stderr, "  H2D fraction:   %.1f%%\n", frac_h2d);
    fprintf(stderr, "  D2H fraction:   %.1f%%\n", frac_d2h);
    fprintf(stderr, "  Kernel fraction:%.1f%%\n", frac_kernel);

    if (time_kernel > time_h2d + time_d2h) {
        fprintf(stderr, "  -> Runtime is kernel-dominated. Focus on kernel latency, occupancy,\n");
        fprintf(stderr, "     memory hierarchy (L2/shared/global), and instruction throughput.\n");
    } else {
        fprintf(stderr, "  -> Runtime is transfer-dominated. PCIe bandwidth/latency is a major\n");
        fprintf(stderr, "     factor. Consider overlapping transfers or reducing traffic.\n");
    }

    // -------------------------
    // Logical work & GUpdates/s
    // -------------------------
    double logical_updates_n3 = (double)n * (double)n * (double)n;
    double logical_updates_est = logical_updates_single + logical_updates_multiple;

    fprintf(stderr, "\n[Logical Work (Floyd–Warshall updates)]\n");
    fprintf(stderr, "  Theoretical n^3 updates:          %.3e\n", logical_updates_n3);
    fprintf(stderr, "  Estimated Phase 1 updates:        %.3e\n", logical_updates_single);
    fprintf(stderr, "  Estimated Phase 2+3 updates:      %.3e\n", logical_updates_multiple);
    fprintf(stderr, "  Estimated total logical updates:  %.3e\n", logical_updates_est);

    fprintf(stderr, "\n[GUpdates/s]\n");
    if (time_kernel > 0.0f) {
        double gups_n3 = logical_updates_n3 / (time_kernel * 1e6);
        double gups_est = logical_updates_est / (time_kernel * 1e6);
        double gups_phase1 = logical_updates_single / (time_phase1_ms * 1e6 + 1e-3);
        double gups_phase23 = logical_updates_multiple / (time_phase23_ms * 1e6 + 1e-3);
        fprintf(stderr, "  Using n^3 / total kernel time:    %.2f GUpdates/s\n", gups_n3);
        fprintf(stderr, "  Using estimated logical updates:  %.2f GUpdates/s\n", gups_est);
        fprintf(stderr, "  Phase 1 (kernel_single) only:     %.2f GUpdates/s\n", gups_phase1);
        fprintf(stderr, "  Phase 2+3 (kernel_multiple) only: %.2f GUpdates/s\n", gups_phase23);
    } else {
        fprintf(stderr, "  Kernel time is 0? (check profiler)\n");
    }

    fprintf(stderr, "\n[Kernel Call Statistics]\n");
    fprintf(stderr, "  kernel_single launches:   %lld\n", single_calls);
    fprintf(stderr, "  kernel_multiple launches: %lld\n", multiple_calls);

    // -------------------------
    // Shared memory, registers, occupancy (kernel_multiple)
    // -------------------------
    cudaFuncAttributes attr_mult;
    cudaFuncGetAttributes(&attr_mult, kernel_multiple);

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int block_size = BLOCK_DIM_X * BLOCK_DIM_Y;
    int max_active_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, kernel_multiple, block_size, 0);

    float occupancy = (float)max_active_blocks_per_sm * block_size / (float)prop.maxThreadsPerMultiProcessor;

    fprintf(stderr, "\n[Kernel_multiple Resource Usage]\n");
    fprintf(stderr, "  SMs:                         %d\n", prop.multiProcessorCount);
    fprintf(stderr, "  Max threads per SM:          %d\n", prop.maxThreadsPerMultiProcessor);
    fprintf(stderr, "  Block size (threads/block):  %d\n", block_size);
    fprintf(stderr, "  Static shared mem per block: %zu bytes\n", attr_mult.sharedSizeBytes);
    fprintf(stderr, "  Registers per thread:        %d\n", attr_mult.numRegs);
    fprintf(stderr, "  Max active blocks per SM:    %d\n", max_active_blocks_per_sm);
    fprintf(stderr, "  Theoretical occupancy:       %.2f %%\n", occupancy * 100.0f);

    // -------------------------
    // Simple bottleneck label
    // -------------------------
    fprintf(stderr, "\n[High-level Bottleneck]\n");
    if (time_h2d >= time_kernel && time_h2d >= time_d2h) {
        fprintf(stderr, "  Dominant: H2D transfer (%.1f%% of total)\n", frac_h2d);
    } else if (time_d2h >= time_kernel && time_d2h >= time_h2d) {
        fprintf(stderr, "  Dominant: D2H transfer (%.1f%% of total)\n", frac_d2h);
    } else {
        fprintf(stderr, "  Dominant: Kernel execution (%.1f%% of total)\n", frac_kernel);
    }

    cudaEventDestroy(event_h2d_start);
    cudaEventDestroy(event_h2d_stop);
    cudaEventDestroy(event_d2h_start);
    cudaEventDestroy(event_d2h_stop);
#endif  // PROFILING

    output(argv[2]);
    free(Dist);
    return 0;
}

// ============================================================
// Blocked FW driver
// ============================================================
void block_FW(
#ifdef PROFILING
    double *time_phase1_ms, double *time_phase23_ms, long long *single_calls, long long *multiple_calls, double *logical_updates_single, double *logical_updates_multiple
#endif
) {
    const int round = (n + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR;

#ifdef PROFILING
    // Per-round timing aggregation via CUDA events
    cudaEvent_t e_p1_start, e_p1_stop;
    cudaEvent_t e_p23_start, e_p23_stop;
    cudaEventCreate(&e_p1_start);
    cudaEventCreate(&e_p1_stop);
    cudaEventCreate(&e_p23_start);
    cudaEventCreate(&e_p23_stop);

    float acc_p1 = 0.0f;
    float acc_p23 = 0.0f;
#endif

    for (int r = 0; r < round; ++r) {

        // Phase 1: pivot block (self-dependent)
#ifdef PROFILING
        cudaEventRecord(e_p1_start);
#endif
        cal(r, r, r, 1, 1, 1
#ifdef PROFILING
            ,
            single_calls, multiple_calls, logical_updates_single, logical_updates_multiple
#endif
        );
#ifdef PROFILING
        cudaEventRecord(e_p1_stop);
        cudaEventSynchronize(e_p1_stop);
        float t_p1 = 0.0f;
        cudaEventElapsedTime(&t_p1, e_p1_start, e_p1_stop);
        acc_p1 += t_p1;
#endif

        // Phase 2 + Phase 3: row/column and remaining blocks
#ifdef PROFILING
        cudaEventRecord(e_p23_start);
#endif

        // Phase 2
        cal(r, r, 0, 1, r, 0
#ifdef PROFILING
            ,
            single_calls, multiple_calls, logical_updates_single, logical_updates_multiple
#endif
        );
        cal(r, r, r + 1, 1, round - r - 1, 0
#ifdef PROFILING
            ,
            single_calls, multiple_calls, logical_updates_single, logical_updates_multiple
#endif
        );
        cal(r, 0, r, r, 1, 0
#ifdef PROFILING
            ,
            single_calls, multiple_calls, logical_updates_single, logical_updates_multiple
#endif
        );
        cal(r, r + 1, r, round - r - 1, 1, 0
#ifdef PROFILING
            ,
            single_calls, multiple_calls, logical_updates_single, logical_updates_multiple
#endif
        );

        // Phase 3
        cal(r, 0, 0, r, r, 0
#ifdef PROFILING
            ,
            single_calls, multiple_calls, logical_updates_single, logical_updates_multiple
#endif
        );
        cal(r, 0, r + 1, r, round - r - 1, 0
#ifdef PROFILING
            ,
            single_calls, multiple_calls, logical_updates_single, logical_updates_multiple
#endif
        );
        cal(r, r + 1, 0, round - r - 1, r, 0
#ifdef PROFILING
            ,
            single_calls, multiple_calls, logical_updates_single, logical_updates_multiple
#endif
        );
        cal(r, r + 1, r + 1, round - r - 1, round - r - 1, 0
#ifdef PROFILING
            ,
            single_calls, multiple_calls, logical_updates_single, logical_updates_multiple
#endif
        );

#ifdef PROFILING
        cudaEventRecord(e_p23_stop);
        cudaEventSynchronize(e_p23_stop);
        float t_p23 = 0.0f;
        cudaEventElapsedTime(&t_p23, e_p23_start, e_p23_stop);
        acc_p23 += t_p23;
#endif
    }

#ifdef PROFILING
    *time_phase1_ms = acc_p1;
    *time_phase23_ms = acc_p23;

    cudaEventDestroy(e_p1_start);
    cudaEventDestroy(e_p1_stop);
    cudaEventDestroy(e_p23_start);
    cudaEventDestroy(e_p23_stop);
#endif
}

// ============================================================
// Block-level scheduling
// ============================================================
void cal(const int current_round,
         const int block_start_row,
         const int block_start_col,
         const int block_height,
         const int block_width,
         const int dependent
#ifdef PROFILING
         ,
         long long *single_calls,
         long long *multiple_calls,
         double *logical_updates_single,
         double *logical_updates_multiple
#endif
) {
    const int start_row = block_start_row * BLOCKING_FACTOR;
    const int start_col = block_start_col * BLOCKING_FACTOR;

    const int end_row = (int)min((long long)(block_start_row + block_height) * BLOCKING_FACTOR, (long long)n);
    const int end_col = (int)min((long long)(block_start_col + block_width) * BLOCKING_FACTOR, (long long)n);

    const int start_k = current_round * BLOCKING_FACTOR;
    const int end_k = (int)min((long long)(current_round + 1) * BLOCKING_FACTOR, (long long)n);

    const int rows = end_row - start_row;
    const int cols = end_col - start_col;

    dim3 threads_per_block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocks_per_grid((cols + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (rows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    if (dependent) {
        // Phase 1: one kernel per k
        for (int k = start_k; k < end_k; ++k) {
            kernel_single<<<blocks_per_grid, threads_per_block>>>(d_Dist, n, start_row, start_col, end_row, end_col, k);
#ifdef PROFILING
            (*single_calls)++;
            (*logical_updates_single) += (double)rows * (double)cols;
#endif
        }
    } else {
        // Phase 2+3: one kernel for k in [start_k, end_k)
        kernel_multiple<<<blocks_per_grid, threads_per_block>>>(d_Dist, n, start_row, start_col, end_row, end_col, start_k, end_k);
#ifdef PROFILING
        (*multiple_calls)++;
        (*logical_updates_multiple) += (double)rows * (double)cols * (double)(end_k - start_k);
#endif
    }
}

// ============================================================
// Device kernels
// ============================================================
__global__ void kernel_single(int *d_Dist, const int n, const int start_row, const int start_col, const int end_row, const int end_col, const int k) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x + start_col;
    const int row = blockDim.y * blockIdx.y + threadIdx.y + start_row;

    if (row >= end_row || col >= end_col) return;

    int ij = d_Dist[row * n + col];
    int ik = d_Dist[row * n + k];
    int kj = d_Dist[k * n + col];
    int alt = ik + kj;
    d_Dist[row * n + col] = (ij < alt) ? ij : alt;
}

__global__ void kernel_multiple(int *d_Dist, const int n, const int start_row, const int start_col, const int end_row, const int end_col, const int start_k, const int end_k) {

    __shared__ int sm_ik[BLOCK_DIM_Y][BLOCKING_FACTOR];
    __shared__ int sm_kj[BLOCKING_FACTOR][BLOCK_DIM_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockDim.x * blockIdx.x + tx + start_col;
    const int row = blockDim.y * blockIdx.y + ty + start_row;

    // Load row segment into shared memory
    for (int tk = tx; tk < BLOCKING_FACTOR; tk += BLOCK_DIM_X) {
        int gk = start_k + tk;
        sm_ik[ty][tk] = (gk < n && row < n) ? d_Dist[row * n + gk] : INF;
    }

    // Load column segment into shared memory
    for (int tk = ty; tk < BLOCKING_FACTOR; tk += BLOCK_DIM_Y) {
        int gk = start_k + tk;
        sm_kj[tk][tx] = (col < n && gk < n) ? d_Dist[gk * n + col] : INF;
    }

    __syncthreads();

    if (row >= end_row || col >= end_col) return;

    int dist_ij = d_Dist[row * n + col];

#pragma unroll
    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        int alt = sm_ik[ty][k] + sm_kj[k][tx];
        dist_ij = (dist_ij < alt) ? dist_ij : alt;
    }

    d_Dist[row * n + col] = dist_ij;
}

// ============================================================
// I/O helpers
// ============================================================
void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    Dist = (int *)malloc((size_t)n * (size_t)n * sizeof(int));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Dist[i * n + j] = (i == j) ? 0 : INF;

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            if (Dist[i * n + j] >= INF) Dist[i * n + j] = INF;
        fwrite(Dist + i * n, sizeof(int), n, outfile);
    }
    fclose(outfile);
}
