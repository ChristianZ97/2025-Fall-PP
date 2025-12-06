// hw4.cu

#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef PROFILING
#include <time.h>
#endif

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *d_Q, float *d_K, float *d_V, float *d_O, const int B, const int N, const int d);

__global__ void flash_attention_kernel(float *d_Q, float *d_K, float *d_V, float *d_O, const int N, const int d);

// Algorithm 1 Line 1: Set block sizes BR, BC
// [Optimization: Tiling / Block Size Configuration]
// Defining block sizes small enough to fit in Shared Memory (SRAM).
// (BR + 2 * BC) * d * 4 <= 48KB
// BR + 2 * BC <= 192, if d = 64
#define BR 64
#define BC 16

// Host Variables
static int B, N, d;
static float *Q, *K, *V, *O;
static float *d_Q, *d_K, *d_V, *d_O;

int main(int argc, char *argv[]) {

#ifdef PROFILING
    // -------------------------
    // CPU-side I/O timing
    // -------------------------
    double time_io_ms = 0.0;

    // -------------------------
    // Device-side commu timing (mem + transfer)
    // -------------------------
    double time_commu_ms = 0.0;

    // -------------------------
    // Device-side compute timing
    // -------------------------
    double time_comp_ms = 0.0;

    // CUDA events for H2D / D2H / compute
    hipEvent_t event_h2d_start, event_h2d_stop;
    hipEvent_t event_d2h_start, event_d2h_stop;
    hipEvent_t event_comp_start, event_comp_stop;

    hipEventCreate(&event_h2d_start);
    hipEventCreate(&event_h2d_stop);
    hipEventCreate(&event_d2h_start);
    hipEventCreate(&event_d2h_stop);
    hipEventCreate(&event_comp_start);
    hipEventCreate(&event_comp_stop);
#endif

#ifdef PROFILING
    // ---- input I/O ----
    clock_t io_start = clock();
#endif

    input(argv[1]);

#ifdef PROFILING
    clock_t io_end = clock();
    time_io_ms += 1000.0 * (io_end - io_start) / CLOCKS_PER_SEC;
#endif

    size_t size = B * N * d * sizeof(float);

#ifdef PROFILING
    // ---- mem alloc (hipMalloc) ----
    clock_t alloc_start = clock();
#endif

    hipMalloc(&d_Q, size);
    hipMalloc(&d_K, size);
    hipMalloc(&d_V, size);
    hipMalloc(&d_O, size);

#ifdef PROFILING
    clock_t alloc_end = clock();
    time_commu_ms += 1000.0 * (alloc_end - alloc_start) / CLOCKS_PER_SEC;
#endif

    // -------------------------
    // H2D transfer (Q, K, V)
    // -------------------------
#ifdef PROFILING
    hipEventRecord(event_h2d_start);
#endif

    hipMemcpy(d_Q, Q, size, hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, size, hipMemcpyHostToDevice);
    hipMemcpy(d_V, V, size, hipMemcpyHostToDevice);

#ifdef PROFILING
    hipEventRecord(event_h2d_stop);
    hipEventSynchronize(event_h2d_stop);
#endif

    // -------------------------
    // Computation (kernel)
    // -------------------------
#ifdef PROFILING
    hipEventRecord(event_comp_start);
#endif

    flash_attention(d_Q, d_K, d_V, d_O, B, N, d);

#ifdef PROFILING
    hipEventRecord(event_comp_stop);
    hipEventSynchronize(event_comp_stop);
#endif

    // -------------------------
    // D2H transfer (O)
    // -------------------------
#ifdef PROFILING
    hipEventRecord(event_d2h_start);
#endif

    hipMemcpy(O, d_O, size, hipMemcpyDeviceToHost);

#ifdef PROFILING
    hipEventRecord(event_d2h_stop);
    hipEventSynchronize(event_d2h_stop);
#endif

#ifdef PROFILING
    // ---- mem free (hipFree) ----
    clock_t free_start = clock();
#endif

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_O);

#ifdef PROFILING
    clock_t free_end = clock();
    time_commu_ms += 1000.0 * (free_end - free_start) / CLOCKS_PER_SEC;
#endif

#ifdef PROFILING
    // ---- output I/O ----
    clock_t io_write_start = clock();
#endif

    output(argv[2]);

#ifdef PROFILING
    clock_t io_write_end = clock();
    time_io_ms += 1000.0 * (io_write_end - io_write_start) / CLOCKS_PER_SEC;
#endif

#ifdef PROFILING
    // ============================================================
    //   - io: input + output
    //   - commu: hipMalloc + hipFree + H2D + D2H
    //   - comp: flash_attention
    // ============================================================
    float t_h2d_f = 0.0f, t_d2h_f = 0.0f, t_comp_f = 0.0f;
    hipEventElapsedTime(&t_h2d_f, event_h2d_start, event_h2d_stop);
    hipEventElapsedTime(&t_d2h_f, event_d2h_start, event_d2h_stop);
    hipEventElapsedTime(&t_comp_f, event_comp_start, event_comp_stop);

    time_commu_ms += (double)t_h2d_f + (double)t_d2h_f;
    time_comp_ms = (double)t_comp_f;
    double total_ms = time_io_ms + time_commu_ms + time_comp_ms;

    // total,io,commu,comp
    fprintf(stderr, "[PROF_RESULT],%.3f,%.3f,%.3f,%.3f\n", total_ms, time_io_ms, time_commu_ms, time_comp_ms);

    hipEventDestroy(event_h2d_start);
    hipEventDestroy(event_h2d_stop);
    hipEventDestroy(event_d2h_start);
    hipEventDestroy(event_d2h_stop);
    hipEventDestroy(event_comp_start);
    hipEventDestroy(event_comp_stop);
#endif

    return 0;
}

__global__ void flash_attention_kernel(float *d_Q, float *d_K, float *d_V, float *d_O, const int N, const int d) {

    // [Optimization: CUDA 2D Alignment / Batch Handling]
    // Uses the y-dimension of the grid (blockIdx.y) to process different batches in parallel.
    // This maps the B x N problem structure to the GPU hardware hierarchy efficiently.
    const int batch_idx = blockIdx.y;
    const long long batch_offset = batch_idx * (N * d);

    d_Q += batch_offset;
    d_K += batch_offset;
    d_V += batch_offset;
    d_O += batch_offset;

    // --- ALGORITHM 1 LOGIC ---

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;  // Block index corresponding to loop "i" in Algorithm

    // Global row index for Q_i
    const int row_offset_Q = bx * BR;

    // [Optimization: Shared Memory]
    // Using 'extern __shared__' to dynamically allocate fast on-chip memory (SRAM).
    // This allows reusable data (Q, K, V blocks) to be stored close to compute units,
    // drastically reducing the number of slow Global Memory (HBM) accesses.
    extern __shared__ float sram[];

    // Line 6 & 8: Buffers for loading blocks into SRAM
    float *sm_Q = sram;
    float *sm_K = sram + BR * d;
    float *sm_V = sram + BR * d + BC * d;

    // Line 2: Initialize O_i, l_i, m_i
    // O_i corresponds to the accumulator for the current row
    // [Optimization: Register Caching]
    // Storing the accumulator O_i in registers (local array) rather than writing to global memory
    // in every iteration reduces memory traffic.
    float O_i[64];

// [Optimization: Loop Unrolling]
// #pragma unroll hints the compiler to expand the loop inline.
// This reduces loop control overhead (branch prediction, index increment)
// and exposes more instructions for pipeline parallelism.
#pragma unroll
    for (int t = 0; t < d; t++)
        O_i[t] = 0.0f;

    // l_i (running sum), m_i (running max)
    float l_i = 0.0f;
    float m_i = -FLT_MAX;

// Line 8: Load Q_i from HBM to on-chip SRAM
// [Optimization: Global to Shared Memory Load]
// Loading the query block once into Shared Memory so it can be reused
// across the inner loop iterations (against many K/V blocks).
#pragma unroll
    for (int t = 0; t < d; t++)
        sm_Q[tx * d + t] = d_Q[(row_offset_Q + tx) * d + t];
    //__syncthreads();

    // Line 5: Outer loop "for 1 <= j <= Tc"
    // [Optimization: Tiling]
    // The core FlashAttention technique: processing the matrix in small tiles (BC)
    // to respect the limited size of Shared Memory.
    for (int j = 0; j < N; j += BC) {

// Line 6: Load K_j, V_j from HBM to SRAM
// [Optimization: Cooperative Loading]
// Threads cooperate to load data into shared memory.
#pragma unroll
        for (int t = tx; t < BC; t += blockDim.x) {
            for (int k = 0; k < d; k++) {
                sm_K[t * d + k] = d_K[(j + t) * d + k];
                sm_V[t * d + k] = d_V[(j + t) * d + k];
            }
        }
        __syncthreads();  // Ensure K and V are fully loaded before computation

        // Line 9: Compute S_ij = Q_i * K_j^T
        float S_ij[BC];
        float m_ij_tilde = -FLT_MAX;  // Line 10: rowmax(S_ij)

        for (int k = 0; k < BC; k++) {

            float score = 0.0f;
            // [Optimization: Shared Memory Access]
            // Accessing operands from sm_Q and sm_K (SRAM) is much faster than HBM.
            for (int t = 0; t < d; t++)
                score += sm_Q[tx * d + t] * sm_K[k * d + t];
            score *= (1.0f / sqrtf((float)d));

            S_ij[k] = score;
            m_ij_tilde = fmaxf(m_ij_tilde, score);
        }

        // Line 11: Compute m_i_new
        // [Optimization: Online Softmax]
        // Updating statistics (max and sum-exp) incrementally.
        // This avoids materializing the full N x N attention matrix in memory,
        // reducing space complexity from O(N^2) to O(N).
        float m_i_new = fmaxf(m_i, m_ij_tilde);

        // Compute terms for Line 11 & 12 updates
        // Term 1: exp(m_i - m_i_new)
        float exp_m_diff_old = expf(m_i - m_i_new);

        // Term 2: P_ij = exp(S_ij - m_i_new)
        // Note: Code optimizes by using m_i_new directly instead of tilde_m_ij
        float P_ij[BC];
        float l_ij_sum_exp = 0.0f;  // Part of Line 11: sum(P_ij)

        for (int k = 0; k < BC; k++) {

            P_ij[k] = expf(S_ij[k] - m_i_new);
            l_ij_sum_exp += P_ij[k];
        }

        // Line 11: Update l_i_new
        float l_i_new = l_i * exp_m_diff_old + l_ij_sum_exp;

        // Line 12: Update O_i
        // O_i = O_i * exp(m_i - m_i_new) + P_ij * V_j
        // Note: Division by l_i_new is done at the very end (Step 4)

        for (int t = 0; t < d; t++) {

            float P_ij_V_j = 0.0f;
            for (int k = 0; k < BC; k++)
                P_ij_V_j += P_ij[k] * sm_V[k * d + t];

            O_i[t] = O_i[t] * exp_m_diff_old + P_ij_V_j;
        }

        // Line 13: Update m_i, l_i
        m_i = m_i_new;
        l_i = l_i_new;
        __syncthreads();
    }

    // Line 12 (Finalize): Write O_i to HBM
    // Normalization by diag(l_i)^{-1}

#pragma unroll
    for (int t = 0; t < d; t++)
        d_O[(row_offset_Q + tx) * d + t] = O_i[t] / l_i;
}

void flash_attention(float *d_Q, float *d_K, float *d_V, float *d_O, const int B, const int N, const int d) {

    // Line 3: Divide Q into Tr blocks
    // [Optimization: Occupancy / Grid Configuration]
    // Calculates the grid dimensions to cover the Sequence Length (N) and Batch Size (B).
    // (N + BR - 1) / BR ensures we have enough blocks to cover N rows.
    dim3 blocks_per_grid((N + BR - 1) / BR, B);
    dim3 threads_per_block(BR);

    // [Optimization: Dynamic Shared Memory Calculation]
    // Calculates exact shared memory usage required for Q, K, V blocks to maximize usage.
    size_t sram_size = (BR * d + BC * d + BC * d) * sizeof(float);
    flash_attention_kernel<<<blocks_per_grid, threads_per_block, sram_size>>>(d_Q, d_K, d_V, d_O, N, d);
}

void input(char *input_filename) {

    FILE *file = fopen(input_filename, "rb");
    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));
    fclose(file);
}

void output(char *output_filename) {

    FILE *file = fopen(output_filename, "wb");
    fwrite(O, sizeof(float), B * N * d, file);
    free(Q);
    free(K);
    free(V);
    free(O);
    fclose(file);
}
