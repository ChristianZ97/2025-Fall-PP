// hw3-3.hip

/* Headers */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

/* Constants & Global Variables */
#define BLOCKING_FACTOR 64  // Matches v2 (64x64 data block)
#define HALF_BLOCK 32       // Thread block dimension (32x32 threads)
#define INF ((1 << 30) - 1)

static int *D;        // Host pointer
static int *d_D[2];   // Device pointers for 2 GPUs
static int V, E;      // Original vertices, edges
static int V_padded;  // Padded vertices (multiple of 64)

hipStream_t stream_main[2], stream_row[2], stream_col[2];
hipEvent_t event_p1_done[2], event_p2_row_done[2], event_p2_col_done[2];

/* Function Prototypes */
__global__ void kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded, const int row_offset, const int row_limit);
__global__ void kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset);

void input(char *infile);
void output(char *outfile);
void block_FW(const int dev_id);

/* Main */
int main(int argc, char *argv[]) {

    /*
    if (argc != 3) {
        printf("Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }
    */

    input(argv[1]);
    // Use size_t for total size calculation
    size_t size = (size_t)V_padded * V_padded * sizeof(int);

    hipSetDevice(0);
    hipDeviceEnablePeerAccess(1, 0);

    hipSetDevice(1);
    hipDeviceEnablePeerAccess(0, 0);

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
        hipMemcpy(d_D[dev_id], D, size, hipMemcpyHostToDevice);

#pragma omp barrier

        block_FW(dev_id);
        hipDeviceSynchronize();

        const int round = V_padded / BLOCKING_FACTOR;
        const int row_mid = round / 2;
        const int row_block_start = (dev_id == 0) ? 0 : row_mid;
        const int row_block_count = (dev_id == 0) ? row_mid : (round - row_mid);
        const int row_start = row_block_start * BLOCKING_FACTOR;
        const int row_num = row_block_count * BLOCKING_FACTOR;

        // FIX: Cast to size_t for pointer arithmetic
        hipMemcpy(D + (size_t)row_start * V_padded, d_D[dev_id] + (size_t)row_start * V_padded, (size_t)row_num * V_padded * sizeof(int), hipMemcpyDeviceToHost);

        hipFree(d_D[dev_id]);
        hipStreamDestroy(stream_main[dev_id]);
        hipStreamDestroy(stream_row[dev_id]);
        hipStreamDestroy(stream_col[dev_id]);
        hipEventDestroy(event_p1_done[dev_id]);
        hipEventDestroy(event_p2_row_done[dev_id]);
        hipEventDestroy(event_p2_col_done[dev_id]);
    }

    output(argv[2]);
    return 0;
}

/* Function Definitions */
void block_FW(const int dev_id) {
    const int round = V_padded / BLOCKING_FACTOR;
    dim3 threads_per_block(HALF_BLOCK, HALF_BLOCK);

    const int row_mid = round / 2;
    const int start_row_block = (dev_id == 0) ? 0 : row_mid;
    const int num_row_blocks = (dev_id == 0) ? row_mid : (round - row_mid);

    for (int r = 0; r < round; ++r) {

        const int owner_id = (r < row_mid) ? 0 : 1;

        if (dev_id == owner_id) {

            kernel_phase1<<<1, threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded);
            hipEventRecord(event_p1_done[dev_id], stream_main[dev_id]);

            hipStreamWaitEvent(stream_row[dev_id], event_p1_done[dev_id], 0);
            kernel_phase2_row<<<dim3(round, 1), threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded);
            hipEventRecord(event_p2_row_done[dev_id], stream_row[dev_id]);

            hipStreamWaitEvent(stream_main[dev_id], event_p2_row_done[dev_id], 0);
            size_t copy_size = (size_t)BLOCKING_FACTOR * V_padded * sizeof(int);
            size_t offset = (size_t)r * BLOCKING_FACTOR * V_padded;
            const int peer_id = 1 - dev_id;

            hipMemcpyPeer(d_D[peer_id] + offset, peer_id, d_D[dev_id] + offset, dev_id, copy_size);
        }

#pragma omp barrier

        hipStreamWaitEvent(stream_main[dev_id], event_p2_row_done[dev_id], 0);
        kernel_phase2_col<<<dim3(num_row_blocks, 1), threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded, start_row_block, num_row_blocks);
        hipEventRecord(event_p2_col_done[dev_id], stream_col[dev_id]);

        hipStreamWaitEvent(stream_main[dev_id], event_p2_col_done[dev_id], 0);
        kernel_phase3<<<dim3(round, num_row_blocks), threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded, start_row_block);

        hipDeviceSynchronize();

#pragma omp barrier
    }
}

void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    // Calculate Padded Size (Round up to multiple of 64)
    V_padded = (V + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR * BLOCKING_FACTOR;

    // Use Pinned Memory for faster host-device transfer
    // FIX: size_t for alloc size
    hipHostAlloc(&D, (size_t)V_padded * V_padded * sizeof(int), hipHostAllocDefault);

    // Initialize with INF (and 0 diagonal)
    // Note: Padding areas are also initialized to avoid side effects
    for (int i = 0; i < V_padded; ++i)
        for (int j = 0; j < V_padded; ++j)
            // FIX: size_t for array indexing
            D[(size_t)i * V_padded + j] = (i == j) ? 0 : INF;

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        // FIX: size_t for array indexing
        D[(size_t)pair[0] * V_padded + pair[1]] = pair[2];
    }

    fclose(file);
}

void output(char *outfile) {
    FILE *f = fopen(outfile, "wb");
    // Write only the valid part (V x V), skipping padding
    for (int i = 0; i < V; ++i)
        // FIX: size_t for pointer arithmetic
        fwrite(&D[(size_t)i * V_padded], sizeof(int), V, f);

    fclose(f);
    hipFreeHost(D);  // Free Pinned Memory
}

/* Kernels */
__global__ void kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;  // 0..31
    const int ty = threadIdx.y;  // 0..31

    // Shared Memory for the 64x64 block
    __shared__ int sm[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    // Global Memory Offset for the Pivot Block (r, r)
    // FIX: Use size_t to prevent overflow
    const size_t b_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);

    // 1. Load Global -> Shared (Each thread loads 4 ints)
    // Access pattern: Top-Left, Top-Right, Bottom-Left, Bottom-Right relative to thread
    sm[ty][tx] = d_D[b_start + ty * V_padded + tx];
    sm[ty][tx + HALF_BLOCK] = d_D[b_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm[ty + HALF_BLOCK][tx] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    __syncthreads();

    // 2. Floyd-Warshall Computation within the block

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        // Read shared memory to registers to avoid bank conflicts
        const int pivot_row_val1 = sm[ty][k];
        const int pivot_row_val2 = sm[ty + HALF_BLOCK][k];
        const int pivot_col_val1 = sm[k][tx];
        const int pivot_col_val2 = sm[k][tx + HALF_BLOCK];

        // Update 4 elements
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

    // Calculate Global Offsets
    // FIX: Use size_t to prevent overflow
    const size_t pivot_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const size_t self_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // 1. Load Data (Pivot & Self)
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

    // 2. Compute

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        int pivot_val1, pivot_val2, self_val1, self_val2;

        // Row Block: self[r][c] = min(self[r][c], pivot[r][k] + self[k][c])
        // Here 'r' is local row index (ty), 'c' is local col index (tx)
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

    // 3. Write Back
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

    // Load Pivot (r, r)
    sm_pivot[ty][tx] = d_D[pivot_start + ty * V_padded + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[pivot_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    // Load Self (Block (y, r))
    sm_self[ty][tx] = d_D[self_start + ty * V_padded + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    __syncthreads();

    // Compute

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

    // Write Back
    d_D[self_start + ty * V_padded + tx] = sm_self[ty][tx];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = sm_self[ty][tx + HALF_BLOCK];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = sm_self[ty + HALF_BLOCK][tx];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK];
}

__global__ void kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset) {
    const int b_idx_x = blockIdx.x;
    const int b_idx_y = blockIdx.y + row_offset;

    if (b_idx_x == r || b_idx_y == r) return;  // Skip Phase 1 & 2 blocks

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ int sm_row[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Row Block (y, r)
    __shared__ int sm_col[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Col Block (r, x)

    // FIX: Use size_t to prevent overflow
    const size_t row_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const size_t col_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);
    const size_t self_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // 1. Load Row & Col Blocks into Shared Memory
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

    // 2. Compute using Registers (Self values are kept in registers)
    int val[2][2];

    // Load Self from Global to Registers directly
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

    // 3. Write Registers -> Global
    d_D[self_start + ty * V_padded + tx] = val[0][0];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = val[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = val[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = val[1][1];
}
