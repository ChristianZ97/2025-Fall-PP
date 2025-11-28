// hw3-2.hip

/* Headers*/
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

/* Constants & Global Variables */
// [OPT] Blocking factor (tile size). Tune this (e.g., 32/64/128) to study "large blocking factor" vs occupancy & bandwidth.
#define BLOCKING_FACTOR 64  // Matches v2 (64x64 data block)

// [OPT] Thread block dimension (32x32 = 1024 threads). This controls CUDA 2D alignment and occupancy.
#define HALF_BLOCK 32  // Thread block dimension (32x32 threads)

#define INF ((1 << 30) - 1)

static int *D;        // Host pointer
static int *d_D;      // Device pointer
static int V, E;      // Original vertices, edges
static int V_padded;  // Padded vertices (multiple of 64)

// [OPT] Multiple streams & events are used to overlap different phases (streaming / reduce idle time).
hipStream_t stream_main, stream_row, stream_col;
hipEvent_t event_p1_done, event_p2_row_done, event_p2_col_done;

/* Function Prototypes */
__global__ void kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase3(int *d_D, const int r, const int V_padded);

void input(char *infile);
void output(char *outfile);
void block_FW();

/* Main */
int main(int argc, char *argv[]) {
    /*
    if (argc != 3) {
        printf("Usage: %s \n", argv[0]);
        return 1;
    }
    */

    input(argv[1]);

    hipStreamCreate(&stream_main);
    hipStreamCreate(&stream_row);
    hipStreamCreate(&stream_col);
    hipEventCreate(&event_p1_done);
    hipEventCreate(&event_p2_row_done);
    hipEventCreate(&event_p2_col_done);

    size_t size = V_padded * V_padded * sizeof(int);
    hipMalloc(&d_D, size);
    hipMemcpy(d_D, D, size, hipMemcpyHostToDevice);

    block_FW();

    hipMemcpy(D, d_D, size, hipMemcpyDeviceToHost);
    output(argv[2]);

    /*
    hipFree(d_D);
    hipStreamDestroy(stream_main);
    hipStreamDestroy(stream_row);
    hipStreamDestroy(stream_col);
    hipEventDestroy(event_p1_done);
    hipEventDestroy(event_p2_row_done);
    hipEventDestroy(event_p2_col_done);
    */

    return 0;
}

/* Function Definitions */
void block_FW() {
    const int round = V_padded / BLOCKING_FACTOR;

    // [OPT] 2D thread block (HALF_BLOCK x HALF_BLOCK) for good CUDA 2D alignment and coalesced accesses.
    dim3 threads_per_block(HALF_BLOCK, HALF_BLOCK);  // 32x32 threads

    for (int r = 0; r < round; ++r) {
        // 1. Phase 1: Pivot Block
        kernel_phase1<<<1, threads_per_block, 0, stream_main>>>(d_D, r, V_padded);
        hipEventRecord(event_p1_done, stream_main);

        // 2. Phase 2: Pivot Row & Col
        // (round-1) blocks per grid to cover all non-pivot blocks in the pivot row/column.
        hipStreamWaitEvent(stream_row, event_p1_done, 0);
        dim3 grid_phase2(round - 1);
        kernel_phase2_row<<<grid_phase2, threads_per_block, 0, stream_row>>>(d_D, r, V_padded);
        hipEventRecord(event_p2_row_done, stream_row);

        hipStreamWaitEvent(stream_col, event_p1_done, 0);
        kernel_phase2_col<<<grid_phase2, threads_per_block, 0, stream_col>>>(d_D, r, V_padded);
        hipEventRecord(event_p2_col_done, stream_col);

        // 3. Phase 3: Independent Blocks
        // (round, round) blocks per grid
        hipStreamWaitEvent(stream_main, event_p2_row_done, 0);
        hipStreamWaitEvent(stream_main, event_p2_col_done, 0);
        dim3 grid_phase3(round, round);
        kernel_phase3<<<grid_phase3, threads_per_block, 0, stream_main>>>(d_D, r, V_padded);
    }

    // hipDeviceSynchronize();
}

void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    // Calculate Padded Size (Round up to multiple of 64)
    // [OPT] Padding V up to a multiple of BLOCKING_FACTOR simplifies index math and improves coalesced access.
    V_padded = (V + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR * BLOCKING_FACTOR;

    // Use Pinned Memory for faster host-device transfer
    // [OPT] Pinned host memory (hipHostAlloc) increases H2D/D2H bandwidth; affects "memory copy" time in profiling.
    hipHostAlloc(&D, V_padded * V_padded * sizeof(int), hipHostAllocDefault);

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
    hipFreeHost(D);  // Free Pinned Memory
}

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;  // 0..31
    const int ty = threadIdx.y;  // 0..31

    // Shared Memory for the 64x64 block
    // [OPT] Shared memory tile + extra column (+1) to reduce global memory traffic and avoid shared memory bank conflicts.
    __shared__ int sm[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

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
    // [OPT] Unrolling inner loop improves ILP and can help hide latency (occupancy / instruction-level parallelism).

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
    // [OPT] Stores are coalesced as each warp writes contiguous segments of the tile back to global memory.
    d_D[b_start + ty * V_padded + tx] = sm[ty][tx];
    d_D[b_start + ty * V_padded + (tx + HALF_BLOCK)] = sm[ty][tx + HALF_BLOCK];
    d_D[b_start + (ty + HALF_BLOCK) * V_padded + tx] = sm[ty + HALF_BLOCK][tx];
    d_D[b_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = sm[ty + HALF_BLOCK][tx + HALF_BLOCK];
}

__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int b_idx_x = blockIdx.x;
    if (b_idx_x >= r) b_idx_x++;  // Skip the pivot block itself (handled in Phase 1)
    const int b_idx_y = r;

    // [OPT] Shared memory tiles for pivot-row block and current row block to exploit data reuse and reduce global bandwidth.
    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    // Calculate Global Offsets
    const int pivot_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const int self_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

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

__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int b_idx_y = blockIdx.x;
    if (b_idx_y >= r) b_idx_y++;  // Skip the pivot block itself (handled in Phase 1)
    const int b_idx_x = r;

    // [OPT] Shared memory tiles for pivot-column block and current column block to exploit data reuse.
    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    // Calculate Global Offsets
    const int pivot_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const int self_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

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

        // Col Block: self[r][c] = min(self[r][c], self[r][k] + pivot[k][c])
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

    // 3. Write Back
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

    // [OPT] Shared memory tiles for row/col blocks; "self" block is kept in registers to minimize memory traffic.
    __shared__ int sm_row[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Row Block (y, r)
    __shared__ int sm_col[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Col Block (r, x)

    const int row_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const int col_start = (r * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);
    const int self_start = (b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

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
    // [OPT] Keeping the self-block in registers reduces shared/global memory accesses and can improve throughput.
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
