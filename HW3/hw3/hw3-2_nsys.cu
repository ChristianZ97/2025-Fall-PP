// hw3-2_nsys.cu

/* Headers*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef PROFILING
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <string.h>
#include <stdint.h>

// NVTX color definitions for CUDA pipeline profiling
#define COLOR_IO     0xFF5C9FFF  // Azure Blue for IO (input/output)
#define COLOR_COPY   0xFFFFA500  // Orange for H2D / D2H copies
#define COLOR_KERNEL 0xFF5FD068  // Green for kernels / compute
#define COLOR_SETUP  0xFFFFBE3D  // Yellow for setup / block_FW
#define COLOR_DEFAULT 0xFFCBD5E0 // Soft gray

// Smart NVTX macro: auto-selects color based on range name pattern
#define NVTX_PUSH(name)                                      \
    do {                                                     \
        uint32_t color = COLOR_DEFAULT;                      \
        if (strstr(name, "IO_") != NULL) {                   \
            color = COLOR_IO;                                \
        } else if (strstr(name, "H2D") != NULL ||            \
                   strstr(name, "D2H") != NULL) {            \
            color = COLOR_COPY;                              \
        } else if (strstr(name, "Kernel_") != NULL ||        \
                   strstr(name, "Block_FW") != NULL) {       \
            color = COLOR_KERNEL;                            \
        } else {                                             \
            color = COLOR_DEFAULT;                           \
        }                                                    \
        nvtxEventAttributes_t eventAttrib = {0};             \
        eventAttrib.version = NVTX_VERSION;                  \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;    \
        eventAttrib.colorType = NVTX_COLOR_ARGB;             \
        eventAttrib.color = color;                           \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;   \
        eventAttrib.message.ascii = name;                    \
        nvtxRangePushEx(&eventAttrib);                       \
    } while (0)

#define NVTX_POP() nvtxRangePop()
#endif

/*
 * Blocked Floyd–Warshall with CUDA
 * Optional high-resolution profiling (enable with -DPROFILING).
 * When PROFILING is not defined, the core algorithm and behavior are unchanged.
 */

/* Constants & Global Variables */
// [OPT] Blocking factor (tile size). Tune this (e.g., 32/64/128) to study "large blocking factor" vs occupancy & bandwidth.
#define BLOCKING_FACTOR 64  // Matches v2 (64x64 data block)

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
void block_FW();

/* Main */

int main(int argc, char *argv[]) {

    /*
        if (argc != 3) {
            printf("Usage: %s \n", argv[0]);
            return 1;
        }
    */

#ifdef PROFILING
    // Input file read (host-side IO)
    NVTX_PUSH("IO_Input");
#endif
    input(argv[1]);
#ifdef PROFILING
    NVTX_POP();
    // CUDA setup: streams, events, device memory
    NVTX_PUSH("Init_CUDA");
#endif

    cudaStreamCreate(&stream_main);
    cudaStreamCreate(&stream_row);
    cudaStreamCreate(&stream_col);
    cudaEventCreate(&event_p1_done);
    cudaEventCreate(&event_p2_row_done);
    cudaEventCreate(&event_p2_col_done);

    size_t size = V_padded * V_padded * sizeof(int);
    cudaMalloc(&d_D, size);

#ifdef PROFILING
    NVTX_POP();
    // Host-to-Device transfer of adjacency matrix
    NVTX_PUSH("H2D_Copy");
#endif
    // -------------------------
    // H2D transfer
    // -------------------------
    cudaMemcpy(d_D, D, size, cudaMemcpyHostToDevice);
#ifdef PROFILING
    NVTX_POP();
    // Main blocked Floyd–Warshall loop on GPU
    NVTX_PUSH("Block_FW");
#endif

    // -------------------------
    // Kernel execution
    // -------------------------
    block_FW();

#ifdef PROFILING
    NVTX_POP();
    // Device-to-Host transfer of result matrix
    NVTX_PUSH("D2H_Copy");
#endif
    // -------------------------
    // D2H transfer
    // -------------------------
    cudaMemcpy(D, d_D, size, cudaMemcpyDeviceToHost);
#ifdef PROFILING
    NVTX_POP();
    // Output write-back (host-side IO)
    NVTX_PUSH("IO_Output");
#endif

    output(argv[2]);

#ifdef PROFILING
    NVTX_POP();
#endif

    /*
        cudaFree(d_D);
        cudaStreamDestroy(stream_main);
        cudaStreamDestroy(stream_row);
        cudaStreamDestroy(stream_col);
        cudaEventDestroy(event_p1_done);
        cudaEventDestroy(event_p2_row_done);
        cudaEventDestroy(event_p2_col_done);
    */

    return 0;
}

/* Function Definitions */
void block_FW() {

    const int round = V_padded / BLOCKING_FACTOR;

    // [OPT] 2D thread block (HALF_BLOCK x HALF_BLOCK) for good CUDA 2D alignment and coalesced accesses.
    dim3 threads_per_block(HALF_BLOCK, HALF_BLOCK);  // 32x32 threads

#ifdef PROFILING
    // Enclose the full FW iteration loop
    NVTX_PUSH("Block_FW_Loop");
#endif
    for (int r = 0; r < round; ++r) {
        // 1. Phase 1: Pivot Block
#ifdef PROFILING
        NVTX_PUSH("Kernel_Phase1_Pivot");
#endif
        kernel_phase1<<<1, threads_per_block, 0, stream_main>>>(d_D, r, V_padded);
#ifdef PROFILING
        NVTX_POP();
#endif
        cudaEventRecord(event_p1_done, stream_main);

        // 2. Phase 2: Pivot Row (row stream)
        cudaStreamWaitEvent(stream_row, event_p1_done, 0);
#ifdef PROFILING
        NVTX_PUSH("Kernel_Phase2_Row");
#endif
        kernel_phase2_row<<<round, threads_per_block, 0, stream_row>>>(d_D, r, V_padded);
#ifdef PROFILING
        NVTX_POP();
#endif
        cudaEventRecord(event_p2_row_done, stream_row);

        // 2. Phase 2: Pivot Col (col stream)
        cudaStreamWaitEvent(stream_col, event_p1_done, 0);
#ifdef PROFILING
        NVTX_PUSH("Kernel_Phase2_Col");
#endif
        kernel_phase2_col<<<round, threads_per_block, 0, stream_col>>>(d_D, r, V_padded);
#ifdef PROFILING
        NVTX_POP();
#endif
        cudaEventRecord(event_p2_col_done, stream_col);

        cudaStreamWaitEvent(stream_main, event_p2_row_done, 0);
        cudaStreamWaitEvent(stream_main, event_p2_col_done, 0);

        // 3. Phase 3: Independent Blocks
        // (round, round) blocks per grid
#ifdef PROFILING
        // This is the largest kernel; use this range for occupancy / SM / bandwidth analysis.
        NVTX_PUSH("Kernel_Phase3_Main");
#endif
        kernel_phase3<<<dim3(round, round), threads_per_block, 0, stream_main>>>(d_D, r, V_padded);
#ifdef PROFILING
        NVTX_POP();
#endif
    }
#ifdef PROFILING
    NVTX_POP();
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
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    // Global Memory Offset for the Pivot Block (r, r)
    const int b_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);

    // 1. Load Global -> Shared (Each thread loads 4 ints)
    // [OPT] Coalesced global memory loads: threads in a warp read contiguous elements of each row in the 64x64 tile.
    // Access pattern: Top-Left, Top-Right, Bottom-Left, Bottom-Right relative to thread
    sm_self[ty][tx] = d_D[b_start + ty * V_padded + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[b_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
    __syncthreads();

    // 2. Floyd-Warshall Computation within the block

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int r0 = sm_self[ty][k];
        const int r1 = sm_self[ty + HALF_BLOCK][k];
        const int c0 = sm_self[k][tx];
        const int c1 = sm_self[k][tx + HALF_BLOCK];

        sm_self[ty][tx] = min(sm_self[ty][tx], r0 + c0);
        sm_self[ty][tx + HALF_BLOCK] = min(sm_self[ty][tx + HALF_BLOCK], r0 + c1);
        sm_self[ty + HALF_BLOCK][tx] = min(sm_self[ty + HALF_BLOCK][tx], r1 + c0);
        sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = min(sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK], r1 + c1);
        __syncthreads();
    }

    // 3. Write Shared -> Global
    d_D[b_start + ty * V_padded + tx] = sm_self[ty][tx];
    d_D[b_start + ty * V_padded + (tx + HALF_BLOCK)] = sm_self[ty][tx + HALF_BLOCK];
    d_D[b_start + (ty + HALF_BLOCK) * V_padded + tx] = sm_self[ty + HALF_BLOCK][tx];
    d_D[b_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK];
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
        const int r0 = sm_pivot[ty][k];
        const int r1 = sm_pivot[ty + HALF_BLOCK][k];
        const int c0 = sm_self[k][tx];
        const int c1 = sm_self[k][tx + HALF_BLOCK];

        sm_self[ty][tx] = min(sm_self[ty][tx], r0 + c0);
        sm_self[ty][tx + HALF_BLOCK] = min(sm_self[ty][tx + HALF_BLOCK], r0 + c1);
        sm_self[ty + HALF_BLOCK][tx] = min(sm_self[ty + HALF_BLOCK][tx], r1 + c0);
        sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = min(sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK], r1 + c1);
        // __syncthreads();
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
        const int r0 = sm_self[ty][k];
        const int r1 = sm_self[ty + HALF_BLOCK][k];
        const int c0 = sm_pivot[k][tx];
        const int c1 = sm_pivot[k][tx + HALF_BLOCK];

        sm_self[ty][tx] = min(sm_self[ty][tx], r0 + c0);
        sm_self[ty][tx + HALF_BLOCK] = min(sm_self[ty][tx + HALF_BLOCK], r0 + c1);
        sm_self[ty + HALF_BLOCK][tx] = min(sm_self[ty + HALF_BLOCK][tx], r1 + c0);
        sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = min(sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK], r1 + c1);
        // __syncthreads();
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
