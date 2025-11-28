#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// #define DEV_NO 0
// cudaDeviceProp prop;

#define BLOCKING_FACTOR 256
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 32

/*
BLOCK_DIM_X * BLOCK_DIM_Y <= 1024 (kernel launch limit)
(BLOCK_DIM_X + BLOCK_DIM_Y) * BLOCKING_FACTOR * 4 <= 49152（shared mem limit）
*/

static int *Dist;
static int *d_Dist;
static int n, m;

cudaStream_t streams[4];
__constant__ const int INF = ((1 << 30) - 1);

void input(char *inFileName);
void output(char *outFileName);

void block_FW();
void cal(const int current_round, const int block_start_row, const int block_start_col, const int block_height, const int block_width, const int dependent, const int stream_id);

__global__ void kernel_single(int *d_Dist, const int n, const int start_row, const int start_col, const int end_row, const int end_col, const int k);
__global__ void kernel_multiple(int *d_Dist, const int n, const int start_row, const int start_col, const int end_row, const int end_col, const int start_k, const int end_k);

int main(int argc, char *argv[]) {

    input(argv[1]);

    for (int i = 0; i < 4; ++i)
        cudaStreamCreate(&streams[i]);

    cudaMalloc(&d_Dist, n * n * sizeof(int));
    cudaMemcpy(d_Dist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    block_FW();

    cudaMemcpy(Dist, d_Dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);

    for (int i = 0; i < 4; ++i)
        cudaStreamDestroy(streams[i]);

    output(argv[2]);
    free(Dist);
    return 0;
}

void block_FW() {

    const int round = (n + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR;
    for (int r = 0; r < round; ++r) {

        /* Phase 1: Self-Dependent Block */
        cal(r, r, r, 1, 1, 1, 0);
        cudaDeviceSynchronize();

        /* Phase 2: Pivot Row & Column Blocks */
        cal(r, r, 0, 1, r, 0, 0);                  // Stream 0: Pivot Row (Left)
        cal(r, r, r + 1, 1, round - r - 1, 0, 1);  // Stream 1: Pivot Row (Right)
        cal(r, 0, r, r, 1, 0, 2);                  // Stream 2: Pivot Col (Top)
        cal(r, r + 1, r, round - r - 1, 1, 0, 3);  // Stream 3: Pivot Col (Bottom)
        cudaDeviceSynchronize();

        /* Phase 3: Remaining Independent Blocks */
        cal(r, 0, 0, r, r, 0, 0);                                  // Stream 0: Top-Left
        cal(r, 0, r + 1, r, round - r - 1, 0, 1);                  // Stream 1: Top-Right
        cal(r, r + 1, 0, round - r - 1, r, 0, 2);                  // Stream 2: Bottom-Left
        cal(r, r + 1, r + 1, round - r - 1, round - r - 1, 0, 3);  // Stream 3: Bottom-Right
        cudaDeviceSynchronize();
    }
}

void cal(const int current_round, const int block_start_row, const int block_start_col, const int block_height, const int block_width, const int dependent, const int stream_id) {

    const int start_row = block_start_row * BLOCKING_FACTOR;
    const int start_col = block_start_col * BLOCKING_FACTOR;

    const int end_row = min((block_start_row + block_height) * BLOCKING_FACTOR, n);
    const int end_col = min((block_start_col + block_width) * BLOCKING_FACTOR, n);

    const int start_k = current_round * BLOCKING_FACTOR;
    const int end_k = min((current_round + 1) * BLOCKING_FACTOR, n);

    dim3 blocks_per_grid((end_col - start_col + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (end_row - start_row + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    dim3 threads_per_block(BLOCK_DIM_X, BLOCK_DIM_Y);

    if (dependent)
        for (int k = start_k; k < end_k; ++k)
            kernel_single<<<blocks_per_grid, threads_per_block, 0, streams[stream_id]>>>(d_Dist, n, start_row, start_col, end_row, end_col, k);
    else
        kernel_multiple<<<blocks_per_grid, threads_per_block, 0, streams[stream_id]>>>(d_Dist, n, start_row, start_col, end_row, end_col, start_k, end_k);
}

__global__ void kernel_single(int *d_Dist, const int n, const int start_row, const int start_col, const int end_row, const int end_col, const int k) {

    const int col = blockDim.x * blockIdx.x + threadIdx.x + start_col;
    const int row = blockDim.y * blockIdx.y + threadIdx.y + start_row;

    if ((row >= end_row) || (col >= end_col)) return;

    d_Dist[row * n + col] = min(d_Dist[row * n + col], d_Dist[row * n + k] + d_Dist[k * n + col]);
}

__global__ void kernel_multiple(int *d_Dist, const int n, const int start_row, const int start_col, const int end_row, const int end_col, const int start_k, const int end_k) {

    __shared__ int sm_ik[BLOCK_DIM_Y][BLOCKING_FACTOR];
    __shared__ int sm_kj[BLOCKING_FACTOR][BLOCK_DIM_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockDim.x * blockIdx.x + tx + start_col;
    const int row = blockDim.y * blockIdx.y + ty + start_row;

    for (int tk = tx; tk < BLOCKING_FACTOR; tk += BLOCK_DIM_X)
        sm_ik[ty][tk] = ((start_k + tk) < n && row < n) ? d_Dist[row * n + (start_k + tk)] : INF;

    for (int tk = ty; tk < BLOCKING_FACTOR; tk += BLOCK_DIM_Y)
        sm_kj[tk][tx] = (col < n && (start_k + tk) < n) ? d_Dist[(start_k + tk) * n + col] : INF;

    __syncthreads();

    if ((row >= end_row) || (col >= end_col)) return;

    int dist_ij = d_Dist[row * n + col];

#pragma unroll

    for (int k = 0; k < BLOCKING_FACTOR; ++k)
        dist_ij = min(dist_ij, sm_ik[ty][k] + sm_kj[k][tx]);

    d_Dist[row * n + col] = dist_ij;
}

void input(char *infile) {

    FILE *file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    Dist = (int *)malloc(n * n * sizeof(int));

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
