#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// #define DEV_NO 0
// cudaDeviceProp prop;

#define BLOCKING_FACTOR 512
#define BLOCK_DIM_X 1
#define BLOCK_DIM_Y 512

const int INF = ((1 << 30) - 1);
static int *Dist;
static int *d_Dist;
static int n, m;

void input(char *inFileName);
void output(char *outFileName);

void block_FW();
void cal(const int round, const int block_start_x, const int block_start_y, const int block_width, const int block_height);
__global__ void kernel(
    int *d_Dist, const int n, const int element_start_x, const int element_start_y, const int element_end_x, const int element_end_y, const int k);

int main(int argc, char *argv[]) {

    input(argv[1]);

    cudaMalloc(&d_Dist, n * n * sizeof(int));
    cudaMemcpy(d_Dist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    block_FW();

    cudaMemcpy(Dist, d_Dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);

    output(argv[2]);
    free(Dist);
    return 0;
}

void block_FW() {

    const int round = (n + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR;
    for (int r = 0; r < round; ++r) {
        // printf("%d %d\n", r, round);
        // fflush(stdout);

        /* Phase 1*/
        cal(r, r, r, 1, 1);

        /* Phase 2*/
        cal(r, r, 0, r, 1);
        cal(r, r, r + 1, round - r - 1, 1);
        cal(r, 0, r, 1, r);
        cal(r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        cal(r, 0, 0, r, r);
        cal(r, 0, r + 1, round - r - 1, r);
        cal(r, r + 1, 0, r, round - r - 1);
        cal(r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(const int current_round, const int block_start_x, const int block_start_y, const int block_width, const int block_height) {

    const int block_end_x = block_start_x + block_height;
    const int block_end_y = block_start_y + block_width;

    for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
        for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {

            const int element_start_x = b_i * BLOCKING_FACTOR;
            const int element_start_y = b_j * BLOCKING_FACTOR;

            const int element_end_x = min(element_start_x + BLOCKING_FACTOR, n);
            const int element_end_y = min(element_start_y + BLOCKING_FACTOR, n);

            dim3 blocks_per_grid((element_end_x - element_start_x + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (element_end_y - element_start_y + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
            dim3 threads_per_block(BLOCK_DIM_X, BLOCK_DIM_Y);

            for (int k = current_round * BLOCKING_FACTOR; k < (current_round + 1) * BLOCKING_FACTOR && k < n; ++k)
                kernel<<<blocks_per_grid, threads_per_block>>>(d_Dist, n, element_start_x, element_start_y, element_end_x, element_end_y, k);
        }
    }
}

__global__ void kernel(
    int *d_Dist, const int n, const int element_start_x, const int element_start_y, const int element_end_x, const int element_end_y, const int k) {

    const int local_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int local_y = blockDim.y * blockIdx.y + threadIdx.y;

    const int x = element_start_x + local_x;
    const int y = element_start_y + local_y;

    if ((x >= element_end_x) || (y >= element_end_y)) return;
    d_Dist[x * n + y] = min(d_Dist[x * n + y], d_Dist[x * n + k] + d_Dist[k * n + y]);
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
