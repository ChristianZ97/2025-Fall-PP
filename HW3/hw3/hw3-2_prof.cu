#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

const int INF = ((1 << 30) - 1);
static int *Dist;
static int *d_Dist;
static int n, m;

void input(char *inFileName);
void output(char *outFileName);
int ceil(int a, int b);

void block_FW(const int B);
void cal(const int B, const int round, const int block_start_x, const int block_start_y, const int block_width, const int block_height);
__global__ void kernel(
    int *d_Dist, const int n, const int B, const int round, const int element_start_x, const int element_start_y, const int element_end_x, const int element_end_y);

int main(int argc, char *argv[]) {

    input(argv[1]);
    const int B = 512;

    cudaMalloc(&d_Dist, (size_t)n * (size_t)n * sizeof(int));

    // === PROFILING EVENTS ===
    cudaEvent_t event_h2d_start, event_h2d_stop;
    cudaEvent_t event_kernel_start, event_kernel_stop;
    cudaEvent_t event_d2h_start, event_d2h_stop;
    cudaEventCreate(&event_h2d_start);
    cudaEventCreate(&event_h2d_stop);
    cudaEventCreate(&event_kernel_start);
    cudaEventCreate(&event_kernel_stop);
    cudaEventCreate(&event_d2h_start);
    cudaEventCreate(&event_d2h_stop);

    // === H2D TRANSFER ===
    cudaEventRecord(event_h2d_start);
    cudaMemcpy(d_Dist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(event_h2d_stop);
    cudaEventSynchronize(event_h2d_stop);

    // === KERNEL EXECUTION ===
    cudaEventRecord(event_kernel_start);
    block_FW(B);
    cudaEventRecord(event_kernel_stop);
    cudaEventSynchronize(event_kernel_stop);

    // === D2H TRANSFER ===
    cudaEventRecord(event_d2h_start);
    cudaMemcpy(Dist, d_Dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(event_d2h_stop);
    cudaEventSynchronize(event_d2h_stop);

    cudaFree(d_Dist);

    // === PROFILING RESULTS ===
    float time_h2d, time_kernel, time_d2h;
    cudaEventElapsedTime(&time_h2d, event_h2d_start, event_h2d_stop);
    cudaEventElapsedTime(&time_kernel, event_kernel_start, event_kernel_stop);
    cudaEventElapsedTime(&time_d2h, event_d2h_start, event_d2h_stop);

    float total_time = time_h2d + time_kernel + time_d2h;
    size_t total_bytes = (size_t)n * (size_t)n * sizeof(int);

    fprintf(stderr, "\n=== PROFILING RESULTS ===\n");
    fprintf(stderr, "n = %d, m = %d\n", n, m);
    fprintf(stderr, "Matrix size: %d x %d, total bytes = %zu\n", n, n, total_bytes);
    fprintf(stderr, "\nTiming:\n");
    fprintf(stderr, "  H2D transfer:  %.3f ms (%.2f GB/s)\n",
            time_h2d,
            (total_bytes / 1e9) / (time_h2d / 1000.0));
    fprintf(stderr, "  Kernel exec:   %.3f ms\n", time_kernel);
    fprintf(stderr, "  D2H transfer:  %.3f ms (%.2f GB/s)\n",
            time_d2h,
            (total_bytes / 1e9) / (time_d2h / 1000.0));
    fprintf(stderr, "  Total:         %.3f ms\n\n", total_time);

    // 以理論 n^3 次更新當作計算量估計
    double logical_updates = (double)n * (double)n * (double)n;
    fprintf(stderr, "Kernel performance:\n");
    fprintf(stderr, "  Approx. FW updates: %.3e\n", logical_updates);
    fprintf(stderr, "  GUpdates/s (n^3 / time): %.2f\n",
            logical_updates / (time_kernel * 1e6));

    // 簡單判斷瓶頸
    if (time_h2d > time_kernel && time_h2d > time_d2h) {
        fprintf(stderr, "\nBOTTLENECK: H2D Transfer (%.1f%% of total)\n", 100 * time_h2d / total_time);
    } else if (time_d2h > time_kernel && time_d2h > time_h2d) {
        fprintf(stderr, "\nBOTTLENECK: D2H Transfer (%.1f%% of total)\n", 100 * time_d2h / total_time);
    } else {
        fprintf(stderr, "\nBOTTLENECK: Kernel execution (%.1f%% of total)\n", 100 * time_kernel / total_time);
    }

    cudaEventDestroy(event_h2d_start);
    cudaEventDestroy(event_h2d_stop);
    cudaEventDestroy(event_kernel_start);
    cudaEventDestroy(event_kernel_stop);
    cudaEventDestroy(event_d2h_start);
    cudaEventDestroy(event_d2h_stop);

    output(argv[2]);
    free(Dist);
    return 0;
}

void block_FW(const int B) {

    const int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        cal(B, r, r, 0, r, 1);
        cal(B, r, r, r + 1, round - r - 1, 1);
        cal(B, r, 0, r, 1, r);
        cal(B, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(const int B, const int round, const int block_start_x, const int block_start_y, const int block_width, const int block_height) {

    const int block_end_x = block_start_x + block_height;
    const int block_end_y = block_start_y + block_width;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {

            const int element_start_x = b_i * B;
            const int element_start_y = b_j * B;

            const int element_end_x = min(element_start_x + B, n);
            const int element_end_y = min(element_start_y + B, n);

            dim3 blocks_per_grid((element_end_x - element_start_x + 7) / 8, (element_end_y - element_start_y + 7) / 8);
            dim3 threads_per_block(8, 8);
            kernel<<<blocks_per_grid, threads_per_block>>>(d_Dist, n, B, round, element_start_x, element_start_y, element_end_x, element_end_y);
        }
    }
}

__global__ void kernel(
    int *d_Dist, const int n, const int B, const int round, const int element_start_x, const int element_start_y, const int element_end_x, const int element_end_y) {

    const int local_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int local_y = blockDim.y * blockIdx.y + threadIdx.y;

    const int x = element_start_x + local_x;
    const int y = element_start_y + local_y;

    if (x >= element_end_x || y >= element_end_y) return;

    for (int k = round * B; k < (round + 1) * B && k < n; ++k)
        d_Dist[x * n + y] = min(d_Dist[x * n + y], d_Dist[x * n + k] + d_Dist[k * n + y]);
}

void input(char *infile) {

    FILE *file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    Dist = (int *)malloc((size_t)n * n * sizeof(int));
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

int ceil(int a, int b) {

    return (a + b - 1) / b;
}
