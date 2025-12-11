#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// 錯誤檢查巨集
#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = (call);                                                                       \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

#define G 1.0
#define SOFTENING 1e-3
#define BLOCK_SIZE 256

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

/* -------------------------------------------------------- */
/* GPU Kernels (Safe Version)                               */
/* -------------------------------------------------------- */

// Kernel 1: 計算力 + 更新速度 (Kick)
// v(t+0.5) = v(t-0.5) + a(t) * dt
__global__ void compute_force_and_kick_kernel(int N,
                                              double dt,
                                              const double *__restrict__ x,
                                              const double *__restrict__ y,
                                              const double *__restrict__ z,
                                              const double *__restrict__ m,
                                              double *__restrict__ vx,
                                              double *__restrict__ vy,
                                              double *__restrict__ vz) {
    // 全域執行緒索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 區塊內索引
    int tid = threadIdx.x;

    // 宣告 Shared Memory
    __shared__ double sh_x[BLOCK_SIZE];
    __shared__ double sh_y[BLOCK_SIZE];
    __shared__ double sh_z[BLOCK_SIZE];
    __shared__ double sh_m[BLOCK_SIZE];

    double my_x, my_y, my_z;
    double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;

    // [Safety] 只有有效粒子才載入自己的位置
    // 注意：不能在這裡 return，否則會導致後面的 __syncthreads() 死鎖
    if (i < N) {
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
    }

    // [Loop] 遍歷所有 Tile
    for (int tile_start = 0; tile_start < N; tile_start += BLOCK_SIZE) {

        // 1. 載入資料到 Shared Memory
        // 每個執行緒負責載入一個位置，即使該執行緒本身 (i) 無效，
        // 只要它對應的載入來源 (src_idx) 有效，就必須幫忙載入。
        int src_idx = tile_start + tid;

        if (src_idx < N) {
            sh_x[tid] = x[src_idx];
            sh_y[tid] = y[src_idx];
            sh_z[tid] = z[src_idx];
            sh_m[tid] = m[src_idx];
        }

        // [Sync] 等待所有資料載入完成
        // 重要：所有執行緒 (包含 i >= N 的) 都必須執行到這裡
        __syncthreads();

        // 2. 計算力
        // 只有有效粒子才需要計算受力
        if (i < N) {
            // 計算這個 tile 內實際有多少有效粒子
            int tileSize = BLOCK_SIZE;
            if (tile_start + BLOCK_SIZE > N) {
                tileSize = N - tile_start;
            }

            // 標準迴圈計算 (無 Loop Unrolling，確保正確性)
            for (int j = 0; j < tileSize; ++j) {
                double dx = sh_x[j] - my_x;
                double dy = sh_y[j] - my_y;
                double dz = sh_z[j] - my_z;

                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;

                // 使用標準 1.0/sqrt 以確保數值精度與 CPU/v0 一致
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                double f = G * sh_m[j] * invDist3;

                acc_x += dx * f;
                acc_y += dy * f;
                acc_z += dz * f;
            }
        }

        // [Sync] 等待計算完成，才能進入下一輪載入
        __syncthreads();
    }

    // 3. 更新速度 (Kick)
    if (i < N) {
        vx[i] += acc_x * dt;
        vy[i] += acc_y * dt;
        vz[i] += acc_z * dt;
    }
}

// Kernel 2: 更新位置 (Drift)
// x(t+1) = x(t) + v(t+0.5) * dt
__global__ void drift_kernel(
    int N, double dt, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z, const double *__restrict__ vx, const double *__restrict__ vy, const double *__restrict__ vz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Drift Kernel 不涉及 Shared Memory 或同步，可以提早 return
    if (i >= N) return;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

/* -------------------------------------------------------- */
/* Host Helpers                                             */
/* -------------------------------------------------------- */

int read_input_soa(
    const char *filename, int *N, double *total_time, double *dt, int *dump_interval, double **h_x, double **h_y, double **h_z, double **h_vx, double **h_vy, double **h_vz, double **h_m) {
    FILE *fin = fopen(filename, "r");
    if (!fin) return 0;
    if (fscanf(fin, "%d", N) != 1) return 0;
    if (fscanf(fin, "%lf %lf %d", total_time, dt, dump_interval) != 3) return 0;

    *h_x = (double *)malloc(*N * sizeof(double));
    *h_y = (double *)malloc(*N * sizeof(double));
    *h_z = (double *)malloc(*N * sizeof(double));
    *h_vx = (double *)malloc(*N * sizeof(double));
    *h_vy = (double *)malloc(*N * sizeof(double));
    *h_vz = (double *)malloc(*N * sizeof(double));
    *h_m = (double *)malloc(*N * sizeof(double));

    for (int i = 0; i < *N; ++i) {
        if (fscanf(fin, "%lf %lf %lf %lf %lf %lf %lf", &(*h_x)[i], &(*h_y)[i], &(*h_z)[i], &(*h_vx)[i], &(*h_vy)[i], &(*h_vz)[i], &(*h_m)[i]) != 7) return 0;
    }
    fclose(fin);
    return 1;
}

void dump_traj_soa(const char *filename, int step, double t, int N, const char *mode, double *x, double *y, double *z, double *vx, double *vy, double *vz, double *m) {
    FILE *ftraj = fopen(filename, mode);
    if (!ftraj) return;
    if (step == 0) fprintf(ftraj, "step,t,id,x,y,z,vx,vy,vz,m\n");
    for (int i = 0; i < N; ++i) {
        fprintf(ftraj, "%d,%.5f,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", step, t, i, x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i]);
    }
    fclose(ftraj);
}

/* -------------------------------------------------------- */
/* Main                                                     */
/* -------------------------------------------------------- */

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    double start = getTimeStamp();

    int N, dump_interval;
    double total_time, dt;
    double *h_x, *h_y, *h_z, *h_vx, *h_vy, *h_vz, *h_m;

    if (!read_input_soa(argv[1], &N, &total_time, &dt, &dump_interval, &h_x, &h_y, &h_z, &h_vx, &h_vy, &h_vz, &h_m)) {
        printf("Error reading input.\n");
        return 1;
    }
    int steps = (int)(total_time / dt);

    double *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_m;
    size_t sz = N * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_x, sz));
    CUDA_CHECK(cudaMalloc(&d_y, sz));
    CUDA_CHECK(cudaMalloc(&d_z, sz));
    CUDA_CHECK(cudaMalloc(&d_vx, sz));
    CUDA_CHECK(cudaMalloc(&d_vy, sz));
    CUDA_CHECK(cudaMalloc(&d_vz, sz));
    CUDA_CHECK(cudaMalloc(&d_m, sz));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, h_m, sz, cudaMemcpyHostToDevice));

    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    // --- 初始化: Half-step Velocity ---
    // v(0.5) = v(0) + 0.5 * a(0) * dt
    compute_force_and_kick_kernel<<<gridSize, blockSize>>>(N, 0.5 * dt, d_x, d_y, d_z, d_m, d_vx, d_vy, d_vz);
    CUDA_CHECK(cudaGetLastError());  // 檢查 kernel 啟動錯誤
    CUDA_CHECK(cudaDeviceSynchronize());

    dump_traj_soa(argv[2], 0, 0.0, N, "w", h_x, h_y, h_z, h_vx, h_vy, h_vz, h_m);

    printf("Running v4 (Safe) simulation: N=%d, steps=%d, grid=%d, block=%d\n", N, steps, gridSize, blockSize);

    double t = 0.0;
    for (int s = 1; s <= steps; ++s) {
        t += dt;

        // 1. Drift: x(t) = x(t-1) + v(t-0.5) * dt
        drift_kernel<<<gridSize, blockSize>>>(N, dt, d_x, d_y, d_z, d_vx, d_vy, d_vz);

        // 2. Kick: v(t+0.5) = v(t-0.5) + a(t) * dt
        compute_force_and_kick_kernel<<<gridSize, blockSize>>>(N, dt, d_x, d_y, d_z, d_m, d_vx, d_vy, d_vz);

        if (s % dump_interval == 0) {
            CUDA_CHECK(cudaMemcpy(h_x, d_x, sz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_y, d_y, sz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_z, d_z, sz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_vx, d_vx, sz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_vy, d_vy, sz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_vz, d_vz, sz, cudaMemcpyDeviceToHost));

            dump_traj_soa(argv[2], s, t, N, "a", h_x, h_y, h_z, h_vx, h_vy, h_vz, h_m);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    double end = getTimeStamp();
    printf("Total Execution Time: %.3f seconds\n", end - start);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_m);
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_vx);
    free(h_vy);
    free(h_vz);
    free(h_m);

    printf("Done.\n");
    return 0;
}
