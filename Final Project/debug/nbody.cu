// nbody_debug.cu

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* --- CUDA 錯誤檢查 (永遠啟用，遇到錯誤直接終止) --- */
#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

/* --- 除錯訊息印出 (只有定義 DEBUG 時啟用) --- */
#ifdef DEBUG
#define DEBUG_PRINT(fmt, ...)                           \
    do {                                                \
        fprintf(stderr, "[DEBUG] " fmt, ##__VA_ARGS__); \
    } while (0)
#else
#define DEBUG_PRINT(fmt, ...) \
    do {                      \
    } while (0)
#endif

/* 單一質點 (Body) 的狀態 */
typedef struct {
    double x, y, z;    /* position */
    double vx, vy, vz; /* velocity */
    double m;          /* mass */
} Body;

/* 物理常數 */
static const double G = 1;
static const double SOFTENING = 1e-3;

int read_input(const char *filename, int *N, double *total_time, double *dt, int *dump_interval, Body **bodies_out);
FILE *init_traj_file(const char *filename);
void dump_traj_step(FILE *ftraj, int step, double t, int N, const Body *bodies);

/* 計算所有粒子的加速度 */
__global__ void compute_acceleration(const Body *device_bodies, int N, double *device_ax, double *device_ay, double *device_az) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    double xi = device_bodies[i].x;
    double yi = device_bodies[i].y;
    double zi = device_bodies[i].z;

    double axi = 0.0, ayi = 0.0, azi = 0.0;

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        double dx = device_bodies[j].x - xi;
        double dy = device_bodies[j].y - yi;
        double dz = device_bodies[j].z - zi;

        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;

        double force = G * device_bodies[j].m * invDist3;
        axi += dx * force;
        ayi += dy * force;
        azi += dz * force;
    }

    device_ax[i] = axi;
    device_ay[i] = ayi;
    device_az[i] = azi;
}

/* Leapfrog 積分步驟 */
void step_leapfrog(Body *bodies, int N, double dt, double *ax, double *ay, double *az) {
    /* 1) v(t+1/2) = v(t) + 0.5 * a(t) * dt */
    for (int i = 0; i < N; ++i) {
        bodies[i].vx += 0.5 * ax[i] * dt;
        bodies[i].vy += 0.5 * ay[i] * dt;
        bodies[i].vz += 0.5 * az[i] * dt;
    }

    /* 2) x(t+1) = x(t) + v(t+1/2) * dt */
    for (int i = 0; i < N; ++i) {
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }

    /* 3) GPU 計算 a(t+1) */
    Body *device_bodies = NULL;
    double *device_ax = NULL, *device_ay = NULL, *device_az = NULL;

    CUDA_CHECK(cudaMalloc(&device_bodies, N * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&device_ax, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_ay, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_az, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(device_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice));

#ifdef DEBUG
    // Optional: Peek first element to ensure copy correctness
    Body host_peek;
    CUDA_CHECK(cudaMemcpy(&host_peek, device_bodies, sizeof(Body), cudaMemcpyDeviceToHost));
    DEBUG_PRINT("Step Leapfrog: Input Body[0].x = %.5f\n", host_peek.x);
#endif

    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    compute_acceleration<<<blocks_per_grid, threads_per_block>>>(device_bodies, N, device_ax, device_ay, device_az);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ax, device_ax, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay, device_ay, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az, device_az, N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_bodies));
    CUDA_CHECK(cudaFree(device_ax));
    CUDA_CHECK(cudaFree(device_ay));
    CUDA_CHECK(cudaFree(device_az));

    /* 4) v(t+1) = v(t+1/2) + 0.5 * a(t+1) * dt */
    for (int i = 0; i < N; ++i) {
        bodies[i].vx += 0.5 * ax[i] * dt;
        bodies[i].vy += 0.5 * ay[i] * dt;
        bodies[i].vz += 0.5 * az[i] * dt;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <traj_output_csv>\n", argv[0]);
        return 1;
    }

    int N, dump_interval;
    double total_time, dt;
    Body *bodies = NULL;

    if (!read_input(argv[1], &N, &total_time, &dt, &dump_interval, &bodies)) {
        fprintf(stderr, "Error reading input.\n");
        return 1;
    }
    DEBUG_PRINT("Input Read Success: N=%d, total_time=%.2f, dt=%.5f\n", N, total_time, dt);

    int steps = (int)(total_time / dt);
    if (steps <= 0) {
        fprintf(stderr, "Error: total_time / dt <= 0\n");
        free(bodies);
        return 1;
    }

    double *ax_buf = (double *)malloc(N * sizeof(double));
    double *ay_buf = (double *)malloc(N * sizeof(double));
    double *az_buf = (double *)malloc(N * sizeof(double));
    if (!ax_buf || !ay_buf || !az_buf) {
        fprintf(stderr, "Error: malloc failed\n");
        free(bodies);
        free(ax_buf);
        free(ay_buf);
        free(az_buf);
        return 1;
    }

    FILE *ftraj = init_traj_file(argv[2]);
    if (!ftraj) return 1;

    printf("Running simulation: N=%d, steps=%d\n", N, steps);

    double t = 0.0;

    /* --- 初始計算 Step 0 --- */
    DEBUG_PRINT("Starting Initial Kernel Launch...\n");
    Body *device_bodies = NULL;
    double *device_ax = NULL, *device_ay = NULL, *device_az = NULL;

    CUDA_CHECK(cudaMalloc(&device_bodies, N * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&device_ax, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_ay, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_az, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(device_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    DEBUG_PRINT("Launching Kernel with Grid=(%d), Block=(%d)\n", blocks_per_grid, threads_per_block);
    compute_acceleration<<<blocks_per_grid, threads_per_block>>>(device_bodies, N, device_ax, device_ay, device_az);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG_PRINT("Kernel Finished & Synced.\n");

    CUDA_CHECK(cudaMemcpy(ax_buf, device_ax, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay_buf, device_ay, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az_buf, device_az, N * sizeof(double), cudaMemcpyDeviceToHost));

    DEBUG_PRINT("Initial Acceleration (First Body): ax=%.5f, ay=%.5f, az=%.5f\n", ax_buf[0], ay_buf[0], az_buf[0]);

    CUDA_CHECK(cudaFree(device_bodies));
    CUDA_CHECK(cudaFree(device_ax));
    CUDA_CHECK(cudaFree(device_ay));
    CUDA_CHECK(cudaFree(device_az));

    /* 輸出 Step 0 */
    dump_traj_step(ftraj, 0, t, N, bodies);

    for (int s = 1; s <= steps; ++s) {
        step_leapfrog(bodies, N, dt, ax_buf, ay_buf, az_buf);
        t += dt;

        if (s % dump_interval == 0) {
            dump_traj_step(ftraj, s, t, N, bodies);
        }
    }

    fclose(ftraj);
    free(bodies);
    free(ax_buf);
    free(ay_buf);
    free(az_buf);
    printf("Done. Output written to %s\n", argv[2]);
    return 0;
}

int read_input(const char *filename, int *N, double *total_time, double *dt, int *dump_interval, Body **bodies_out) {
    FILE *fin = fopen(filename, "r");
    if (!fin) return 0;
    if (fscanf(fin, "%d", N) != 1) return 0;
    if (fscanf(fin, "%lf %lf %d", total_time, dt, dump_interval) != 3) return 0;
    *bodies_out = (Body *)malloc((*N) * sizeof(Body));
    for (int i = 0; i < *N; ++i) {
        if (fscanf(fin, "%lf %lf %lf %lf %lf %lf %lf", &(*bodies_out)[i].x, &(*bodies_out)[i].y, &(*bodies_out)[i].z, &(*bodies_out)[i].vx, &(*bodies_out)[i].vy, &(*bodies_out)[i].vz,
                   &(*bodies_out)[i].m) != 7)
            return 0;
    }
    fclose(fin);
    return 1;
}

FILE *init_traj_file(const char *filename) {
    FILE *ftraj = fopen(filename, "w");
    if (ftraj) fprintf(ftraj, "step,t,id,x,y,z,vx,vy,vz,m\n");
    return ftraj;
}

void dump_traj_step(FILE *ftraj, int step, double t, int N, const Body *bodies) {
    for (int i = 0; i < N; ++i) {
        fprintf(ftraj, "%d,%.5f,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", step, t, i, bodies[i].x, bodies[i].y, bodies[i].z, bodies[i].vx, bodies[i].vy, bodies[i].vz, bodies[i].m);
    }
}
