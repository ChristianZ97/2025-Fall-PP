// nbody.cu

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* 單一質點 (Body) 的狀態：位置 + 速度 + 質量 */
typedef struct {
    double x, y, z;    /* position */
    double vx, vy, vz; /* velocity */
    double m;          /* mass */
} Body;

/* 物理常數：萬有引力常數 & softening (可調整) */
static const double G = 1;
static const double SOFTENING = 1e-3;

int read_input(const char *filename, int *N, double *total_time, double *dt, int *dump_interval, Body **bodies_out);
FILE *init_traj_file(const char *filename);
void dump_traj_step(FILE *ftraj, int step, double t, int N, const Body *bodies);

/* 計算所有粒子的加速度 */
__global__ void compute_acceleration(const Body *device_bodies, int N, double *device_ax, double *device_ay, double *device_az) {

    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= N || j >= N || i == j) return;

    double xi = device_bodies[i].x;
    double yi = device_bodies[i].y;
    double zi = device_bodies[i].z;

    double axi = 0.0, ayi = 0.0, azi = 0.0;

    for (int j = 0; j < N; ++j) {

        if (j == i) continue; /* 自己對自己沒有重力 */

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

/* 單一步 Leapfrog / Velocity-Verlet 積分：
   事先假設 ax,ay,az 內含 a(t)，執行完後：
   - bodies 變成時間 t+dt 的位置 / 速度
   - ax,ay,az 更新成 a(t+dt)，供下一步使用
*/
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

    /* 3) 用新位置 x(t+1) 計算 a(t+1) */

    Body *device_bodies = NULL;
    double *device_ax = NULL;
    double *device_ay = NULL;
    double *device_az = NULL;

    cudaMalloc(&device_bodies, N * sizeof(Body));
    cudaMalloc(&device_ax, N * sizeof(double));
    cudaMalloc(&device_ay, N * sizeof(double));
    cudaMalloc(&device_az, N * sizeof(double));

    cudaMemcpy(device_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ax, ax, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ay, ay, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_az, az, N * sizeof(double), cudaMemcpyHostToDevice);

    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    compute_acceleration<<<blocks_per_grid, threads_per_block>>>(device_bodies, N, device_ax, device_ay, device_az);

    cudaMemcpy(ax, device_ax, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ay, device_ay, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(az, device_az, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_bodies);
    cudaFree(device_ax);
    cudaFree(device_ay);
    cudaFree(device_az);

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

    int steps = (int)(total_time / dt);
    if (steps <= 0) {
        fprintf(stderr, "Error: total_time / dt <= 0\n");
        free(bodies);
        return 1;
    }

    /* 加速度 buffer：重複使用，避免每步 malloc/free */
    double *ax_buf = (double *)malloc(N * sizeof(double));
    double *ay_buf = (double *)malloc(N * sizeof(double));
    double *az_buf = (double *)malloc(N * sizeof(double));
    if (!ax_buf || !ay_buf || !az_buf) {
        fprintf(stderr, "Error: malloc failed for acceleration buffers\n");
        free(bodies);
        free(ax_buf);
        free(ay_buf);
        free(az_buf);
        return 1;
    }

    FILE *ftraj = init_traj_file(argv[2]);
    if (!ftraj) {
        free(bodies);
        free(ax_buf);
        free(ay_buf);
        free(az_buf);
        return 1;
    }

    printf("Running simulation: N=%d, steps=%d, dump_interval=%d\n", N, steps, dump_interval);

    double t = 0.0;

    /* 先計算 t=0 時刻的加速度 a(0)，供第一步 step_leapfrog 使用 */

    Body *device_bodies = NULL;
    double *device_ax = NULL;
    double *device_ay = NULL;
    double *device_az = NULL;

    cudaMalloc(&device_bodies, N * sizeof(Body));
    cudaMalloc(&device_ax, N * sizeof(double));
    cudaMalloc(&device_ay, N * sizeof(double));
    cudaMalloc(&device_az, N * sizeof(double));

    cudaMemcpy(device_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ax, ax_buf, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ay, ay_buf, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_az, az_buf, N * sizeof(double), cudaMemcpyHostToDevice);

    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    compute_acceleration<<<blocks_per_grid, threads_per_block>>>(device_bodies, N, device_ax, device_ay, device_az);

    cudaMemcpy(ax_buf, device_ax, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ay_buf, device_ay, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(az_buf, device_az, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_bodies);
    cudaFree(device_ax);
    cudaFree(device_ay);
    cudaFree(device_az);

    /* 輸出初始狀態 (step 0) */
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

/* 讀取輸入檔：
   格式：
   N
   total_time dt dump_interval
   x y z vx vy vz m   (共 N 行)
*/
int read_input(const char *filename, int *N, double *total_time, double *dt, int *dump_interval, Body **bodies_out) {
    FILE *fin = fopen(filename, "r");
    if (!fin) {
        fprintf(stderr, "Error: cannot open input file %s\n", filename);
        return 0;
    }

    if (fscanf(fin, "%d", N) != 1 || *N <= 0) {
        fprintf(stderr, "Error: invalid N in input file\n");
        fclose(fin);
        return 0;
    }
    if (fscanf(fin, "%lf %lf %d", total_time, dt, dump_interval) != 3 || *total_time <= 0.0 || *dt <= 0.0 || *dump_interval <= 0) {
        fprintf(stderr, "Error: invalid total_time / dt / dump_interval\n");
        fclose(fin);
        return 0;
    }

    Body *bodies = (Body *)malloc((*N) * sizeof(Body));
    if (!bodies) {
        fprintf(stderr, "Error: malloc failed for %d bodies\n", *N);
        fclose(fin);
        return 0;
    }

    for (int i = 0; i < *N; ++i) {
        if (fscanf(fin, "%lf %lf %lf %lf %lf %lf %lf", &bodies[i].x, &bodies[i].y, &bodies[i].z, &bodies[i].vx, &bodies[i].vy, &bodies[i].vz, &bodies[i].m) != 7) {
            fprintf(stderr, "Error: invalid body line at index %d\n", i);
            free(bodies);
            fclose(fin);
            return 0;
        }
    }

    fclose(fin);
    *bodies_out = bodies;
    return 1;
}

/* 初始化軌跡輸出檔 (CSV) */
FILE *init_traj_file(const char *filename) {
    FILE *ftraj = fopen(filename, "w");
    if (!ftraj) {
        fprintf(stderr, "Error: cannot open trajectory file %s\n", filename);
        return NULL;
    }
    /* 每一列格式：step,t,id,x,y,z,vx,vy,vz,m */
    fprintf(ftraj, "step,t,id,x,y,z,vx,vy,vz,m\n");
    return ftraj;
}

/* 輸出某一個 time step 的所有粒子狀態 */
void dump_traj_step(FILE *ftraj, int step, double t, int N, const Body *bodies) {
    for (int i = 0; i < N; ++i) {
        fprintf(ftraj, "%d,%.5f,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", step, t, i, bodies[i].x, bodies[i].y, bodies[i].z, bodies[i].vx, bodies[i].vy, bodies[i].vz, bodies[i].m);
    }
}
