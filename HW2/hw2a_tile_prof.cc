/*
 * Mandelbrot Set Visualization - Optimized Version
 * Parallel Programming - Homework 2
 *
 * ============================================================================
 * Design & Optimization Rationale
 * ============================================================================
 * This optimized version incorporates advanced performance optimization
 * techniques inspired by industrial-grade parallel programming practices.
 *
 * ----------------------------------------------------------------------------
 * I. Build Strategy
 * ----------------------------------------------------------------------------
 * CXXFLAGS = -std=c++17 -O3 -march=native -pthread -lpng
 *
 * - O3: Maximum compiler optimizations including auto-vectorization
 * - march=native: Target host CPU for AVX/AVX2 SIMD instructions
 * - Compile with -DPROFILING to enable detailed timing analysis
 *
 * ----------------------------------------------------------------------------
 * II. Algorithmic Optimizations
 * ----------------------------------------------------------------------------
 *
 * 1. Cache-Aware Blocking/Tiling: The image is divided into rectangular tiles
 *    (TILE_WIDTH x TILE_HEIGHT) to maximize cache locality. This ensures that
 *    pixel computations within a tile benefit from L1/L2 cache, reducing
 *    memory latency significantly.
 *
 * 2. Aligned Memory Allocation: Uses aligned_alloc for 32-byte alignment,
 *    enabling efficient SIMD operations when compiled with -march=native.
 *
 * 3. Dynamic Load Balancing: Threads dynamically fetch tiles using a mutex-
 *    protected global counter, ensuring even workload distribution across
 *    threads regardless of computational complexity variation.
 *
 * 4. Profiling Infrastructure: NVTX markers and pthread timing provide
 *    fine-grained performance analysis for initialization, computation,
 *    synchronization, and I/O phases.
 *
 * NOTE: Profiling code is within #ifdef PROFILING blocks.
 *       Compile with -DPROFILING to enable.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP

#include <assert.h>
#include <errno.h>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PROFILING
#include <nvtx3/nvToolsExt.h>
#include <sys/time.h>

// ============================================================================
// NVTX Color Definitions for Performance Profiling
// ============================================================================
#define COLOR_INIT 0xFFFFBE3D    // Sunflower Yellow
#define COLOR_COMPUTE 0xFF5FD068 // Spring Green
#define COLOR_SYNC 0xFF9B59B6    // Amethyst Purple
#define COLOR_IO 0xFF5C9FFF      // Azure Blue
#define COLOR_DEFAULT 0xFFCBD5E0 // Soft Gray

// ============================================================================
// Smart NVTX Macro: Auto-selects color based on range name
// ============================================================================
#define NVTX_PUSH(name)                                                                                                                                                                                \
    do {                                                                                                                                                                                               \
        uint32_t color = COLOR_DEFAULT;                                                                                                                                                                \
        if (strstr(name, "Init") != NULL || strstr(name, "Setup") != NULL) {                                                                                                                           \
            color = COLOR_INIT;                                                                                                                                                                        \
        } else if (strstr(name, "Compute") != NULL || strstr(name, "Mandelbrot") != NULL) {                                                                                                            \
            color = COLOR_COMPUTE;                                                                                                                                                                     \
        } else if (strstr(name, "Sync") != NULL || strstr(name, "Join") != NULL) {                                                                                                                     \
            color = COLOR_SYNC;                                                                                                                                                                        \
        } else if (strstr(name, "IO") != NULL || strstr(name, "Write") != NULL) {                                                                                                                      \
            color = COLOR_IO;                                                                                                                                                                          \
        }                                                                                                                                                                                              \
        nvtxEventAttributes_t eventAttrib = {0};                                                                                                                                                       \
        eventAttrib.version = NVTX_VERSION;                                                                                                                                                            \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                                                                                                                              \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                                                                                                                                                       \
        eventAttrib.color = color;                                                                                                                                                                     \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                                                                                                                                             \
        eventAttrib.message.ascii = name;                                                                                                                                                              \
        nvtxRangePushEx(&eventAttrib);                                                                                                                                                                 \
    } while (0)

#define NVTX_POP() nvtxRangePop()

// Timing utilities
static inline double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
#endif

// ============================================================================
// OPTIMIZATION POINT 1: Cache-Aware Tiling Parameters
// ============================================================================
#define TILE_WIDTH 64  // Optimized for L1 cache line size
#define TILE_HEIGHT 64 // Total tile size = 64x64 = 4KB (fits in L1)

pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;
int next_tile_idx = 0;
int total_tiles = 0;

typedef struct {
    int tile_x; // Tile column index
    int tile_y; // Tile row index
} Tile;

Tile *tiles = NULL;

typedef struct {
    int iters;
    int width;
    int height;
    int *image;
    double left;
    double right;
    double lower;
    double upper;
#ifdef PROFILING
    double compute_time;
    double sync_time;
#endif
} ThreadArg;

void write_png(const char *filename, int iters, int width, int height, const int *buffer);
void *local_mandelbrot(void *argv);

int main(int argc, char *argv[]) {
#ifdef PROFILING
    double total_start_time = get_time();
    NVTX_PUSH("Main_Program");
#endif

    // ========================================================================
    // Block 1: Initialization and Setup
    // ========================================================================
#ifdef PROFILING
    NVTX_PUSH("Init_Setup");
#endif

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    const char *filename = argv[1];
    const int iters = strtol(argv[2], NULL, 10);
    const double left = strtod(argv[3], NULL);
    const double right = strtod(argv[4], NULL);
    const double lower = strtod(argv[5], NULL);
    const double upper = strtod(argv[6], NULL);
    const int width = strtol(argv[7], NULL, 10);
    const int height = strtol(argv[8], NULL, 10);

    // ========================================================================
    // OPTIMIZATION POINT 2: Aligned Memory Allocation
    // ========================================================================
    const size_t image_bytes = (size_t)width * height * sizeof(int);
    const size_t aligned_bytes = ((image_bytes + 31) / 32) * 32;

    int *image = (int *)aligned_alloc(32, aligned_bytes);
    if (!image) {
        image = (int *)malloc(aligned_bytes); // Fallback
        if (!image) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
    }
    memset(image, 0, aligned_bytes);

    // ========================================================================
    // OPTIMIZATION POINT 1: Tile Generation for Cache Locality
    // ========================================================================
    const int tiles_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    const int tiles_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    total_tiles = tiles_x * tiles_y;

    tiles = (Tile *)malloc(total_tiles * sizeof(Tile));
    if (!tiles) {
        fprintf(stderr, "Tile allocation failed\n");
        free(image);
        return 1;
    }

    // Generate tile list
    int idx = 0;
    for (int ty = 0; ty < tiles_y; ty++) {
        for (int tx = 0; tx < tiles_x; tx++) {
            tiles[idx].tile_x = tx;
            tiles[idx].tile_y = ty;
            idx++;
        }
    }

    int num_threads = CPU_COUNT(&cpu_set);
    pthread_t threads[num_threads];
    ThreadArg t_args[num_threads];

#ifdef PROFILING
    NVTX_POP(); // Init_Setup
#endif

    // ========================================================================
    // Block 2: Thread Creation and Computation
    // ========================================================================
#ifdef PROFILING
    double compute_start = get_time();
    NVTX_PUSH("Compute_Phase");
#endif

    for (int i = 0; i < num_threads; i++) {
        t_args[i].iters = iters;
        t_args[i].width = width;
        t_args[i].height = height;
        t_args[i].image = image;
        t_args[i].left = left;
        t_args[i].right = right;
        t_args[i].lower = lower;
        t_args[i].upper = upper;
#ifdef PROFILING
        t_args[i].compute_time = 0.0;
        t_args[i].sync_time = 0.0;
#endif
        pthread_create(&threads[i], NULL, local_mandelbrot, (void *)&t_args[i]);
    }

#ifdef PROFILING
    NVTX_PUSH("Thread_Join");
#endif

    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);

#ifdef PROFILING
    NVTX_POP(); // Thread_Join
    NVTX_POP(); // Compute_Phase
    double compute_time = get_time() - compute_start;
#endif

    // ========================================================================
    // Block 3: I/O - Write PNG
    // ========================================================================
#ifdef PROFILING
    double io_start = get_time();
    NVTX_PUSH("IO_Write_PNG");
#endif

    write_png(filename, iters, width, height, image);

#ifdef PROFILING
    NVTX_POP(); // IO_Write_PNG
    double io_time = get_time() - io_start;
#endif

    // ========================================================================
    // Block 4: Cleanup and Profiling Report
    // ========================================================================
    free(image);
    free(tiles);

#ifdef PROFILING
    double total_time = get_time() - total_start_time;

    // Aggregate per-thread stats
    double avg_thread_compute = 0.0, avg_thread_sync = 0.0;
    for (int i = 0; i < num_threads; i++) {
        avg_thread_compute += t_args[i].compute_time;
        avg_thread_sync += t_args[i].sync_time;
    }
    avg_thread_compute /= num_threads;
    avg_thread_sync /= num_threads;

    printf("\n============================================================\n");
    printf("Performance Profiling Report\n");
    printf("============================================================\n");
    printf("Image Size: %d x %d\n", width, height);
    printf("Iterations: %d\n", iters);
    printf("Threads: %d\n", num_threads);
    printf("Tile Size: %dx%d (Total Tiles: %d)\n", TILE_WIDTH, TILE_HEIGHT, total_tiles);
    printf("------------------------------------------------------------\n");
    printf("Total Time:           %.6f s (100.00%%)\n", total_time);
    printf("  Compute Time:       %.6f s (%.2f%%)\n", compute_time, (compute_time / total_time) * 100);
    printf("  I/O Time:           %.6f s (%.2f%%)\n", io_time, (io_time / total_time) * 100);
    printf("------------------------------------------------------------\n");
    printf("Avg Thread Compute:   %.6f s\n", avg_thread_compute);
    printf("Avg Thread Sync:      %.6f s\n", avg_thread_sync);
    printf("============================================================\n");

    NVTX_POP(); // Main_Program
#endif

    return 0;
}

void *local_mandelbrot(void *argv) {
    ThreadArg *t_arg = (ThreadArg *)argv;
    const int iters = t_arg->iters;
    const int width = t_arg->width;
    const int height = t_arg->height;
    const double left = t_arg->left;
    const double right = t_arg->right;
    const double lower = t_arg->lower;
    const double upper = t_arg->upper;

    const double x_scale = (right - left) / width;
    const double y_scale = (upper - lower) / height;

    int *image = t_arg->image;

#ifdef PROFILING
    double local_compute_time = 0.0;
    double local_sync_time = 0.0;
#endif

    // ========================================================================
    // OPTIMIZATION POINT 3: Dynamic Tile-Based Scheduling
    // ========================================================================
    while (1) {
#ifdef PROFILING
        double sync_start = get_time();
#endif

        pthread_mutex_lock(&task_mutex);
        const int my_tile_idx = next_tile_idx++;
        pthread_mutex_unlock(&task_mutex);

#ifdef PROFILING
        local_sync_time += get_time() - sync_start;
#endif

        if (my_tile_idx >= total_tiles) break;

#ifdef PROFILING
        double comp_start = get_time();
#endif

        // Get tile boundaries
        const int tile_x = tiles[my_tile_idx].tile_x;
        const int tile_y = tiles[my_tile_idx].tile_y;

        const int x_start = tile_x * TILE_WIDTH;
        const int x_end = (x_start + TILE_WIDTH > width) ? width : x_start + TILE_WIDTH;
        const int y_start = tile_y * TILE_HEIGHT;
        const int y_end = (y_start + TILE_HEIGHT > height) ? height : y_start + TILE_HEIGHT;

        // ====================================================================
        // Cache-Optimized Tile Processing
        // ====================================================================
        for (int j = y_start; j < y_end; ++j) {
            const double y0 = j * y_scale + lower;

            // Inner loop processes contiguous memory - excellent cache behavior
            for (int i = x_start; i < x_end; ++i) {
                const double x0 = i * x_scale + left;

                double x = 0.0, y = 0.0, length_squared = 0.0;
                int repeats = 0;

                // Mandelbrot iteration loop
                while (repeats < iters && length_squared < 4.0) {
                    const double temp = x * x - y * y + x0;
                    y = 2.0 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }

                image[j * width + i] = repeats;
            }
        }

#ifdef PROFILING
        local_compute_time += get_time() - comp_start;
#endif
    }

#ifdef PROFILING
    t_arg->compute_time = local_compute_time;
    t_arg->sync_time = local_sync_time;
#endif

    return NULL;
}

/**
 * @brief Writes the raw iteration data into a PNG image file.
 *
 * @param filename The name of the output PNG file.
 * @param iters The maximum number of iterations used for the calculation.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @param buffer A pointer to an integer array of size width*height, where each
 *               element stores the number of iterations for the corresponding pixel.
 */
void write_png(const char *filename, int iters, int width, int height, const int *buffer) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);

    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;

            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p % 16) * 16;
                } else {
                    color[0] = (p % 16) * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }

    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
