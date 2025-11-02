#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cmath>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

__constant__ int mask[MASK_N][MASK_X][MASK_Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};

int read_png(const char *filename, unsigned char **image, unsigned *height, unsigned *width, unsigned *channels) {
    unsigned char sig[8];
    FILE *infile = fopen(filename, "rb");
    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1;

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char *filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels) {
    FILE *fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels;
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {
    int x, y, i, v, u;
    int R, G, B;
    double val[MASK_N * 3] = {0.0};
    int adjustX, adjustY, xBound, yBound;

    /* Hint 6 */
    // parallel job by blockIdx, blockDim, threadIdx

    x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height) return;

    adjustX = (MASK_X % 2) ? 1 : 0;
    adjustY = (MASK_Y % 2) ? 1 : 0;
    xBound = MASK_X / 2;
    yBound = MASK_Y / 2;

    for (i = 0; i < MASK_N; ++i) {

        val[i * 3 + 2] = 0.0;
        val[i * 3 + 1] = 0.0;
        val[i * 3] = 0.0;

        for (v = -yBound; v < yBound + adjustY; ++v) {
            for (u = -xBound; u < xBound + adjustX; ++u) {
                if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
                    R = s[channels * (width * (y + v) + (x + u)) + 2];
                    G = s[channels * (width * (y + v) + (x + u)) + 1];
                    B = s[channels * (width * (y + v) + (x + u)) + 0];
                    val[i * 3 + 2] += R * mask[i][u + xBound][v + yBound];
                    val[i * 3 + 1] += G * mask[i][u + xBound][v + yBound];
                    val[i * 3 + 0] += B * mask[i][u + xBound][v + yBound];
                }
            }
        }
    }

    double totalR = 0.0;
    double totalG = 0.0;
    double totalB = 0.0;
    for (i = 0; i < MASK_N; ++i) {
        totalR += val[i * 3 + 2] * val[i * 3 + 2];
        totalG += val[i * 3 + 1] * val[i * 3 + 1];
        totalB += val[i * 3 + 0] * val[i * 3 + 0];
    }

    totalR = sqrt(totalR) / SCALE;
    totalG = sqrt(totalG) / SCALE;
    totalB = sqrt(totalB) / SCALE;
    const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
    const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
    const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
    t[channels * (width * y + x) + 2] = cR;
    t[channels * (width * y + x) + 1] = cG;
    t[channels * (width * y + x) + 0] = cB;
}

int main(int argc, char *argv[]) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *host_s = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels);
    unsigned char *host_t = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));

    unsigned char *device_src = NULL;
    unsigned char *device_dst = NULL;
    cudaMalloc(&device_src, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&device_dst, height * width * channels * sizeof(unsigned char));

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
    cudaMemcpy(device_src, host_s, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaEventRecord(event_h2d_stop);
    cudaEventSynchronize(event_h2d_stop);

    // === KERNEL EXECUTION ===
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16, 1);
    
    cudaEventRecord(event_kernel_start);
    sobel<<<gridDim, blockDim>>>(device_src, device_dst, height, width, channels);
    cudaEventRecord(event_kernel_stop);
    cudaEventSynchronize(event_kernel_stop);

    // === D2H TRANSFER ===
    cudaEventRecord(event_d2h_start);
    cudaMemcpy(host_t, device_dst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaEventRecord(event_d2h_stop);
    cudaEventSynchronize(event_d2h_stop);

    // === PROFILING RESULTS ===
    float time_h2d, time_kernel, time_d2h;
    cudaEventElapsedTime(&time_h2d, event_h2d_start, event_h2d_stop);
    cudaEventElapsedTime(&time_kernel, event_kernel_start, event_kernel_stop);
    cudaEventElapsedTime(&time_d2h, event_d2h_start, event_d2h_stop);

    float total_time = time_h2d + time_kernel + time_d2h;
    size_t total_bytes = height * width * channels;
    
    // 每個像素 25 次讀 + 3 次寫 = 28 次訪問 (每次 1 byte)
    float theoretical_bytes = (float)total_bytes * 25 * 3;  // input + output
    float bw_utilization = (theoretical_bytes / (1024*1024*1024)) / (time_kernel / 1000.0);

    fprintf(stderr, "\n=== PROFILING RESULTS ===\n");
    fprintf(stderr, "Image: %u x %u, channels=%u\n", width, height, channels);
    fprintf(stderr, "Block: %u x %u, Grid: %u x %u\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    fprintf(stderr, "Total pixels: %.2f M\n", (float)total_bytes / 1e6);
    fprintf(stderr, "\nTiming:\n");
    fprintf(stderr, "  H2D transfer:  %.3f ms (%.2f GB/s)\n", time_h2d, (total_bytes/1e9)/(time_h2d/1000.0));
    fprintf(stderr, "  Kernel exec:   %.3f ms\n", time_kernel);
    fprintf(stderr, "  D2H transfer:  %.3f ms (%.2f GB/s)\n", time_d2h, (total_bytes/1e9)/(time_d2h/1000.0));
    fprintf(stderr, "  Total:         %.3f ms\n\n", total_time);
    
    fprintf(stderr, "Kernel performance:\n");
    fprintf(stderr, "  Est. bandwidth: %.2f GB/s (25 reads/3 writes per pixel)\n", bw_utilization);
    fprintf(stderr, "  GPixel/s: %.2f\n", (float)total_bytes / (time_kernel * 1e6));
    
    // 找出瓶頸
    if (time_h2d > time_kernel) {
        fprintf(stderr, "\n⚠ BOTTLENECK: H2D Transfer (%.1f%% of total)\n", 100*time_h2d/total_time);
    } else if (time_d2h > time_kernel) {
        fprintf(stderr, "\n⚠ BOTTLENECK: D2H Transfer (%.1f%% of total)\n", 100*time_d2h/total_time);
    } else {
        fprintf(stderr, "\n⚠ BOTTLENECK: Kernel execution (%.1f%% of total)\n", 100*time_kernel/total_time);
    }

    write_png(argv[2], host_t, height, width, channels);

    cudaFree(device_src);
    cudaFree(device_dst);
    cudaEventDestroy(event_h2d_start);
    cudaEventDestroy(event_h2d_stop);
    cudaEventDestroy(event_kernel_start);
    cudaEventDestroy(event_kernel_stop);
    cudaEventDestroy(event_d2h_start);
    cudaEventDestroy(event_d2h_stop);
    
    return 0;
}
