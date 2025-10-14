/*
 * This program generates a visualization of the Mandelbrot set.
 * It is a sequential implementation that calculates the set for a specified
 * region of the complex plane and saves the output as a PNG image.
 */

// Define _GNU_SOURCE to get access to GNU-specific functions, like
// sched_getaffinity.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

// Define PNG_NO_SETJMP to avoid using setjmp/longjmp for error handling in
// libpng.
#define PNG_NO_SETJMP

#include <assert.h>  // For assertions (assert)
#include <png.h>     // For the libpng library, used to write PNG files
#include <pthread.h>
#include <sched.h>  // For CPU affinity functions (sched_getaffinity, CPU_COUNT)
#include <stdio.h>  // For standard I/O (printf, fopen, etc.)
#include <stdlib.h>  // For general utilities (strtol, strtod, malloc, free)
#include <string.h>  // For memory manipulation (memset)

typedef struct {
  int tid;
  int iters;
  int width;  // fix, since we divide by y
  int height;
  int my_height_start;
  int my_height_end;
  int* image;
  // for main loop usage
  double left;   // Real part start (x0)
  double right;  // Real part end (x1)
  double lower;  // Imaginary part start (y0)
  double upper;  // Imaginary part end (y1)
} ThreadArg;

void write_png(const char* filename, int iters, int width, int height,
               const int* buffer);
void* local_mandelbrot(void* argv);

int main(int argc, char* argv[]) {
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
  printf("%d cpus available\n", CPU_COUNT(&cpu_set));

  assert(argc == 9);
  const char* filename = argv[1];
  const int iters = strtol(argv[2], NULL, 10);
  const double left = strtod(argv[3], NULL);   // Real part start (x0)
  const double right = strtod(argv[4], NULL);  // Real part end (x1)
  const double lower = strtod(argv[5], NULL);  // Imaginary part start (y0)
  const double upper = strtod(argv[6], NULL);  // Imaginary part end (y1)
  const int width = strtol(argv[7], NULL, 10);
  const int height = strtol(argv[8], NULL, 10);

  int* image = (int*)malloc(width * height * sizeof(int));
  assert(image);  // Ensure memory was allocated.

  const int num_threads = CPU_COUNT(&cpu_set);
  pthread_t threads[num_threads];
  ThreadArg t_args[num_threads];

  // pthread init
  for (int i = 0; i < num_threads; i++) {
    t_args[i].tid = i;
    t_args[i].iters = iters;
    t_args[i].width = width;
    t_args[i].height = height;

    const int chunk = height / num_threads;
    const int remainder = height % num_threads;
    t_args[i].my_height_start = chunk * i + (i < remainder ? i : remainder);
    t_args[i].my_height_end =
        t_args[i].my_height_start + chunk - 1 + (i < remainder ? 1 : 0);

    t_args[i].image = image;

    t_args[i].left = left;
    t_args[i].right = right;
    t_args[i].lower = lower;
    t_args[i].upper = upper;

    pthread_create(&threads[i], NULL, local_mandelbrot, (void*)&t_args[i]);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }

  write_png(filename, iters, width, height, image);

  // Free the dynamically allocated memory for the image buffer.
  free(image);

  return 0;
}

void* local_mandelbrot(void* argv) {
  ThreadArg* t_arg = (ThreadArg*)argv;
  const int tid = t_arg->tid;
  const int iters = t_arg->iters;
  const int width = t_arg->width;  // fix, since we divide by y
  const int height = t_arg->height;
  const int my_height_start = t_arg->my_height_start;
  const int my_height_end = t_arg->my_height_end;
  int* image = t_arg->image;
  // for main loop usage
  const double left = t_arg->left;    // Real part start (x0)
  const double right = t_arg->right;  // Real part end (x1)
  const double lower = t_arg->lower;  // Imaginary part start (y0)
  const double upper = t_arg->upper;  // Imaginary part end (y1)

  // divide this for loop into pthread
  for (int j = my_height_start; j <= my_height_end;
       ++j) {  // Loop over rows (y-axis)

    // Map the current pixel row 'j' to its corresponding coordinate 'y0' in the
    // complex plane.
    const double y0 = j * ((upper - lower) / height) + lower;
    for (int i = 0; i < width; ++i) {  // Loop over columns (x-axis)
      // Map the current pixel column 'i' to its corresponding coordinate 'x0'
      // in the complex plane.
      const double x0 = i * ((right - left) / width) + left;

      int repeats = 0;
      double x = 0;
      double y = 0;
      double length_squared = 0;

      // This is the core Mandelbrot iteration for the complex number c = x0 +
      // i*y0. The loop continues until the point escapes (its distance from
      // origin > 2) or the maximum number of iterations is reached.
      while (repeats < iters && length_squared < 4) {
        // z_new = z^2 + c
        // z = x + iy
        // z^2 = (x^2 - y^2) + i(2xy)
        // z_new = (x^2 - y^2 + x0) + i(2xy + y0)
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++repeats;
      }

      // Store the number of iterations it took for the point to escape.
      // This value will be used to determine the pixel's color.
      image[j * width + i] = repeats;
    }
  }
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
 *               element stores the number of iterations for the corresponding
 * pixel.
 */
void write_png(const char* filename, int iters, int width, int height,
               const int* buffer) {
  // Open the file for writing in binary mode.
  FILE* fp = fopen(filename, "wb");
  assert(fp);  // Ensure the file was opened successfully.

  // Initialize the libpng structures for writing.
  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  assert(png_ptr);  // Ensure the png_struct was created.

  png_infop info_ptr = png_create_info_struct(png_ptr);
  assert(info_ptr);  // Ensure the info_struct was created.

  // Set up the output stream for libpng.
  png_init_io(png_ptr, fp);

  // Write the PNG header information.
  // This includes width, height, bit depth (8 bits per channel), color type
  // (RGB), and other standard PNG settings.
  png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  // Disable PNG filtering for simplicity and speed.
  png_set_filter(png_ptr, 0, PNG_NO_FILTERS);

  // Write the info chunk to the file.
  png_write_info(png_ptr, info_ptr);

  // Set the compression level (1 is a good balance between speed and size).
  png_set_compression_level(png_ptr, 1);

  // Allocate a buffer for a single row of the image. Each pixel is 3 bytes (R,
  // G, B).
  size_t row_size = 3 * width * sizeof(png_byte);
  png_bytep row = (png_bytep)malloc(row_size);

  // Iterate through each row of the image data.
  for (int y = 0; y < height; ++y) {
    // Clear the row buffer to black (0,0,0).
    memset(row, 0, row_size);

    // Iterate through each pixel in the row.
    for (int x = 0; x < width; ++x) {
      // Get the iteration count for the current pixel.
      // The buffer is read in reverse 'y' order to match typical image
      // coordinate systems (origin at top-left).
      int p = buffer[(height - 1 - y) * width + x];

      // Get a pointer to the start of the color data for this pixel.
      png_bytep color = row + x * 3;

      // If the point escaped (p != iters), assign a color to it.
      // Points that reached the max iteration count remain black, representing
      // the Mandelbrot set itself.
      if (p != iters) {
        if (p & 16) {
          color[0] = 240;                       // Red component
          color[1] = color[2] = (p % 16) * 16;  // Green and Blue components
        } else {
          color[0] = (p % 16) * 16;  // Red component
        }
      }
    }
    // Write the completed row to the PNG file.
    png_write_row(png_ptr, row);
  }

  // Free the memory for the row buffer.
  free(row);

  // Finalize the PNG file.
  png_write_end(png_ptr, NULL);

  // Clean up the libpng structures.
  png_destroy_write_struct(&png_ptr, &info_ptr);

  // Close the file.
  fclose(fp);
}
