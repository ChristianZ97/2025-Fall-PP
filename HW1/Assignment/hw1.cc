/**
 * Parallel Odd-Even Sort using MPI
 * CS542200 Parallel Programming - Homework 1
 *
 * ============================================================================
 * Build Strategy and Compiler Flags:
 * ============================================================================
 * This program is compiled with a set of aggressive optimization flags to
 * maximize performance on the target architecture.
 *
 * CXXFLAGS = -std=c++17 -O3 -march=native -flto -fomit-frame-pointer -funroll-loops
 *
 * - -std=c++17:         Ensures code is compiled using the C++17 standard.
 * - -O3:                 Enables the highest level of general compiler optimizations.
 * - -march=native:       CRITICAL. Generates code specifically for the CPU it's
 *                        compiled on, allowing use of advanced instruction sets like
 *                        AVX2 for faster floating-point and memory operations.
 * - -flto:               Link-Time Optimization. Performs optimizations across the
 *                        entire program, including library code, during the final
 *                        link stage.
 * - -fomit-frame-pointer: Frees up an extra CPU register for the optimizer by omitting
 *                         the frame pointer, potentially reducing memory access.
 * - -funroll-loops:      Aggressively unrolls loops to reduce branching overhead,
 *                        which is beneficial for the tight loops in the merge logic.
 *
 * ============================================================================
 * Key Optimizations for Demonstration:
 * ============================================================================
 * 1. Conditional Merge-Split: A massive optimization that avoids unnecessary work.
 *    Before performing a full data exchange and merge, processes first exchange
 *    only their boundary elements (a single float). The expensive merge-split
 *    operation is completely skipped if the boundary data is already in order.
 * 2. Zero-Copy Merge-Split: The cornerstone of this implementation. It uses a custom
 *    partial merge and an O(1) pointer swap (std::swap) to completely eliminate
 *    memory copy overhead in the merge step, significantly reducing memory bandwidth
 *    and CPU cycles.
 * 3. Adaptive Local Sort: A smart sorting strategy that employs different algorithms
 *    (Insertion Sort, std::sort, boost::spreadsort) based on the local data size.
 *    This ensures the best sorting performance for any data partition size.
 * 4. Aligned Memory Allocation: Uses aligned_alloc to ensure data buffers are
 *    32-byte aligned. This is a crucial micro-optimization that can significantly
 *    improve performance for modern CPUs by enabling more efficient SIMD instructions.
 * 5. Efficient Early Exit: A global sortedness check is performed periodically after
 *    a sufficient number of phases. This avoids unnecessary computation and communication
 *    by terminating the sort as soon as the data is globally ordered.
 */

/* Headers */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <float.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

#define min(a, b) ((a) < (b) ? (a) : (b))

/* Function Prototypes */
void local_sort(float local_data[], const int my_count);
const int sorted_check(float *local_data, const int my_count, const int my_rank, const int numtasks, MPI_Comm comm);
void merge_sort_split(float *&local_data, const int my_count, float *recv_data, const int recv_count, float *&temp, const int is_left);

/* Main Function */
int main(int argc, char *argv[]) {

    if (argc != 4) return -1;


    /* MPI Init and Grouping */
    MPI_Group orig_group, active_group;
    MPI_Comm active_comm;
    const int N = atoi(argv[1]); if (N < 1) return -1;
    int numtasks, my_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    const int active_numtasks = min(numtasks, N);
    int *active_ranks = (int *)malloc(active_numtasks * sizeof(int));
    for (int i = 0; i < active_numtasks; i++) active_ranks[i] = i;

    MPI_Group_incl(orig_group, active_numtasks, active_ranks, &active_group);
    MPI_Comm_create(MPI_COMM_WORLD, active_group, &active_comm);
    
    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_size(active_comm, &numtasks);
        MPI_Comm_rank(active_comm, &my_rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Data distribution and I/O */
    const int base_chunk_size = N / numtasks;
    const int remainder = N % numtasks;
    const int my_count = (my_rank < remainder) ? base_chunk_size + 1 : base_chunk_size;
    const int my_start_index = my_rank * base_chunk_size + min(my_rank, remainder);
    const int is_active = (active_comm != MPI_COMM_NULL) && (my_count > 0);
    const int max_phases = numtasks + (numtasks / 2);
    const int is_odd_rank = (my_rank % 2) ? 1 : 0;
    float partner_boundary;

    const size_t max_needed = (size_t)(base_chunk_size + 1) * 2;
    if (max_needed > SIZE_MAX / sizeof(float)) MPI_Abort(MPI_COMM_WORLD, -1);

    MPI_File input_file, output_file;
    const char *const input_filename = argv[2];
    const char *const output_filename = argv[3];
    float *temp = NULL;
    float *local_data = NULL;
    float *recv_data = NULL;

    if (is_active) {
        
        // DEMO POINT: Smart Memory Allocation for Zero-Copy
        const size_t two_chunk = (((base_chunk_size + 1) * 2 * sizeof(float) + 31) / 32) * 32;
        const size_t one_chunk = (((base_chunk_size + 1) * sizeof(float) + 31) / 32) * 32;

        // DEMO POINT: Aligned Memory
        temp = (float *)aligned_alloc(32, two_chunk);
        local_data = (float *)aligned_alloc(32, two_chunk);
        recv_data = (float *)aligned_alloc(32, one_chunk);

        // DEMO POINT: Pointer Swapping Strategy
        if (!temp) temp = (float *)malloc(two_chunk);
        if (!local_data) local_data = (float *)malloc(two_chunk);
        if (!recv_data) recv_data = (float *)malloc(one_chunk);
        
        // Use robust, synchronous I/O for simplicity and correctness.
        MPI_File_open(active_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_File_read_at(input_file, my_start_index * sizeof(float), local_data, my_count, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&input_file);
        
        // DEMO POINT: Adaptive Local Sort
        local_sort(local_data, my_count);
    }

    /* Main loop */
    if (is_active) {

        for (int phase = 0; phase < max_phases; phase++) {

            const int mpi_boundary_tag = 2 * phase;
            const int mpi_data_tag = 2 * phase + 1;
            const int is_odd_phase = (phase % 2) ? 1 : 0;
            const float my_first = local_data[0];
            const float my_last = local_data[my_count - 1];

            /* Partner ID */
            int partner = -1;
            if (is_odd_phase) partner = (is_odd_rank) ? my_rank + 1 : my_rank - 1;
            else partner = (is_odd_rank) ? my_rank - 1 : my_rank + 1;  
            if ((partner < 0) || (partner >= numtasks)) partner = MPI_PROC_NULL;

            const int partner_exist = (partner != MPI_PROC_NULL) ? 1 : 0;
            const int is_left = (my_rank < partner) ? 1 : 0;
            const int recv_count = (partner < remainder) ? base_chunk_size + 1 : base_chunk_size;
            
            // DEMO POINT: Conditional Merge-Split
            // This is the core of the final optimization. Instead of blindly merging,
            // we first perform a cheap check by exchanging only the boundary elements.
            if (partner_exist && is_left) { // I am the left process of a pair.

                // Exchange my last element with my partner's first element.
                MPI_Sendrecv(&my_last, 1, MPI_FLOAT, partner, mpi_boundary_tag,
                             &partner_boundary, 1, MPI_FLOAT, partner, mpi_boundary_tag,
                             active_comm, MPI_STATUS_IGNORE);

                // Only perform the expensive full exchange and merge if my data
                // might flow into my partner's partition.
                if (my_last > partner_boundary) {

                    MPI_Sendrecv(local_data, my_count, MPI_FLOAT, partner, mpi_data_tag,
                                 recv_data, recv_count, MPI_FLOAT, partner, mpi_data_tag,
                                 active_comm, MPI_STATUS_IGNORE);

                    merge_sort_split(local_data, my_count, recv_data, recv_count, temp, is_left);
                }

            } else if (partner_exist) { // I am the right process of a pair.

                // Exchange my first element with my partner's last element.
                MPI_Sendrecv(&my_first, 1, MPI_FLOAT, partner, mpi_boundary_tag,
                             &partner_boundary, 1, MPI_FLOAT, partner, mpi_boundary_tag,
                             active_comm, MPI_STATUS_IGNORE);

                // Only perform the expensive full exchange and merge if my partner's data
                // might flow into my partition.
                if (my_first < partner_boundary) {

                    MPI_Sendrecv(local_data, my_count, MPI_FLOAT, partner, mpi_data_tag,
                                 recv_data, recv_count, MPI_FLOAT, partner, mpi_data_tag,
                                 active_comm, MPI_STATUS_IGNORE);

                    merge_sort_split(local_data, my_count, recv_data, recv_count, temp, is_left);
                }
            }

            /* Early Exit */
            int done = 0;
            // DEMO POINT: Optimized Early Exit Strategy
            if (phase >= numtasks / 2 && !(phase % 2)) done = sorted_check(local_data, my_count, my_rank, numtasks, active_comm);
            if (done) break;
        }
    }

    /* MPI I/O and Finalize */
    if (is_active) {
        MPI_Barrier(active_comm);
        MPI_File_open(active_comm, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
        MPI_File_write_at(output_file, my_start_index * sizeof(float), local_data, my_count, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&output_file);

        free(local_data);
        free(recv_data);
        free(temp);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    free(active_ranks);
    MPI_Group_free(&active_group);
    MPI_Group_free(&orig_group);
    MPI_Finalize();
    return 0;
}

/* Function Definitions */

// DEMO POINT: Adaptive Sorting (Function Definition)
void local_sort(float local_data[], const int my_count) {
    if (my_count < 33) {
        for (int i = 1; i < my_count; i++) {
            float temp = local_data[i];
            int j = i - 1;
            while (j >= 0 && temp < local_data[j]) { 
                local_data[j + 1] = local_data[j];
                j--; 
            }
            local_data[j + 1] = temp;
        }
    }
    else if (my_count < 1025) std::sort(local_data, local_data + my_count);
    else boost::sort::spreadsort::float_sort(local_data, local_data + my_count);
}

// DEMO POINT: Zero-Copy Merge-Split (Function Definition)
void merge_sort_split(float *&local_data, const int my_count, float *recv_data, const int recv_count, float *&temp, const int is_left) {

    if (my_count < 1) return;

    if (is_left) {
        int i = 0, j = 0, k = 0;
        while (k < my_count && i < my_count && j < recv_count) {
            if (local_data[i] <= recv_data[j]) temp[k++] = local_data[i++];
            else temp[k++] = recv_data[j++];
        }
        while (k < my_count && i < my_count) temp[k++] = local_data[i++];
        while (k < my_count && j < recv_count) temp[k++] = recv_data[j++];

    } else { // is_right
        int i = my_count - 1, j = recv_count - 1, k = my_count - 1;
        while (k >= 0 && i >= 0 && j >= 0) {
            if (local_data[i] >= recv_data[j]) temp[k--] = local_data[i--];
            else temp[k--] = recv_data[j--];
        }
        while (k >= 0 && i >= 0) temp[k--] = local_data[i--];
        while (k >= 0 && j >= 0) temp[k--] = recv_data[j--];
    }
    
    // DEMO POINT: The O(1) Pointer Swap
    std::swap(local_data, temp);
}

// Checks if the global array is sorted by verifying the boundary elements.
const int sorted_check(float *local_data, const int my_count, const int my_rank, const int numtasks, MPI_Comm comm) {

    const int mpi_tag = 0;
    const int prev_rank = (my_rank > 0) ? my_rank - 1 : MPI_PROC_NULL;
    const int next_rank = (my_rank < numtasks - 1) ? my_rank + 1 : MPI_PROC_NULL;
    const float my_last_element = (my_count > 0) ? local_data[my_count - 1] : -FLT_MAX;
    int boundary_sorted = 1;
    int global_sorted;
    float prev_last_element;

    MPI_Sendrecv(&my_last_element, 1, MPI_FLOAT, next_rank, mpi_tag,
                 &prev_last_element, 1, MPI_FLOAT, prev_rank, mpi_tag,
                 comm, MPI_STATUS_IGNORE);

    if (my_count > 0 && my_rank > 0 && prev_last_element > local_data[0]) boundary_sorted = 0;
    MPI_Allreduce(&boundary_sorted, &global_sorted, 1, MPI_INT, MPI_LAND, comm);
    
    return global_sorted;
}
