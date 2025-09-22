/**
 * Parallel Odd-Even Sort using MPI (Revised Optimization)
 * CS542200 Parallel Programming - Homework 1
 *
 * Key Optimizations Implemented:
 * 1. Zero-Copy Merge-Split: Uses a custom partial merge and O(1) pointer swapping
 *    (std::swap) to eliminate memory copy overhead in the merge step.
 * 2. Adaptive Local Sort: Employs different sorting algorithms (Insertion, std::sort,
 *    boost::spreadsort) based on the local data size for optimal performance.
 * 3. Aligned Memory Allocation: Uses aligned_alloc to ensure data buffers are
 *    32-byte aligned, which can improve performance for SIMD instructions.
 * 4. Efficient Early Exit: A global sortedness check is performed periodically
 *    to terminate the sort as soon as the data is globally ordered.
 * 5. CPU Cache Prefetching: Hints the CPU to pre-load data into the cache
 *    before it is accessed in the merge loop to reduce memory latency.
 */

/* Headers */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>

#define min(a, b) ((a) < (b) ? (a) : (b))

/* Function Prototypes */
void local_sort(float local_data[], const int my_count);
const int sorted_check(float *local_data, const int my_count, const int my_rank, const int numtasks, MPI_Comm comm);
void merge_sort_split(float *&local_data, const int my_count, float *recv_data, const int recv_count, float *&temp, const int is_left);

/* Main Function */
int main(int argc, char *argv[]) {


    /* MPI Init and Grouping */
    MPI_Group orig_group, active_group;
    MPI_Comm active_comm;
    const int N = atoi(argv[1]);
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

    MPI_File input_file, output_file;
    const char *const input_filename = argv[2];
    const char *const output_filename = argv[3];
    float *temp = NULL;
    float *local_data = NULL;
    float *recv_data = NULL;

    if (is_active) {
        
        // Memory is allocated based on worst-case scenarios for chunk sizes.
        // 'two_chunk' is large enough to hold the result of merging two chunks.
        const size_t two_chunk = (((base_chunk_size + 1) * 2 * sizeof(float) + 31) / 32) * 32;
        // 'one_chunk' is for receiving data from a partner, which is at most one chunk.
        const size_t one_chunk = (((base_chunk_size + 1) * sizeof(float) + 31) / 32) * 32;

        // Allocate local_data and temp with the same large size. This is crucial
        // for the pointer swapping (std::swap) optimization to work safely.
        temp = (float *)aligned_alloc(32, two_chunk);
        local_data = (float *)aligned_alloc(32, two_chunk);
        // The receive buffer only needs to be large enough for one chunk.
        recv_data = (float *)aligned_alloc(32, one_chunk);

        // Fallback to standard malloc if aligned_alloc fails.
        if (!temp) temp = (float *)malloc(two_chunk * sizeof(float));
        if (!local_data) local_data = (float *)malloc(two_chunk * sizeof(float));
        if (!recv_data) recv_data = (float *)malloc(one_chunk * sizeof(float));

        // Read the assigned partition of data from the input file.
        MPI_File_open(active_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_File_read_at(input_file, my_start_index * sizeof(float), local_data, my_count, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&input_file);
        
        // Perform the initial local sort on the data each process holds.
        local_sort(local_data, my_count);
    }

    /* Main loop */
    if (is_active) {

        for (int phase = 0; phase < max_phases; phase++) {

            /* Partner ID */
            int partner = -1;
            if (phase % 2) partner = (my_rank % 2) ? my_rank + 1 : my_rank - 1;
            else partner = (my_rank % 2) ? my_rank - 1 : my_rank + 1;  
            if ((partner < 0) || (partner >= numtasks)) partner = MPI_PROC_NULL;

            /* MPI and subarrays merge */
            if (partner != MPI_PROC_NULL) {

                const int recv_count = (partner < remainder) ? base_chunk_size + 1 : base_chunk_size;
                const int is_left = (my_rank < partner) ? 1 : 0;
                
                // Exchange data with the partner process. MPI_Sendrecv is used for its
                // simplicity and deadlock-free nature in this paired communication pattern.
                MPI_Sendrecv(local_data, my_count, MPI_FLOAT, partner, phase,
                             recv_data, recv_count, MPI_FLOAT, partner, phase,
                             active_comm, MPI_STATUS_IGNORE);
                
                // This is the core Zero-Copy merge-split operation.
                merge_sort_split(local_data, my_count, recv_data, recv_count, temp, is_left);
            }

            /* Early Exit */
            int done = 0;
            // The sortedness check starts after numtasks/2 phases and is performed
            // only on even phases to reduce the overhead of global synchronization.
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

// This function uses an adaptive sorting strategy based on the array size.
void local_sort(float local_data[], const int my_count) {

    // For very small arrays, simple Insertion Sort is efficient due to low overhead.
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
    // For medium-sized arrays, std::sort (Introsort) is a robust and fast choice.
    else if (my_count < 1025) std::sort(local_data, local_data + my_count);
    // For larger arrays, boost::spreadsort is often faster for floating-point types.
    else if (my_count < 10000) boost::sort::spreadsort::spreadsort(local_data, local_data + my_count);
    else boost::sort::spreadsort::float_sort(local_data, local_data + my_count);
}

// Implements the zero-copy merge-split.
// It uses a custom partial merge and an O(1) pointer swap to avoid memory copies.
void merge_sort_split(float *&local_data, const int my_count, float *recv_data, const int recv_count, float *&temp, const int is_left) {

    if (my_count < 1) return;
    
    // Prefetching hints the CPU to load data into cache before it's needed,
    // potentially reducing latency from memory access.
    // rw=1 for write, rw=0 for read. locality=0 means low temporal locality.
    if (my_count > 0) __builtin_prefetch(temp, 1, 0);
    if (my_count > 0) __builtin_prefetch(local_data, 0, 0);
    if (recv_count > 0) __builtin_prefetch(recv_data, 0, 0);

    if (is_left) {
        // This process is on the "left" of the pair, so it needs to keep the
        // 'my_count' smallest elements from the combined data.
        int i = 0, j = 0, k = 0;
        
        // Partially merge only the smallest 'my_count' elements into the temp buffer.
        while (k < my_count && i < my_count && j < recv_count) {
            if (local_data[i] <= recv_data[j]) temp[k++] = local_data[i++];
            else temp[k++] = recv_data[j++];
        }
        // Fill the rest of temp if one of the source arrays was exhausted early.
        while (k < my_count && i < my_count) temp[k++] = local_data[i++];
        while (k < my_count && j < recv_count) temp[k++] = recv_data[j++];

    } else { // is_right
        // This process is on the "right," so it needs the 'my_count' largest elements.
        int i = my_count - 1, j = recv_count - 1, k = my_count - 1;

        // Merge is done backwards, from largest to smallest, filling the temp buffer.
        while (k >= 0 && i >= 0 && j >= 0) {
            if (local_data[i] >= recv_data[j]) temp[k--] = local_data[i--];
            else temp[k--] = recv_data[j--];
        }
        // Fill the rest of temp if one of the source arrays was exhausted early.
        while (k >= 0 && i >= 0) temp[k--] = local_data[i--];
        while (k >= 0 && j >= 0) temp[k--] = recv_data[j--];
    }
    
    // The core of the zero-copy optimization: an O(1) pointer swap.
    // Instead of an O(N) copy, we just swap which memory buffer 'local_data' points to.
    std::swap(local_data, temp);
}

// Checks if the global array is sorted by verifying the boundary elements
// between adjacent processes.
const int sorted_check(float *local_data, const int my_count, const int my_rank, const int numtasks, MPI_Comm comm) {

    const int mpi_tag = 0;
    const int prev_rank = (my_rank > 0) ? my_rank - 1 : MPI_PROC_NULL;
    const int next_rank = (my_rank < numtasks - 1) ? my_rank + 1 : MPI_PROC_NULL;
    const float my_last_element = (my_count > 0) ? local_data[my_count - 1] : -1.0f;
    int boundary_sorted = 1;
    int global_sorted;
    float prev_last_element;

    // Each process sends its last element to the next process and receives
    // the last element from the previous process.
    MPI_Sendrecv(&my_last_element, 1, MPI_FLOAT, next_rank, mpi_tag,
                 &prev_last_element, 1, MPI_FLOAT, prev_rank, mpi_tag,
                 comm, MPI_STATUS_IGNORE);

    // Check if the local boundary is sorted. If the last element from the previous
    // process is greater than my first element, the boundary is not sorted.
    if (my_count > 0 && my_rank > 0 && prev_last_element > local_data[0]) {
        boundary_sorted = 0;
    }
    
    // Use MPI_Allreduce with logical AND to get a global consensus.
    // If all processes report their boundaries are sorted, 'global_sorted' will be 1.
    MPI_Allreduce(&boundary_sorted, &global_sorted, 1, MPI_INT, MPI_LAND, comm);
    
    return global_sorted;
}
