/**
 * Parallel Odd-Even Sort using MPI
 * CS542200 Parallel Programming - Homework 1
 * 
 * Algorithm Overview:
 * - Odd-even sort alternates between two phases: odd phase and even phase
 * - Odd phase: compare pairs (1,2), (3,4), (5,6), ... (odd index with even index)
 * - Even phase: compare pairs (0,1), (2,3), (4,5), ... (even index with odd index)
 * - Continue until no swaps occur in both phases globally
 */

/* Headers */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

#define min(a, b) ((a) < (b) ? (a) : (b))


/* Function Prototypes */
void local_sort(float local_data[], const int my_count);
void merge_sort_split(float *local_data, const int my_count, float *recv_data, const int recv_count, float *temp, const int is_left);
const int global_sorted_check(float *local_data, const int my_count, MPI_Comm comm);


/* Main Function */
int main(int argc, char *argv[]) {


    /* MPI init and grouping */
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
    MPI_Barrier(MPI_COMM_WORLD);

    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_size(active_comm, &numtasks);
        MPI_Comm_rank(active_comm, &my_rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    /* Calaulate data distribution */
    const int base_chunk_size = N / numtasks; // 8 / 3 = 2
    const int remainder = N % numtasks; // 8 % 3 = 2
    const int my_count = (my_rank < remainder) ? base_chunk_size + 1 : base_chunk_size; // Processes with my_rank < remainder get one extra element
    const int my_start_index = my_rank * base_chunk_size + min(my_rank, remainder);
    const int is_active = (active_comm != MPI_COMM_NULL) && (my_count > 0);
    const int max_phases = numtasks + (numtasks / 2);
    int done = 0;


    /* MPI I/O */
    MPI_File input_file, output_file;
    const char *const input_filename = argv[2];
    const char *const output_filename = argv[3];
    float *local_data = NULL;
    float *recv_data = NULL;
    float *temp = NULL;

    if (is_active) {

        const size_t local_size = ((my_count * sizeof(float) + 31) / 32) * 32;
        local_data = (float *)aligned_alloc(32, local_size);
        if (!local_data) local_data = (float *)malloc(my_count * sizeof(float));

        const int max_recv_count = base_chunk_size + 1;
        const size_t recv_size = ((max_recv_count * sizeof(float) + 31) / 32) * 32;
        recv_data = (float *)aligned_alloc(32, recv_size);
        if (!recv_data) recv_data = (float *)malloc(max_recv_count * sizeof(float));

        const int max_temp_count = my_count + base_chunk_size + 1;
        const size_t temp_size = ((max_temp_count * sizeof(float) + 31) / 32) * 32;
        temp = (float *)aligned_alloc(32, temp_size);
        if (!temp) temp = (float *)malloc(max_temp_count * sizeof(float));

        MPI_File_open(active_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_File_read_at(input_file, my_start_index * sizeof(float), local_data, my_count, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&input_file);

        /* Input preprocessing */
        local_sort(local_data, my_count);
    }


    /* Main loop */
    MPI_Barrier(MPI_COMM_WORLD);
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

                MPI_Sendrecv(local_data, my_count, MPI_FLOAT, partner, phase,
                            recv_data, recv_count, MPI_FLOAT, partner, phase,
                            active_comm, MPI_STATUS_IGNORE);

                merge_sort_split(local_data, my_count, recv_data, recv_count, temp, is_left);
            }

            /* Early Exit */
            if (phase >= numtasks) done = global_sorted_check(local_data, my_count, active_comm);
            if (done) break;
        }
    }


    /* MPI I/O */
    if (is_active) {
        MPI_Barrier(active_comm);
        MPI_File_open(active_comm, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
        MPI_File_write_at(output_file, my_start_index * sizeof(float), local_data, my_count, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&output_file);

        free(local_data);
        free(recv_data);
        free(temp);
    }


    /* MPI Finalize and clean */
    MPI_Barrier(MPI_COMM_WORLD);
    free(active_ranks);
    MPI_Group_free(&active_group);
    MPI_Group_free(&orig_group);
    MPI_Finalize();
    return 0;
}


/* Function Definitions */
void local_sort(float local_data[], const int my_count) {

    if (my_count < 33) { // insertion sort
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
    else if (my_count < 10000) boost::sort::spreadsort::spreadsort(local_data, local_data + my_count);
    else boost::sort::spreadsort::float_sort(local_data, local_data + my_count);
}

void merge_sort_split(float *local_data, const int my_count, float *recv_data, const int recv_count, float *temp, const int is_left) {
    
    if (my_count < 1 || recv_count < 1) return;

    int i = 0, j = 0, k = 0;
    while (i < my_count && j < recv_count) {

        if (local_data[i] <= recv_data[j]) temp[k++] = local_data[i++];    
        else temp[k++] = recv_data[j++];
    }
    
    while (i < my_count) temp[k++] = local_data[i++];
    while (j < recv_count) temp[k++] = recv_data[j++];

    if (is_left) memcpy(local_data, temp, my_count * sizeof(float));
    else memcpy(local_data, temp + recv_count, my_count * sizeof(float));
}

const int global_sorted_check(float *local_data, const int my_count, MPI_Comm comm) {

    int local_sorted = 1;
    for (int i = 1; i < my_count; i++) {

        if (local_data[i - 1] > local_data[i]) {
            local_sorted = 0;
            break;
        }
    }
    
    int global_sorted;
    MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI_INT, MPI_LAND, comm);
    return global_sorted;
}

