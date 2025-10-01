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
#include <mpi.h>
#include <cstdlib>
#include <cstring>
#ifdef PROFILING
#include <nvtx3/nvToolsExt.h>
// ============================================================================
// NVTX Color Definitions for Performance Profiling
// ============================================================================
#define COLOR_IO 0xFF5C9FFF           // Azure Blue
#define COLOR_BOUNDARY 0xFFFF6B81     // Rose Red (Boundary Check)
#define COLOR_DATA_EXC 0xFFFFA500     // Orange (Data Exchange)
#define COLOR_LOCAL_SORT 0xFF5FD068   // Spring Green (Local Sort)
#define COLOR_MERGE_SPLIT 0xFF32CD32  // Lime Green (Merge-Split)
#define COLOR_SETUP 0xFFFFBE3D        // Sunflower Yellow
#define COLOR_DEFAULT 0xFFCBD5E0      // Soft Gray
// ============================================================================
// Smart NVTX Macro: Auto-selects color based on range name pattern
// ============================================================================
#define NVTX_PUSH(name) \
do { \
    uint32_t color = COLOR_DEFAULT; \
    if (strstr(name, "IO_") != NULL) { \
        color = COLOR_IO; \
    } else if (strcmp(name, "Boundary_Check") == 0) { \
        color = COLOR_BOUNDARY; \
    } else if (strcmp(name, "Data_Exchange") == 0) { \
        color = COLOR_DATA_EXC; \
    } else if (strcmp(name, "Sorted_Check") == 0) { \
        color = COLOR_BOUNDARY; \
    } else if (strcmp(name, "Local_Sort") == 0) { \
        color = COLOR_LOCAL_SORT; \
    } else if (strcmp(name, "Merge_Split") == 0) { \
        color = COLOR_MERGE_SPLIT; \
    } else if (strcmp(name, "MPI_Setup") == 0 || strcmp(name, "Mem_Alloc") == 0) { \
        color = COLOR_SETUP; \
    } else { \
        color = COLOR_DEFAULT; \
    } \
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = color; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
} while(0)
#define NVTX_POP() nvtxRangePop()
#endif

#define min(a, b) ((a) < (b) ? (a) : (b))
static float temp_swap;
#define FAST_SWAP(x, y) do { \
    temp_swap = (x); \
    (x) = (y); \
    (y) = temp_swap; \
} while(0)

int compare_floats(const void *a, const void *b);
void local_sort(float arr[], const int arr_count);
int sequential_swaps(float arr[], const int arr_count);
int odd_phase_swaps(float arr[], const int arr_count, const int my_start_index);
int even_phase_swaps(float arr[], const int arr_count, const int my_start_index);

int compare_floats(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

void local_sort(float arr[], const int arr_count) {
    qsort(arr, arr_count, sizeof(float), compare_floats);
}

int sequential_swaps(float arr[], const int arr_count) {
    if (arr_count < 2) return 0;
    int is_swap = 0;
    // Compare and swap consecutive pairs: (0,1), (2,3), (4,5), ...
    for (int i = 0; i + 1 < arr_count; i += 2) {
        if (arr[i] > arr[i + 1]) {
            FAST_SWAP(arr[i], arr[i + 1]);
            is_swap = 1;
        }
    }
    return is_swap;
}

int odd_phase_swaps(float arr[], const int arr_count, const int my_start_index) {
    if (arr_count < 2) return 0;
    int is_swap = 0;
    // If we start with odd index, process pairs: (0,1), (2,3), ... within local array
    if (my_start_index % 2) is_swap = sequential_swaps(arr, arr_count);
    // If we start with even index, skip first element, process: (1,2), (3,4), ... within local array
    else is_swap = sequential_swaps(arr + 1, arr_count - 1);
    return is_swap;
}

int even_phase_swaps(float arr[], const int arr_count, const int my_start_index) {
    if (arr_count < 2) return 0;
    int is_swap = 0;
    // If we start with odd index, skip first element, process: (1,2), (3,4), ... within local array
    if (my_start_index % 2) is_swap = sequential_swaps(arr + 1, arr_count - 1);
    // If we start with even index, process pairs: (0,1), (2,3), ... within local array
    else is_swap = sequential_swaps(arr, arr_count);
    return is_swap;
}

int main(int argc, char *argv[]) {
    /* MPI init and grouping */
    MPI_Init(&argc, &argv);
    
#ifdef PROFILING
    double total_start_time, temp_start, io_time = 0.0, comm_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before starting the main timer
    total_start_time = MPI_Wtime();
    NVTX_PUSH("MPI_Setup");
#endif

    int numtasks, my_rank;
    const int N = atoi(argv[1]); // Total number of elements to sort
    MPI_Group orig_group, active_group;
    MPI_Comm active_comm;
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
    
#ifdef PROFILING
    NVTX_POP(); // end MPI_Setup
#endif

    /* Calaulate my offset */
    // Example: 8 elements, 3 processes -> 3, 3, 2
    const int base_chunk_size = N / numtasks; // 8 / 3 = 2
    const int remainder = N % numtasks; // 8 % 3 = 2
    // Processes with my_rank < remainder get one extra element
    const int my_count = (my_rank < remainder) ? base_chunk_size + 1 : base_chunk_size;
    const int my_start_index = my_rank * base_chunk_size + min(my_rank, remainder);
    const int my_end_index = my_start_index + my_count - 1;
    
    /* Important active tag!! */
    const int is_active = (active_comm != MPI_COMM_NULL) && (my_count > 0);
    
    /* MPI I/O */
    const char *const input_filename = argv[2], *const output_filename = argv[3];
    MPI_File input_file, output_file;
    float *local_data = NULL;
    
#ifdef PROFILING
    NVTX_PUSH("Mem_Alloc");
#endif
    
    if (is_active) {
        const size_t size = ((my_count * sizeof(float) + 31) / 32) * 32;
        local_data = (float *)aligned_alloc(32, size);
        if (!local_data) local_data = (float *)malloc(my_count * sizeof(float));
        
#ifdef PROFILING
        NVTX_POP(); // end Mem_Alloc
        temp_start = MPI_Wtime();
        NVTX_PUSH("IO_Read");
#endif
        
        MPI_File_open(active_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_File_read_at(input_file, sizeof(float) * my_start_index, local_data, my_count, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&input_file);
        
#ifdef PROFILING
        io_time += MPI_Wtime() - temp_start;
        NVTX_POP(); // end IO_Read
#endif
    }
#ifdef PROFILING
    else {
        NVTX_POP(); // end Mem_Alloc
    }
#endif
    
    /* Main loop variables */
    int iteration = 1; // start with odd, or 01.txt will fail
    int is_swap = 0, is_global_swap = 1;
    static float boundary_buffer[2];
    const int mpi_even_tag = 0, mpi_odd_tag = 1;
    const int check_interval = (N < 10000) ? 4
                             : (N < 100000) ? 16
                             : (N < 1000000) ? 128
                             : 256;
    
    /* Input preprocessing */
#ifdef PROFILING
    NVTX_PUSH("Local_Sort");
#endif
    
    if (is_active) local_sort(local_data, my_count);
    
#ifdef PROFILING
    NVTX_POP(); // end Local_Sort
#endif
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Main loop */
#ifdef PROFILING
    NVTX_PUSH("Main_Loop");
#endif
    
    while (is_active && is_global_swap) {
        is_swap = 0;
        const float my_last_data = local_data[my_count - 1];
        const float my_first_data = local_data[0];
        
        if (iteration % 2) { // odd phase
            // Case 1: I'm the left process in a cross-boundary pair
            if (my_end_index % 2 && my_rank + 1 < numtasks) {
#ifdef PROFILING
                temp_start = MPI_Wtime();
                NVTX_PUSH("Boundary_Check");
#endif
                // Exchange boundary elements with right neighbor
                MPI_Sendrecv(&my_last_data, 1, MPI_FLOAT, my_rank + 1, mpi_odd_tag,
                            &boundary_buffer[1], 1, MPI_FLOAT, my_rank + 1, mpi_odd_tag,
                            active_comm, MPI_STATUS_IGNORE);
#ifdef PROFILING
                comm_time += MPI_Wtime() - temp_start;
                NVTX_POP(); // end Boundary_Check
#endif
                is_swap = (my_last_data > boundary_buffer[1]) ? 1 : 0;
                if (is_swap) local_data[my_count - 1] = boundary_buffer[1];
            }
            
            // Case 2: I'm the right process in a cross-boundary pair
            if (!(my_start_index % 2) && my_rank > 0) {
#ifdef PROFILING
                temp_start = MPI_Wtime();
                NVTX_PUSH("Boundary_Check");
#endif
                // Exchange boundary elements with left neighbor
                MPI_Sendrecv(&my_first_data, 1, MPI_FLOAT, my_rank - 1, mpi_odd_tag,
                            &boundary_buffer[0], 1, MPI_FLOAT, my_rank - 1, mpi_odd_tag,
                            active_comm, MPI_STATUS_IGNORE);
#ifdef PROFILING
                comm_time += MPI_Wtime() - temp_start;
                NVTX_POP(); // end Boundary_Check
#endif
                is_swap = (my_first_data < boundary_buffer[0]) ? 1 : 0;
                if (is_swap) local_data[0] = boundary_buffer[0];
            }
            
            // Handle internal odd phase swaps within this process
            is_swap |= odd_phase_swaps(local_data, my_count, my_start_index); // check every time !!!!
        } else { // even phase
            // Case 1: I'm the left process in a cross-boundary pair
            if (!(my_end_index % 2) && my_rank + 1 < numtasks) {
#ifdef PROFILING
                temp_start = MPI_Wtime();
                NVTX_PUSH("Boundary_Check");
#endif
                // Exchange boundary elements with right neighbor
                MPI_Sendrecv(&my_last_data, 1, MPI_FLOAT, my_rank + 1, mpi_even_tag,
                            &boundary_buffer[1], 1, MPI_FLOAT, my_rank + 1, mpi_even_tag,
                            active_comm, MPI_STATUS_IGNORE);
#ifdef PROFILING
                comm_time += MPI_Wtime() - temp_start;
                NVTX_POP(); // end Boundary_Check
#endif
                is_swap = (my_last_data > boundary_buffer[1]) ? 1 : 0;
                if (is_swap) local_data[my_count - 1] = boundary_buffer[1];
            }
            
            // Case 2: I'm the right process in a cross-boundary pair
            if (my_start_index % 2 && my_rank > 0) {
#ifdef PROFILING
                temp_start = MPI_Wtime();
                NVTX_PUSH("Boundary_Check");
#endif
                // Exchange boundary elements with left neighbor
                MPI_Sendrecv(&my_first_data, 1, MPI_FLOAT, my_rank - 1, mpi_even_tag,
                            &boundary_buffer[0], 1, MPI_FLOAT, my_rank - 1, mpi_even_tag,
                            active_comm, MPI_STATUS_IGNORE);
#ifdef PROFILING
                comm_time += MPI_Wtime() - temp_start;
                NVTX_POP(); // end Boundary_Check
#endif
                is_swap = (my_first_data < boundary_buffer[0]) ? 1 : 0;
                if (is_swap) local_data[0] = boundary_buffer[0];
            }
            
            // Handle internal even phase swaps within this process
            is_swap |= even_phase_swaps(local_data, my_count, my_start_index); // check every time !!!!
        }
        
        iteration++;
        if (!(iteration % check_interval)) {
            if (my_count > 2) local_sort(local_data, my_count);
#ifdef PROFILING
            temp_start = MPI_Wtime();
            NVTX_PUSH("Sorted_Check");
#endif
            MPI_Allreduce(&is_swap, &is_global_swap, 1, MPI_INT, MPI_LOR, active_comm);
#ifdef PROFILING
            comm_time += MPI_Wtime() - temp_start;
            NVTX_POP(); // end Sorted_Check
#endif
        }
    } // end while loop
    
#ifdef PROFILING
    NVTX_POP(); // end Main_Loop
#endif
    
    /* MPI I/O */
    if (is_active) {
#ifdef PROFILING
        temp_start = MPI_Wtime();
        NVTX_PUSH("IO_Write");
#endif
        MPI_Barrier(active_comm);
        MPI_File_open(active_comm, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
        MPI_File_write_at(output_file, my_start_index * sizeof(float), local_data, my_count, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&output_file);
#ifdef PROFILING
        io_time += MPI_Wtime() - temp_start;
        NVTX_POP(); // end IO_Write
#endif
    }
    
    /* MPI Finalize and clean */
    MPI_Barrier(MPI_COMM_WORLD);
    
#ifdef PROFILING
    double total_time = MPI_Wtime() - total_start_time;
    double cpu_time = total_time - io_time - comm_time;
    if (my_rank == 0) {
        printf("N=%d, Procs=%d, Total Time: %f s\n", N, numtasks, total_time);
        printf("  IO Time: %f s (%f %%)\n", io_time, (io_time / total_time) * 100);
        printf("  Comm Time: %f s (%f %%)\n", comm_time, (comm_time / total_time) * 100);
        printf("  CPU Time: %f s (%f %%)\n", cpu_time, (cpu_time / total_time) * 100);
    }
    NVTX_PUSH("MPI_Finalize");
#endif
    
    free(active_ranks);
    free(local_data);
    MPI_Group_free(&active_group);
    MPI_Group_free(&orig_group);
    
#ifdef PROFILING
    NVTX_POP(); // end MPI_Finalize
#endif
    
    MPI_Finalize();
    return 0;
}
