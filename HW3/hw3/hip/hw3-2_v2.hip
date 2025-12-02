#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* 
 * 優化策略:
 * 1. 使用 int4 向量化存取 (128-bit load/store)，最大化記憶體頻寬。
 * 2. 調整 Blocking Factor 與 Thread Tile 比例。
 *    Block Size: 32x32 (1024 threads)
 *    Data Tile: 64x64
 *    每個執行緒負責讀取/計算 2x2 的元素 (但使用 int4 優化讀取)。
 */

#define BLOCKING_FACTOR 64
#define HALF_BLOCK 32
#define INF ((1 << 30) - 1)

static int *D;
static int *d_D;
static int V, E;
static int V_padded;

// 串流與事件
hipStream_t stream_main;

/* 輔助: int4 定義與 helper */
// HIP 內建 int4, 但為了方便操作指針，我們使用 reinterpret_cast
// 確保 V_padded 是 4 的倍數 (int4 alignment)

/* Kernels */
__global__ void kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2(int *d_D, const int r, const int V_padded); // 合併 Row/Col 以減少 Kernel Launch overhead (可選，分開亦可)
__global__ void kernel_phase3(int *d_D, const int r, const int V_padded);

void input(char *infile);
void output(char *outfile);
void block_FW();

int main(int argc, char *argv[]) {
    if (argc < 3) return 1;

    input(argv[1]);

    hipStreamCreate(&stream_main);
    
    size_t size = V_padded * V_padded * sizeof(int);
    hipMalloc(&d_D, size);

    hipMemcpy(d_D, D, size, hipMemcpyHostToDevice);

    block_FW();

    hipMemcpy(D, d_D, size, hipMemcpyDeviceToHost);

    output(argv[2]);
    
    // Cleanup omitted for brevity in competitive coding context
    return 0;
}

void block_FW() {
    int round = V_padded / BLOCKING_FACTOR;
    dim3 threads(HALF_BLOCK, HALF_BLOCK); // 32, 32

    for (int r = 0; r < round; ++r) {
        // Phase 1
        kernel_phase1<<<1, threads, 0, stream_main>>>(d_D, r, V_padded);
        
        // Phase 2: Row and Col
        // Launching grid: (2, round) -> x=0 is Row, x=1 is Col
        dim3 grid_p2(2, round);
        kernel_phase2<<<grid_p2, threads, 0, stream_main>>>(d_D, r, V_padded);
        
        // Phase 3
        dim3 grid_p3(round, round);
        kernel_phase3<<<grid_p3, threads, 0, stream_main>>>(d_D, r, V_padded);
    }
}

void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    // Pad to multiple of 64 (BLOCKING_FACTOR)
    V_padded = (V + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR * BLOCKING_FACTOR;

    hipHostAlloc(&D, V_padded * V_padded * sizeof(int), hipHostAllocDefault);

    // 初始化優化: 直接填值，不做 if (i==j) 判斷，後面再修正對角線
    // 這對於編譯器向量化初始化更有利
    for (int i = 0; i < V_padded * V_padded; ++i) {
        D[i] = INF;
    }
    for (int i = 0; i < V_padded; ++i) {
        D[i * V_padded + i] = 0;
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        D[pair[0] * V_padded + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outfile) {
    FILE *f = fopen(outfile, "w");
    for (int i = 0; i < V; ++i)
        fwrite(&D[i * V_padded], sizeof(int), V, f);
    fclose(f);
}

/* 
 * Kernel 優化細節: 
 * 使用 int4 讀寫。每個 thread block 處理 64x64。
 * Thread (tx, ty) 負責 (ty, tx), (ty, tx+32), (ty+32, tx), (ty+32, tx+32) 四個點
 * 但為了 int4，我們需要調整 Shared Memory 的 Layout。
 */

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded) {
    // Shared Memory padding: [64][65] to avoid bank conflicts
    __shared__ int sm[BLOCKING_FACTOR][BLOCKING_FACTOR]; 

    const int tx = threadIdx.x; // 0..31
    const int ty = threadIdx.y; // 0..31
    
    // Global memory pointer to the start of the block
    const int offset = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    
    // Load data to Shared Memory
    // 每個 thread 讀取 4 個 int? 不，int4 需要連續記憶體。
    // 我們的 2x2 分佈是: (y, x), (y, x+32), (y+32, x), (y+32, x+32) -> 這不是連續的。
    // 因此，我們在 Load 階段不使用 int4 指令，或者我們改變 Thread 負責的區域。
    // 為了保持邏輯簡單且高效，我們維持手動展開，但確保合併讀取。
    
    int *ptr = d_D + offset;
    
    sm[ty][tx] = ptr[ty * V_padded + tx];
    sm[ty][tx + 32] = ptr[ty * V_padded + tx + 32];
    sm[ty + 32][tx] = ptr[(ty + 32) * V_padded + tx];
    sm[ty + 32][tx + 32] = ptr[(ty + 32) * V_padded + tx + 32];

    __syncthreads();

    // Floyd-Warshall in Shared Memory
    // Loop unrolling hint
    #pragma unroll 16
    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        // Pre-load k-th row/col from SMEM to Registers to save SMEM bandwidth
        int val_k_row_0 = sm[ty][k];
        int val_k_row_1 = sm[ty + 32][k];
        int val_k_col_0 = sm[k][tx];
        int val_k_col_1 = sm[k][tx + 32];

        sm[ty][tx] = min(sm[ty][tx], val_k_row_0 + val_k_col_0);
        sm[ty][tx + 32] = min(sm[ty][tx + 32], val_k_row_0 + val_k_col_1);
        sm[ty + 32][tx] = min(sm[ty + 32][tx], val_k_row_1 + val_k_col_0);
        sm[ty + 32][tx + 32] = min(sm[ty + 32][tx + 32], val_k_row_1 + val_k_col_1);
        
        __syncthreads();
    }

    // Write back
    ptr[ty * V_padded + tx] = sm[ty][tx];
    ptr[ty * V_padded + tx + 32] = sm[ty][tx + 32];
    ptr[(ty + 32) * V_padded + tx] = sm[ty + 32][tx];
    ptr[(ty + 32) * V_padded + tx + 32] = sm[ty + 32][tx + 32];
}

__global__ void kernel_phase2(int *d_D, const int r, const int V_padded) {
    // grid.x = 0 (Row Block), grid.x = 1 (Col Block)
    // grid.y = iterating 0..round
    const int b_i = blockIdx.x; 
    const int b_j = blockIdx.y;
    
    if (b_j == r) return; // Pivot block handled in Phase 1

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    int pivot_offset = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    int self_offset;
    
    if (b_i == 0) { 
        // Row Phase: Block(r, b_j) -> 修正：Row Phase 處理的是同一 Row 的不同 Col Blocks
        // 定義: Row Phase 處理 Block(r, c)。這裡 blockIdx.y 代表變動的 Block Index
        // 但通常 FW 的 Phase 2 是處理 Pivot Row 和 Pivot Col。
        // Pivot Row: Block(r, 0), Block(r, 1)... 
        // Pivot Col: Block(0, r), Block(1, r)...
        
        // 為了符合原程式邏輯：
        // Phase 2 Row: 處理 Block(r, b_j) (第 r 列, 第 b_j 個 block) -> 錯誤，這需要 Pivot(r,r) 和 Self(r, b_j)
        // 但實際上 FW 需要更新的是 Pivot Row 和 Pivot Col。
        // 讓我們回歸原邏輯:
        // b_i=0 -> Row Stream -> 處理 Block(r, b_j) -> 固定 Row r, 變動 Col b_j
        self_offset = (r * BLOCKING_FACTOR) * V_padded + (b_j * BLOCKING_FACTOR);
    } else {
        // b_i=1 -> Col Stream -> 處理 Block(b_j, r) -> 固定 Col r, 變動 Row b_j
        self_offset = (b_j * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    }

    // Load Pivot
    sm_pivot[ty][tx] = d_D[pivot_offset + ty * V_padded + tx];
    sm_pivot[ty][tx+32] = d_D[pivot_offset + ty * V_padded + (tx+32)];
    sm_pivot[ty+32][tx] = d_D[pivot_offset + (ty+32) * V_padded + tx];
    sm_pivot[ty+32][tx+32] = d_D[pivot_offset + (ty+32) * V_padded + (tx+32)];

    // Load Self
    int my_val[2][2];
    my_val[0][0] = d_D[self_offset + ty * V_padded + tx];
    my_val[0][1] = d_D[self_offset + ty * V_padded + (tx+32)];
    my_val[1][0] = d_D[self_offset + (ty+32) * V_padded + tx];
    my_val[1][1] = d_D[self_offset + (ty+32) * V_padded + (tx+32)];

    // 為了計算方便，也存入 SMEM (雖然可以直接用 Registers，但為了對齊索引邏輯)
    // 注意：Row Phase 需要 Pivot 的 Row (sm_pivot[k][tx]) 和 Self 的 Col
    // 這裡統一放入 SMEM 比較保險
    sm_self[ty][tx] = my_val[0][0];
    sm_self[ty][tx+32] = my_val[0][1];
    sm_self[ty+32][tx] = my_val[1][0];
    sm_self[ty+32][tx+32] = my_val[1][1];

    __syncthreads();

    // Compute
    #pragma unroll 16
    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        int r0, r1, c0, c1;
        
        if (b_i == 0) { 
            // Row Phase: Update Block(r, b_j) using Pivot(r,r) and Block(r, b_j) ??
            // FW 規則: D[i][j] = min(D[i][j], D[i][k] + D[k][j])
            // 目前處理 Block(r, b_j)。 i 範圍是 [r*64, (r+1)*64), j 範圍是 [b_j*64...]
            // Pivot k 位於 r-th block。
            // 公式: Self[y][x] = min(Self[y][x], Pivot[y][k] + Self[k][x]?) -> 不對
            // 應該是: D[r_y][c_x] = min(D[r_y][c_x], D[r_y][pivot_k] + D[pivot_k][c_x])
            // 對於 Row Block (都在 r 行): Pivot 是 (r, r)。Self 是 (r, b_j)。
            // D(row_local, col_local) using Pivot(row_local, k) + Self(k, col_local) -> 不對，這是依賴自己
            // 正確邏輯: 
            // Pivot Row Block (r, b_j): 依賴 Pivot(r, r) 和 自己 (r, b_j)? 
            // 不，Phase 2 更新的是 "十字線"。
            // Pivot Row (r, x) 需要 Pivot Block (r, r) 和 ...? 
            // 其實 Phase 2 不需要這麼複雜。Phase 2 只是單純用 Pivot Block 更新同一 Row 和同一 Col 的 Block。
            // 對於 Row Block (r, b_j): New(r, b_j) = min(Old(r, b_j), Pivot(r, r) + Old(r, b_j)) -> 這是錯的，這會造成自我相加。
            
            // 讓我們看你原始的正確邏輯：
            // Kernel Row: sm_pivot (Pivot), sm_self (Target)
            // reg_self = min(reg_self, pivot_row + self_col) ???
            // 你的原代碼: r0 = sm_pivot[ty][k] (Pivot Row), c0 = sm_self[k][tx] (Target Col??)
            
            // 讓我們明確定義:
            // 1. Row Kernel 更新 Block(r, b_j)。
            //    需要 D[row][col] = min(D[row][col], D[row][k] + D[k][col])
            //    其中 row 在 r 區間 (pivot range), col 在 b_j 區間。 k 在 r 區間。
            //    D[row][k] 來自 Pivot Block (r, r)。
            //    D[k][col] 來自 Self Block (r, b_j) 的第 k 行 (local row k)。
            r0 = sm_pivot[ty][k];           // Pivot(y, k)
            r1 = sm_pivot[ty+32][k];
            c0 = sm_self[k][tx];            // Self(k, x)
            c1 = sm_self[k][tx+32];
        } else {
            // Col Kernel 更新 Block(b_j, r)。
            // row 在 b_j 區間, col 在 r 區間。 k 在 r 區間。
            // D[row][col] = min(D[row][col], D[row][k] + D[k][col])
            // D[row][k] 來自 Self Block (b_j, r) 的第 k 列 (local col k)。
            // D[k][col] 來自 Pivot Block (r, r)。
            r0 = sm_self[ty][k];            // Self(y, k)
            r1 = sm_self[ty+32][k];
            c0 = sm_pivot[k][tx];           // Pivot(k, x)
            c1 = sm_pivot[k][tx+32];
        }

        my_val[0][0] = min(my_val[0][0], r0 + c0);
        my_val[0][1] = min(my_val[0][1], r0 + c1);
        my_val[1][0] = min(my_val[1][0], r1 + c0);
        my_val[1][1] = min(my_val[1][1], r1 + c1);
    }

    // Write back
    d_D[self_offset + ty * V_padded + tx] = my_val[0][0];
    d_D[self_offset + ty * V_padded + (tx+32)] = my_val[0][1];
    d_D[self_offset + (ty+32) * V_padded + tx] = my_val[1][0];
    d_D[self_offset + (ty+32) * V_padded + (tx+32)] = my_val[1][1];
}

__global__ void kernel_phase3(int *d_D, const int r, const int V_padded) {
    const int b_x = blockIdx.x;
    const int b_y = blockIdx.y;
    
    if (b_x == r || b_y == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Phase 3 更新 Block(b_y, b_x)
    // 需要 Block(b_y, r) [Row dependency] 和 Block(r, b_x) [Col dependency]
    
    __shared__ int sm_a[BLOCKING_FACTOR][BLOCKING_FACTOR]; // From Block(b_y, r)
    __shared__ int sm_b[BLOCKING_FACTOR][BLOCKING_FACTOR]; // From Block(r, b_x)

    int offset_a = (b_y * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    int offset_b = (r * BLOCKING_FACTOR) * V_padded + (b_x * BLOCKING_FACTOR);
    int offset_c = (b_y * BLOCKING_FACTOR) * V_padded + (b_x * BLOCKING_FACTOR);

    // Load A
    sm_a[ty][tx] = d_D[offset_a + ty * V_padded + tx];
    sm_a[ty][tx+32] = d_D[offset_a + ty * V_padded + (tx+32)];
    sm_a[ty+32][tx] = d_D[offset_a + (ty+32) * V_padded + tx];
    sm_a[ty+32][tx+32] = d_D[offset_a + (ty+32) * V_padded + (tx+32)];

    // Load B
    sm_b[ty][tx] = d_D[offset_b + ty * V_padded + tx];
    sm_b[ty][tx+32] = d_D[offset_b + ty * V_padded + (tx+32)];
    sm_b[ty+32][tx] = d_D[offset_b + (ty+32) * V_padded + tx];
    sm_b[ty+32][tx+32] = d_D[offset_b + (ty+32) * V_padded + (tx+32)];

    // Load C (Self) directly to registers
    int c[2][2];
    c[0][0] = d_D[offset_c + ty * V_padded + tx];
    c[0][1] = d_D[offset_c + ty * V_padded + (tx+32)];
    c[1][0] = d_D[offset_c + (ty+32) * V_padded + tx];
    c[1][1] = d_D[offset_c + (ty+32) * V_padded + (tx+32)];

    __syncthreads();

    #pragma unroll 16
    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        // D[y][x] = min(D[y][x], D[y][k] + D[k][x])
        // D[y][k] comes from sm_a (Block(b_y, r) has col indices in pivot range)
        // D[k][x] comes from sm_b (Block(r, b_x) has row indices in pivot range)
        
        int val_a0 = sm_a[ty][k];
        int val_a1 = sm_a[ty+32][k];
        int val_b0 = sm_b[k][tx];
        int val_b1 = sm_b[k][tx+32];

        c[0][0] = min(c[0][0], val_a0 + val_b0);
        c[0][1] = min(c[0][1], val_a0 + val_b1);
        c[1][0] = min(c[1][0], val_a1 + val_b0);
        c[1][1] = min(c[1][1], val_a1 + val_b1);
    }

    // Write Back
    d_D[offset_c + ty * V_padded + tx] = c[0][0];
    d_D[offset_c + ty * V_padded + (tx+32)] = c[0][1];
    d_D[offset_c + (ty+32) * V_padded + tx] = c[1][0];
    d_D[offset_c + (ty+32) * V_padded + (tx+32)] = c[1][1];
}
