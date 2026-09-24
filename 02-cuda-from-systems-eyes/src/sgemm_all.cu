// sgemm_all.cu —— 第 3~6 章的七个 kernel + cuBLAS 基准，同一台机器一次跑完
//
//   ⚠ 这是唯一的源文件。原先 P0 补测用的 sgemm_all_k0.cu 已并入这里 ——
//     两份源文件意味着两个二进制、两次编译，而"占 cuBLAS 百分之多少"
//     这类跨 kernel 的比值，只有在同一个二进制、同一次开机下才成立。
//
//   k0   cuBLAS 基准（不是我们写的 kernel，是靶子）                       —— 上限参照
//   k1   第 3 章 Fig. 3.11：一个线程一个输出，索引映射故意写反 → 不合并   —— 基线
//   k2   第 4 章：把映射换回来，其余一字不改                              —— 只动合并
//   k3   第 5 章：shared 分块，每线程 1 个输出                            —— 只动字节
//   k3c  第 6 章 Fig. 6.13 的线程粗化（每线程 4 个输出，M tile 复用）     —— 砍字节，不砍条数
//   k4   1D 寄存器粗化（每线程 8 行，N 元素进寄存器用 8 次）              —— 砍条数，不砍字节
//   k5   2D 寄存器粗化（每线程 8 行 x 4 列 = 32 个输出）                  —— 两本账一起动
//   k6   在 k5 上加 float4 向量化访存                                     —— 只砍指令条数
//
//   为什么要合并成一份：K1/K2/K3 原先分散在三章、三个可执行文件、三次开机跑的，
//   机器状态（时钟、温度、驱动、后台占用）都不保证一致。放进同一个程序、同一个脚本，
//   七个 kernel 的数字才真正可比。
//
//   编译:   nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v sgemm_all.cu -o sgemm_all -lcublas
//             ↑ -lcublas 必须放在源文件后面（链接顺序有讲究），k0 要用
//   运行:   ./sgemm_all <kernel> [N] [ITERS]        方阵，M = N = K
//           ./sgemm_all <kernel> M N K [ITERS]      任意形状（prefill / decode）
//           kernel = k0|k1|k2|k3|k3c|k4|k5|k6       默认 N = 4096，ITERS = 11
//   ⚠ 跑 k0 前先 export NVIDIA_TF32_OVERRIDE=0，否则 cuBLAS 可能偷偷用 TF32
//   查竞态: compute-sanitizer --tool racecheck ./sgemm_all k4 128 1
//   注意    k6 要求 N、K 是 4 的倍数（向量化访存要 16 字节对齐）
//   注意    k1 在 N=8192 上约 1.6 s/次，扫大尺寸时把 ITERS 调小

// ---- 给不熟悉 CUDA / C++ 的读者：这份文件用到的语法，一次讲清 ----------------
//
//   __global__ void f(...)   kernel：在 GPU 上跑、由 CPU 调用。返回类型必须是 void，
//                            "结果"只能写进传进去的指针里。
//   __device__ float g(...)  只能在 GPU 上被调用的辅助函数（本文件没有，第 10 章那份有）。
//   __shared__ float s[N]    block 内共享的片上内存：同 block 的线程都看得见，
//                            block 之间互相看不见，kernel 一结束就没了。
//   f<<<grid, block>>>(...)  启动 kernel。grid = 有几个 block，block = 每个 block 几个线程。
//                            两者都是 dim3（三维），dim3(32,4) 即 x=32, y=4, z=1。
//   threadIdx.x / .y         我在 block 内的坐标；blockIdx 是我这个 block 在 grid 内的坐标；
//                            blockDim 是 block 的尺寸。三个都是 CUDA 自动提供的内置变量。
//   __syncthreads()          屏障：同一 block 的线程在这里等齐。要求所有线程都能到达 ——
//                            写在 if 里面就可能死锁。
//   #pragma unroll           让编译器展开循环。循环次数必须是编译期常量才有效；展开后
//                            数组下标变成常量，acc[r][j] 才进得了寄存器而不是显存。
//   const float *A           A 指向的内容不可改（A 自己指向哪可以改）。既是给读者的契约，
//                            也让编译器敢做更多优化。
//   (size_t)M * K            先转 64 位再乘 —— 两个 int 相乘会按 int 算，4096*4096*4 直接溢出。
//                            这是 CUDA 代码最常见的坑之一。
//   float4 / make_float4     16 字节的向量类型，一条指令搬 4 个 float（k6 用它）。
//   reinterpret_cast<T*>(p)  把指针按另一种类型重新解释：不做转换，一个 bit 都不改。
//   cudaMalloc(&dA, bytes)   注意那个 & —— 它要改的是"dA 这个指针本身"，所以得把指针的
//                            地址传进去。
//   CUDA_CHECK(...)          本文件自定义的宏，每个 CUDA 调用包一层，出错立刻打印行号退出。
//                            CUDA 的错误默认是静默的，不查就只会看到结果不对。
//   static_assert(条件, 说明) 编译期断言，条件不成立就编译不过 —— 用来锁住那些
//                            "改了这个常量就必须同步改 kernel"的约定。
//
// -----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>   // K0 用

#define TILE    32      // 五个 kernel 共用的 tile 边长，也是 K 方向每阶段的步数
#define COARSE   4      // k3c：每线程负责同一行上的 4 个输出（跨 4 个 tile）
#define CF       8      // k4/k5/k6：每线程负责同一列上的 8 行
#define TN       4      // k5/k6：每线程负责的列数

// 宏体写成 do { ... } while (0)：这是 C 的老惯例，为的是让整个宏在语法上
// 等价于"一条语句"，于是 if (x) CUDA_CHECK(...); else ... 也不会散架。
// 行尾那些反斜杠是续行符，表示"这一行还没完"。
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
        printf("[CUDA 错误] %s:%d  %s\n  ↳ %s\n",                              \
               __FILE__, __LINE__, cudaGetErrorString(err_), #call);           \
        exit(1);                                                               \
    }                                                                          \
} while (0)

// cuBLAS 的返回值是 cublasStatus_t，不是 cudaError_t，所以要单独包一个宏。
// 写法和 CUDA_CHECK 一样：do{...}while(0) 让它在语法上等价于一条语句。
#define CUBLAS_CHECK(call) do {                                                \
    cublasStatus_t st_ = (call);                                               \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
        printf("[cuBLAS 错误] %s:%d  status=%d\\n  ↳ %s\\n",                     \
               __FILE__, __LINE__, (int)st_, #call);                           \
        exit(1);                                                               \
    }                                                                          \
} while (0)

// ---------------------------------------------------------------------------
// K0 · cuBLAS 基准 —— 这不是 kernel，是一次库调用，所以单独写两个函数
//
// 坑 ①：cuBLAS 认为矩阵是【列主序】，我们的 A/B/C 全是行主序。
//       不做任何转置拷贝的解法 —— 利用这个恒等式：
//           行主序的 C = A x B   <=>   列主序看过去就是  C^T = B^T x A^T
//       因为"一个行主序矩阵，按列主序去读，读到的就是它的转置"。
//       所以把 B、A 对调着传进去，cuBLAS 写回的内存布局正好就是我们要的行主序 C。
//
// 坑 ②：Ampere 起 cuBLAS 对 FP32 GEMM 默认可能走 TF32（10 位尾数），
//       那样和我们纯 FP32 的 k1..k6 就不是 apples-to-apples 了。
//       两道保险：CUBLAS_PEDANTIC_MATH + 环境变量 NVIDIA_TF32_OVERRIDE=0。
// ---------------------------------------------------------------------------
static cublasHandle_t g_cublas = nullptr;      // 全局 handle，建一次用到底

static void k0_init(void) {
    CUBLAS_CHECK(cublasCreate(&g_cublas));
    // 禁 TF32（坑 ②）：强制走真正的 FP32 管线
    CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_PEDANTIC_MATH));
}

static void k0_launch(int M, int N, int K, float alpha,
                      const float *A, const float *B, float beta, float *C) {
    // 列主序换位法（坑 ①）：
    //   三个维度按"转置后"的形状填：m = N, n = M, k = K
    //   三个 leading dimension 填"行主序下一行有几个元素"：B->N, A->K, C->N
    CUBLAS_CHECK(cublasSgemm(g_cublas,
                             CUBLAS_OP_N, CUBLAS_OP_N,   // 都不转置（转置靠换位实现了）
                             N, M, K,                    // <- m, n, k：注意 N 在前
                             &alpha,
                             B, N,                       // <- 第一个矩阵传 B，lda = N
                             A, K,                       // <- 第二个矩阵传 A，ldb = K
                             &beta,
                             C, N));                     //    结果写回 C，ldc = N
}

static int coresPerSM(int major, int minor) {
    switch (major * 10 + minor) {
        case 70: case 72: case 75: case 80: return 64;
        case 86: case 87: case 89:          return 128;
        case 90: case 100: case 120:        return 128;
        default:                            return 0;
    }
}

// ---------------------------------------------------------------------------
// K1 · naive —— 第 3 章 Fig. 3.11 原样，一字未改
//
//   这里把 x 当「行」、y 当「列」—— 和书里正好相反，这不是笔误：
//   同一个 warp 里 threadIdx.x 连续，于是相邻线程算的是 C 的相邻「行」，
//   地址相隔 N 个 float —— A 的读和 C 的写都完全不合并。
//   A[x*K+i] 一条 LDG 展开成 32 个独立请求；B[i*N+y] 在 warp 内 y 相同 → 广播 1 个请求。
//   合计 33 个请求 / FMA，这就是 K1 慢的全部原因。
// ---------------------------------------------------------------------------
__global__ void sgemm_k1(int M, int N, int K, float alpha,
                         const float *A, const float *B, float beta, float *C) {
    // 这两行是 CUDA 里最常见的一句话："我是谁"。
    // blockIdx 是我这个 block 在 grid 里的编号，blockDim 是 block 的宽度，
    // threadIdx 是我在 block 内的编号 —— 三者拼出全局唯一的坐标。
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // 行（M 方向）
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // 列（N 方向）

    if (x < M && y < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i)
            acc += A[x * K + i] * B[i * N + y];            // A 走一行，B 跨一列
        C[x * N + y] = alpha * acc + beta * C[x * N + y];
    }
}

// ---------------------------------------------------------------------------
// K2 · 合并 —— 第 4 章原样，一字未改
//
//   和 K1 的差别只有两行：x 和 y 的来源对调。
//   y（列）来自 threadIdx.x → warp 内连续 → B 的读、C 的写合并成 4 个 32B 事务；
//   x（行）来自 threadIdx.y → warp 内相同 → A 的读是广播，1 个请求。
//   请求数从 33 降到 2，而字节账、指令条数账一个字都没变。
// ---------------------------------------------------------------------------
__global__ void sgemm_k2(int M, int N, int K, float alpha,
                         const float *A, const float *B, float beta, float *C) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;   // 列（N 方向）
    const int x = blockIdx.y * blockDim.y + threadIdx.y;   // 行（M 方向）

    if (x < M && y < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i)
            acc += A[x * K + i] * B[i * N + y];
        C[x * N + y] = alpha * acc + beta * C[x * N + y];
    }
}

// ---------------------------------------------------------------------------
// K3 · shared 分块 —— 第 5 章原样，一字未改
// ---------------------------------------------------------------------------
__global__ void sgemm_k3(int M, int N, int K, float alpha,
                         const float *A, const float *B, float beta, float *C) {
    __shared__ float Mds[TILE][TILE];
    __shared__ float Nds[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int Col = blockIdx.x * TILE + tx;
    const int Row = blockIdx.y * TILE + ty;

    float acc = 0.0f;

    for (int ph = 0; ph < (K + TILE - 1) / TILE; ++ph) {
        const int aCol = ph * TILE + tx;
        const int bRow = ph * TILE + ty;

        Mds[ty][tx] = (Row < M && aCol < K) ? A[Row * K + aCol] : 0.0f;
        Nds[ty][tx] = (bRow < K && Col < N) ? B[bRow * N + Col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += Mds[ty][k] * Nds[k][tx];

        __syncthreads();
    }

    if (Row < M && Col < N)
        C[Row * N + Col] = alpha * acc + beta * C[Row * N + Col];
}

// ---------------------------------------------------------------------------
// K3c · 书 Fig. 6.13 的线程粗化，补上三处边界检查
//
//   block 仍是 32×32；线程 (ty,tx) 负责第 Row 行上的 4 个输出：
//   col = colStart + c*TILE，c = 0..3。M 的 tile 每阶段只搬一次，4 个 c 共用。
//
//   ⚠ 书里的 float Pvalue[COARSE_FACTOR] 是自动数组 → local memory（Table 5.1）。
//     这里换成 4 个标量寄存器，每个 c 结束时轮转一次：4 轮之后顺序复原。
// ---------------------------------------------------------------------------
__global__ void sgemm_k3c(int M, int N, int K, float alpha,
                          const float *A, const float *B, float beta, float *C) {
    __shared__ float Mds[TILE][TILE];
    __shared__ float Nds[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int Row      = blockIdx.y * TILE + ty;
    const int colStart = blockIdx.x * TILE * COARSE + tx;     // Fig. 6.13 第 13 行

    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;         // 代替 Pvalue[4]

    for (int ph = 0; ph < (K + TILE - 1) / TILE; ++ph) {
        const int aCol = ph * TILE + tx;
        const int bRow = ph * TILE + ty;

        // ---- M tile：每阶段只搬一次 —— 粗化省下的就是这里 ----
        Mds[ty][tx] = (Row < M && aCol < K) ? A[Row * K + aCol] : 0.0f;

        for (int c = 0; c < COARSE; ++c) {                     // 粗化循环
            const int col = colStart + c * TILE;

            // ---- N tile：随列变，每个 c 各搬一次 ----
            Nds[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

            __syncthreads();                                   // ① 写后读

            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < TILE; ++k)
                acc += Mds[ty][k] * Nds[k][tx];                // 内层和 K3 一字不差

            __syncthreads();                                   // ② 读后写

            // 寄存器轮转：p0 加上本轮结果后排到队尾
            const float t = p0 + acc;
            p0 = p1; p1 = p2; p2 = p3; p3 = t;
        }
    }

    if (Row < M) {
        const int c0 = colStart, c1 = c0 + TILE, c2 = c1 + TILE, c3 = c2 + TILE;
        if (c0 < N) C[Row * N + c0] = alpha * p0 + beta * C[Row * N + c0];
        if (c1 < N) C[Row * N + c1] = alpha * p1 + beta * C[Row * N + c1];
        if (c2 < N) C[Row * N + c2] = alpha * p2 + beta * C[Row * N + c2];
        if (c3 < N) C[Row * N + c3] = alpha * p3 + beta * C[Row * N + c3];
    }
}

// ---------------------------------------------------------------------------
// K4 · 1D 寄存器粗化
//
//   block 32×4：一个 warp = 同一个 ty 的 32 个线程，所有全局地址以 tx 结尾 → 合并。
//   线程 (ty,tx) 负责第 Col 列上的 8 个输出：行 Row0..Row0+7。
//   内层每步读 1 个 N 进寄存器，配 8 个 M 用 8 次：load/FMA = 9/8。
//
//   8 个累加器写成 8 个标量，不用数组 —— 理由同 K3c。
// ---------------------------------------------------------------------------
__global__ void sgemm_k4(int M, int N, int K, float alpha,
                         const float *A, const float *B, float beta, float *C) {
    __shared__ float Mds[TILE][TILE];
    __shared__ float Nds[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int Col  = blockIdx.x * TILE + tx;                  // 列：tx 连续
    const int i0   = ty * CF;                                 // tile 内第 i0..i0+7 行归我
    const int Row0 = blockIdx.y * TILE + i0;                  // 全局第 Row0..Row0+7 行

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f,
          a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    for (int ph = 0; ph < (K + TILE - 1) / TILE; ++ph) {
        const int aCol = ph * TILE + tx;

        // ---- 协同加载：128 个线程 × 每人 16 格 = 两块 32×32 的 tile ----
        for (int r = 0; r < CF; ++r) {
            const int row  = Row0 + r;
            const int bRow = ph * TILE + i0 + r;
            Mds[i0 + r][tx] = (row  < M && aCol < K) ? A[row  * K + aCol] : 0.0f;
            Nds[i0 + r][tx] = (bRow < K && Col  < N) ? B[bRow * N + Col ] : 0.0f;
        }

        __syncthreads();                                      // ① 写后读

        // ---- 内层：1 个 N 进寄存器，用 8 次 ----
        for (int k = 0; k < TILE; ++k) {
            const float n = Nds[k][tx];
            a0 += Mds[i0    ][k] * n;
            a1 += Mds[i0 + 1][k] * n;
            a2 += Mds[i0 + 2][k] * n;
            a3 += Mds[i0 + 3][k] * n;
            a4 += Mds[i0 + 4][k] * n;
            a5 += Mds[i0 + 5][k] * n;
            a6 += Mds[i0 + 6][k] * n;
            a7 += Mds[i0 + 7][k] * n;
        }

        __syncthreads();                                      // ② 读后写
    }

    if (Col < N) {
        if (Row0     < M) C[(Row0    ) * N + Col] = alpha * a0 + beta * C[(Row0    ) * N + Col];
        if (Row0 + 1 < M) C[(Row0 + 1) * N + Col] = alpha * a1 + beta * C[(Row0 + 1) * N + Col];
        if (Row0 + 2 < M) C[(Row0 + 2) * N + Col] = alpha * a2 + beta * C[(Row0 + 2) * N + Col];
        if (Row0 + 3 < M) C[(Row0 + 3) * N + Col] = alpha * a3 + beta * C[(Row0 + 3) * N + Col];
        if (Row0 + 4 < M) C[(Row0 + 4) * N + Col] = alpha * a4 + beta * C[(Row0 + 4) * N + Col];
        if (Row0 + 5 < M) C[(Row0 + 5) * N + Col] = alpha * a5 + beta * C[(Row0 + 5) * N + Col];
        if (Row0 + 6 < M) C[(Row0 + 6) * N + Col] = alpha * a6 + beta * C[(Row0 + 6) * N + Col];
        if (Row0 + 7 < M) C[(Row0 + 7) * N + Col] = alpha * a7 + beta * C[(Row0 + 7) * N + Col];
    }
}

// ---------------------------------------------------------------------------
// K5 · 2D 寄存器粗化
//
//   block 仍是 32×4 = 128 个线程；输出块 BM = TILE = 32 行 × BN = TILE*TN = 128 列。
//   线程 (ty,tx) 负责 CF = 8 行 × TN = 4 列 = 32 个输出：
//     行 = Row0 + r          （r = 0..7）
//     列 = colBase + j*TILE + tx （j = 0..3，每隔 TILE 列取一个，保证读 Nds 仍是 32 个连号）
//   内层每步 k：8 条广播 + 4 条 32 路 → 32 个 FMA。
//
//   ⚠ acc[CF][TN] 是自动数组，靠 #pragma unroll 把下标变成常量才能进寄存器；
//     编译后必须确认 0 bytes stack frame。
// ---------------------------------------------------------------------------
__global__ void sgemm_k5(int M, int N, int K, float alpha,
                         const float *A, const float *B, float beta, float *C) {
    __shared__ float Mds[TILE][TILE];               //  4 KB
    __shared__ float Nds[TILE][TILE * TN];          // 16 KB

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i0      = ty * CF;
    const int Row0    = blockIdx.y * TILE + i0;
    const int colBase = blockIdx.x * TILE * TN;

    float acc[CF][TN];
    #pragma unroll
    for (int r = 0; r < CF; ++r)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[r][j] = 0.0f;

    for (int ph = 0; ph < (K + TILE - 1) / TILE; ++ph) {
        const int aCol = ph * TILE + tx;

        // ---- 搬 A 的 tile：每线程 8 格（和 K4 一样）----
        #pragma unroll
        for (int r = 0; r < CF; ++r) {
            const int row = Row0 + r;
            Mds[i0 + r][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        }

        // ---- 搬 B 的 tile：每线程 8 行 × 4 段 = 32 格，段内仍以 tx 结尾 → 合并 ----
        #pragma unroll
        for (int r = 0; r < CF; ++r) {
            const int bRow = ph * TILE + i0 + r;
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = colBase + j * TILE + tx;
                Nds[i0 + r][j * TILE + tx] =
                    (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
            }
        }

        __syncthreads();                            // ① 写后读

        // ---- 内层：一次 8×4 的外积 ----
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float m[CF], n[TN];
            #pragma unroll
            for (int r = 0; r < CF; ++r) m[r] = Mds[i0 + r][k];          // 8 条广播
            #pragma unroll
            for (int j = 0; j < TN; ++j) n[j] = Nds[k][j * TILE + tx];   // 4 条 32 路

            #pragma unroll
            for (int r = 0; r < CF; ++r)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[r][j] += m[r] * n[j];                            // 32 个 FMA
        }

        __syncthreads();                            // ② 读后写
    }

    #pragma unroll
    for (int r = 0; r < CF; ++r) {
        const int row = Row0 + r;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = colBase + j * TILE + tx;
            if (col < N) C[row * N + col] = alpha * acc[r][j] + beta * C[row * N + col];
        }
    }
}

// ---------------------------------------------------------------------------
// K6 · 在 K5 上加 float4 向量化访存
//
//   两处结构变化：
//     ① A 的 tile 转置存成 Ads[k][行] —— 同一步 k 要用的 8 个 m 才连续，能合成 float4
//     ② 每线程改回「连续的 4 列」col0..col0+3 —— n[0..3] 连续，写回 C 也能用 float4
//   内层每步 k：2 条 128 位广播 + 1 条 128 位 32 路 → 32 个 FMA。
//
//   ⚠ 要求 N % 4 == 0 且 K % 4 == 0；main 里会先检查。
//     真实的库用另一个标量 kernel 处理边角，这里直接拒绝。
// ---------------------------------------------------------------------------
__global__ void sgemm_k6(int M, int N, int K, float alpha,
                         const float *A, const float *B, float beta, float *C) {
    __shared__ __align__(16) float Ads[TILE][TILE];           // 转置存：Ads[k][行]
    __shared__ __align__(16) float Bds[TILE][TILE * TN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid     = ty * TILE + tx;                       // 0..127
    const int i0      = ty * CF;
    const int Row0    = blockIdx.y * TILE + i0;
    const int colBase = blockIdx.x * TILE * TN;
    const int col0    = colBase + tx * TN;                    // 连续 4 列

    float acc[CF][TN];
    #pragma unroll
    for (int r = 0; r < CF; ++r)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[r][j] = 0.0f;

    const int phases = (K + TILE - 1) / TILE;
    for (int ph = 0; ph < phases; ++ph) {
        // ---- 搬 A：TILE×TILE 个 float = 256 个 float4，128 线程各 2 条 ----
        #pragma unroll
        for (int s = 0; s < (TILE * TILE / 4) / 128; ++s) {
            const int idx4 = tid + s * 128;                   // 0..255
            const int r    = idx4 / (TILE / 4);               // tile 内行号 0..31
            const int c    = (idx4 % (TILE / 4)) * 4;         // tile 内列号，4 的倍数
            const int row  = blockIdx.y * TILE + r;
            const int aCol = ph * TILE + c;
            float4 a4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row < M && aCol + 3 < K)
                a4 = *reinterpret_cast<const float4 *>(&A[row * K + aCol]);
            Ads[c + 0][r] = a4.x;   Ads[c + 1][r] = a4.y;     // 转置写入
            Ads[c + 2][r] = a4.z;   Ads[c + 3][r] = a4.w;
        }

        // ---- 搬 B：TILE×(TILE*TN) 个 float = 1024 个 float4，128 线程各 8 条 ----
        #pragma unroll
        for (int s = 0; s < (TILE * TILE * TN / 4) / 128; ++s) {
            const int idx4 = tid + s * 128;                   // 0..1023
            const int r    = idx4 / (TILE * TN / 4);          // tile 内行号 0..31
            const int c    = (idx4 % (TILE * TN / 4)) * 4;    // tile 内列号 0..124
            const int bRow = ph * TILE + r;
            const int col  = colBase + c;
            float4 b4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (bRow < K && col + 3 < N)
                b4 = *reinterpret_cast<const float4 *>(&B[bRow * N + col]);
            *reinterpret_cast<float4 *>(&Bds[r][c]) = b4;
        }

        __syncthreads();                                      // ① 写后读

        // ---- 内层：每步 3 条 128 位读 + 32 个 FMA ----
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            // reinterpret_cast 把"4 个连续 float 的地址"重新看作"1 个 float4 的地址"，
            // 前面的 * 再把它读出来 —— 一条 128 位指令顶四条 32 位。
            // 前提是地址必须 16 字节对齐，所以 Ads/Bds 声明时加了 __align__(16)。
            const float4 m0 = *reinterpret_cast<const float4 *>(&Ads[k][i0    ]);  // 广播
            const float4 m1 = *reinterpret_cast<const float4 *>(&Ads[k][i0 + 4]);  // 广播
            const float4 n0 = *reinterpret_cast<const float4 *>(&Bds[k][tx * TN]); // 32 路
            const float m[CF] = { m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w };
            const float n[TN] = { n0.x, n0.y, n0.z, n0.w };
            #pragma unroll
            for (int r = 0; r < CF; ++r)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[r][j] += m[r] * n[j];
        }

        __syncthreads();                                      // ② 读后写
    }

    // ---- 写回：每线程 8 行 × 一条 float4 ----
    #pragma unroll
    for (int r = 0; r < CF; ++r) {
        const int row = Row0 + r;
        if (row >= M || col0 + TN > N) continue;
        const float4 old = *reinterpret_cast<const float4 *>(&C[row * N + col0]);
        float4 out;
        out.x = alpha * acc[r][0] + beta * old.x;
        out.y = alpha * acc[r][1] + beta * old.y;
        out.z = alpha * acc[r][2] + beta * old.z;
        out.w = alpha * acc[r][3] + beta * old.w;
        *reinterpret_cast<float4 *>(&C[row * N + col0]) = out;
    }
}

// 抽样校验：和前几章完全一样
static int verifySamples(int M, int N, int K, float alpha, float beta,
                         const std::vector<float> &A, const std::vector<float> &B,
                         const std::vector<float> &Cin, const std::vector<float> &Cout,
                         int samples) {
    int bad = 0;
    srand(12345);
    for (int s = 0; s < samples; ++s) {
        int r = rand() % M, c = rand() % N;
        double acc = 0.0;
        for (int i = 0; i < K; ++i)
            acc += (double)A[r * K + i] * (double)B[i * N + c];
        double want = alpha * acc + beta * (double)Cin[r * N + c];
        double got  = Cout[r * N + c];
        double tol  = 1e-3 * std::max(1.0, std::fabs(want));
        if (std::fabs(got - want) > tol) {
            if (bad < 3)
                printf("  ✗ 不符 @(%d,%d): GPU %.4f vs CPU %.4f\n", r, c, got, want);
            bad++;
        }
    }
    return bad;
}

enum Kind { KIND_K0, KIND_K1, KIND_K2, KIND_K3, KIND_K3C, KIND_K4, KIND_K5, KIND_K6 };

static void launch(Kind kind, dim3 grid, dim3 block, int M, int N, int K, float alpha,
                   const float *A, const float *B, float beta, float *C) {
    // <<<grid, block>>> 是 CUDA 独有的启动语法（普通 C++ 编译器看不懂，所以要用 nvcc）。
    // 调用会立刻返回，kernel 在 GPU 上异步跑 —— 所以计时必须靠 cudaEvent 或 Synchronize。
    switch (kind) {
        case KIND_K0:  k0_launch(M, N, K, alpha, A, B, beta, C); break;   // 不用 <<<>>>
        case KIND_K1:  sgemm_k1 <<<grid, block>>>(M, N, K, alpha, A, B, beta, C); break;
        case KIND_K2:  sgemm_k2 <<<grid, block>>>(M, N, K, alpha, A, B, beta, C); break;
        case KIND_K3:  sgemm_k3 <<<grid, block>>>(M, N, K, alpha, A, B, beta, C); break;
        case KIND_K3C: sgemm_k3c<<<grid, block>>>(M, N, K, alpha, A, B, beta, C); break;
        case KIND_K4:  sgemm_k4 <<<grid, block>>>(M, N, K, alpha, A, B, beta, C); break;
        case KIND_K5:  sgemm_k5 <<<grid, block>>>(M, N, K, alpha, A, B, beta, C); break;
        case KIND_K6:  sgemm_k6 <<<grid, block>>>(M, N, K, alpha, A, B, beta, C); break;
    }
}

int main(int argc, char **argv) {
    const char *name = (argc > 1) ? argv[1] : "k4";

    // 两种调用形式，靠参数个数区分：
    //   ./sgemm_all k5 4096 11          方阵（M=N=K=4096，11 次迭代）—— 向后兼容
    //   ./sgemm_all k5 32 18944 3584 5  任意形状（M N K，5 次迭代）
    // 生产里的 GEMM 几乎从不是方阵：decode 阶段 M 可能只有 1，而 N/K 是几千 —— 见导学 §5
    int M, N, K, ITERS;
    if (argc >= 5) {                                // 给了三个维度
        M = atoi(argv[2]);
        N = atoi(argv[3]);
        K = atoi(argv[4]);
        ITERS = (argc > 5) ? atoi(argv[5]) : 11;
    } else {                                        // 只给一个 → 当方阵
        N = (argc > 2) ? atoi(argv[2]) : 4096;
        M = K = N;
        ITERS = (argc > 3) ? atoi(argv[3]) : 11;
    }
    // 预热次数。只有 ITERS==1 才允许 0 —— 那是 racecheck 专用（./sgemm_all k3c 128 1），
    // 它只关心有没有竞态，不关心耗时。
    //
    // ⚠ 这里踩过一次坑：原来写的是 (ITERS <= 2) ? 0 : 3，于是 512 那趟正确性检查
    //   （ITERS=2）完全没有预热。自己写的 kernel 冷启动只差几十微秒无所谓，
    //   但 cuBLAS 第一次调 cublasSgemm 要惰性加载它那个巨大的 kernel 库，
    //   一次几十毫秒 —— K0 @ 512 因此报出了 55 ms / 4.9 GFLOP/s 这种荒唐数字。
    int WARMUP = (ITERS <= 1) ? 0 : (ITERS <= 2 ? 1 : 3);
    const float alpha = 1.0f, beta = 0.0f;

    Kind kind;
    if      (!strcmp(name, "k0"))  kind = KIND_K0;
    else if (!strcmp(name, "k1"))  kind = KIND_K1;
    else if (!strcmp(name, "k2"))  kind = KIND_K2;
    else if (!strcmp(name, "k3"))  kind = KIND_K3;
    else if (!strcmp(name, "k3c")) kind = KIND_K3C;
    else if (!strcmp(name, "k4"))  kind = KIND_K4;
    else if (!strcmp(name, "k5"))  kind = KIND_K5;
    else if (!strcmp(name, "k6"))  kind = KIND_K6;
    else { printf("用法: %s <kernel> [N] [ITERS]           方阵 M=N=K=N\n"
               "      %s <kernel> M N K [ITERS]         任意形状\n"
               "  kernel = k0|k1|k2|k3|k3c|k4|k5|k6\n"
               "  例： %s k5 4096 11              方阵基准\n"
               "       %s k5 2048 18944 3584 11   Qwen2.5-7B prefill 的 FFN up\n"
               "       %s k5 32 18944 3584 11     batch=32 decode（M 只有 32）\n",
               argv[0], argv[0], argv[0], argv[0], argv[0]); return 2; }

    // static_assert 在编译期检查，不成立就编译不过（运行时一点开销都没有）。
    // 这四条锁的都是"手写死了的常量"：改了它们就必须同步改 kernel 里的代码。
    static_assert(TILE * TILE <= 1024, "block 最多 1024 个线程，所以 TILE ≤ 32");
    static_assert(TILE % CF == 0,      "k4/k5/k6 要求 TILE 是 CF 的整数倍");
    static_assert(COARSE == 4,         "k3c 的 p0..p3 轮转和写回按 4 手写：改 COARSE 必须同步改 kernel");
    static_assert(CF == 8,             "k4 的 a0..a7、k6 的 m[] 拼装按 8 手写：改 CF 必须同步改 kernel");
    static_assert(TN == 4,             "k6 的 float4 读写按 TN = 4 写死");

    if (M <= 0 || N <= 0 || K <= 0) {
        printf("M/N/K 必须都是正数（拿到的是 %d %d %d）\n", M, N, K);
        return 2;
    }

    if (kind == KIND_K6 && (N % 4 || K % 4)) {
        printf("k6 需要 N、K 是 4 的倍数（向量化访存要 16 字节对齐）；边界测试请用 4100\n");
        return 2;
    }

    // cublasCreate 第一次调用要花掉几百毫秒（分配 workspace、加载 kernel），
    // 所以放在这里，不能放进计时循环
    if (kind == KIND_K0) k0_init();

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int memClkKHz = 0, gpuClkKHz = 0;
    cudaDeviceGetAttribute(&memClkKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&gpuClkKHz, cudaDevAttrClockRate, 0);
    cudaGetLastError();

    double l2MB   = prop.l2CacheSize / 1.0e6;
    double peakBW = memClkKHz * 2.0 * (prop.memoryBusWidth / 8) / 1.0e6;
    int    cps    = coresPerSM(prop.major, prop.minor);
    double peakTF = cps * prop.multiProcessorCount * 2.0 * gpuClkKHz * 1e3 / 1e12;

    printf("GPU: %s  sm_%d%d  %d SM  L2 %.1f MB   时钟 %.2f GHz\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, l2MB, gpuClkKHz / 1e6);

    // ---- 每个 kernel 的 block / grid，以及三本账的理论值 ----
    //      dist  = 每个 FMA 的「32 路」shared load 条数
    //      bcast = 每个 FMA 的「广播」shared load 条数
    //      ldg   = 每个 FMA 的全局 load 条数
    //
    //      bytesPerOut = 每个输出元素摊到多少字节的"名义"全局访存
    //
    // 这五个量怎么从 kernel 代码一步步推出来（以及"名义"两个字的分量），
    // 见导学 §14 · 七「七个指标是怎么算出来的」—— 那里有逐行的推导。
    dim3 block, grid;
    double bytesPerOut, dist, bcast, ldg;
    double reqPerFMA = 0.0;          // 只对 k1/k2 有意义：内存"请求"数（不是指令数）
    size_t smemPerBlock = 2 * TILE * TILE * sizeof(float);
    switch (kind) {
        case KIND_K0:
            // cuBLAS 内部怎么分块是它自己的事，我们既不知道也不该猜。
            // 这几个量只是占位，下面打印时会整段跳过。
            block = dim3(0, 0);  grid = dim3(0, 0);
            bytesPerOut = 0.0;
            dist = bcast = ldg = 0.0;
            smemPerBlock = 0;
            break;
        case KIND_K1:
            block = dim3(32, 32);
            grid  = dim3((M + 31) / 32, (N + 31) / 32);   // 注意 k1 的 x 是行
            bytesPerOut = 2.0 * K * 4.0 + 4.0;            // 每个输出读 2K 个 float
            dist = 0.0;  bcast = 0.0;  ldg = 2.0;         // 无 shared，每个 FMA 两条全局 load
            reqPerFMA   = 33.0;                           // A 非合并 32 + B 广播 1
            smemPerBlock = 0;
            break;
        case KIND_K2:
            block = dim3(32, 32);
            grid  = dim3((N + 31) / 32, (M + 31) / 32);   // k2 的 x 是列
            bytesPerOut = 2.0 * K * 4.0 + 4.0;            // 字节账和 k1 一模一样
            dist = 0.0;  bcast = 0.0;  ldg = 2.0;         // 指令条数账也一模一样
            reqPerFMA   = 2.0;                            // A 广播 1 + B 合并 1
            smemPerBlock = 0;
            break;
        case KIND_K3:
            block = dim3(TILE, TILE);
            grid  = dim3((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            bytesPerOut = 2.0 * K / TILE * 4.0 + 4.0;
            dist = 1.0;  bcast = 1.0;  ldg = 2.0 / TILE;
            break;
        case KIND_K3C:
            block = dim3(TILE, TILE);
            grid  = dim3((N + TILE * COARSE - 1) / (TILE * COARSE), (M + TILE - 1) / TILE);
            bytesPerOut = (double)K / TILE * (1.0 + COARSE) / COARSE * 4.0 + 4.0;
            dist = 1.0;  bcast = 1.0;  ldg = (1.0 + COARSE) / (COARSE * TILE);
            break;
        case KIND_K4:
            block = dim3(TILE, TILE / CF);
            grid  = dim3((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            bytesPerOut = 2.0 * K / TILE * 4.0 + 4.0;
            dist = 1.0 / CF;  bcast = 1.0;  ldg = 2.0 / TILE;
            break;
        case KIND_K5:
        case KIND_K6:
            block = dim3(TILE, TILE / CF);
            grid  = dim3((N + TILE * TN - 1) / (TILE * TN), (M + TILE - 1) / TILE);
            bytesPerOut = (double)K / TILE * (1.0 + 1.0 / TN) * 4.0 + 4.0;
            smemPerBlock = (TILE * TILE + TILE * TILE * TN) * sizeof(float);   // 20 KB
            if (kind == KIND_K5) {          // 12 条 32 位 / 32 个 FMA
                dist = 1.0 / CF;  bcast = 1.0 / TN;  ldg = (double)(CF + CF * TN) / (CF * TN * TILE);
            } else {                        // 3 条 128 位 / 32 个 FMA（单价另算）
                dist = 1.0 / (CF * TN);  bcast = 2.0 / (CF * TN);
                ldg  = ((TILE * TILE / 4) / 128 + (TILE * TILE * TN / 4) / 128) / (double)(CF * TN * TILE);
            }
            break;
    }
    const double loadsPerFMA = dist + bcast + ldg;

    // ---- 三类 load 的单价（32 位），由实测反推，不是查手册查来的 ----
    //   LDG   由 k2 直接给出（它一条 shared load 都没有）：节拍 ÷ load/FMA
    //   32路 / 广播  解 k3、k4 的二元一次方程组 —— 这两个 kernel 的 bcast、ldg 相同，
    //                只有 dist 差 8 倍，是个天然的控制变量实验。
    //   标定过程见导学 §14 · 七 ⑦。
    //
    //   2026-09-21 本批标定： 32路 2.61   广播 0.53   LDG 1.89   ← 下面用的这组
    //   2026-09-14 上一批：   32路 2.54   广播 0.50   LDG 2.00
    //   三个数各动几个百分点（跨批漂移约 4.5%），但"32 路是广播的 5 倍"两批都成立 ——
    //   本章的结论依赖的是这个倍数关系，不是具体数值。
    //
    //   ⚠ 导学 §16 存档的那份日志是用上一批单价跑的，所以它打印的"模型预测"是旧值；
    //     下一轮跑出来的才会是新值。那份存档不回头修改 —— 它是那次运行的忠实记录。
    const double predicted = 2.61 * dist + 0.53 * bcast + 1.89 * ldg;

    if (kind == KIND_K0)
        printf("[%s] 矩阵 %d×%d×%d   （cuBLAS 自己决定 tile 和 grid，我们看不见）\n",
               name, M, N, K);
    else
        printf("[%s] 矩阵 %d×%d×%d   TILE %d   block %u×%u   grid %u×%u\n",
               name, M, N, K, TILE, block.x, block.y, grid.x, grid.y);

    // ---- occupancy：在启动之前就把"每个 SM 能同时驻留几个 block"算出来 ----
    //
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&out, kernel, blockSize, dynSmem)
    //   &out       输出参数（同 cudaMalloc 那个 & ）：结果由它写回来，所以传地址
    //   kernel     直接写函数名 —— C++ 里函数名传参时自动退化成函数指针，不用写 &
    //   blockSize  假设用这个 block 大小启动（是"假设"，不是查询已经跑过的 kernel）
    //   dynSmem    动态 shared 的字节数。本文件用静态 __shared__，大小已编进 kernel
    //              元数据，所以填 0；若改用 extern __shared__，这里必须填实际字节数。
    // 它不启动 kernel —— 只拿编译期的资源用量（寄存器数、静态 shared）配设备属性算，
    // 开销近乎为零，放在正式计时之前很安全。
    //
    // 写成 switch 是因为第二个参数的类型跟着 kernel 走（这个 API 是模板，七个 kernel
    // 会实例化出七份）。这是必须同步维护的第五张登记表：
    //   enum Kind / launch() / 命令行解析 / block-grid 那个 switch / 这里。
    // 漏掉这里不会报错，只会把 occupancy 打印成 0。
    //
    // API 内部取四条限制的最小值，谁先到顶谁说了算（4090：每 SM 65536 个寄存器、
    // 100 KB shared、1536 个线程）：
    //   K4  寄存器 54 → 每 warp 32×54=1728，按 256 的粒度取整 1792，每 block 4 warp = 7168
    //       65536/7168 = 9.1 → 9 个 block → 9×128 = 1152/1536 = 75.0%     ← 寄存器卡住
    //   K6  寄存器 111 → 每 block 14336，65536/14336 = 4.5 → 4 → 512/1536 = 33.3%  ← 同上
    //   K5  寄存器算得 7、shared 算得 102400/20480 = 5，而 API 返回 4
    //       （少的那一个多半是驱动每 SM 保留的约 1 KB shared —— 20480×5 正好顶满上限）
    // 最后这条正是"要调 API 而不是手算"的理由：分配粒度、保留量、L1/shared carveout
    // 这些手算极容易漏。预估值和实测值对不上，本身就是信息（见 §14 · 六 意外三）。
    const int tpb = block.x * block.y;        // threads per block；z 恒为 1，故只乘 x*y
    int blocksPerSM = 0;
    // K0 不是我们的 kernel，cudaOccupancy* 查不了它，整段跳过
    if (kind != KIND_K0)
    switch (kind) {
        case KIND_K1:  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, sgemm_k1,  tpb, 0)); break;
        case KIND_K2:  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, sgemm_k2,  tpb, 0)); break;
        case KIND_K3:  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, sgemm_k3,  tpb, 0)); break;
        case KIND_K3C: CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, sgemm_k3c, tpb, 0)); break;
        case KIND_K4:  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, sgemm_k4,  tpb, 0)); break;
        case KIND_K5:  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, sgemm_k5,  tpb, 0)); break;
        case KIND_K6:  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, sgemm_k6,  tpb, 0)); break;
    }
    // %zu 是 size_t 专用（用 %d / %lu 在别的平台会错位）；%% 输出一个字面的百分号。
    // 末尾那个 100.0 不能写成 100 —— 整数除法会把 512/1536 直接截成 0。
    if (kind == KIND_K0)
        printf("cuBLAS（列主序换位调用，TF32 已禁用）—— 没有我们自己的 block/grid\n");
    else
        printf("shared/block = %zu B（每 SM 上限 %zu B）　occupancy: %d block × %d 线程 = %d / %d = %.1f%%\n",
               smemPerBlock, prop.sharedMemPerMultiprocessor, blocksPerSM, tpb, blocksPerSM * tpb,
               prop.maxThreadsPerMultiProcessor, 100.0 * blocksPerSM * tpb / prop.maxThreadsPerMultiProcessor);

    // (size_t) 必须放在第一个乘数上：M * K 若按 int 算，8192*8192 就已经溢出了，
    // 再转成 size_t 也救不回来 —— 溢出发生在转换之前。
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);
    double footMB = (bytesA + bytesB + bytesC) / 1.0e6;
    printf("footprint = A+B+C = %.1f MB（L2 %.1f MB）%s\n", footMB, l2MB,
           footMB < l2MB ? "  ⚠ 装得下 → 数字会虚高，主基准请用 4096 以上"
                         : "  → 装不下，必须走显存");

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N),
                       hC((size_t)M * N), hCin((size_t)M * N);
    srand(1);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < hCin.size(); ++i) hCin[i] = 0.0f;

    float *dA, *dB, *dC;
    // 传 &dA 而不是 dA：cudaMalloc 要把"新分配到的显存地址"写回给我们，
    // 所以它需要的是"指针变量本身的地址"（float** ）。
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hCin.data(), bytesC, cudaMemcpyHostToDevice));

    for (int i = 0; i < WARMUP; ++i)
        launch(kind, grid, block, M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    std::vector<float> samples;
    for (int it = 0; it < ITERS; ++it) {
        CUDA_CHECK(cudaMemcpy(dC, hCin.data(), bytesC, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev0));
        launch(kind, grid, block, M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        samples.push_back(ms);
    }
    // 取中位数而不是平均：一次偶发的卡顿会把平均值整个带偏，中位数不受影响。
    // std::sort 原地排序，samples.begin()/end() 是 vector 的首尾迭代器。
    std::sort(samples.begin(), samples.end());
    // 样本数是偶数时取中间两个的平均。原来直接写 samples[n/2]，取到的是偏大的那一个 ——
    // ITERS=2 时等于"取两次里较慢的那次"，恰好把冷启动那次挑了出来。
    size_t nsm = samples.size();
    float ms = (nsm % 2) ? samples[nsm / 2]
                         : 0.5f * (samples[nsm / 2 - 1] + samples[nsm / 2]);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    int bad = verifySamples(M, N, K, alpha, beta, hA, hB, hCin, hC, 100);
    printf("%s（抽样 100 个元素）\n", bad == 0 ? "✓ 抽样校验通过" : "✗ 抽样校验失败");

    // ---- 三本账同时打出来 ----
    double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;

    printf("中位耗时 %.3f ms   %.1f GFLOP/s", ms, gflops);
    if (peakTF > 0) printf(" = FP32 峰值的 %.2f%%", 100.0 * gflops / (peakTF * 1000.0));
    printf("\n");

    if (kind == KIND_K0) {
        // K0 是靶子，不是被分析的对象：cuBLAS 内部怎么搬数据我们不知道，
        // 硬给它记账只会得到一个看起来很精确的假数（bytesPerOut=0 还会算出 inf）。
        printf("（cuBLAS 内部实现未知，不记三本账 —— 它在这里的角色是分母）\n");
    } else {
        double ai   = 2.0 * K / bytesPerOut;
        double beat = 32.0 * prop.multiProcessorCount * gpuClkKHz * 1e3 * (ms / 1000.0)
                    / ((double)M * N * K);
        if (peakBW > 0)
            printf("字节账: 算术强度 %.3f FLOP/byte → roofline 上限 %.0f GFLOP/s，达成率 %.1f%%\n",
                   ai, ai * peakBW, 100.0 * gflops / (ai * peakBW));
        if (reqPerFMA > 0.0)
            printf("请求账: 内存请求 %.0f 个/FMA（k1 的 A 不合并 → 一条 LDG 展开成 32 个请求）\n",
                   reqPerFMA);
        printf("条数账: load/FMA %.3f（32 路 %.3f + 广播 %.3f + 全局 %.3f）\n",
               loadsPerFMA, dist, bcast, ldg);
        printf("        节拍 %.2f 周期/warp圈 → 每条 load %.2f 周期   模型预测 %.2f%s\n",
               beat, beat / loadsPerFMA, predicted,
               kind == KIND_K6 ? "（k6 是 128 位 load，单价未标定，这个数只当下界）" : "");
    }

    if (kind == KIND_K0) CUBLAS_CHECK(cublasDestroy(g_cublas));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return bad ? 1 : 0;
}