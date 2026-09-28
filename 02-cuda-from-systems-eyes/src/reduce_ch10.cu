// =====================================================================
// reduce_ch10.cu —— PMPP 第 10 章配套：六个归约 kernel
//
//   R1  Fig. 10.6   朴素             单 block，控制发散 + 访存发散
//   R2  Fig. 10.9   收敛             单 block，两个发散都治好
//   R3  Fig. 10.11  shared           单 block，全局访问降到 N+1
//   R4  Fig. 10.13  分段 + 原子加     多 block  ← 唯一动到瓶颈的一步
//   R5  Fig. 10.15  线程粗化          多 block
//   R6  （书里没有） warp shuffle      多 block，最后 5 轮交给 warp
//   none            空 kernel         只用来量启动开销
//
// 和书的三处不同（都在注释里标了 ※）：
//   ※1 书里 Fig. 10.11/10.13/10.15 的签名有笔误（漏 output / 漏 void），已补
//   ※2 把写死的 BLOCK_DIM 换成 blockDim.x + 动态 shared，
//       这样同一份代码能用任意 block 大小跑，方便验证 occupancy 那一条
//   ※3 加了越界保护之外的输入校验，书里的 kernel 假设 N 正好整除
//
// 编译：nvcc -O3 -arch=sm_89 -lineinfo -o reduce_ch10 reduce_ch10.cu
// 运行：./reduce_ch10 <r1|r2|r3|r4|r5|r6|none> <N> [ITERS] [BLOCK]
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t e_ = (x);                                                \
        if (e_ != cudaSuccess) {                                             \
            printf("CUDA 错误：%s @ %s:%d\n",                                \
                   cudaGetErrorString(e_), __FILE__, __LINE__);              \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define COARSE_FACTOR 4        // R5/R6 每个 block 多吃几倍；书里 Fig.10.15 用 2
#define PEAK_BW_GBS   1008.0   // RTX 4090 理论带宽，第 6 章 §4·三 标定

// ---------------------------------------------------------------------
// R1 · Fig. 10.6 —— 朴素归约
//   owner 是偶数位置 2t；stride 由小变大；活跃线程越来越稀疏
//   注意：它原地修改 input，跑完输入就毁了（§7·错法一）
// ---------------------------------------------------------------------
__global__ void reduce_r1(float *input, float *output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0)          // 编号是 stride 倍数的才干活
            input[i] += input[i + stride];      // → warp 内部分裂 = 控制发散
        __syncthreads();                        // 必须在 if 外面
    }
    if (threadIdx.x == 0) *output = input[0];
}

// ---------------------------------------------------------------------
// R2 · Fig. 10.9 —— 收敛归约
//   只改了三处：owner 变 threadIdx.x、stride 由大变小、条件变 <
//   一处改动治两个病：控制发散（warp 整批退休）+ 访存发散（地址连号）
// ---------------------------------------------------------------------
__global__ void reduce_r2(float *input, float *output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride)               // 活跃的永远是编号最小的一批
            input[i] += input[i + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *output = input[0];
}

// ---------------------------------------------------------------------
// R3 · Fig. 10.11 —— 搬进 shared（※1 补了 output 参数，※2 用动态 shared）
//   第 1 轮在循环外做掉，顺手把结果写进 shared；之后全在片上
//   全局访问 = N 读 + 1 写，字节账到此触底
// ---------------------------------------------------------------------
__global__ void reduce_r3(const float *input, float *output) {
    extern __shared__ float s[];
    unsigned int t = threadIdx.x;
    s[t] = input[t] + input[t + blockDim.x];    // 第 1 轮 + 搬进 shared
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();                        // 挪到循环头：同步上一行那次写
        if (t < stride) s[t] += s[t + stride];
    }
    if (t == 0) *output = s[0];
}

// ---------------------------------------------------------------------
// R4 · Fig. 10.13 —— 分段 + 原子加（※1 补了 void）
//   每个 block 领 2*blockDim.x 个元素，独立跑完一棵树，再原子加汇总
//   跨 block 没有屏障可用，所以只能这样两层
// ---------------------------------------------------------------------
__global__ void reduce_r4(const float *input, float *output) {
    extern __shared__ float s[];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;   // 我这个 block 的段起点
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    s[t] = input[i] + input[i + blockDim.x];
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) s[t] += s[t + stride];
    }
    if (t == 0) atomicAdd(output, s[0]);        // 每个 block 交一次答卷
}

// ---------------------------------------------------------------------
// R5 · Fig. 10.15 —— 线程粗化（※1 补了 void）
//   段长 ×COARSE_FACTOR；多出来的部分用一个全员活跃、无需同步的循环吃掉
//   注意 sum 累加在寄存器里，整个粗化循环一次 shared 都没碰
// ---------------------------------------------------------------------
__global__ void reduce_r5(const float *input, float *output) {
    extern __shared__ float s[];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = input[i];                                  // 粗化循环：全员活跃
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile)
        sum += input[i + tile * blockDim.x];               // 步长 = blockDim.x → 合并
    s[t] = sum;

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) s[t] += s[t + stride];
    }
    if (t == 0) atomicAdd(output, s[0]);
}

// ---------------------------------------------------------------------
// R6 · 书里没有 —— warp shuffle
//   树的最后 5 层（32→16→8→4→2→1）本来就困在一个 warp 里，
//   而 warp 内的 32 个线程天然同步、寄存器又能互相读 —— 根本不需要 shared
// ---------------------------------------------------------------------
__global__ void reduce_r6(const float *input, float *output) {
    extern __shared__ float warp_s[];           // 只需要 blockDim.x/32 个 float
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    unsigned int lane = t & 31, warp = t >> 5;
    unsigned int nWarps = blockDim.x >> 5;

    float sum = input[i];                                  // ① 粗化循环，同 R5
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile)
        sum += input[i + tile * blockDim.x];

    for (int off = 16; off > 0; off >>= 1)                 // ② warp 内 5 轮
        sum += __shfl_down_sync(0xffffffffu, sum, off);    //    全在寄存器里

    if (lane == 0) warp_s[warp] = sum;                     // ③ 每个 warp 交一个数
    __syncthreads();                                       //    全 kernel 唯一一次同步

    if (warp == 0) {                                       // ④ 头一个 warp 收尾
        sum = (lane < nWarps) ? warp_s[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) atomicAdd(output, sum);
    }
}

// ---------------------------------------------------------------------
// 空 kernel —— 用来量 kernel 启动开销
//   N 小的时候启动开销和 kernel 本身同量级，不减掉就什么也比不出来
// ---------------------------------------------------------------------
__global__ void reduce_none(const float *, float *) {}

// =====================================================================
enum Kind { K_R1, K_R2, K_R3, K_R4, K_R5, K_R6, K_NONE, K_BAD };

static Kind parseKind(const char *s) {
    if (!strcmp(s, "r1"))   return K_R1;
    if (!strcmp(s, "r2"))   return K_R2;
    if (!strcmp(s, "r3"))   return K_R3;
    if (!strcmp(s, "r4"))   return K_R4;
    if (!strcmp(s, "r5"))   return K_R5;
    if (!strcmp(s, "r6"))   return K_R6;
    if (!strcmp(s, "none")) return K_NONE;
    return K_BAD;
}
static const char *kindName(Kind k) {
    static const char *n[] = {"R1 朴素", "R2 收敛", "R3 shared",
                              "R4 分段", "R5 粗化", "R6 shuffle", "空 kernel"};
    return n[k];
}
static bool singleBlock(Kind k) { return k == K_R1 || k == K_R2 || k == K_R3; }

static void launch(Kind k, dim3 grid, dim3 block, size_t smem,
                   float *in, float *out) {
    switch (k) {
        case K_R1:   reduce_r1  <<<grid, block, smem>>>(in, out); break;
        case K_R2:   reduce_r2  <<<grid, block, smem>>>(in, out); break;
        case K_R3:   reduce_r3  <<<grid, block, smem>>>(in, out); break;
        case K_R4:   reduce_r4  <<<grid, block, smem>>>(in, out); break;
        case K_R5:   reduce_r5  <<<grid, block, smem>>>(in, out); break;
        case K_R6:   reduce_r6  <<<grid, block, smem>>>(in, out); break;
        case K_NONE: reduce_none<<<grid, block, smem>>>(in, out); break;
        default: break;
    }
}

int main(int argc, char **argv) {
    const char *name = (argc > 1) ? argv[1] : "r4";
    long long N      = (argc > 2) ? atoll(argv[2]) : (1LL << 28);
    int  ITERS       = (argc > 3) ? atoi(argv[3]) : (N <= (1 << 16) ? 1000 : 11);
    int  BLOCK       = (argc > 4) ? atoi(argv[4]) : 256;
    const int WARMUP = 3;

    Kind kind = parseKind(name);
    if (kind == K_BAD) {
        printf("用法：%s <r1|r2|r3|r4|r5|r6|none> <N> [ITERS] [BLOCK]\n", argv[0]);
        return 1;
    }

    // ---- 参数体检：书里的 kernel 全都假设整除，不整除就直接拒绝 ----
    if (N <= 0 || (N & (N - 1))) {
        printf("N 必须是 2 的幂（书里的归约树没有处理尾巴）\n");
        return 1;
    }
    if (BLOCK % 32 || BLOCK < 32 || BLOCK > 1024) {
        printf("BLOCK 必须是 32 的倍数且在 32..1024 之间\n");
        return 1;
    }

    dim3 block, grid;
    long long perBlock;              // 一个 block 吃多少元素
    if (kind == K_NONE) {            // 空 kernel：只量启动开销，形状无所谓
        block = dim3((unsigned)BLOCK);
        grid  = dim3(1);
        perBlock = N;
    } else if (singleBlock(kind)) {
        if (N > 2048) {
            printf("R1/R2/R3 只能跑一个 block，N 最大 2048（这正是 §10.7 要解决的事）\n");
            return 1;
        }
        block = dim3((unsigned)(N / 2));
        grid  = dim3(1);
        perBlock = N;
    } else {
        perBlock = (kind == K_R4) ? 2LL * BLOCK : (long long)COARSE_FACTOR * 2 * BLOCK;
        if (N % perBlock) {
            printf("N 必须能被每 block 的段长 %lld 整除（课后第 5 题让你去掉这个限制）\n",
                   perBlock);
            return 1;
        }
        block = dim3((unsigned)BLOCK);
        grid  = dim3((unsigned)(N / perBlock));
    }
    // R1/R2 不用 shared，多申请会白白压低 occupancy
    size_t smem = (kind == K_R1 || kind == K_R2 || kind == K_NONE) ? 0
                : (kind == K_R6) ? (block.x / 32) * sizeof(float)
                                 : block.x * sizeof(float);

    // ---- 准备数据 ----
    printf("=== %s · N = %lld · block = %u · grid = %u ===\n",
           kindName(kind), N, block.x, grid.x);

    // 用一个自带的 LCG 填数：比 rand() 快得多，而且跨平台结果一致
    std::vector<float> h_in((size_t)N);
    unsigned int rng = 20260918u;
    double want = 0.0;                                       // 参考答案用 double 累加
    for (long long j = 0; j < N; ++j) {
        rng = rng * 1664525u + 1013904223u;
        float v = (float)(rng >> 8) * (1.0f / 16777216.0f);  // [0,1)
        h_in[(size_t)j] = v;
        want += (double)v;
    }

    // 单 block 的 R1/R2 会原地毁掉输入，所以每次计时都用一份全新的拷贝
    // （§5·① 那个"多份拷贝"的技巧 —— 不这么做，第 2 次起算的就是垃圾）
    int copies = singleBlock(kind) ? (ITERS + WARMUP) : 1;
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)N * copies * sizeof(float)));
    for (int c = 0; c < copies; ++c)
        CUDA_CHECK(cudaMemcpy(d_in + (size_t)N * c, h_in.data(),
                              (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // 每次迭代用一个独立的 output 槽位，省掉循环里那次 memset
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)(ITERS + WARMUP) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, (size_t)(ITERS + WARMUP) * sizeof(float)));

    // ---- 预热 ----
    for (int it = 0; it < WARMUP; ++it)
        launch(kind, grid, block, smem,
               d_in + (size_t)N * (copies > 1 ? it : 0), d_out + it);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int it = 0; it < ITERS; ++it)
        launch(kind, grid, block, smem,
               d_in + (size_t)N * (copies > 1 ? (WARMUP + it) : 0),
               d_out + WARMUP + it);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= ITERS;
    CUDA_CHECK(cudaGetLastError());

    // ---- 校验（拿最后一次的结果）----
    float got = 0.0f;
    CUDA_CHECK(cudaMemcpy(&got, d_out + WARMUP + ITERS - 1, sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (kind != K_NONE) {
        double rel = fabs((double)got - want) / (want != 0.0 ? fabs(want) : 1.0);
        printf("校验：GPU %.6f  vs  CPU(double) %.6f  相对误差 %.2e  %s\n",
               (double)got, want, rel, rel < 1e-4 ? "通过" : "不通过 ✗");
    }

    // ---- 记账 ----
    double sec   = ms / 1e3;
    double bytes = 4.0 * (double)N;                 // 必须搬的字节：每个 float 读一次
    double effBW = bytes / sec / 1e9;               // 有效带宽（不是实际流量！）
    double floorMs = bytes / (PEAK_BW_GBS * 1e9) * 1e3;

    printf("耗时      %9.4f ms\n", ms);
    if (kind != K_NONE) {
        printf("下限      %9.4f ms   （4N 字节 ÷ %.0f GB/s）\n", floorMs, PEAK_BW_GBS);
        printf("有效带宽  %9.1f GB/s  达成率 %.1f%%\n",
               effBW, effBW / PEAK_BW_GBS * 100.0);
        if (singleBlock(kind))
            printf("          ↑ 注意：只用了 1/128 的 SM，这个达成率没有可比性\n");
        if (kind == K_R4 || kind == K_R5 || kind == K_R6)
            printf("原子加    %9u 次（= block 数，全部打在同一个地址上）\n", grid.x);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
