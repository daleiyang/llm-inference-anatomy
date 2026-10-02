// fused_ops.cu —— 两个访存密集算子的融合阶梯：RMSNorm 与 online softmax
//
//   这份文件是第 10 章（Reduction）的实际应用。第 10 章练的是"把一个长数组归约成
//   一个标量"，这里练的是"把每一行归约成一个标量，然后立刻拿它去缩放这一行" ——
//   归约不再是目的，而是中间步骤，这才是它在真实推理栈里的样子。
//
//   ---- RMSNorm 阶梯（Qwen2.5 用的就是它，作用在 hidden=3584 上）----
//   n1   两趟两个 kernel：先算 sum(x²)，再归一化          —— 基线   12N 字节
//   n2   融成一个 kernel，但 x 仍然从 global 读两次        —— 只省 launch，不省字节  12N
//   n3   x 留在寄存器，读一次                              —— 这一步才动字节账   8N
//   n4   n3 + float4 向量化                                —— 只动条数账，字节不变  8N
//
//   ---- online softmax 阶梯（attention 打分和 logits 采样都要用）----
//   s1   三趟三个 kernel：max → sum(exp) → 归一化          —— 基线   16N 字节
//   s2   online 递推，两趟两个 kernel：(m,s) → 归一化      —— 省一趟  12N
//   s3   一趟，x 留寄存器                                  —— 8N
//   s4   s3 + float4                                       —— 8N
//
//   ⚠ n2 和 n1 的字节数【完全一样】。这是故意安排的一级台阶：
//     它用来证明"融合"本身不等于省带宽 —— 省带宽的是"让数据留在片上"，
//     而不是"把两个 kernel 写成一个"。这两件事经常被混为一谈。
//
//   评价指标是 MBU（Memory Bandwidth Utilization）= 实测 GB/s ÷ 理论峰值 GB/s。
//   访存密集算子的唯一目标就是把 MBU 顶上去 —— 和 SGEMM 看"占峰值算力"正好是
//   roofline 的两端。
//
//   编译:   nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v fused_ops.cu -o fused_ops
//   运行:   ./fused_ops <kernel> [M] [N] [ITERS]
//             M = 行数（token 数 / 打分矩阵的行数），N = 每行长度
//             RMSNorm 默认 M=4096  N=3584   （Qwen2.5-7B 的 hidden）
//             softmax 默认 M=4096  N=2048   （一段 2048 上下文的打分行）
//   查竞态: compute-sanitizer --tool racecheck ./fused_ops n3 8 3584 1
//
//   语法速查见 sgemm_all.cu 开头那一段，这里只标注新出现的写法。

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cfloat>

// ---------------------------------------------------------------------------
// 出错就地报，不要等到后面某个莫名其妙的地方才崩
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
        printf("[CUDA 错误] %s:%d  %s\n  ↳ %s\n",                              \
               __FILE__, __LINE__, cudaGetErrorString(e_), #call);             \
        exit(1);                                                               \
    }                                                                          \
} while (0)

// 每个 block 固定 256 线程。一个 block 负责一行。
static const int BLOCK = 256;
// 寄存器驻留版每线程最多存多少个 float。32 个 float = 32 个寄存器，
// 加上其余开销还远没到 255 的上限，不会 spill。
// 于是单行最长 = BLOCK × MAXS = 256 × 32 = 8192。
static const int MAXS = 32;
// float4 版：8 个 float4 也是 32 个 float，两条路径的寄存器压力刻意做成一样，
// 这样 n3→n4 / s3→s4 就是干净的单变量对照（只动指令条数，不动寄存器）。
static const int MAXV = 8;

// ===========================================================================
//                            归约积木
// ===========================================================================
//
// 和第 10 章 R6 的区别，就一处，但很关键：
//
//   R6 用的是 __shfl_down_sync —— 归约结果只汇聚到 0 号 lane，因为那里只需要
//   一个线程把结果 atomicAdd 出去，其余线程的值是垃圾无所谓。
//
//   这里用 __shfl_xor_sync（蝶形/butterfly）—— 归约完【每个 lane 手里都是全和】。
//   因为接下来 32 个线程都要拿这个和去缩放自己负责的那几个元素，
//   少一个线程知道都不行。
//
//   代价是一样的：同样 5 轮，同样每轮一条 shuffle 指令。xor 版不多花钱，
//   只是把"漏斗"换成了"全交换"。
// ---------------------------------------------------------------------------
__inline__ __device__ float warpReduceSum(float v) {
    // off 依次取 16 8 4 2 1；lane ^ off 是"异或伙伴"，
    // 第 1 轮 0↔16、1↔17……第 5 轮 0↔1、2↔3……五轮之后所有 lane 都持有全和。
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__inline__ __device__ float warpReduceMax(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// ---------------------------------------------------------------------------
// online softmax 的联合归约：把两个 (m, s) 合成一个
//
//   合并规则：
//       m = max(m1, m2)
//       s = s1·exp(m1 − m) + s2·exp(m2 − m)
//
//   为什么可以拿它做树形归约？因为这个运算【满足结合律】。
//   不满足的话，warp 内的蝶形归约就是错的 —— 它的合并顺序和串行扫描完全不同。
//   结合律的证明见导学 §5 · 三，一句话版本：
//   两个 (m,s) 合并后等价于"把两段数据拼起来重新算一遍 max 和 sum"，
//   而"拼起来"这个操作本身显然是可结合的。
// ---------------------------------------------------------------------------
// 注意这里【不能】用 -INFINITY 当初值。
//   没分到元素的线程 m 保持初值、s=0；两个这样的线程合并时
//   m - m_n = (-inf) - (-inf) = NaN，s = 0×NaN + 0×NaN = NaN，然后污染整行。
//   换成 -FLT_MAX：相减得 0，exp(0)=1，s = 0×1 + 0×1 = 0，安全退化。
__inline__ __device__ void warpReduceMaxSum(float &m, float &s) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float m_o = __shfl_xor_sync(0xffffffffu, m, off);
        float s_o = __shfl_xor_sync(0xffffffffu, s, off);
        float m_n = fmaxf(m, m_o);
        // 两边都先缩放到新的 m_n 再相加。谁的 m 小，谁的 exp 就是个小于 1 的数，
        // 不会溢出；相等时两个 exp 都是 1，退化成普通求和。
        s = s * __expf(m - m_n) + s_o * __expf(m_o - m_n);
        m = m_n;
    }
}

// ---------------------------------------------------------------------------
// block 级归约：先 warp 内，再跨 warp。
//   BLOCK=256 → 8 个 warp。第一轮每个 warp 归出一个值放进 shared，
//   第二轮让 0 号 warp 把这 8 个值再归一次，最后广播给全 block。
// ---------------------------------------------------------------------------
__inline__ __device__ float blockReduceSum(float v, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warpReduceSum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    // 只有前 8 个 lane 有真值，其余喂 0（加法的幺元）
    v = (threadIdx.x < (BLOCK >> 5)) ? smem[lane] : 0.0f;
    if (warp == 0) v = warpReduceSum(v);
    // 0 号 warp 算完了，但全 block 都要用，所以再过一次 shared 广播
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}

__inline__ __device__ void blockReduceMaxSum(float &m, float &s, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = BLOCK >> 5;
    warpReduceMaxSum(m, s);
    if (lane == 0) { smem[warp] = m; smem[nwarp + warp] = s; }
    __syncthreads();
    if (warp == 0) {
        // 幺元是 (-inf, 0)：最大值取 -inf 不影响 max，和取 0 不影响求和
        m = (lane < nwarp) ? smem[lane]         : -FLT_MAX;
        s = (lane < nwarp) ? smem[nwarp + lane] : 0.0f;
        warpReduceMaxSum(m, s);
        if (lane == 0) { smem[2 * nwarp] = m; smem[2 * nwarp + 1] = s; }
    }
    __syncthreads();
    m = smem[2 * nwarp];
    s = smem[2 * nwarp + 1];
}

// ===========================================================================
//                            RMSNorm
//                y = x · rsqrt(mean(x²) + eps) · w
// ===========================================================================

// ---- N1：两趟，两个 kernel ------------------------------------------------
// 第一趟只算每行的 rstd，写进一个 M 长的小数组。
// 字节：读 x 一次 = 4N/行
__global__ void rms_n1_pass1(const float *__restrict__ x, float *__restrict__ rstd,
                             int N, float eps) {
    __shared__ float smem[BLOCK / 32];
    const long long row = blockIdx.x;
    const float *xr = x + row * (long long)N;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = xr[i];
        acc += v * v;
    }
    acc = blockReduceSum(acc, smem);
    if (threadIdx.x == 0)
        rstd[row] = rsqrtf(acc / N + eps);
}

// 第二趟再把 x 读一遍，乘上第一趟算好的 rstd 和权重，写出去。
// 字节：读 x 一次 + 写 y 一次 = 8N/行   （合计 12N）
__global__ void rms_n1_pass2(const float *__restrict__ x, const float *__restrict__ w,
                             const float *__restrict__ rstd, float *__restrict__ y, int N) {
    const long long row = blockIdx.x;
    const float *xr = x + row * (long long)N;
    float       *yr = y + row * (long long)N;
    const float r = rstd[row];
    for (int i = threadIdx.x; i < N; i += BLOCK)
        yr[i] = xr[i] * r * w[i];
}

// ---- N2：融成一个 kernel，但 x 还是读两次 ---------------------------------
// 和 N1 的字节数【一模一样】。省掉的只有：一次 kernel launch、
// rstd 数组的一写一读（4M 字节，相对 12NM 可以忽略）。
// 留着这一级，就是为了让"融合 ≠ 省带宽"这件事有个可量化的对照。
__global__ void rms_n2(const float *__restrict__ x, const float *__restrict__ w,
                       float *__restrict__ y, int N, float eps) {
    __shared__ float smem[BLOCK / 32];
    const long long row = blockIdx.x;
    const float *xr = x + row * (long long)N;
    float       *yr = y + row * (long long)N;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = xr[i];                       // ← 第一次读
        acc += v * v;
    }
    const float r = rsqrtf(blockReduceSum(acc, smem) / N + eps);
    for (int i = threadIdx.x; i < N; i += BLOCK)
        yr[i] = xr[i] * r * w[i];              // ← 第二次读，同样的地址
}

// ---- N3：x 留在寄存器，只读一次 -------------------------------------------
// 字节：读 x 一次 + 写 y 一次 = 8N/行。相对 N1/N2 的 12N，理论加速 1.5×。
// 代价是每线程多占 MAXS 个寄存器，且单行长度被 BLOCK×MAXS 卡死。
__global__ void rms_n3(const float *__restrict__ x, const float *__restrict__ w,
                       float *__restrict__ y, int N, float eps) {
    __shared__ float smem[BLOCK / 32];
    const long long row = blockIdx.x;
    const float *xr = x + row * (long long)N;
    float       *yr = y + row * (long long)N;

    // 循环必须写成"固定 MAXS 次 + 边界判断"，不能写成 for(i=tid; i<N; i+=BLOCK)
    // 配一个 xs[cnt++]。后者的下标是运行期算出来的，而【寄存器不能动态寻址】——
    // 编译器只好把 xs 放进 local memory（也就是显存），ptxas 会报 spill stores，
    // 这一级"数据留在片上"的前提当场作废。
    // 写成常量下标 xs[c] 并 #pragma unroll，编译器才会把它真正分配到寄存器里。
    float xs[MAXS];
    float acc = 0.0f;
    #pragma unroll
    for (int c = 0; c < MAXS; ++c) {
        int i = threadIdx.x + c * BLOCK;
        float v = (i < N) ? xr[i] : 0.0f;      // 越界的线程喂 0，不影响平方和
        xs[c] = v;
        acc += v * v;
    }
    const float r = rsqrtf(blockReduceSum(acc, smem) / N + eps);
    #pragma unroll
    for (int c = 0; c < MAXS; ++c) {
        int i = threadIdx.x + c * BLOCK;
        if (i < N) yr[i] = xs[c] * r * w[i];
    }
}

// ---- N4：N3 + float4 -------------------------------------------------------
// 字节数和 N3 完全相同，变的只有指令条数和内存请求数：
// 4 个 32 位 load 并成 1 个 128 位 load。
__global__ void rms_n4(const float *__restrict__ x, const float *__restrict__ w,
                       float *__restrict__ y, int N, float eps) {
    __shared__ float smem[BLOCK / 32];
    const long long row = blockIdx.x;
    const int n4 = N >> 2;                     // 一行有多少个 float4

    // reinterpret_cast 把 float* 重新解释成 float4*，地址必须 16 字节对齐 ——
    // cudaMalloc 返回的指针是 256 字节对齐的，而 row*N 只要 N%4==0 就仍然对齐。
    const float4 *xr = reinterpret_cast<const float4 *>(x + row * (long long)N);
    float4       *yr = reinterpret_cast<float4 *>(y + row * (long long)N);
    const float4 *wr = reinterpret_cast<const float4 *>(w);

    float4 xs[MAXV];
    float acc = 0.0f;
    #pragma unroll
    for (int c = 0; c < MAXV; ++c) {
        int i = threadIdx.x + c * BLOCK;
        float4 v = (i < n4) ? xr[i] : make_float4(0.f, 0.f, 0.f, 0.f);
        xs[c] = v;
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float r = rsqrtf(blockReduceSum(acc, smem) / N + eps);
    #pragma unroll
    for (int c = 0; c < MAXV; ++c) {
        int i = threadIdx.x + c * BLOCK;
        if (i >= n4) continue;
        float4 v = xs[c], g = wr[i], o;
        o.x = v.x * r * g.x;  o.y = v.y * r * g.y;
        o.z = v.z * r * g.z;  o.w = v.w * r * g.w;
        yr[i] = o;
    }
}

// ===========================================================================
//                          online softmax
//                  y_i = exp(x_i − max) / Σ exp(x_j − max)
// ===========================================================================

// ---- S1：三趟，三个 kernel -------------------------------------------------
// 字节：4N + 4N + (4N+4N) = 16N/行
__global__ void sm_s1_max(const float *__restrict__ x, float *__restrict__ mx, int N) {
    __shared__ float smem[BLOCK / 32];
    const long long row = blockIdx.x;
    const float *xr = x + row * (long long)N;
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += BLOCK) m = fmaxf(m, xr[i]);

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    m = warpReduceMax(m);
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    m = (threadIdx.x < BLOCK / 32) ? smem[lane] : -FLT_MAX;
    if (warp == 0) m = warpReduceMax(m);
    if (threadIdx.x == 0) mx[row] = m;
}

__global__ void sm_s1_sum(const float *__restrict__ x, const float *__restrict__ mx,
                          float *__restrict__ sm, int N) {
    __shared__ float smem[BLOCK / 32];
    const long long row = blockIdx.x;
    const float *xr = x + row * (long long)N;
    const float m = mx[row];
    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK) s += __expf(xr[i] - m);
    s = blockReduceSum(s, smem);
    if (threadIdx.x == 0) sm[row] = s;
}

__global__ void sm_s1_div(const float *__restrict__ x, const float *__restrict__ mx,
                          const float *__restrict__ sm, float *__restrict__ y, int N) {
    const long long row = blockIdx.x;
    const float *xr = x + row * (long long)N;
    float       *yr = y + row * (long long)N;
    const float m = mx[row], inv = 1.0f / sm[row];
    for (int i = threadIdx.x; i < N; i += BLOCK)
        yr[i] = __expf(xr[i] - m) * inv;
}

// ---- S2：online 递推，两趟 -------------------------------------------------
// 第一趟【一次扫描同时得到 max 和 sum】—— 这就是 online softmax 的全部内容。
// 字节：4N + (4N+4N) = 12N/行
__global__ void sm_s2_ms(const float *__restrict__ x, float *__restrict__ mx,
                         float *__restrict__ sm, int N) {
    __shared__ float smem[BLOCK / 32 * 2 + 2];
    const long long row = blockIdx.x;
    const float *xr = x + row * (long long)N;

    float m = -FLT_MAX, s = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = xr[i];
        // 递推三行：见到更大的 max，就把已经累好的 s 按比例缩回去
        float m_n = fmaxf(m, v);
        s = s * __expf(m - m_n) + __expf(v - m_n);
        m = m_n;
    }
    blockReduceMaxSum(m, s, smem);
    if (threadIdx.x == 0) { mx[row] = m; sm[row] = s; }
}

// ---- S3：一趟，x 留寄存器 ---------------------------------------------------
// 字节：4N + 4N = 8N/行。相对 S1 的 16N，理论加速 2.0×。
__global__ void sm_s3(const float *__restrict__ x, float *__restrict__ y, int N) {
    __shared__ float smem[BLOCK / 32 * 2 + 2];
    const long long row = blockIdx.x;
    const float *xr = x + row * (long long)N;
    float       *yr = y + row * (long long)N;

    // 常量下标 + #pragma unroll，理由同 rms_n3
    float xs[MAXS];
    float m = -FLT_MAX, s = 0.0f;
    #pragma unroll
    for (int c = 0; c < MAXS; ++c) {
        int i = threadIdx.x + c * BLOCK;
        if (i >= N) { xs[c] = -FLT_MAX; continue; }  // 越界位置塞 -FLT_MAX：
        float v = xr[i];                             // 既不会成为 max，
        xs[c] = v;                                   // exp 出来也是 0
        float m_n = fmaxf(m, v);
        s = s * __expf(m - m_n) + __expf(v - m_n);
        m = m_n;
    }
    blockReduceMaxSum(m, s, smem);
    const float inv = 1.0f / s;
    #pragma unroll
    for (int c = 0; c < MAXS; ++c) {
        int i = threadIdx.x + c * BLOCK;
        if (i < N) yr[i] = __expf(xs[c] - m) * inv;
    }
}

// ---- S4：S3 + float4 -------------------------------------------------------
__global__ void sm_s4(const float *__restrict__ x, float *__restrict__ y, int N) {
    __shared__ float smem[BLOCK / 32 * 2 + 2];
    const long long row = blockIdx.x;
    const int n4 = N >> 2;
    const float4 *xr = reinterpret_cast<const float4 *>(x + row * (long long)N);
    float4       *yr = reinterpret_cast<float4 *>(y + row * (long long)N);

    float4 xs[MAXV];
    float m = -FLT_MAX, s = 0.0f;
    #pragma unroll
    for (int c = 0; c < MAXV; ++c) {
        int i = threadIdx.x + c * BLOCK;
        if (i >= n4) { xs[c] = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX); continue; }
        float4 v = xr[i];
        xs[c] = v;
        // 一次进来 4 个元素，先在寄存器里把这 4 个合成一个 (m,s)，再并进累计值。
        // 这已经是 FlashAttention 分块合并的最小形态了 —— 只不过块大小是 4。
        float mv = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        float sv = __expf(v.x - mv) + __expf(v.y - mv) + __expf(v.z - mv) + __expf(v.w - mv);
        float m_n = fmaxf(m, mv);
        s = s * __expf(m - m_n) + sv * __expf(mv - m_n);
        m = m_n;
    }
    blockReduceMaxSum(m, s, smem);
    const float inv = 1.0f / s;
    #pragma unroll
    for (int c = 0; c < MAXV; ++c) {
        int i = threadIdx.x + c * BLOCK;
        if (i >= n4) continue;
        float4 v = xs[c], o;
        o.x = __expf(v.x - m) * inv;  o.y = __expf(v.y - m) * inv;
        o.z = __expf(v.z - m) * inv;  o.w = __expf(v.w - m) * inv;
        yr[i] = o;
    }
}

// ===========================================================================
//                         CPU 参照与抽样校验
// ===========================================================================
enum Kind { K_N1, K_N2, K_N3, K_N4, K_S1, K_S2, K_S3, K_S4 };
static bool isNorm(Kind k) { return k <= K_N4; }

// 只抽查若干行。全查的话 M=4096、N=3584 要在 CPU 上算 1400 万次 exp，太慢。
static int verifyRows(Kind kind, int M, int N, const std::vector<float> &hx,
                      const std::vector<float> &hw, const std::vector<float> &hy,
                      int nrows) {
    std::mt19937 rng(20260925);
    int bad = 0;
    for (int t = 0; t < nrows; ++t) {
        long long row = rng() % (unsigned)M;
        const float *xr = hx.data() + row * (long long)N;
        const float *yr = hy.data() + row * (long long)N;

        std::vector<double> ref(N);
        if (isNorm(kind)) {
            double acc = 0.0;
            for (int i = 0; i < N; ++i) acc += (double)xr[i] * xr[i];
            double r = 1.0 / std::sqrt(acc / N + 1e-6);
            for (int i = 0; i < N; ++i) ref[i] = xr[i] * r * hw[i];
        } else {
            double m = -1e300;
            for (int i = 0; i < N; ++i) m = std::max(m, (double)xr[i]);
            double s = 0.0;
            for (int i = 0; i < N; ++i) s += std::exp(xr[i] - m);
            for (int i = 0; i < N; ++i) ref[i] = std::exp(xr[i] - m) / s;
        }
        // __expf / rsqrtf 是快速近似指令，容差要比普通浮点宽一点。
        // 这里用相对误差 2e-3，softmax 的绝对值很小，再加一个 1e-7 的绝对兜底。
        for (int i = 0; i < N; ++i) {
            double d = std::fabs(ref[i] - yr[i]);
            if (d > 2e-3 * std::fabs(ref[i]) + 1e-7) {
                if (bad < 3)
                    printf("  行 %lld 列 %d: GPU %.6g  CPU %.6g\n", row, i, yr[i], ref[i]);
                ++bad;
            }
        }
    }
    return bad;
}

// ===========================================================================
//                                main
// ===========================================================================
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("用法: %s <kernel> [M] [N] [ITERS]\n"
               "  kernel = n1|n2|n3|n4   RMSNorm     默认 M=4096 N=3584\n"
               "           s1|s2|s3|s4   softmax     默认 M=4096 N=2048\n"
               "  M = 行数（token 数 / 打分矩阵的行数），N = 每行长度\n"
               "  例： %s n4 4096 3584 11      Qwen2.5-7B 的一次 RMSNorm\n"
               "       %s n4 1 3584 11         decode batch=1（只有 1 个 block！）\n"
               "       %s s4 114688 2048 11    28 头 × 4096 个 query 的打分\n",
               argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    const char *name = argv[1];
    Kind kind;
    if      (!strcmp(name, "n1")) kind = K_N1;
    else if (!strcmp(name, "n2")) kind = K_N2;
    else if (!strcmp(name, "n3")) kind = K_N3;
    else if (!strcmp(name, "n4")) kind = K_N4;
    else if (!strcmp(name, "s1")) kind = K_S1;
    else if (!strcmp(name, "s2")) kind = K_S2;
    else if (!strcmp(name, "s3")) kind = K_S3;
    else if (!strcmp(name, "s4")) kind = K_S4;
    else { printf("不认识的 kernel: %s\n", name); return 2; }

    int M     = (argc > 2) ? atoi(argv[2]) : 4096;
    int N     = (argc > 3) ? atoi(argv[3]) : (isNorm(kind) ? 3584 : 2048);
    int ITERS = (argc > 4) ? atoi(argv[4]) : 11;
    if (M <= 0 || N <= 0) { printf("M/N 必须为正\n"); return 2; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int memClkKHz = 0, gpuClkKHz = 0;
    cudaDeviceGetAttribute(&memClkKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&gpuClkKHz, cudaDevAttrClockRate, 0);
    // 峰值带宽 = 内存时钟 × 2（DDR 双沿）× 位宽/8。和 sgemm_all.cu 用同一个口径。
    const double peakBW = memClkKHz * 2.0 * (prop.memoryBusWidth / 8) / 1.0e6;

    printf("GPU: %s  sm_%d%d  %d SM  L2 %.1f MB   显存带宽峰值 %.0f GB/s\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize / 1048576.0, peakBW);
    printf("[%s] %s   M=%d 行 × N=%d 列   block %d 线程   grid %d 个 block\n",
           name, isNorm(kind) ? "RMSNorm" : "softmax", M, N, BLOCK, M);

    // ---- 两个会让结果作废的前提，先查 ----
    const bool needReg = (kind == K_N3 || kind == K_N4 || kind == K_S3 || kind == K_S4);
    if (needReg && N > BLOCK * MAXS) {
        printf("✗ 这一级把整行存在寄存器里，单行最长 %d，而 N=%d。\n",
               BLOCK * MAXS, N);
        printf("  这不是实现偷懒，是硬件约束 —— 也正是 FlashAttention 必须分块的原因，\n"
               "  详见导学 §7 · 三。想跑长行请用 n1/n2 或 s1/s2。\n");
        return 3;
    }
    const bool needVec4 = (kind == K_N4 || kind == K_S4);
    if (needVec4 && (N & 3)) {
        printf("✗ float4 版要求 N 是 4 的倍数（当前 N=%d）\n", N);
        return 3;
    }

    // ---- 数据 ----
    const size_t nElem = (size_t)M * N;
    const size_t bytes = nElem * sizeof(float);
    std::vector<float> hx(nElem), hw(N), hy(nElem);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (size_t i = 0; i < nElem; ++i) hx[i] = dist(rng);
    for (int i = 0; i < N; ++i)        hw[i] = 0.5f + 0.5f * dist(rng) / 3.0f;

    float *dx = nullptr, *dw = nullptr, *dy = nullptr, *dm = nullptr, *ds = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMalloc(&dw, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dm, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, hw.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const float eps = 1e-6f;
    // lambda：把"发射这一级的全部 kernel"打包成一个可重复调用的东西。
    // [&] 表示按引用捕获外面所有用到的变量。
    auto launch = [&]() {
        switch (kind) {
            case K_N1: rms_n1_pass1<<<M, BLOCK>>>(dx, dm, N, eps);
                       rms_n1_pass2<<<M, BLOCK>>>(dx, dw, dm, dy, N);          break;
            case K_N2: rms_n2<<<M, BLOCK>>>(dx, dw, dy, N, eps);               break;
            case K_N3: rms_n3<<<M, BLOCK>>>(dx, dw, dy, N, eps);               break;
            case K_N4: rms_n4<<<M, BLOCK>>>(dx, dw, dy, N, eps);               break;
            case K_S1: sm_s1_max<<<M, BLOCK>>>(dx, dm, N);
                       sm_s1_sum<<<M, BLOCK>>>(dx, dm, ds, N);
                       sm_s1_div<<<M, BLOCK>>>(dx, dm, ds, dy, N);             break;
            case K_S2: sm_s2_ms <<<M, BLOCK>>>(dx, dm, ds, N);
                       sm_s1_div<<<M, BLOCK>>>(dx, dm, ds, dy, N);             break;
            case K_S3: sm_s3<<<M, BLOCK>>>(dx, dy, N);                         break;
            case K_S4: sm_s4<<<M, BLOCK>>>(dx, dy, N);                         break;
        }
    };

    // ---- occupancy：这一级理论上每个 SM 能同时跑几个 block ----
    int blocksPerSM = 0;
    const void *rep = (kind == K_N1) ? (const void *)rms_n1_pass1
                    : (kind == K_N2) ? (const void *)rms_n2
                    : (kind == K_N3) ? (const void *)rms_n3
                    : (kind == K_N4) ? (const void *)rms_n4
                    : (kind == K_S1) ? (const void *)sm_s1_max
                    : (kind == K_S2) ? (const void *)sm_s2_ms
                    : (kind == K_S3) ? (const void *)sm_s3
                                     : (const void *)sm_s4;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, rep, BLOCK, 0));
    printf("occupancy: %d block × %d 线程 = %d / %d = %.1f%%",
           blocksPerSM, BLOCK, blocksPerSM * BLOCK, prop.maxThreadsPerMultiProcessor,
           100.0 * blocksPerSM * BLOCK / prop.maxThreadsPerMultiProcessor);
    // grid 太小的话，占用率再高也没用 —— 卡根本没被填满。这是 decode 的常态。
    const double smFill = (double)M / (prop.multiProcessorCount * blocksPerSM);
    printf("   grid 填充度 %d / (%d SM × %d) = %.2f%s\n",
           M, prop.multiProcessorCount, blocksPerSM, smFill,
           smFill < 1.0 ? "   ⚠ 填不满，卡在空转" : "");

    printf("footprint = x + y = %.1f MB（L2 %.1f MB）%s\n",
           2.0 * bytes / 1048576.0, prop.l2CacheSize / 1048576.0,
           2.0 * bytes > prop.l2CacheSize ? "  → 装不下，必须走显存"
                                          : "  ⚠ 装得下 → 数字会虚高");

    // ---- 计时：和 sgemm_all.cu 同一套（预热 + 中位数）----
    const int WARMUP = (ITERS <= 1) ? 0 : (ITERS <= 2 ? 1 : 3);
    for (int i = 0; i < WARMUP; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    std::vector<float> samples;
    for (int it = 0; it < ITERS; ++it) {
        CUDA_CHECK(cudaEventRecord(ev0));
        launch();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float t; CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
        samples.push_back(t);
    }
    std::sort(samples.begin(), samples.end());
    const size_t ns = samples.size();
    const double ms = (ns % 2) ? samples[ns / 2]
                               : 0.5 * (samples[ns / 2 - 1] + samples[ns / 2]);

    CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
    const int bad = verifyRows(kind, M, N, hx, hw, hy, 8);
    printf("%s（抽查 8 行）\n", bad == 0 ? "✓ 抽样校验通过" : "✗ 抽样校验失败");

    // ---- 字节账：这是这两个算子唯一重要的一本账 ----
    // 每行搬多少字节，按"必须经过显存的量"算。权重 w 只有 4N 字节，
    // 所有行共用、稳稳待在 L2 里，不计入。
    double bytesPerRow;
    switch (kind) {
        case K_N1: case K_N2: bytesPerRow = 12.0 * N; break;   // 读两次 + 写一次
        case K_N3: case K_N4: bytesPerRow =  8.0 * N; break;   // 读一次 + 写一次
        case K_S1:            bytesPerRow = 16.0 * N; break;   // 读三次 + 写一次
        case K_S2:            bytesPerRow = 12.0 * N; break;
        default:              bytesPerRow =  8.0 * N; break;
    }
    const double moved = bytesPerRow * M;
    const double gbps  = moved / (ms / 1000.0) / 1.0e9;
    const double mbu   = 100.0 * gbps / peakBW;

    printf("中位耗时 %.4f ms   搬运 %.1f MB   %.1f GB/s   MBU %.1f%%\n",
           ms, moved / 1048576.0, gbps, mbu);
    printf("字节账: %.0fN 字节/行 × %d 行 = %.1f MB；"
           "理论下限 %.4f ms（按 %.0f GB/s 跑满）\n",
           bytesPerRow / N, M, moved / 1048576.0,
           moved / (peakBW * 1e9) * 1000.0, peakBW);

    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(dm)); CUDA_CHECK(cudaFree(ds));
    return bad == 0 ? 0 : 1;
}
