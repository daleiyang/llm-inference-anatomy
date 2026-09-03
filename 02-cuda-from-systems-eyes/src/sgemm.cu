// sgemm.cu —— 六级 SGEMM 优化的基准框架
//
//   nvcc -O3 -arch=sm_89 -lcublas -Xptxas -v sgemm.cu -o sgemm
//   NVIDIA_TF32_OVERRIDE=0 ./sgemm <kernel_id> <N>
//
// kernel_id:  0=cuBLAS  1=naive  2=coalesced  3=shared  4=1D-tile  5=2D-tile  6=vectorized
//
// ─────────────────────────────────────────────────────────────────────────────
// 这个文件里【框架已经写好】：计时、正确性校验、cuBLAS 基准、CSV 输出、尺寸扫描。
// 【六个 kernel 留给你自己写】—— 那才是这一阶段要学的东西。
//
// 本仓库的规矩（沿用 01 阶段）：工具可以让 AI 写，理解必须归自己。
// 框架是工具；kernel 和"为什么这么写"是理解。
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while (0)

// ============================================================================
// K1 · naive —— 已给出，作为你的起点和对照
// 每个线程算 C 的一个元素。算术强度 0.25 FLOP/byte。
// ============================================================================
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B, float beta, float *C) {
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) acc += A[row * K + i] * B[i * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// ============================================================================
// TODO K2 · 全局内存合并访存
//   只改索引映射，让同一 warp 内相邻线程访问 C 的相邻列。
//   预期：数倍提升。用 ncu 的
//   smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct 对比 K1/K2。
// ============================================================================

// ============================================================================
// TODO K3 · shared memory 分块
//   把 A/B 的块搬进 __shared__ 复用。别漏 __syncthreads()（加载后 + 计算后各一次）。
//   ⚠️ 预期提升不如想象 —— 32×32 分块算术强度只有 8，离脊点 82 还远。
//      这个"落空"本身是要写进报告的证据。
// ============================================================================

// ============================================================================
// TODO K4 · 1D 寄存器分块（每线程算 TM=8 个输出）
//   A 的元素读进寄存器后复用 TM 次。算术强度 8 → 16。
//   预期：整个序列里提升最大的一步。
// ============================================================================

// ============================================================================
// TODO K5 · 2D 寄存器分块（每线程算 TM×TN=8×8）
//   ⚠️ 会撞寄存器压力。编译时看 -Xptxas -v 的输出：
//      出现 "spill stores/loads" 就是警报，减小 TM/TN。
//   ⚠️ occupancy 会下降 —— 这是正确的取舍，不是 bug。
// ============================================================================

// ============================================================================
// TODO K6 · 向量化访存（float4 / LDS.128）
//   前提是地址 16 字节对齐。用 ncu 看 SASS 里是否真的出现了 LDG.E.128。
// ============================================================================

// ---------------------------------------------------------------------------
// cuBLAS 基准（列主序 → 用 C^T = B^T · A^T 技巧得到行主序结果）
// ---------------------------------------------------------------------------
static cublasHandle_t g_handle;

void run_cublas(int M, int N, int K, float alpha,
                const float *A, const float *B, float beta, float *C) {
    cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

// ---------------------------------------------------------------------------
// kernel 分发
// ---------------------------------------------------------------------------
void run_kernel(int id, int M, int N, int K, float alpha,
                const float *A, const float *B, float beta, float *C) {
    switch (id) {
    case 0:
        run_cublas(M, N, K, alpha, A, B, beta, C);
        break;
    case 1: {
        dim3 block(32, 32);
        dim3 grid((M + 31) / 32, (N + 31) / 32);
        sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    // TODO: case 2..6 —— 你写的 kernel 接到这里
    default:
        printf("kernel %d 还没实现\n", id);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// 正确性校验 —— 规矩 1：没验证过的"快"是没有意义的
// ---------------------------------------------------------------------------
bool verify(const std::vector<float> &got, const std::vector<float> &ref, float tol = 1e-3f) {
    for (size_t i = 0; i < ref.size(); ++i) {
        float d = std::fabs(got[i] - ref[i]);
        float m = std::max(1.0f, std::fabs(ref[i]));
        if (d / m > tol) {
            printf("  ✗ 校验失败 @%zu: 实测 %.6f vs 参考 %.6f\n", i, got[i], ref[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    const int kid = (argc > 1) ? atoi(argv[1]) : 1;
    const int N   = (argc > 2) ? atoi(argv[2]) : 4096;
    const int M = N, K = N;
    const float alpha = 1.0f, beta = 0.0f;

    const int WARMUP = 5, ITERS = 20;   // 规矩 2

    cublasCreate(&g_handle);
    // 规矩 5：锁死精度模式，禁止 cuBLAS 偷偷用 TF32 张量核心
    cublasSetMathMode(g_handle, CUBLAS_DEFAULT_MATH);

    size_t bytes = (size_t)M * N * sizeof(float);
    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f), hRef(M * N), hGot(M * N);
    for (auto &v : hA) v = (float)rand() / RAND_MAX - 0.5f;
    for (auto &v : hB) v = (float)rand() / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- 参考结果：先用 cuBLAS 算一遍 ----
    CUDA_CHECK(cudaMemset(dC, 0, bytes));
    run_cublas(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // ---- 校验待测 kernel ----
    CUDA_CHECK(cudaMemset(dC, 0, bytes));
    run_kernel(kid, M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hGot.data(), dC, bytes, cudaMemcpyDeviceToHost));
    if (!verify(hGot, hRef)) { printf("kernel %d 结果不对，不测性能\n", kid); return 1; }

    // ---- 计时：CUDA event（规矩 3），warmup + 多次取中位（规矩 2）----
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    for (int i = 0; i < WARMUP; ++i) run_kernel(kid, M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(s));
        run_kernel(kid, M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        samples.push_back(ms);
    }
    std::sort(samples.begin(), samples.end());
    float ms = samples[samples.size() / 2];                 // 中位数

    double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
    // CSV: kernel,N,ms_median,gflops   —— 直接 tee 进 Results/
    printf("%d,%d,%.4f,%.1f\n", kid, N, ms, gflops);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(g_handle);
    return 0;
}
