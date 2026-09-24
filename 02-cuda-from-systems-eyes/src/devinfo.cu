// devinfo.cu —— 设备自查：把规格表上的数字换成这台机器实测的数字
//
//   nvcc devinfo.cu -o devinfo && ./devinfo | tee ../Results/device.txt
//
// 为什么要有这一步：本仓库的规矩是"先算再测然后对账"。roofline 的分母
// （峰值算力、峰值带宽）必须来自这台机器，不能抄网上的规格表 —— 与 01 阶段
// "KV 池必须每台机器现读启动日志"是同一条原则。
//
// ⚠️ vast.ai 上的卡可能有功耗墙或降频，实测 boost 时钟低于标称是正常的。
//    用实测值算脊点。

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int dev = 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        printf("无法读取设备属性\n");
        return 1;
    }

    // CUDA 13 起 cudaDeviceProp 里删掉了 clockRate / memoryClockRate 两个字段
    // （12.x 就已标记 deprecated）。改用 cudaDeviceGetAttribute 查，单位同样是 kHz，
    // 这条路径在老版本 CUDA 上一样能编译。
    int smClockKHz = 0, memClockKHz = 0;
    cudaDeviceGetAttribute(&smClockKHz,  cudaDevAttrClockRate,       dev);
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);

    printf("=== 设备 ===\n");
    printf("名称                  %s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("SM 数量               %d\n", p.multiProcessorCount);
    printf("warp 大小             %d\n", p.warpSize);

    printf("\n=== 并发容量 ===\n");
    printf("每 SM 最大线程        %d\n", p.maxThreadsPerMultiProcessor);
    printf("每 SM 最大 warp       %d\n", p.maxThreadsPerMultiProcessor / p.warpSize);
    printf("每 block 最大线程     %d\n", p.maxThreadsPerBlock);
    printf("每 SM 寄存器          %d (32-bit)\n", p.regsPerMultiprocessor);
    printf("每 block 寄存器       %d\n", p.regsPerBlock);

    printf("\n=== 内存层级 ===\n");
    printf("每 block shared       %zu KB (可选入上限 %zu KB)\n",
           p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
    printf("每 SM shared          %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("L2                    %.1f MB\n", p.l2CacheSize / 1048576.0);
    printf("全局显存              %.1f GB\n", p.totalGlobalMem / 1073741824.0);

    printf("\n=== 时钟与带宽 ===\n");
    printf("显存时钟              %.2f GHz\n", memClockKHz / 1e6);
    printf("显存位宽              %d bit\n", p.memoryBusWidth);
    double bw = 2.0 * memClockKHz * (p.memoryBusWidth / 8) / 1.0e6;   // GB/s
    printf("峰值带宽              %.0f GB/s\n", bw);
    printf("SM 时钟(标称)         %.2f GHz\n", smClockKHz / 1e6);

    // FP32 峰值：cudaDeviceProp 不直接给，按 SM 数 × 每 SM CUDA 核 × 2(FMA) × 时钟 估算。
    // 每 SM 的 CUDA 核数随架构不同：Ada/Ampere GA10x = 128，Volta/Turing = 64。
    int coresPerSM = (p.major == 8 || p.major == 9) ? 128 : 64;
    double tflops = 2.0 * coresPerSM * p.multiProcessorCount * (smClockKHz / 1e6) / 1000.0;
    printf("FP32 峰值(估算)       %.1f TFLOP/s   [%d 核/SM × %d SM × 2 × %.2f GHz]\n",
           tflops, coresPerSM, p.multiProcessorCount, smClockKHz / 1e6);

    printf("\n=== roofline 脊点（本阶段的指北针）===\n");
    printf("FP32 脊点             %.1f FLOP/byte   [%.1f TFLOP/s ÷ %.0f GB/s]\n",
           tflops * 1000.0 / bw, tflops, bw);
    printf("\n⚠️  上面的 SM 时钟是标称 boost。实际运行时可能因功耗墙降频，\n");
    printf("    请用 `nvidia-smi -q -d CLOCK` 看压测中的真实时钟再修正脊点。\n");

    return 0;
}
