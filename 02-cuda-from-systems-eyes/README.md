[← 返回 llm-inference-anatomy](../)

# 02 · CUDA from a Systems Engineer's Eyes

**我写过内存池和 lock-free hash table。这一阶段想回答：那些 CPU 上的性能直觉，搬到 GPU 上还剩多少？**
答案是：一半能直接用，另一半会主动害你。

在 RTX 4090 上手写了三类 kernel。每一类都**先算出理论上限、写下预测，再上机测量、逐条对账**：

| 算子 | 性质 | 结果 |
|---|---|---|
| **SGEMM**（矩阵乘，六级优化） | 算力密集 | 从 naive 到 **cuBLAS 的 71.5%**，共快 **63 倍** |
| **归约**（PMPP 第 10 章，六级） | 纯带宽 | 收在带宽理论下限的 **95%** |
| **RMSNorm / online softmax**（融合算子） | 访存密集 | 跑到峰值带宽的 **90–91%** |

---

## 完成情况

对照开工前定下的完成标志：

| 目标 | 结果 | |
|---|---|---|
| 六级 SGEMM 全部通过正确性校验 | 全部通过，含 N=4097 边界 | ✅ |
| 最快版本达到 cuBLAS 的 70%+ | **71.5%** | ✅ |
| 理论预测与实测对账，差异有解释 | 对账了 —— **结论是预测被推翻**，见下面的发现 1 | ✅ |
| RMSNorm / softmax 融合版 MBU 70%+ | **91.0%** / **90.3%** | ✅ |
| 并入仓库，路线图更新 | 4 份报告 + 全部代码、脚本、原始数据 | ✅ |
| 每一级都有 Nsight 指标佐证 | **没做到**：租用实例没有性能计数器权限，[见下文](#没做到的nsight) | ❌ |

计划外多做了三件事：六级归约（作为带宽侧的对照）、`compute-sanitizer` 竞态检查、
SASS 反汇编计数（用来补 Nsight 缺掉的那部分证据）。

---

## 四份报告

| # | 报告 | 讲什么 |
|---|---|---|
| 01 | [核心代码的演化](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/01-SGEMM六级优化-核心代码的演化.html) | SGEMM 六级，每一级改了哪一行、为什么这样改 |
| 02 | [实测与三本账](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html) | SGEMM 的每个数字从哪来，哪本账真的准 |
| 03 | [归约六级优化：一道纯带宽题](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/03-归约六级优化-一道纯带宽题.html) | 没有数据复用可做时，该看什么 |
| 04 | [融合算子：RMSNorm 与 online softmax](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/04-融合算子-RMSNorm与OnlineSoftmax.html) | 算子融合到底省了什么 |

建议按顺序读；03 和 04 也可以单独读。报告是自包含的单页 HTML，经 GitHub Pages 渲染，
直接在仓库里点 `.html` 只会看到源码。

---

## 五个最重要的发现

**1 · 「提高算术强度就能更快」，七个 kernel 里错了五次。**
收益最大的两步（8.2× 和 3.4×）算术强度一点没变；算术强度涨了 1.6 倍的那一步只快了 8%。
原因是这些 kernel 根本不在带宽墙上 —— 数据大多命中了 L2。真正决定速度的是另外两本账：
**一个 warp 的访存被拆成几个请求**，以及**每次乘加要配几条 load 指令**。→ 报告 01、02

**2 · 合并访存带来的 8 倍里，只有 2.4 倍是合并本身。**
另外 3.4 倍来自矩阵宽度恰好是 2 的幂：把 N 从 4096 改成 4097，naive 版快了 3.33 倍。
具体是哪个硬件结构在冲突，没有 Nsight 判断不了，如实标为悬案。→ 报告 02

**3 · occupancy 不是越高越好。**
K5 把占用率从 75% 降到 33%，反而快了 1.66 倍 —— 每个线程多算几个结果带来的复用更值钱。→ 报告 01

**4 · 带宽受限的算子，上限一秒钟就能算出来。**
归约的上限就是 `数据量 ÷ 1008 GB/s`。书里的五步优化只有一步打中瓶颈（快 292 倍），
其余四步收益为零。→ 报告 03

**5 · 融合省的不是访存次数，是两次访存之间的距离。**
一个账面上一个字节都没省的融合版快了 1.49 倍：它的第二次读离第一次只隔 14 KB，落进了 L2。
开跑前写下的 14 条预测落空了 5 条，全是这个原因。FlashAttention 押的也是这一点。→ 报告 04

还有一条适用边界：**「优化了 63 倍」只在方阵上成立。** 换成 decode 形状（每次只算 1 行）只剩 3.4 倍 ——
那一侧的墙是带宽，cuBLAS 也绕不过去，我们的 K4 甚至跑到了它的 102.6%。

---

## 开工前那句话，改写成了什么

开工前我写下的是：

> 「SGEMM 在 4090 上摸不到 FP32 峰值，因为分块后的算术强度只有 32，而脊点是 82。」

**结论对，因果错。** 实测之后应该这么说：

> 「摸不到峰值不是因为带宽 —— 最快的 K6 早就不在带宽墙上了。
> 真正的墙是 load 指令占掉的发射槽。要再往上走，只能继续减少 load 条数，或者换用张量核心。」

---

## 没做到的：Nsight

租用的 vast.ai 容器没有 `CAP_SYS_ADMIN`，`ncu` 报 `ERR_NVGPUCTRPERM`，用 root 也没用。

计划里的六个 Nsight 指标，真正拿不到的只有两个（DRAM 吞吐、合并访存效率），
其余四个有别的来源。缺的那两个，用三条独立证据替代了：SASS 反汇编计数、访存请求数的推算、
N=4097 对照实验，详见[报告 02 · §7](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html)。
roofline 图用手算的算术强度加实测吞吐画出。

---

## 怎么复现

```bash
# SGEMM（RTX 4090）
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v sgemm_all.cu -o sgemm_all -lcublas
NVIDIA_TF32_OVERRIDE=0 bash run_p0.sh       # 九个阶段，50–80 分钟

# 归约
nvcc -O3 -arch=sm_89 -lineinfo reduce_ch10.cu -o reduce_ch10
./reduce_ch10 r4 268435456 11

# 融合算子
bash run_fused.sh                           # 编译、校验、计时、形状扫描、racecheck 一次跑完
```

`src/` 和 `Scripts/` 的文件放在同一个目录下运行。从开机、传文件、取回到关机的完整命令串，
以及几条踩坑得来的规矩，见 [`Scripts/build-and-bench.md`](Scripts/build-and-bench.md)。

---

## 目录

```
02-cuda-from-systems-eyes/
├── Reports/     4 份 HTML 报告
├── src/         sgemm_all.cu · reduce_ch10.cu · fused_ops.cu · devinfo.cu
├── Scripts/     build-and-bench.md + 各个驱动脚本
└── Results/     原始数据，一条没删
    ├── Round1/      SGEMM（2026-09-24）
    ├── Reduction/   归约（2026-09-21）
    └── Fused/       融合算子（2026-10-01）
```

三批数据来自三台不同的 4090，所以绝对耗时不跨批比较；报告里的比值都取自同一批。
每个文件的说明见 [`Results/README.md`](Results/README.md)。

---

## 环境

| | |
|---|---|
| **GPU** | NVIDIA RTX 4090 · 24 GB · sm_89 · 128 SM · L2 72 MiB（vast.ai 按小时租用） |
| **口径** | FP32 峰值 82.6 TFLOP/s · 显存带宽 1008 GB/s |
| **编译** | `nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v` · 所有 kernel 0 spill |

---

## 参考

- **PMPP 第 4 版** 第 1–6 章（执行模型、内存、性能）+ 第 10 章（归约）
- **siboehm, CUDA matmul** — <https://siboehm.com/articles/22/CUDA-MMM>（K4–K6 的思路来源；他在 A6000 上做到 93%，我在 4090 上做到 71.5%）
- **GPU MODE** L2 / L3 / L8 / L9 — <https://www.gpumode.com/lectures>
