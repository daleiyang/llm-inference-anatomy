[← 返回 llm-inference-anatomy](../)

# 02 · CUDA from a Systems Engineer's Eyes

> **SGEMM 部分已完成。** 同一块 RTX 4090、同一个可执行文件、同一次开机，
> 把一个 FP32 矩阵乘从 naive 改到 **cuBLAS 的 71.5%**，共 **63.1 倍**。
>
> 但这份记录的价值不在这两个数字，而在于：**教科书给的那本账（算术强度 → roofline）
> 在七个 kernel 里失效了五次。** 我把它替换成了三本账，并且让每一级优化只动其中一本 ——
> 于是"这一步的收益归谁所有"没有含糊的余地。

我写过内存池、lock-free hash table，也逐行读过 word2vec.c。这一阶段要回答的问题是：
**那些 CPU 上的性能直觉，搬到 GPU 上还剩多少？**

答案是——**一半能直接用，另一半会主动害你**。下面是把这条线划清楚的过程。

---

## 30 秒结论

| 问题 | 答案 | 出处 |
|---|---|---|
| **手写 kernel 能追到 cuBLAS 多近？** | **71.5%**（K6 3.358 ms vs cuBLAS 2.401 ms）。cuBLAS 只比它快 **1.40 倍** | [报告 02 · §1](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html) |
| **提高算术强度就能更快吗？** | **不能**。收益最大的两步（8.2× 和 3.1×）**算术强度一个字都没变**；<br>算术强度真涨 1.6 倍的那一步只换来 8% | [报告 01 · §1](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/01-SGEMM六级优化-核心代码的演化.html) |
| **那什么才是墙？** | 三本账：字节账 / 请求账 / **条数账**。K3 之后全靠第三本，<br>外推误差 **±10% 以内** | [报告 02 · §3](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html) |
| **naive kernel 慢在哪？** | 那 8.2 倍里**只有 2.4 倍是"不合并"**，另外 **3.4 倍是 2 的幂跨度冲突** ——<br>把 N 从 4096 改成 4097，naive 快 3.33 倍 | [报告 02 · §6](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html) |
| **occupancy 越高越好吗？** | **反过来**。K5 把占用率从 75% 砍到 33.3%，速度涨 1.66 倍 | [报告 01 · §7](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/01-SGEMM六级优化-核心代码的演化.html) |
| **"优化了 63 倍"是真的吗？** | **只在方阵上成立**。换成 decode 形状只剩 **3.4 倍**，<br>而且 K6 反而比 K4 **慢** | [报告 02 · §5](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html) |
| **cuBLAS 强在哪一侧？** | 只在**算力受限**那一侧。decode 上我们追平它，K4 甚至跑到 **102.6%** | [报告 02 · §5](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html) |

---

## 六个最硬的发现

### 1 · 教科书那本账，七次里失效五次

roofline 达成率超过 100%，意味着这个 kernel **根本没在带宽墙上** ——
那些"逻辑上要读的字节"绝大部分命中了 4090 的 75.5 MB L2，压根没走到显存。

```
K1 258%    K2 2113%    K3 77%    K3c 52%    K4 261%    K5 271%    K6 322%
                ↑ 超了二十倍
```

**一个不受带宽限制的 kernel，你把它的 DRAM 算术强度提高 32 倍，换不来时间。**
K2 → K3 就是这样：算术强度 0.250 → 7.969（涨 32 倍），实测只快 **1.16 倍**。

### 2 · 三本账，每一级只动一本

| 这一步 | 字节账<br>算术强度 | 请求账<br>请求/FMA | 条数账<br>load/FMA | 实测 |
|---|---|---|---|---|
| K1→K2 | 不变 0.25 | **33 → 2** | 不变 2.000 | **8.21×** |
| K2→K3 | 0.25 → 7.97 | 不变 | 2.000 → 2.062 | 1.16× |
| K3→K3c | 7.97 → 12.72 | 不变 | 2.062 → 2.039 | 1.08× |
| K3→K4 | 不变 7.97 | 不变 | **2.062 → 1.188** | **3.10×** |
| K4→K5 | 7.97 → 12.72 | 不变 | **1.188 → 0.414** | 1.66× |
| K5→K6 | 不变 12.72 | 不变 | **0.414 → 0.104** | 1.19× |

**收益大的每一步都动了请求账或条数账；只动字节账的两步，收益都在 10% 上下。**

条数账的三个单价（32 路 shared / 广播 shared / 全局 load）用 K2、K3、K4 三点解出：

```
32 路 LDS  2.69 周期      广播 LDS  0.53 周期      全局 LDG  1.94 周期
```

**外推验收**：两个没参与标定的点，K3c 误差 +6.6%、K5 误差 −7.6%，都在 ±10% 以内。

### 3 · naive 的 8 倍劣势里，四分之三和"不合并"无关

边界测试本来只查越界，结果 **K1 在 N=4097 上比 N=4096 快 3.33 倍**（63.675 vs 211.911 ms），
而其他 kernel 在 4097 上只慢 2.8%–7.1%（cuBLAS 慢 15.7%）。

```
            N=4096（2 的幂）    N=4097（打破对齐）
K1/K2          8.209×             2.392×        ← 后者才是"不合并"本身的代价

8.209 ÷ 2.392 = 3.43   ← 2 的幂跨度的额外惩罚
回乘检验：2.392 × 3.43 = 8.209，逐位相同
```

N=4096 时 A 的行跨度恰好 16384 字节 = 2¹⁴，4097 把它变成 16388 —— 唯一改动的变量就是这个。
**所以"合并访存是性价比最高的一步"要加个注**：那 8 倍里只有 2.4 倍是合并本身。

> **机制没查出来，如实标注。** 最顺手的解释是"warp 内 32 个地址撞同一个 L2 set"，
> 但把地址逐位算一遍会发现：两种 N 的**行号模式完全相同**（都是 `base + 128t`），
> sector 效率也都是 12.5% —— 差别只在 128 字节行的**内部**，而那几位不参与 set/bank 选择。
> 剩下的候选是 L2 slice 哈希或 DRAM row buffer，**两个都要 ncu 才能判，这一批没有**。
> 留一个悬案，比给个听起来合理的解释诚实。

### 4 · 编译器早就替我做了一半的事

`cuobjdump -sass` 数一遍静态指令（K5 和 K6 的 FFMA 条数完全相同，可直接比）：

```
kernel     FFMA  LDS.128   LDS.32  LDG.128   LDG.32    TOTAL
k5         1056       64      128        -       72     1856   ← 预测说这里该是 0
k6         1056       96        -       18        -     1520
```

**K5 本来就有 64 条 `LDS.128`** —— 编译器自己把它三分之一的 shared load 向量化了。
所以 K5→K6 不是"标量 vs 向量化"，是"部分 vs 完全向量化"：
**真实访存指令比 2.32 倍，而非手推条数账说的 3.98 倍。**

**一个没预测到的旁证**：K1 和 K2 的 SASS **逐项完全相同**（30 FFMA / 59 LDG.32 / 216 条）。
它们只差一个索引映射。**所以那 8.2 倍里，指令条数的贡献严格为零，100% 是访存行为** ——
这是"请求账必须独立于条数账"最干净的证据。

### 5 · "优化了 63 倍"只在方阵上成立

N/K 取自 Qwen2.5-7B 的 FFN 维度（18944 × 3584）：

```
方阵 4096³     K1 → K6   63.4×
prefill        K1 → K6   49.2×
decode b32     K1 → K6   23.8×
decode b1      K1 → K6    3.4×   ← 只剩零头
```

**而且 decode 上 K6 反而比 K4 慢（0.95×）。** K6 多出来的本事全部用来砍指令条数，
而那一侧的墙是带宽 —— 砍条数不但没用，更大的 tile 在 M=1 时还浪费更多线程。
**这把"三本账只在算力受限一侧有效"钉死了。**

**占 cuBLAS（%）**：

| | 方阵 | prefill | decode-b32 | decode-b1 |
|---|---|---|---|---|
| K4 | 36.2% | 25.5% | **102.6%** | **100.7%** |
| K6 | **71.5%** | 57.1% | 97.5% | 94.7% |

decode 上我们几乎追平 cuBLAS，K4 甚至超过它 —— 这和事前预测正好相反。
原因不难想：**带宽墙对谁都一样高**，cuBLAS 那些精巧的分块调度在这里无处施展。
**它的优势只在算力受限那一侧兑现。**

### 6 · 给下一阶段的接口

```
decode b1   0.302 ms        decode b32  0.313 ms        耗时比 1.04×
                                                        吞吐比 32×
```

**batch 32 和 batch 1 花一样的时间，吞吐差 32 倍。**
这就是 continuous batching 必须存在的量化理由 ——
也和 [01 阶段](../01-decode-latency-anatomy/) 在 vLLM 吞吐曲线上看到的宏观现象对上了：
**那边是现象，这里是机制。**

---

## 那句该改写的话

开工前我写下的推论是：

> "SGEMM 在 4090 上摸不到 FP32 峰值，因为 128×128 分块的算术强度只有 32，
> 而脊点是 82 —— 剩下的靠 L2 和寄存器复用补。"

**结论对了，因果链错了。** 实测之后应该这么说：

> **"SGEMM 在 4090 上摸不到 FP32 峰值，不是因为带宽 —— K6 的 roofline 达成率 322%，
> 说明它早就不在那堵墙上了。真正的墙在 LSU：算力地板是 0.25 周期/warp圈，K6 实测 0.50，
> 差的那一倍是 3 条 128 位 shared load 占掉的发射槽。
> 要再往上走，只能继续砍 load 条数（warptiling）或者换张量核心。"**

开工前的另外几条预测，逐条对账：

| 预测 | 结果 |
|---|---|
| naive 算术强度 0.25 → 上限约 **252 GFLOP/s** | ❌ 实测 **648.6**，超了 2.6 倍（L2 命中，没走 DRAM） |
| K3 的提升会不如预期 | ✅ 1.16×，**但理由错了**（不是"离脊点远"，是 K2 本来就不在带宽墙上） |
| K4 是提升最大的一步 | ✅ 3.10×，**但机制完全不同**（算术强度一动没动，动的是指令条数） |
| K5 occupancy 下降但更快 | ✅ 75% → 33.3%，快 1.66 倍 |
| K6 小幅 | ✅ 1.19× |
| 能到 cuBLAS 的 **80–90%** | ❌ 做到 **71.5%**，差的 21 个点是 warptiling + double buffering |

---

## 报告目录

> 所有报告都是**自包含单文件**：数据与 SVG 全部内嵌，无外部依赖，支持明暗两套主题。
> 仓库里直接点 `.html` 只会看到源码，需经 GitHub Pages 渲染。

| # | 报告 | 回答什么 | 一句话 |
|---|---|---|---|
| **01** | [**核心代码的演化**](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/01-SGEMM六级优化-核心代码的演化.html) | 每一级改了哪一行、为什么那样改 | 9 张原理图，六级代码逐个拆 |
| **02** | [**实测与三本账**](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html) | 数字从哪来、八个指标怎么算 | 从代码数出七个指标，再和实测对账 |

**建议读法**：先 01 建立"代码长什么样"的直觉，再 02 看"凭什么这么判断"。
02 的 §3 是全套方法的核心 —— **八个指标里只有一个是量出来的，其余七个全部从代码数出来**。

---

## 一个没拿到的东西

**Nsight Compute 的六个指标和 roofline 图没采到。**

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
          NVIDIA GPU Performance Counters on the target device 0.
```

租用实例的容器没有 `CAP_SYS_ADMIN`（实测 `CapEff = 0xa80405fb`，正是 Docker 默认集），
是 root 也没用 —— 这个能力在 `docker run` 创建容器时就定死了。

**六个指标里只有两个是真的拿不到**（DRAM 吞吐、每 sector 有效字节），其余四个都有替代来源。
那两个想证明的事（"K1 的劣势来自访存不合并"），用三条独立证据替代了，
**而且比 ncu 单个百分比证得更完整** —— 详见[报告 02 · §7](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html)。

**没有藏这件事**：脚本里的自检在编译后 5 秒就探明了权限问题并跳过后两步，
前面三步的数据一个没丢。

---

## 怎么复现

```bash
# 测试机（RTX 4090）
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v sgemm_all.cu -o sgemm_all -lcublas
export NVIDIA_TF32_OVERRIDE=0
bash run_p0.sh                  # 九个阶段，50~80 分钟

# 本地
python parse_results.py all_*.txt shapes_*.txt     # 文本 → CSV
```

四条关键规矩：

1. **八个 kernel 必须进同一个可执行文件。**
   "K6 是 cuBLAS 的百分之多少"这类比值，只有在同一个二进制、同一次开机下才成立。
   *实测：跨批次绝对耗时漂 2–3%，同批内比值只漂 0.5%。*
2. **cuBLAS 必须双保险锁死 FP32**：源码里 `cublasSetMathMode(CUBLAS_PEDANTIC_MATH)`，
   环境里 `NVIDIA_TF32_OVERRIDE=0`。*否则 Ada 会偷偷用 TF32 张量核心，那就不是同一个东西在比。*
3. **预热至少一次，哪怕只跑 2 轮。**
   cuBLAS 第一次调用要惰性加载 kernel 库，一次几十毫秒。
   *这个坑真踩过 —— K0 在 512 上报出过 55 ms，实际是 0.017 ms。*
4. **`ncu` 的权限自检要放在最前面。**
   编译完立刻探 5 秒，不行就当场换机器。*放在后面会白等四十分钟。*

---

## 目录

```
02-cuda-from-systems-eyes/
├── README.md                    # 本页
├── Reports/                     # 2 份自包含 HTML 报告
│   ├── 01-SGEMM六级优化-核心代码的演化.html
│   └── 02-SGEMM实测与三本账.html
├── src/
│   ├── sgemm_all.cu             # 八个 kernel（K0 cuBLAS + K1–K6）+ 测试脚手架
│   ├── devinfo.cu               # 设备自查
│   └── sgemm.cu                 # 开工前的空框架（kernel 留白），留作对照
├── Scripts/
│   ├── run_p0.sh                # 总驱动，九个阶段
│   ├── run_all.sh               # 计时引擎
│   ├── sweep_shapes.sh          # 形状扫描
│   ├── check_sass.sh            # 反汇编计数
│   ├── collect_ncu.sh           # Nsight 采集（本轮因权限未成功）
│   └── parse_results.py         # 文本 → CSV
└── Results/Round1/              # 原始数据，一条没删
    ├── all_20260924_180114.txt          # 计时主表（1555 行）
    ├── all_20260924_180114_clocks.csv   # 1 Hz 时钟 / 温度 / 功耗采样
    ├── shapes_20260924_180619.txt       # 形状扫描
    ├── sass_20260924_180113{,_raw}.txt  # 指令统计 + 完整反汇编
    ├── ncu_20260924_180816.log          # 权限失败的现场
    ├── fingerprint.txt                  # 环境指纹
    └── csv/bench_*.csv                  # 解析后：逐轮的是证据，中位数的是结论
```

---

## 完成标志

- [x] 六级 SGEMM 全部通过正确性校验（对 CPU 参照抽样，含 N=4097 边界）
- [x] 最快版本达到 cuBLAS 的 **70%+** —— 71.5%
- [ ] 每一级都有 Nsight 指标佐证 —— **未完成**，实例无性能计数器权限（已用 SASS + 请求账 + 4097 实验替代）
- [x] 算术强度的理论预测与实测对账，差异有解释 —— **对账结果是预测被推翻，解释见上**
- [ ] RMSNorm / online softmax 融合版 MBU 70%+ —— 代码与导学已完成，**待上机**
- [x] SGEMM 两份报告产出

---

## 环境

| | |
|---|---|
| **GPU** | NVIDIA RTX 4090 · 24 GB · driver 580.159.03 · sm_89 · 128 SM · L2 75.5 MB |
| **口径** | FP32 峰值 82,575 GFLOP/s（按设备报告的 2.52 GHz）· 显存带宽 1008 GB/s |
| **编译** | `nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v` · 七个 kernel 全部 0 spill |
| **规模** | 8 个 kernel × 4 个尺寸 × 3 轮 + 4 组形状 + 边界 + racecheck，同批同二进制 |
| **数据** | `Results/Round1/`，原始输出 1555 行 + 完整 SASS 反汇编 8981 行 |

---

## 参考

- **PMPP 第 4 版** 第 1–6 章（执行模型、内存、性能）+ 第 10 章（归约）
  *K1/K2/K3/K3c 分别出自第 3/4/5/6 章；K4/K5/K6 书上没有。*
- **siboehm, CUDA matmul** — <https://siboehm.com/articles/22/CUDA-MMM>
  *K4/K5/K6 的思路来源。他的结果在 A6000（Ampere）上，4090 是 Ada、L2 大得多 ——
  **数字必然不同，量出自己的数才是这个项目的价值**。他做到 93%，我做到 71.5%。*
- **GPU MODE** L2 / L3 / L8 / L9 — <https://www.gpumode.com/lectures>
- **Nsight Compute** metrics 文档
