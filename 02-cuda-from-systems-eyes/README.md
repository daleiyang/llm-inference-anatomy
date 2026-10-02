[← 返回 llm-inference-anatomy](../)

# 02 · CUDA from a Systems Engineer's Eyes

> **SGEMM、归约、融合算子三条线已完成。** 同一块 RTX 4090、同一个可执行文件、同一次开机，
> 把一个 FP32 矩阵乘从 naive 改到 **cuBLAS 的 71.5%**，共 **63.1 倍**。
>
> 但这份记录的价值不在这两个数字，而在于：**教科书给的那本账（算术强度 → roofline）
> 在七个 kernel 里失效了五次。** 我把它替换成了三本账，并且让每一级优化只动其中一本 ——
> 于是"这一步的收益归谁所有"没有含糊的余地。
>
> 然后拿 **PMPP 第 10 章的六级归约**做反面对照 —— 一个**真的**受带宽限制的算子：
> 算术强度恒为 0.25 FLOP/B（脊点的 **1/334**），一步也挪不动。
> 结果是书里五步优化**只有一步打中瓶颈**，其余四步收益为零。
> **同一套账，两种命运；差别只在瓶颈在哪一侧。**
>
> 最后用 **RMSNorm 与 online softmax** 两个融合算子把字节账用到推理里的真实算子上：
> 寄存器驻留版收在峰值带宽的 **90–91%**。但开跑前写死的十四条预测落空了五条，
> **全部是同一个原因：L2** —— 字节账应该算到达 DRAM 的字节，不是访存指令发出的字节。

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
| **真·带宽受限长什么样？** | 归约：算术强度恒为 **0.25 FLOP/B = 脊点的 1/334**，六个版本一步也挪不动。<br>上限一秒钟就能算出来：`4N ÷ 1008 GB/s` | [报告 03 · §1](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/03-归约六级优化-一道纯带宽题.html) |
| **那五步优化值多少？** | **只有一步打中瓶颈**（R3→R4 分段，每元素快 **292 倍**）。<br>其余四步收益为零 —— R4/R5/R6 全部收在下限的 **95%** | [报告 03 · §1](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/03-归约六级优化-一道纯带宽题.html) |
| **融合算子能跑到多少带宽？** | RMSNorm 与 online softmax 的寄存器驻留版都在峰值带宽的 **90–91%**（MBU），<br>相对基线分别快 **1.51×**（理论 1.50×）和 **1.70×**（理论 2.00×） | [报告 04 · §9](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/04-融合算子-RMSNorm与OnlineSoftmax.html) |
| **融合到底省的是什么？** | 省的是**重用距离**，访存指令一条没少。N2 的账面字节一点没省（仍是 12N），实测却快 **1.49×**：<br>第二次读和第一次读之间的距离从 112 MiB 缩到 14 KB，第二次读落进了 L2 | [报告 04 · §9](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/04-融合算子-RMSNorm与OnlineSoftmax.html) |

---

## 八个最硬的发现

### 1 · 教科书那本账，七次里失效五次

roofline 达成率超过 100%，意味着这个 kernel **根本没在带宽墙上** ——
那些"逻辑上要读的字节"绝大部分命中了 4090 的 75.5 MB L2，压根没走到显存。

```
K1 257%    K2 2112%    K3 77%    K3c 52%    K4 259%    K5 269%    K6 319%
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

**外推验收**：两个没参与标定的点，K3c 误差 +6.6%、K5 误差 −9.2%，都在 ±10% 以内。

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

### 6 · 换一个真的受带宽限制的算子，同一套方法给出相反的结论

归约（PMPP 第 10 章 R1–R6，N = 2²⁸ = 2.68 亿个 float）。
和 SGEMM 最大的不同是：**这里没有复用可做，每个元素只读一次**。

```
算术强度   0.250 FLOP/B   ← 六个版本完全相同，而 4090 的脊点是 83.4
                            0.250 ÷ 83.4 = 1/334 —— 不是"离得远"，是数学上挪不动

下限 = 4N 字节 ÷ 1008 GB/s = 1.0652 ms      ← 这一章的上限，一秒钟就能算出来
```

于是这一章的成绩单只有一个百分比，所有含糊的地方全部暴露：

| 这一步 | 书里衡量的 | 实测 |
|---|---|---|
| R1→R2 消除发散 | 资源利用率 29.5% → 66.4% | 净耗时 **4.40×**（但两者都只用 1 个 SM，微秒级，定量不可信） |
| R2→R3 搬进 shared | 全局访问 ~N·log N → N+1 | **0.92×** —— 8 KB 工作集本来就全在 L1 里，甚至略负 |
| **R3→R4 分段 + 原子加** | 书里只说"能处理任意长度" | **每元素快 292 倍** ← 唯一打中瓶颈的一步 |
| R4→R5 线程粗化 | 同步与原子加都除以 4 | 1.1203 → 1.1211 ms，**零** |
| R5→R6 warp shuffle | 最后 5 轮不走 shared | 1.1211 → 1.1212 ms，**零** |

R4 之后达成率 **95.1%**，剩下那 5% 是 L2/DRAM 的现实开销 —— **没有地方可以变快了**。
连"提高 occupancy"这条通用建议也只值 **0.1%**（66.7% 与 100% 打平）。

**两条线正好照出对方的形状**：SGEMM 教你怎么把复用做出来，
归约教你在**没有复用可做**的时候该看什么 —— 而推理里一大半算子属于后者。

### 7 · 融合算子：字节账要算到 DRAM，不能只算到指令

RMSNorm（N=3584）与 online softmax（N=2048）都取 Qwen2.5-7B 的真实维度，M=8192 行。
这两个算子都是访存密集型，所以指标从「占峰值算力」换成 **MBU = 实测 GB/s ÷ 1008**。

```
               字节/行    耗时 ms     MBU     相对基线   理论
N1 两趟           12N    0.3859    90.6%      基准
N2 融合但读两次    12N    0.2591   134.9%     1.49×    1.00×   ← 账面一字节没省，却快了一半
N3 寄存器驻留      8N    0.2560    91.0%     1.51×    1.50×
S1 三趟           16N    0.2509   106.1%      基准
S2 online 两趟    12N    0.2202    90.7%     1.14×    1.33×
S3 一趟            8N    0.1475    90.3%     1.70×    2.00×
```

**MBU 超过 100%，说明分子已经不是 DRAM 流量了。** 把 N3 实测的 917.5 GB/s 当作这张卡的实际天花板，
用耗时反推每一级真正搬运的字节数：八级里有六级落在账面的 100.0%–100.8%，只有两级明显偏离 ——

- **N2**：账面 12N，实际只搬了 **8.1N**。两次读之间，这个 block 只处理了自己那一行（14 KB），
  第二次读全部命中 L2；N1 是两个 kernel，两次读之间隔着全部 8192 行（112 MiB），L2 早被冲掉了。
- **S1**：x 只有 64 MiB，**刚好装得进 72 MiB 的 L2**，所以基线被抬快了，比值跟着缩水。
  把 M 加到 16384（x 涨到 128 MiB）后，S3 对 S1 立刻回到 **1.97×**，几乎正好是理论值。

十四条预测里成立 9 条、落空 4 条、口径失效 1 条，**没成立的 5 条都是被 L2 绊倒的**。
**融合省带宽的机制不是「少发访存指令」，而是「把重用距离压进 cache 装得下的范围」** ——
FlashAttention 押的也是这一点：它并没有减少读 Q/K/V 的次数，减少的是两次读之间的距离。

### 8 · 给下一阶段的接口

```
cuBLAS   decode b1   M=1    0.302 ms   449.6 GFLOP/s
         decode b32  M=32   0.313 ms   13882.7 GFLOP/s

  活多了 32×（M 从 1 变成 32），耗时只多 1.04×
  →  实测吞吐比 = 13882.7 ÷ 449.6 = 30.9×        // = 32 ÷ 1.04
```

**batch 32 和 batch 1 几乎花一样的时间，吞吐却差 30.9 倍。**
这就是 continuous batching 必须存在的量化理由 ——
也和 [01 阶段](../01-decode-latency-anatomy/) 在 vLLM 吞吐曲线上看到的宏观现象对上了：
**那边是现象，这里是机制。**

---

## 那句该改写的话

开工前我写下的推论是：

> "SGEMM 在 4090 上摸不到 FP32 峰值，因为 128×128 分块的算术强度只有 32，
> 而脊点是 82 —— 剩下的靠 L2 和寄存器复用补。"

**结论对了，因果链错了。** 实测之后应该这么说：

> **"SGEMM 在 4090 上摸不到 FP32 峰值，不是因为带宽 —— K6 的 roofline 达成率 319%，
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
| **01** | [**核心代码的演化**](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/01-SGEMM六级优化-核心代码的演化.html) | 每一级改了哪一行、为什么那样改 | 17 张原理图，六级代码逐个拆 |
| **02** | [**实测与三本账**](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html) | 数字从哪来、八个指标怎么算 | 从代码数出七个指标，再和实测对账 |
| **03** | [**归约六级优化：一道纯带宽题**](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/03-归约六级优化-一道纯带宽题.html) | 没有复用可做的算子该看什么 | 五步优化只有一步打中瓶颈 |
| **04** | [**融合算子：RMSNorm 与 online softmax**](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/04-融合算子-RMSNorm与OnlineSoftmax.html) | 归约之后拿结果回头缩放整行，融合到底省了什么 | 收在峰值带宽 90–91%，五条落空预测全栽在 L2 |

**建议读法**：先 01 建立"代码长什么样"的直觉，再 02 看"凭什么这么判断"。
02 的 §3 是全套方法的核心 —— **八个指标里只有一个是量出来的，其余七个全部从代码数出来**。
03 可以单独读：它换了一个**没有复用可做**的算子，把同一套账跑了一遍，结论正好相反。
04 接在 03 后面：block 级归约之后多了一步「拿归约结果回头缩放整行」，指标换成 MBU。
它的 §9 是这一阶段最有价值的一次「对不上」—— 字节账和实测之间还隔着一层 L2。

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
# 测试机（RTX 4090）—— src/ 和 Scripts/ 的内容放在同一个目录下跑
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v sgemm_all.cu -o sgemm_all -lcublas
export NVIDIA_TF32_OVERRIDE=0
bash run_p0.sh                  # 九个阶段，50~80 分钟

# 本地
python parse_results.py --outdir Results/Round1/csv all_*.txt shapes_*.txt
```

归约那条线独立于上面这一整套，单文件、无依赖：

```bash
nvcc -O3 -arch=sm_89 -lineinfo reduce_ch10.cu -o reduce_ch10
# 用法：reduce_ch10 <r1|r2|r3|r4|r5|r6|none> <N> [ITERS] [BLOCK]

./reduce_ch10 none 2048 1000          # 实验 A 的基线：空 kernel 的启动开销
./reduce_ch10 r3   2048 1000          # 实验 A：单 block，N=2048，量微观差异
./reduce_ch10 r4 268435456 11         # 实验 B：整卡，N=2²⁸，量带宽达成率
./reduce_ch10 r5 268435456 11 1024    # 第四个参数是 BLOCK —— 动态 shared，不必重编译
```

**实验 A 必须减掉空 kernel 的基线**：N=2048 时 kernel 本身和启动开销同量级，
不减就什么也比不出来。完整命令串见 [`Results/Reduction/reduce_20260921.txt`](Results/Reduction/reduce_20260921.txt) ——
那是原样的终端输出，报告 03 的每个数字都由构建脚本从它现场解析。

融合算子那条线也是单文件，外加一个总驱动脚本：

```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v fused_ops.cu -o fused_ops
# 用法：fused_ops <n1|n2|n3|n4|s1|s2|s3|s4> [M] [N] [ITERS]

./fused_ops n3 8192 3584 11           # RMSNorm 主阶梯：M=8192 行 × hidden 3584
./fused_ops s3 8192 2048 11           # softmax 主阶梯：M=8192 行 × seq 2048

bash run_fused.sh                     # 全套：编译 + spill 检查 → 正确性与边界 → 主阶梯 × 3 轮
                                      #       → 形状扫描 → 长行拒绝 → racecheck
ROUNDS=1 bash run_fused.sh            # 快速过一遍
```

**spill 检查是这一套的命门**：n3/n4/s3/s4 的前提是整行留在寄存器里，一旦 spill，数据就落回显存，
1.5× 和 2.0× 都不会出现，而且从耗时上看不出原因。十一个 kernel 必须全部 `0 bytes spill stores`。
原样日志见 [`Results/Fused/fused_20261001_005225.txt`](Results/Fused/fused_20261001_005225.txt)。

> 从开机到收工的完整命令串（含传文件、取回、关机）在 [`Scripts/build-and-bench.md`](Scripts/build-and-bench.md)。

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
├── Reports/                     # 4 份自包含 HTML 报告
│   ├── README.md                # 四份报告的分工与建议读法
│   ├── 01-SGEMM六级优化-核心代码的演化.html
│   ├── 02-SGEMM实测与三本账.html
│   ├── 03-归约六级优化-一道纯带宽题.html
│   └── 04-融合算子-RMSNorm与OnlineSoftmax.html
├── src/
│   ├── sgemm_all.cu             # 八个 kernel（K0 cuBLAS + K1–K6）+ 测试脚手架
│   ├── reduce_ch10.cu           # 归约 R1–R6 + 空 kernel 基线
│   ├── fused_ops.cu             # RMSNorm n1–n4 + online softmax s1–s4（11 个 kernel）
│   ├── devinfo.cu               # 设备自查
│   └── sgemm.cu                 # 开工前的空框架（kernel 留白），留作对照
├── Scripts/
│   ├── build-and-bench.md       # 从开机到收工的完整命令串
│   ├── run_p0.sh                # 总驱动，九个阶段
│   ├── run_all.sh               # 计时引擎
│   ├── sweep_shapes.sh          # 形状扫描
│   ├── check_sass.sh            # 反汇编计数
│   ├── collect_ncu.sh           # Nsight 采集（本轮因权限未成功）
│   ├── run_fused.sh             # 融合算子总驱动，六个阶段
│   └── parse_results.py         # 文本 → CSV
└── Results/
    ├── README.md                # 每个文件是什么、两份 CSV 的分工
    ├── Round1/                  # SGEMM（2026-09-24）原始数据，一条没删
    │   ├── all_20260924_180114.txt          # 计时主表（1555 行）
    │   ├── all_20260924_180114_clocks.csv   # 1 Hz 时钟 / 温度 / 功耗采样
    │   ├── shapes_20260924_180619.txt       # 形状扫描
    │   ├── sass_20260924_180113{,_raw}.txt  # 指令统计 + 完整反汇编
    │   ├── ncu_20260924_180816.log          # 权限失败的现场
    │   ├── fingerprint.txt                  # 环境指纹
    │   └── csv/bench_*.csv                  # 解析后：逐轮的是证据，中位数的是结论
    ├── Reduction/               # 归约（2026-09-21，另一台 4090）
    │   └── reduce_20260921.txt              # 原样终端输出，报告 03 现场解析
    └── Fused/                   # 融合算子（2026-10-01，又一台 4090）
        └── fused_20261001_005225.txt        # run_fused.sh 原样日志（787 行）
```

---

## 完成标志

- [x] 六级 SGEMM 全部通过正确性校验（对 CPU 参照抽样，含 N=4097 边界）
- [x] 最快版本达到 cuBLAS 的 **70%+** —— 71.5%
- [ ] 每一级都有 Nsight 指标佐证 —— **未完成**，实例无性能计数器权限（已用 SASS + 请求账 + 4097 实验替代）
- [x] 算术强度的理论预测与实测对账，差异有解释 —— **对账结果是预测被推翻，解释见上**
- [x] 六级归约全部通过 double 参照校验，R4 之后收在带宽下限的 **95%**
- [x] 归约的五条预测在开跑前写死判决标准，事后逐条对账 —— **押中四条**，没押中的是 R2→R3
- [x] RMSNorm / online softmax 融合版 MBU 70%+ —— N3/N4 **91.0%**、S3/S4 **90.3%**；十一个 kernel 全部 0 spill，racecheck 全部 0 hazards
- [x] 融合算子的十四条预测开跑前写死、事后逐条判决 —— **成立 9 条**，落空的 5 条全部追到 L2
- [x] SGEMM 两份 + 归约一份 + 融合算子一份，共四份报告产出

---

## 环境

| | |
|---|---|
| **GPU** | NVIDIA RTX 4090 · 24 GB · driver 580.159.03 · sm_89 · 128 SM · L2 75.5 MB |
| **口径** | FP32 峰值 82,575 GFLOP/s（按设备报告的 2.52 GHz）· 显存带宽 1008 GB/s |
| **编译** | `nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v` · 七个 kernel 全部 0 spill |
| **规模** | 8 个 kernel × 4 个尺寸 × 3 轮 + 4 组形状 + 边界 + racecheck，同批同二进制 |
| **数据** | `Results/Round1/`，原始输出 1555 行 + 完整 SASS 反汇编 8981 行 |
| **归约那一轮** | 2026-09-21，另一台 RTX 4090（口径相同）· 9 组测量：实验 A 单 block × 5 + 实验 B 整卡 × 3 + BLOCK 对照 × 1 |
| **归约数据** | `Results/Reduction/reduce_20260921.txt`，171 行原样终端输出 |
| **融合算子那一轮** | 2026-10-01，又一台 RTX 4090 · driver 595.91.07 · 口径相同 · 8 级 × 主阶梯 3 轮 + 4 组形状 + 边界 + 长行拒绝 + racecheck |
| **融合算子数据** | `Results/Fused/fused_20261001_005225.txt`，787 行原样日志 |

---

## 参考

- **PMPP 第 4 版** 第 1–6 章（执行模型、内存、性能）+ 第 10 章（归约；融合算子是它的应用）
  *K1/K2/K3/K3c 分别出自第 3/4/5/6 章；K4/K5/K6 书上没有。*
- **siboehm, CUDA matmul** — <https://siboehm.com/articles/22/CUDA-MMM>
  *K4/K5/K6 的思路来源。他的结果在 A6000（Ampere）上，4090 是 Ada、L2 大得多 ——
  **数字必然不同，量出自己的数才是这个项目的价值**。他做到 93%，我做到 71.5%。*
- **GPU MODE** L2 / L3 / L8 / L9 — <https://www.gpumode.com/lectures>
- **Nsight Compute** metrics 文档
