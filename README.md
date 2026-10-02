# llm-inference-anatomy

**把大模型推理拆开看：先从模型结构和硬件规格算出该有的数，再上机器量，看两边对不对得上。**

大多数推理性能仓库是「我跑了一堆 benchmark」。这个仓库反过来做 ——
对上了，这个数就能直接拿去做容量规划；对不上，那个缺口本身就是发现。

**对上的例子**（01 阶段）：手算 KV Cache 每 token 占多少显存，和 vLLM 启动日志只差 **0.03%**。

```
2 (K,V) × 28 层 × 4 KV头 × 128 dim × 2 B = 56 KiB / token
6.41 GiB ÷ 56 KiB = 120,025 token        vLLM 日志实测 120,064 token
```

于是「这张卡能扛多少并发」变成一道除法，**不需要压测**。

**对不上的例子**（02 阶段）：教科书说「提高算术强度就能更快」。手写六级矩阵乘实测下来，**七个 kernel 里错了五次** ——
收益最大的两步（8.2× 和 3.1×）算术强度一点没变。顺着这个缺口，找到了真正决定速度的另外两本账。

> **先算，再测，然后对账。对上了拿去用；对不上，就去查为什么 —— 后者的收获通常更大。**

> **在线阅读**：<https://daleiyang.github.io/llm-inference-anatomy/>
> 报告都是自包含的单页 HTML，经 GitHub Pages 渲染；直接在仓库里点 `.html` 只会看到源码。

---

## 已完成的两个阶段

### [01 · Decode Latency Anatomy](01-decode-latency-anatomy/) —— 一张 4090 上的推理性能解剖

把 Qwen2.5-7B 放在一张 RTX 4090 上拆开量：1 份结构拆解 + 6 份压测报告，
7,594 条请求，**两轮跨机器独立复现**。

| | |
|---|---|
| **容量** | 能扛多少并发不用压测，可以算 —— 实测与理论差 **0.045%** |
| **延迟** | 过载时 **98.9%** 的首字延迟是在排队；要限流，不是换更快的卡 |
| **过载** | 显存超卖 4.4 倍，**零失败** —— 代价全部记在延迟上 |
| **选型** | vLLM 与 llama.cpp 的分界线是延迟预算（p99 首字延迟 ≈ **730 ms**），不是并发数 |
| **量化** | 单流快 **2.71 倍**，但省不了调度：64 并发时 roofline 达成率 81% vs 25% |

→ **[进入第一阶段](01-decode-latency-anatomy/)**

### [02 · CUDA from a Systems Engineer's Eyes](02-cuda-from-systems-eyes/) —— 手写 GPU kernel

在 RTX 4090 上手写三类 kernel，每一类都先算理论上限、再测、再对账。4 份报告。

| | |
|---|---|
| **矩阵乘** | 从 naive 优化到 **cuBLAS 的 71.5%**，共快 63 倍 |
| **归约** | 收在带宽理论下限的 **95%**；书里五步优化只有一步真正有用 |
| **融合算子** | RMSNorm / online softmax 跑到峰值带宽的 **90–91%** |
| **最反直觉的一条** | 「提高算术强度就能更快」在 7 个 kernel 里错了 5 次 |
| **没做到的** | Nsight 指标：租用实例没有性能计数器权限，已用三条独立证据替代 |

→ **[进入第二阶段](02-cuda-from-systems-eyes/)**

---

## 路线图

| 阶段 | 内容 | 要证明什么 | 状态 |
|---|---|---|---|
| **01** | [**Decode Latency Anatomy**](01-decode-latency-anatomy/)<br>推理成本模型：prefill / decode、KV Cache、roofline | 我懂成本模型 | ✅ 已完成 |
| **02** | [**CUDA from a Systems Engineer's Eyes**](02-cuda-from-systems-eyes/)<br>手写矩阵乘逼近 cuBLAS；归约与融合算子做带宽侧 | 我能写 GPU 代码 | ✅ 已完成<br>（Nsight 指标因权限未采到） |
| 03 | **Quantization Shootout**<br>同一模型跨 GGUF-Q4_K_M / AWQ / GPTQ / FP16，比质量、速度、显存 | 我有 serving 判断力 | 📋 计划中 |
| 04 | **FlashAttention Forward in Triton**<br>实现 FA-2 前向，对 `sdpa` 验证并基准 | 我是真懂，不是会调库 | 📋 计划中 |
| 05 | **Speed Up a Real Deployment**<br>baseline → 调 batching / prefix-cache + 量化 + 投机解码，报告每一步的收益 | 这就是客户交付物本身 | 📋 计划中 |

---

## 方法：四条规矩

1. **先算，再测，然后对账。** 任何数上机器之前，先从模型结构或硬件规格推一遍；对不上的缺口单独拎出来查。
2. **自检不过的数据不进报告。** 每轮跑完立刻查输出 token 数、采样参数、缓存开关、失败数 ——
   01 阶段靠这条查出了 `--ignore-eos` 在两个引擎上行为不一致。
3. **原始数据一条不删。** 原始输出、服务端采样、完整日志、环境指纹全部入库，报告里每个数都能追回源文件。
4. **写明不知道什么。** 每份报告最后都有「没搞清的、没测的」，标明哪些只是相关性、哪个数不该被引用。

---

## 目录约定

```
llm-inference-anatomy/
├── README.md              # 本页
└── NN-<阶段名>/
    ├── README.md          # 该阶段的入口：结论、报告索引
    ├── Reports/           # 自包含 HTML 报告
    ├── Results/           # 原始数据，一条不删
    └── Scripts/           # 可复制粘贴的完整命令串
```

**环境**：NVIDIA RTX 4090 · 24 GB（vast.ai 按小时租用）· Qwen2.5-7B-Instruct · vLLM 0.27.1 · llama.cpp。
报告为中文，代码与命令为英文。
