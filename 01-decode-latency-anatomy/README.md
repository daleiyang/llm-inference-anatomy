[← 返回 llm-inference-anatomy](../)

# 01 · 把一个 7B 拆开量：一张 4090 上的推理性能解剖

**同一张 RTX 4090、同一个 Qwen2.5-7B，吞吐能差 22 倍，首字延迟能差 215 倍 —— 全看你怎么发请求。**

这一阶段做了两件事：**先从模型结构把关键常数算出来，再用压测逐一验证。**

| | |
|---|---|
| **算** | 1 份逐层结构拆解：76 亿参数摊开，推出每 token 的 KV 占用（56 KiB）和计算量 |
| **测** | 10 组压测、7,594 条请求，vLLM 与 llama.cpp 对比，**两轮跨机器独立复现** |
| **对账** | 手算的 KV 池容量与 vLLM 启动日志差 **0.03%**，KV 占用的线性拟合与理论差 **0.045%** |

---

## 六个结论

| 问题 | 答案 | 报告 |
|---|---|---|
| **这张卡能扛多少并发？** | **不用压测，可以算**：KV 池 ÷（每 token 56 KiB × 上下文长度）。实测与理论差 0.045% | [KV 的线性](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/03-vllm-KV的线性.html) |
| **并发加到多少开始亏？** | 吞吐到 64 并发都还在涨，但**从 8 并发起，延迟尾部就裂开了**（p99 ITL 17.6 → 102.1 ms） | [并发曲线](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/01-vllm-Qwen2.5-7B并发曲线.html) |
| **首字延迟慢，该换卡吗？** | 不该。过载时 **98.9% 的首字延迟是在排队**，真正计算只要 1.1 秒。要的是限流，不是更快的卡 | [TTFT 解剖](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/02-vllm-TTFT解剖.html) |
| **显存（KV 池）压爆会怎样？** | 超卖 4.4 倍、抢占 38 次，**零失败，输出一个 token 不少**。代价全部记在延迟上 | [断崖](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/04-vllm-断崖.html) |
| **量化省下了什么？** | 省字节（单流快 **2.71 倍**），**省不了调度**（64 并发时 roofline 达成率 vLLM 81% vs llama.cpp 25%） | [llama.cpp 压测](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/05-llama-cpp-压测.html) |
| **选 vLLM 还是 llama.cpp？** | 看延迟预算，不看并发数：p99 首字延迟要求比 **730 ms** 紧选 llama.cpp，比它松选 vLLM | [选型指南](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/06-选型指南-vllm-llama-cpp.html) |

还有两条反直觉的观察：
- **p99 ITL 看不见抢占。** 38 次抢占混在 26 万个 token 间隔里只占 0.0145%，连 p99 都够不着 —— 别拿它做抢占告警。
- **两个计数器会打架。** 抢占导致的重算真实发生了（FLOPs 计数器多出 29%），但调度器的 token 计数器一个没记。指标本身也要交叉验证。

---

## 报告

报告是自包含的单页 HTML，经 GitHub Pages 渲染，点开即读。

**先读理论底座：** [**Qwen2.5-7B 结构拆解**](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Qwen2.5-7B%20结构拆解.html) ——
后面所有压测用到的两个常数（每 token 56 KiB 的 KV、每 token 13–14 GFLOP）都是在这里从模型结构算出来的。

**第二轮压测（最终版，建议从 06 读起）：**

| # | 报告 | 一句话 |
|---|---|---|
| 06 | [选型指南](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/06-选型指南-vllm-llama-cpp.html) | 分界线是 p99 首字延迟 ≈ 730 ms，不是并发数 |
| 01 | [并发曲线](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/01-vllm-Qwen2.5-7B并发曲线.html) | 拐点在 8 并发，卡住它的是调度器而不是显存 |
| 02 | [TTFT 解剖](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/02-vllm-TTFT解剖.html) | 首字延迟拆成排队 / prefill / 前端三段，排队是主因 |
| 03 | [KV 的线性](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/03-vllm-KV的线性.html) | KV 占用严格线性（R²=0.99993），容量可以算 |
| 04 | [断崖](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/04-vllm-断崖.html) | 超卖 4.4 倍仍零失败，抢占是一个稳定的控制环 |
| 05 | [llama.cpp 压测](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round2/05-llama-cpp-压测.html) | 量化让单流快 2.71 倍，但批处理效率差 3 倍多 |

**第一轮**保留了全过程，包括后来被第二轮修正的判断：
[并发曲线](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round1/01-vllm-Qwen2.5-7B%20并发曲线.html) ·
[TTFT 解剖](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round1/02-vllm-TTFT%20解剖.html) ·
[KV 的线性](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round1/03-vllm-KV%20的线性.html) ·
[断崖](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round1/04-vllm-断崖.html) ·
[llama.cpp 与选型](https://daleiyang.github.io/llm-inference-anatomy/01-decode-latency-anatomy/Reports/Round1/05-llama-cpp-压测-选型指南.html)

---

## 如果你要在生产上跑它

1. **先定延迟目标，再反推并发。** 把 `p99 TTFT < X` 写进配置文档，比写 `max-num-seqs = N` 有意义得多。
2. **容量上限直接抄 vLLM 启动日志。** `Maximum concurrency for N tokens per request` 那一行，两轮实测只比它高 5–7%。
3. **缩短 `--max-model-len` 比加卡便宜。** 8192 → 4096，可同时驻留的请求数直接翻倍。
4. **`--max-num-seqs` 设成真实容量。** 留着默认值 128，调度器会先放进太多请求、再靠抢占踢出去 —— 38 次抢占全是这么来的。
5. **监控只盯三个指标：** `ITL p50`（解码健不健康）、`waiting` 队列长度（容量够不够）、`num_preemptions_total`（有没有在白白重算）。
6. **过载时要限流，不是加速。** 用户等的 169 秒里有 168 秒在排队；超出容量的请求直接返回 429，比让它排队好得多。

一句话诊断：**ITL 正常、TTFT 爆炸 → 容量不够在排队；ITL 也慢 → 解码本身有问题。**

---

## 为什么这些数可信

- **两轮跨机器复现。** 相隔十天、换了实例和驱动，七个并发点的吞吐差异都在 **±1.2%** 以内，关键现象全部复现。
- **自检不过的数据不进报告。** 每轮都查输出 token 数、采样参数、缓存开关、失败数、抢占数。靠这条查出了 `--ignore-eos` 在两个引擎上行为不一致（工作量差 1.35%）。
- **原始数据一条没删。** bench 原始 JSON、1 Hz 服务端采样、完整服务端日志、环境指纹都在 [`Results/`](https://github.com/daleiyang/llm-inference-anatomy/tree/main/01-decode-latency-anatomy/Results)。

## 没搞清的、没测的

- **没搞清：** llama.cpp 在 32 并发时异常塌陷（比 16 并发还慢），与 CUDA graph 复用率归零高度相关，但缺对照实验，**这一档不写进任何结论**。
- **没测：** KV 量化、受控的 KV 池大小实验、多卡、投机解码、chunked prefill 调参、其他模型规模。

---

## 怎么复现

完整命令串（开机 → 装环境 → 起服务 → 验证 → 压测 + 采样 → 切换引擎 → 收工）见
[`Scripts/第二轮测试脚本.md`](https://github.com/daleiyang/llm-inference-anatomy/blob/main/01-decode-latency-anatomy/Scripts/第二轮测试脚本.md)。三条关键规矩：

1. **两个引擎用同一份压测脚本**，显式写死 `--temperature 0`、`--random-range-ratio 0`、每档独立 seed。
2. **压测期间不改服务端配置**，一个参数都不动。
3. **每台机器现读启动日志里的 KV 池大小**，凡是从它推出来的数（并发上限、抢占阈值）都不能跨机器搬。

```
01-decode-latency-anatomy/
├── Reports/      12 份 HTML：结构拆解 + 第二轮 6 份 + 第一轮 5 份
├── Results/      原始数据，94 个文件，一条没删
└── Scripts/      第二轮测试脚本.md（可复制粘贴）
```

## 环境

| | |
|---|---|
| **GPU** | NVIDIA RTX 4090 · 24 GB（vast.ai 按小时租用）· CUDA 13.0 |
| **模型** | Qwen2.5-7B-Instruct（vLLM 用 bf16，llama.cpp 用 Q4_K_M） |
| **引擎** | vLLM 0.27.1（V1 engine）· llama.cpp `eab8ee41f` |
| **负载** | 1024 输入 / 256 输出（主线）· 输入 256→4096 扫描 · 6144 / 2048 过载 |
