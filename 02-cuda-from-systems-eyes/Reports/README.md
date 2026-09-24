[← 返回 02 阶段](../)

# Reports —— 自包含 HTML 报告

沿用 01 阶段的版式：单文件、数据与 SVG 内嵌、无外部依赖、支持明暗主题。
经 GitHub Pages 提供（仓库里直接点 `.html` 只会看到源码）。

## SGEMM 部分 · 已完成

| # | 报告 | 回答什么 | 规模 |
|---|---|---|---|
| **01** | [核心代码的演化](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/01-SGEMM六级优化-核心代码的演化.html) | 每一级改了哪一行、为什么那样改、原理是什么 | 10 节 · 9 图 · 六级代码全文 |
| **02** | [实测与三本账](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html) | 数字从哪来、八个指标各自怎么算、哪本账真的准 | 8 节 · 8 表 · 八步推导 |

**建议读法**：先 01 建立"代码长什么样"的直觉，再 02 看"凭什么这么判断"。

> **02 的 §3 是全套方法的核心**：八个指标里只有"耗时"是量出来的，
> 其余七个（dist / bcast / ldg / load-per-FMA / bytesPerOut / 算术强度 / 节拍 / 每条 load 周期）
> 全部从代码数出来 —— 数完再和实测对账，对上了才说明理解是对的。

## 融合算子部分 · 待上机

| # | 报告 | 状态 |
|---|---|---|
| 03 | RMSNorm / online softmax · MBU 视角 | 代码（`fused_ops.cu`，11 个 kernel）与上机导学已完成，**等机器** |

## 一处诚实标注

原计划的第三份报告「Nsight roofline 解读」**做不出来**：
租用实例的容器没有 `CAP_SYS_ADMIN`，`ncu` 报 `ERR_NVGPUCTRPERM`。

六个指标里只有两个是真的拿不到，其余四个有替代来源；
那两个想证明的事用三条独立证据替代了，详见[报告 02 · §7](https://daleiyang.github.io/llm-inference-anatomy/02-cuda-from-systems-eyes/Reports/02-SGEMM实测与三本账.html)。
**roofline 图本身是用手推的算术强度和实测 GFLOP/s 画的，该说的话一句不少。**
