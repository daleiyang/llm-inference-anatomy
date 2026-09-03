[← 返回 02 阶段](../)

# Results —— 原始数据

本仓库的规矩：**原始数据一条不删，结论必须可追溯。** 这一阶段会落在这里的东西：

| 文件 | 内容 | 谁产出 |
|---|---|---|
| `device.txt` | 设备自查输出，roofline 分母的唯一来源 | `src/devinfo.cu` |
| `bench_4096.csv` | 主基准：七个 kernel @ 4096 的中位耗时与 GFLOP/s | `src/sgemm.cu` |
| `bench_sweep.csv` | 尺寸扫描 1024/2048/4096/8192（看 72MB L2 的影响） | `src/sgemm.cu` |
| `ncu/metrics.csv` | Nsight 六个关键指标逐 kernel 导出 | `ncu --csv` |
| `ncu/prof_*.ncu-rep` | 全量采集，用于出官方 roofline 图 | `ncu --set full` |

CSV 表头：`kernel,N,ms_median,gflops`

> 采集命令见 [`../Scripts/build-and-bench.md`](../Scripts/build-and-bench.md)。
