[← 返回 02 阶段](../)

# 编译、跑分、采集：完整命令串

沿用 01 阶段的成本纪律：**收工必 `vastai destroy`**。

---

## 0 · 开机（要 `-devel` 镜像，`runtime` 里没有 nvcc）

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.95' -o 'dph+'

vastai create instance <OFFER_ID> \
  --image nvidia/cuda:12.6.2-devel-ubuntu22.04 \
  --disk 40 --ssh --direct

vastai show instances          # 等 running
vastai ssh-url <INSTANCE_ID>
```

> 磁盘 40G 就够 —— 这一阶段不下模型权重，比 01 阶段省得多。

## 1 · 装工具

```bash
apt update && apt install -y build-essential cmake git
apt install -y nsight-compute || true      # 装不上就从 NVIDIA 官网下 .run 包
nvcc --version && nvidia-smi
which ncu || ls /opt/nvidia/nsight-compute/*/ncu
```

## 2 · 设备自查（第一件事）

```bash
nvcc devinfo.cu -o devinfo
./devinfo | tee ../Results/device.txt
nvidia-smi -q -d CLOCK | grep -A3 "Clocks Event Reasons\|Max Clocks"
```

**把输出抄进 `Results/device.txt`。** 后面所有 roofline 的分母都从这里来，
不从网上的规格表抄 —— 与 01 阶段「KV 池必须每台机器现读日志」同一条原则。

## 3 · 编译（务必看寄存器与 spill）

```bash
nvcc -O3 -arch=sm_89 -lcublas -Xptxas -v sgemm.cu -o sgemm 2>&1 | grep -E "registers|spill"
```

**出现 `spill stores` / `spill loads` 就是警报** —— 寄存器溢出到 local memory
（实际在 HBM 上），性能会崩。减小 K5 的 `TM`/`TN`。

## 4 · 跑分

```bash
# ⚠️ 禁用 TF32，否则 cuBLAS 会偷偷用张量核心，不是同一个东西在比
export NVIDIA_TF32_OVERRIDE=0

# 主基准：各 kernel @ 4096
echo "kernel,N,ms_median,gflops" > ../Results/bench_4096.csv
for k in 0 1 2 3 4 5 6; do ./sgemm $k 4096 >> ../Results/bench_4096.csv; done

# 尺寸扫描：4090 有 72MB L2，小矩阵会让 naive 看起来没那么糟
echo "kernel,N,ms_median,gflops" > ../Results/bench_sweep.csv
for k in 1 5; do for n in 1024 2048 4096 8192; do ./sgemm $k $n >> ../Results/bench_sweep.csv; done; done
```

## 5 · Nsight 采集

```bash
mkdir -p ../Results/ncu

# 全量（慢，出 roofline 图用）
ncu --set full -o ../Results/ncu/prof_k5 ./sgemm 5 4096

# 关键指标导 CSV（快，迭代时用）
for k in 1 2 3 4 5 6; do
  ncu --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_shared.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
launch__registers_per_thread \
  --csv ./sgemm $k 4096 >> ../Results/ncu/metrics.csv
done
```

**六个指标各回答什么：**

| 指标 | 问题 | 期望走向 K1→K6 |
|---|---|---|
| `sm__throughput...pct_of_peak` | 算力用了几成（MFU） | 单调上升 |
| `gpu__dram_throughput...pct_of_peak` | 带宽用了几成（MBU） | 先高后低 |
| `smsp__sass_average_data_bytes_per_sector...` | 合并访存效率 | K1 低 → K2 起接近满 |
| `l1tex__data_bank_conflicts_pipe_lsu_shared.sum` | bank conflict 次数 | 应接近 0 |
| `sm__warps_active...pct_of_peak` | 实际占用率 | **K5 会下降（正常）** |
| `launch__registers_per_thread` | 每线程寄存器 | K5 显著上升 |

> `ncu` 报权限错误时：容器需要 `--cap-add=SYS_ADMIN`，或换一台 vast.ai 机器。

## 6 · 收工

```bash
tar czf results.tgz ../Results ../src
# 本地：scp -P <PORT> root@<HOST>:/workspace/results.tgz .
vastai destroy instance <INSTANCE_ID>      # ← 别忘。忘关一夜 ≈ $8
```

---

## 成本提醒

这一阶段的陷阱和 01 阶段不同：01 是"跑长基准中间不用管"，
这里是**改代码 → 编译 → 跑 2 秒 → 想 20 分钟**，机器空转远多于计算。

**对策三选一：**
- **最省**：本地写代码，`rsync` 上去批量编译测试，一次会话集中跑完即销毁
- **折中**：挂持久卷存代码，每天开机 3–4 小时专注块，收工即 destroy
- **别做**：开着机器慢慢想 —— 那是每小时 $0.35 的思考费
