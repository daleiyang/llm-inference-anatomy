[← 返回 02 阶段](../)

# 编译、跑分、采集：完整命令串

沿用 01 阶段的成本纪律：**收工必 `vastai destroy`**。

---

## 0 · 开机（要 `-devel` 镜像，`runtime` 里没有 nvcc）

```bash
vastai search offers "gpu_name=RTX_4090 num_gpus=1 rentable=true \
  reliability>0.97 driver_version>=580.65.06 disk_space>=100" -o dph

vastai create instance <OFFER_ID> \
  --image nvidia/cuda:13.0.3-devel-ubuntu24.04 \
  --disk 100 --ssh --direct

vastai show instances          # 等 running
vastai ssh-url <INSTANCE_ID>
```

> **租到手第一件事：探 ncu 权限，五秒就知道行不行。**
>
> ```bash
> nsys profile --gpu-metrics-devices=help
> ```
>
> 看到 GPU 名字就能用；看到 `Insufficient privilege` 就是容器没有 `CAP_SYS_ADMIN` ——
> **是 root 也没用**，那是宿主机创建容器时决定的，销毁换一台。
> Round1 这一批就栽在这里，`ncu` 的两个阶段整个作废（其余七个阶段不受影响）。

## 1 · 传代码

```bash
# 本地
tar czf src.tgz sgemm_all.cu run_all.sh run_p0.sh sweep_shapes.sh \
                collect_ncu.sh check_sass.sh parse_results.py
scp -P <PORT> -i <KEY.pem> src.tgz root@<HOST>:/root/

# 远端
cd /root && tar xzf src.tgz
nvcc --version && nvidia-smi
```

## 2 · 一条命令跑完

```bash
export NVIDIA_TF32_OVERRIDE=0
bash run_p0.sh                   # 九个阶段，50~80 分钟
```

九个阶段，**顺序不能改**：

| 阶段 | 做什么 | 为什么在这个位置 |
|---|---|---|
| 0 | 编译 + spill 检查 | 七个 kernel 必须全部 `0 bytes spill stores` |
| 0.5 | **ncu 权限自检** | 5 秒。不行就当场停，别等四十分钟 |
| 1 | SASS 反汇编计数 | 纯 CPU，不占 GPU，放最前 |
| 2 | 计时主表 | `run_all.sh`：8 kernel × 4 尺寸 × 3 轮 + 边界 + racecheck |
| 3 | 形状扫描 | 也是计时，必须在 ncu 之前 |
| 4 | ncu 六指标 | **从这里开始所有耗时数字都不可信** |
| 5 | roofline 采集 | 逐个查 `.ncu-rep` 是否真的生成 |
| 6 | 解析成 CSV | `parse_results.py` |
| 7 | 产物清单 + 打包 | 六样东西谁有谁没有一目了然 |

## 3 · 取回与解析

```bash
# 本地
scp -P <PORT> -i <KEY.pem> root@<HOST>:/root/p0_<stamp>.tgz .
tar xzf p0_<stamp>.tgz
python parse_results.py all_*.txt shapes_*.txt     # 若远端没装 python
```

## 4 · 收工

```bash
vastai destroy instance <INSTANCE_ID>
```

---

## 各脚本的分工

| 脚本 | 干什么 | 单独跑的用法 |
|---|---|---|
| `run_p0.sh` | 总驱动 | `bash run_p0.sh` |
| `run_all.sh` | 计时引擎 | `ROUNDS=1 SIZES=4096 bash run_all.sh` |
| `sweep_shapes.sh` | 四组形状 | `KERNELS="k0 k6" bash sweep_shapes.sh` |
| `check_sass.sh` | 反汇编分类计数 | `bash check_sass.sh`（纯 CPU） |
| `collect_ncu.sh` | Nsight 六指标 | `N=1024 bash collect_ncu.sh` |
| `parse_results.py` | 文本 → 两份 CSV | `python parse_results.py all_*.txt` |

---

## 四条规矩

1. **八个 kernel 必须进同一个可执行文件。**
   "K6 是 cuBLAS 的百分之多少"这类比值，只有在同一个二进制、同一次开机下才成立。
   *实测：跨批次绝对耗时漂 2–3%，同批内比值只漂 0.5%。*

2. **cuBLAS 双保险锁死 FP32**：源码里 `cublasSetMathMode(CUBLAS_PEDANTIC_MATH)`，
   环境里 `NVIDIA_TF32_OVERRIDE=0`。
   *否则 Ada 会偷偷用 TF32 张量核心，那就不是同一个东西在比。*

3. **预热至少一次，哪怕只跑 2 轮。**
   cuBLAS 第一次调用要惰性加载 kernel 库，一次几十毫秒。
   *这个坑真踩过 —— K0 在 512 上报出过 55 ms，修复后是 0.017 ms，差 3249 倍。*

4. **看结果，不看命令有没有报错。**
   roofline 那一步在权限不足时会跑完六个 kernel、打印正常输出、看着像成功，
   **实际一个 `.ncu-rep` 都没生成**。脚本现在逐个查文件。
