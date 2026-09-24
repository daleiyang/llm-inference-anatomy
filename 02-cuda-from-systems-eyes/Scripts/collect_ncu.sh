#!/usr/bin/env bash
# =============================================================================
# collect_ncu.sh —— 采集 Nsight Compute 的六个关键指标，输出一份 csv
#
#   前提：① ncu 有权限（容器要 --cap-add=SYS_ADMIN，详见上机导学 §3·一）
#         ② sgemm_all 已经编译好
#
#   用法：bash collect_ncu.sh                 # 八个 kernel（含 k0）× N=4096
#         N=1024 bash collect_ncu.sh          # 先用小尺寸验证脚本能跑通
#         KERNELS="k1 k2" bash collect_ncu.sh # 只采某几个
#
#   ⚠ 这一步必须在计时基准【之后】跑：ncu 会重放 kernel，慢几十倍。
# =============================================================================
set -u

KERNELS=${KERNELS:-"k0 k1 k2 k3 k3c k4 k5 k6"}   # k0 = cuBLAS，采它是为了有个参照
N=${N:-4096}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="ncu_${STAMP}.csv"
LOG="ncu_${STAMP}.log"

# 六个指标写成一个变量。行尾反斜杠是续行符，后面不能有空格
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
l1tex__data_bank_conflicts_pipe_lsu_shared.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread"

echo "=== Nsight Compute 指标采集  $(date -Is) ===" | tee "$LOG"
echo "N = $N   kernels: $KERNELS" | tee -a "$LOG"
ncu --version 2>&1 | head -3 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 先探一次权限：用最小规模试，失败就立刻退出，别浪费后面几分钟
if ! ncu --metrics launch__registers_per_thread ./sgemm_all k3 512 1 >/dev/null 2>&1; then
  echo "!! ncu 跑不起来 —— 多半是权限（ERR_NVGPUCTRPERM），见上机导学 §3·一" | tee -a "$LOG"
  ncu ./sgemm_all k3 512 1 2>&1 | tail -20 | tee -a "$LOG"
  exit 1
fi
echo ">>> 权限自检通过" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for k in $KERNELS; do
  echo "--- $k @ $N ---" | tee -a "$LOG"
  # ITERS 必须是 1：ncu 会重放 kernel，多迭代只是白白多花几十倍时间
  # --csv 输出逗号分隔；--target-processes all 保证子进程也被抓到
  # sed 在每行前插一列 kernel 名，否则八次输出混在一起分不清谁是谁
  # k0 会多出几行：cuBLAS 内部可能发射不止一个 kernel，每个都会被单独采一次
  ncu --metrics "$METRICS" --csv --target-processes all \
      ./sgemm_all "$k" "$N" 1 2>>"$LOG" \
      | sed "s/^/${k},/" >> "$OUT"
done

echo "" | tee -a "$LOG"
echo "完成：指标 → $OUT    日志 → $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "自检：先看 launch__registers_per_thread 那几行，应该是" | tee -a "$LOG"
echo "      k1 40 / k2 40 / k3 38 / k3c 38 / k4 54 / k5 72 / k6 111" | tee -a "$LOG"
echo "      对不上就是采错了二进制，别看别的指标了。" | tee -a "$LOG"
echo "      k0 是 cuBLAS 自己的 kernel，寄存器数由 NVIDIA 决定，没有预期值 ——" | tee -a "$LOG"
echo "      它在这里的作用是给 sm__throughput / dram_throughput 两列一个上限参照。" | tee -a "$LOG"
