#!/usr/bin/env bash
# =============================================================================
# run_p0.sh —— 一台机器、一次开机、一个二进制，把全部测试做完
#
#   为什么要合成一个脚本：
#       主表里"K6 是 cuBLAS 的百分之多少"、"K6 比 K1 快 62 倍"这类比值，
#       只有在同一块卡、同一个驱动、同一次编译、时钟状态相近的前提下才成立。
#       分几次跑、分两个源文件编译，比值就掺进了机器状态的差异 ——
#       而这个差异我们实测过：绝对耗时跨批漂 ±4.5%，比值才漂 ±0.5%。
#
#   九个阶段，顺序不能改：
#       0 编译          唯一的 sgemm_all.cu（含 K0），唯一的二进制
#     0.5 ncu 自检      5 秒，决定第 4、5 步做不做得了 —— 不行就现在停，别等 40 分钟
#       1 SASS 静态检查 不占 GPU，先做完，顺便证明 float4 真的编成了 128 位
#       2 计时主表      run_all.sh：k0~k6 × 4 个尺寸 × 3 轮 + 边界 + racecheck
#       3 形状扫描      sweep_shapes.sh：方阵 / prefill / decode 四组形状
#       4 ncu 六指标    collect_ncu.sh —— 会重放 kernel、慢几十倍
#       5 roofline      ncu --set roofline，给博文封面图用
#       6 解析成 CSV    parse_results.py，产出 Results/bench_*.csv
#       7 打包          先打一张产物清单（缺哪个一眼看出来），再压成一个 tgz
#
#   ⚠ 2 和 3 都是计时，必须排在 4、5 之前。ncu 一旦介入，
#     同一次运行里后面所有的耗时数字都不再可信。
#
#   预计总时长 50–80 分钟（roofline 那步占大头，K1 在 2048 上也要重放十几遍）
#
#   跑之前：nvidia-smi 确认卡上没有别人。
# =============================================================================
set -u

# TF32 会让 cuBLAS 偷偷用张量核心算 FP32，K0 的数字就不是同一件事了。
# 源码里已经 cublasSetMathMode(CUBLAS_PEDANTIC_MATH)，这个环境变量是第二道保险。
export NVIDIA_TF32_OVERRIDE=0

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="p0_${STAMP}.log"
log()  { echo "$@" | tee -a "$LOG"; }
rule() { log "-------------------------------------------------------------"; }

log "============================================================="
log " 全量测试   $(date -Is)"
log " 主机 $(hostname)   $(uname -sr)"
log " 时间戳 $STAMP"
log "============================================================="

# ---------------------------------------------------------------- 前置检查
rule
log "前置检查"
rule
for f in sgemm_all.cu run_all.sh sweep_shapes.sh collect_ncu.sh check_sass.sh parse_results.py; do
  # [ -f 文件 ] 判断普通文件存不存在
  if [ ! -f "$f" ]; then log "!! 缺少 $f，停在这里"; exit 1; fi
done
log "  六个文件都在。"
log ""
log "卡上还有谁在跑（下面应该只有表头）："
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------- 0 编译
log ""
log "=== 0. 编译（唯一源文件 sgemm_all.cu，-lcublas 要放在源文件后面）==="
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v sgemm_all.cu -o sgemm_all -lcublas \
     2>&1 | tee -a "$LOG" || { log "!! 编译失败，停在这里"; exit 1; }
log ""
log ">>> 七个 kernel（k1~k6 加 k3c）都要是 0 bytes stack frame / 0 bytes spill stores"
log ">>> K0 走 cuBLAS，不是我们的 kernel，不会出现在 ptxas 的列表里 —— 这是正常的"

# ---------------------------------------------------------------- 0.5 ncu 自检
# 这一步花 5 秒，但能省掉 40 分钟的白等。
#
# 教训来自 2026-09-24：ncu 在 vast.ai 的实例上报 ERR_NVGPUCTRPERM（是 root 也没用，
# SYS_ADMIN 是平台建容器时决定的）。当时自检写在 collect_ncu.sh 里，
# 要等前面三步跑完四十分钟才轮到它 —— 那时候才知道后两步做不了，已经晚了。
# 现在挪到最前面：不行就早知道，可以立刻换实例重来。
log ""
log "=== 0.5 ncu 可用性自检（5 秒，但决定第 4、5 步做不做得了）==="
NCU_OK=0
if ! command -v ncu >/dev/null 2>&1; then
  log "  !! 这台机器上没有 ncu（Nsight Compute CLI），第 4、5 步会跳过。"
elif ncu --metrics launch__registers_per_thread ./sgemm_all k3 512 1 >/dev/null 2>&1; then
  NCU_OK=1
  log "  ✓ ncu 可用，第 4、5 步照常做。"
else
  log "  ✗ ncu 跑不起来。下面是它的原话："
  ncu ./sgemm_all k3 512 1 2>&1 | grep -E "ERROR|ERR_" | head -5 | tee -a "$LOG"
  log ""
  log "  多半是 ERR_NVGPUCTRPERM —— 容器缺 SYS_ADMIN capability。"
  log "  是 root 也没用：那是平台创建实例时决定的，换镜像改不了。"
  log ""
  log "  >>> 现在有两个选择："
  log "  >>>   A. 按 Ctrl-C 停下，换一台支持 profiling 的实例重来（推荐，现在停损失最小）"
  log "  >>>   B. 什么都不做，脚本会跳过第 4、5 步，其余四项照常拿到数据"
  log ""
  log "  等 20 秒，不按就当选 B 继续。"
  sleep 20
  log "  继续（第 4、5 步将跳过）。"
fi

# ---------------------------------------------------------------- 1 SASS
log ""
log "=== 1. SASS 静态检查（纯 CPU，不占 GPU，所以放最前面）==="
log "    它验证的是报告里唯一一条从没被证实过的断言："
log "    K6 的 float4 到底有没有被编译成 128 位访存指令。"
bash check_sass.sh 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------- 2 计时主表
log ""
log "=== 2. 计时主表（k0~k6 × 1024/2048/4096/8192 × 3 轮）==="
log "    run_all.sh 里已经含了正确性检查、边界测试和 racecheck，不用另跑。"
bash run_all.sh 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------- 3 形状扫描
log ""
log "=== 3. 形状扫描（方阵 / prefill / decode-b32 / decode-b1）==="
log "    这一步也是计时，所以必须排在 ncu 之前；"
log "    它回答的是'这套为方阵调的优化，到了真实推理负载上还剩多少'。"
bash sweep_shapes.sh 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------- 4 ncu 指标
log ""
log "=== 4. Nsight 六指标（从这里开始，所有耗时数字都不再可信）==="
if [ "$NCU_OK" = "1" ]; then
  bash collect_ncu.sh 2>&1 | tee -a "$LOG"
else
  log "  跳过 —— 0.5 步的自检没过。"
fi

# ---------------------------------------------------------------- 5 roofline
log ""
log "=== 5. Roofline 采集 ==="
log "    六个点全部用 N=2048：K1 在 4096 上要 205 ms，重放十几遍太久；"
log "    而 2048 的 footprint 50 MB 仍接近 L2 的 75.5 MB，趋势和 4096 一致。"
if [ "$NCU_OK" != "1" ]; then
  log "  跳过 —— 0.5 步的自检没过。"
  log "  替代方案：本地 make_report.py 会用手推的算术强度画一张 roofline（导学 §8 · 五），"
  log "  该说的话一句不少，只是横轴不是硬件实测的算术强度。"
else
  ROOF_OK=0
  for k in k1 k2 k3 k4 k5 k6; do
    log "--- roofline $k ---"
    ncu --set roofline -o "prof_$k" ./sgemm_all "$k" 2048 1 2>&1 | tail -3 | tee -a "$LOG"
    # ⚠ 这里必须查文件，不能只看命令有没有报错。
    #   2026-09-24 那次：ncu 权限不足，但六个 kernel 的程序输出照样正常打印，
    #   tail -3 看到的是「条数账 ...」这种正常行，看着像成功 ——
    #   而一个 .ncu-rep 都没生成。看结果，别看过程。
    if [ -f "prof_$k.ncu-rep" ]; then
      ROOF_OK=$((ROOF_OK + 1))
      log "    ✓ prof_$k.ncu-rep ($(du -h "prof_$k.ncu-rep" | cut -f1))"
    else
      log "    ✗ 没有生成 prof_$k.ncu-rep"
    fi
  done
  log ""
  log ">>> 六个点里成功了 $ROOF_OK 个。不是 6 就去看上面哪一行是 ✗。"
fi

# ---------------------------------------------------------------- 6 解析
log ""
log "=== 6. 解析成 CSV ==="
# command -v 查命令在不在 PATH 里；|| true 保证查不到时不触发 set -e 之类的中断
PY=$(command -v python3 || command -v python || true)
if [ -n "$PY" ]; then
  "$PY" parse_results.py all_*.txt shapes_*.txt 2>&1 | tee -a "$LOG"
else
  log "  !! 这台机器上没有 python，跳过。把 all_*.txt / shapes_*.txt 带回本地再解析即可。"
fi

# ---------------------------------------------------------------- 产物清单
log ""
log "-------------------------------------------------------------"
log "产物清单 —— 缺哪个一眼看出来"
log "-------------------------------------------------------------"
# have 文件通配符 说明 —— 有就打勾，没有就打叉
# ls $1 故意不加引号，就是要让通配符展开；2>/dev/null 吞掉"找不到"的抱怨
have() {
  local f
  f=$(ls $1 2>/dev/null | head -1)
  if [ -n "$f" ]; then log "  ✓ $2   ($f)"; else log "  ✗ $2   —— 没有"; fi
}
have "all_*.txt"                   "计时主表"
have "shapes_*.txt"                "形状扫描"
have "sass_*.txt"                  "SASS 统计"
have "Results/bench_*_median.csv"  "解析后的 CSV"
have "ncu_*.csv"                   "ncu 六指标"
have "prof_k6.ncu-rep"             "roofline 报告"

# ---------------------------------------------------------------- 7 打包
log ""
log "=== 7. 打包 ==="
# 2>/dev/null：某些通配符可能一个文件都没匹配上，tar 会抱怨，但不影响结果
tar czf "p0_${STAMP}.tgz" \
    all_*.txt all_*_clocks.csv shapes_*.txt \
    sass_*.txt ncu_*.csv ncu_*.log prof_*.ncu-rep \
    Results/ "$LOG" 2>/dev/null

log ""
log "============================================================="
log " 完成 → p0_${STAMP}.tgz"
log ""
log " 本地取回："
log "   scp -P <PORT> root@<HOST>:$(pwd)/p0_${STAMP}.tgz ."
log ""
log " 回到本地之后："
log "   tar xzf p0_${STAMP}.tgz"
log "   python make_report.py          # 生成表格和图，直接写进补测导学"
log ""
log " roofline 图要用本地 Nsight Compute 打开 prof_k6.ncu-rep，"
log " Page 选 Details → 找 GPU Speed Of Light Roofline Chart。"
log "============================================================="
