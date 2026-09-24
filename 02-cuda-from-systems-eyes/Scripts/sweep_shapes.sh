#!/usr/bin/env bash
# =============================================================================
# sweep_shapes.sh —— 形状扫描：同一批 kernel 在方阵 / prefill / decode 上各跑一遍
#
#   为什么要做：K1–K6 全是为【方阵】调的（TILE=32, CF=8, TN=4），
#   而生产里的 GEMM 几乎从不是方阵 ——
#     prefill    M=2048  N=18944 K=3584   算术强度 610，算力受限（和方阵同类）
#     decode b32 M=32    N=18944 K=3584   算术强度 15.8，带宽受限（换了一堵墙）
#     decode b1  M=1     N=18944 K=3584   算术强度 0.50，极度带宽受限
#   形状取自 Qwen2.5-7B 的 FFN up/gate：[T, hidden] x [hidden, intermediate]
#   hidden=3584、intermediate=18944，T = batch x seq。
#
#   用法：bash sweep_shapes.sh
#         ROUNDS=1 bash sweep_shapes.sh
#         KERNELS="k0 k5 k6" bash sweep_shapes.sh    # 只关心两端
#
#   前提：sgemm_all 已编译好（sgemm_all.cu 的 main 支持 M N K 三个参数）
# =============================================================================
set -u
export NVIDIA_TF32_OVERRIDE=0

ROUNDS=${ROUNDS:-3}
KERNELS=${KERNELS:-"k0 k1 k2 k3 k3c k4 k5 k6"}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="shapes_${STAMP}.txt"

log()  { echo "$@" | tee -a "$OUT"; }
run()  { echo "\$ $*" >>"$OUT"; "$@" 2>&1 | tee -a "$OUT"; }

# 四组形状："名字 M N K"。decode 两组的 M 极小，这正是要看的地方
SHAPES=(
  "方阵基准     4096  4096 4096"
  "prefill      2048 18944 3584"
  "decode-b32     32 18944 3584"
  "decode-b1       1 18944 3584"
)

# K1 在大 N 上慢得离谱（decode 形状下它要读满整个权重矩阵却几乎不复用），
# 所以按 kernel 给不同迭代次数
iters_for() {
  case "$1" in
    k1) echo 3 ;;
    k2) echo 5 ;;
    *)  echo 11 ;;
  esac
}

log "============================================================="
log " 形状扫描   $(date -Is)"
log " kernels: $KERNELS"
log "============================================================="
log ""
log "四组形状的理论账（跑之前先算，见导学 §5·二）："
log "  形状          M     N     K    算术强度   瓶颈      下限"
log "  方阵基准    4096  4096  4096     682.7   算力    1.546 ms"
log "  prefill     2048 18944  3584     609.7   算力    3.128 ms"
log "  decode-b32    32 18944  3584      15.8   带宽    0.272 ms"
log "  decode-b1      1 18944  3584       0.5   带宽    0.270 ms"
log ""
log "  ⚠ decode 两组的下限几乎相同（0.272 vs 0.270 ms）——"
log "    都被'把权重矩阵读一遍'钉死。batch=32 和 batch=1 花一样的时间、"
log "    吞吐却差 32 倍，这就是 decode 必须批量化的根本原因。"
log ""

# ---------------------------------------------------------------- 正确性
log "=== 正确性：非方阵的边界检查有没有写对，全看这一步 ==="
for sh in "${SHAPES[@]}"; do
  set -- $sh                      # 把 "名字 M N K" 拆成 $1 $2 $3 $4
  nm=$1; m=$2; n=$3; k=$4
  log "--- $nm ($m x $n x $k) · 用 k3 抽查 ---"
  run ./sgemm_all k3 "$m" "$n" "$k" 2
done

# ---------------------------------------------------------------- 主扫描
for r in $(seq 1 "$ROUNDS"); do
  log ""
  log "##################### 第 $r 轮 #####################"
  for sh in "${SHAPES[@]}"; do
    set -- $sh
    nm=$1; m=$2; n=$3; k=$4
    log ""
    log "========== $nm : M=$m N=$n K=$k =========="
    for kern in $KERNELS; do
      # k6 要求 N、K 都是 4 的倍数；18944 和 3584 都满足，4096 也满足
      it=$(iters_for "$kern")
      log ""
      log "--- round $r · $nm · $kern · ITERS=$it ---"
      run ./sgemm_all "$kern" "$m" "$n" "$k" "$it"
    done
  done
done

log ""
log "============================================================="
log " 完成 → $OUT"
log ""
log " 重点看三件事："
log "   ① prefill 的 GFLOP/s 是不是接近方阵（都是算力受限，应该接近）"
log "   ② decode-b32 上 K4/K5/K6 的差距是不是缩小了"
log "      （方阵上 K6/K4 = 1.96x；如果 decode 上明显变小，"
log "        说明我们优化的'条数账'在带宽墙面前不起作用）"
log "   ③ decode-b1 和 decode-b32 的耗时是不是接近"
log "      （理论下限只差 0.7%，实测若也接近，就坐实了'权重读取钉死一切'）"
log "============================================================="
