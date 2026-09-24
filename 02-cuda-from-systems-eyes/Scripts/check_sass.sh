#!/usr/bin/env bash
# =============================================================================
# check_sass.sh —— 反汇编数指令，验证"条数账"里假设的那些指令真的被生成了
#
#   这一步回答的是报告里唯一一个纯靠推理、从没被证实过的断言：
#
#       "K6 的 float4 会被编译成 128 位的 LDS.128 / LDG.E.128"
#
#   ——如果 ptxas 其实把它拆成了 4 条 32 位 load，那么 K6 相对 K5 的 1.18×
#     就另有原因，§14 里那套"4 条并成 1 条、单价 3.38"的账全部作废。
#
#   顺带把每个 kernel 的 FFMA / LDS / LDG / BAR 条数也数出来，
#   和"条数账"里手推的 load/FMA 比值对一次 —— 手推的是每个 FMA 摊几条，
#   这里数的是整个 kernel 体内的静态条数，两者的比值应该对得上。
#
#   ⚠ 数的是静态指令条数（循环体里写了几条），不是动态执行次数。
#     循环展开会让静态条数翻倍，所以只看同一 kernel 内部各类指令的比例，
#     不要拿 K5 的绝对条数去减 K6 的绝对条数。
#
#   用法：
#       bash check_sass.sh              # 默认反汇编 ./sgemm_all
#       BIN=./some_other_binary bash check_sass.sh
#
#   前提：sgemm_all 已经编译好（cuobjdump 随 CUDA Toolkit 一起装，不用额外装）
# =============================================================================
set -u

BIN=${BIN:-./sgemm_all}
OUT="sass_$(date +%Y%m%d_%H%M%S).txt"

# tee -a 同时写屏幕和文件；后面每一句 log 都这样走
log() { echo "$@" | tee -a "$OUT"; }

log "============================================================="
log " SASS 指令统计   $(date -Is)"
log " 二进制：$BIN"
log "============================================================="

# command -v 查命令在不在 PATH 里；不在就返回非 0
if ! command -v cuobjdump >/dev/null 2>&1; then
  log "!! 找不到 cuobjdump —— 它在 CUDA Toolkit 的 bin 目录下，"
  log "   通常是 /usr/local/cuda/bin，把它加进 PATH 再跑。"
  exit 1
fi
if [ ! -f "$BIN" ]; then
  log "!! 找不到 $BIN，先编译：nvcc -O3 -arch=sm_89 sgemm_all.cu -o sgemm_all"
  exit 1
fi

# ---------------------------------------------------------------- 原始反汇编存档
RAW="${OUT%.txt}_raw.txt"
cuobjdump -sass "$BIN" > "$RAW" 2>&1 || { log "!! cuobjdump 失败"; exit 1; }
log ""
log "完整反汇编已存 $RAW（$(wc -l < "$RAW") 行），下面是统计。"

# ---------------------------------------------------------------- 逐 kernel 统计
# awk 的思路：
#   1. 看到 "Function : _Z8sgemm_k6iiif..." 就切换当前 kernel
#   2. 指令行的特征是带 /*地址*/ 前缀，把前缀和谓词(@P0 / @!P0)剥掉，取第一个词做操作码
#   3. 用 cnt[kernel, 操作码类别] 累加，END 里排版输出
log ""
log "-------------------------------------------------------------"
log "每个 kernel 的静态指令条数（按类别归并）"
log "-------------------------------------------------------------"

awk '
  # ---- 1) 识别 kernel 名 ----
  /Function : / {
      fn = $0
      sub(/.*Function : /, "", fn)
      # _Z8sgemm_k6iiifPKfS0_fPf -> k6：砍掉前缀 _Z<数字>sgemm_ 和后缀 iiif...
      sub(/^_Z[0-9]+sgemm_/, "", fn)
      sub(/iiif.*$/, "", fn)
      cur = fn
      if (!(cur in seen)) { seen[cur] = 1; order[++n] = cur }
      next
  }

  # ---- 2) 只处理带地址注释的指令行 ----
  #
  # SASS 的一行长这样（前后各有一段 /* */）：
  #     /*0010*/   S2UR UR5, SR_CTAID.Y ;    /* 0x00000000000579c3 */
  #     ^^^^^^^^ 地址                        ^^^^^^^^^^^^^^^^^^^^^^^ 机器码
  #
  # ⚠ 这里踩过一次坑：原来写 sub(/.*\\*\//, "", line) 想砍掉地址，
  #   但 .* 是贪婪的，它一路匹配到【行尾那段机器码注释】的 */，
  #   于是整行被砍光，op 变成空字符串，表里七个 kernel 全是 0。
  #   正确做法是锚定行首砍地址、再单独砍掉行尾的注释。
  cur != "" && /^[ 	]*\/\*[0-9a-f]+\*\// {
      line = $0
      sub(/^[ 	]*\/\*[0-9a-f]+\*\/[ 	]*/, "", line)   # 砍行首 /*0010*/
      sub(/\/\*.*$/, "", line)                        # 砍行尾 /* 0x... */
      sub(/;[ 	]*$/, "", line)                         # 砍结尾的分号
      gsub(/^[ 	]+|[ 	]+$/, "", line)
      sub(/^@![A-Za-z0-9_]+[ 	]+/, "", line)           # 砍掉 @!P0 谓词
      sub(/^@[A-Za-z0-9_]+[ 	]+/, "", line)            # 砍掉 @P0 谓词
      split(line, a, /[ 	]+/)
      op = a[1]
      if (op == "") next

      total[cur]++

      # ---- 3) 归类。只挑和三本账有关的，其余不单独记 ----
      if      (op ~ /^LDS\.128/)   { c[cur "|LDS.128"]++ }
      else if (op ~ /^LDS\.64/)    { c[cur "|LDS.64"]++  }
      else if (op ~ /^LDS/)        { c[cur "|LDS.32"]++  }
      else if (op ~ /^LDG.*\.128/) { c[cur "|LDG.128"]++ }
      else if (op ~ /^LDG.*\.64/)  { c[cur "|LDG.64"]++  }
      else if (op ~ /^LDG/)        { c[cur "|LDG.32"]++  }
      else if (op ~ /^STG.*\.128/) { c[cur "|STG.128"]++ }
      else if (op ~ /^STG/)        { c[cur "|STG.32"]++  }
      else if (op ~ /^STS/)        { c[cur "|STS"]++     }
      else if (op ~ /^FFMA/)       { c[cur "|FFMA"]++    }
      else if (op ~ /^BAR/)        { c[cur "|BAR"]++     }
      next
  }

  END {
      # 列顺序固定，方便跨批次 diff
      nk = split("FFMA LDS.128 LDS.64 LDS.32 LDG.128 LDG.64 LDG.32 STS STG.128 STG.32 BAR", K, " ")
      printf "%-6s", "kernel"
      for (i = 1; i <= nk; i++) printf "%9s", K[i]
      printf "%9s\n", "TOTAL"   # 用 ASCII：awk 的 %9s 按字节算宽度，中文会把对齐挤掉
      for (j = 1; j <= n; j++) {
          k = order[j]
          printf "%-6s", k
          for (i = 1; i <= nk; i++) {
              v = c[k "|" K[i]]
              printf "%9s", (v ? v : "-")
          }
          printf "%9d\n", total[k]
      }
  }
' "$RAW" | tee -a "$OUT"

# ---------------------------------------------------------------- 结论判定
log ""
log "-------------------------------------------------------------"
log "判定"
log "-------------------------------------------------------------"

# grep -c 数匹配行数；|| true 是因为一个都没匹配时 grep 返回 1，会被 set -e 杀掉
n128=$(grep -c "LDS\.128\|LDG\.E\.128\|STG\.E\.128" "$RAW" || true)
log "  128 位访存指令总条数：$n128"
if [ "$n128" -gt 0 ]; then
  log "  ✓ float4 确实被编译成了 128 位访存 —— §14 里 K6 的'4 条并 1 条'成立。"
else
  log "  ✗ 一条 128 位指令都没有！"
  log "    说明 float4 被拆成了 4 条 32 位 load，K6 相对 K5 的 1.18× 另有原因，"
  log "    §14·七 里 K6 那一步的推导需要推翻重写。"
fi

log ""
log "  另外两条要自己核一眼（上表里）："
log "    · K5 应该有 LDS.32 / LDS.64、但没有 LDS.128 —— 它没做向量化"
log "    · 七个 kernel 都不应该出现 LDL / STL（local memory），"
log "      出现了就等于 spill，前面 ptxas 报的 0 bytes spill 是假的"
log ""
log "  LDL/STL 出现次数：$(grep -c "LDL\|STL" "$RAW" || true)   （必须是 0）"

log ""
log "============================================================="
log " 完成 → $OUT（统计） / $RAW（完整反汇编）"
log "============================================================="
