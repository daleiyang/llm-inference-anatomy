#!/usr/bin/env bash
# =============================================================================
# run_all.sh —— 第 3~6 章统一上机脚本（K1 … K6 一次跑完）
#
#   这是 run_ch6.sh 的扩展版。run_ch6.sh 只跑 k3~k6，K1/K2/K3 的数字分别来自
#   第 3/4/5 章三次独立的运行 —— 机器状态不保证一致，跨章比较其实站不住脚。
#   这一版把七个 kernel 放进同一个可执行文件、同一次开机、同一个时钟采样窗口。
#
#   沿用 run_ch6.sh 的四条纪律：
#     ① 七个 kernel 全进同一个循环
#     ② 主基准就是扫尺寸里 N=4096 那一档 —— 不再有单独的一批
#     ③ 全程后台采样 SM 时钟 / 温度 / 功耗，结尾打印统计
#     ④ kernel 顺序每轮轮转，热漂移不会固定惩罚排在最后的那个
#
#   新增：K1 慢两个数量级（N=8192 时约 1.6 s/次），按尺寸自动降低 ITERS，
#         否则光 K1 一个就要跑掉大半小时。
#
#   用法：  bash run_all.sh                   # 全套，约 15–25 分钟
#           ROUNDS=1 bash run_all.sh          # 快速过一遍
#           SIZES="4096" bash run_all.sh      # 只跑主基准
#           SKIP_RACECHECK=1 bash run_all.sh
# =============================================================================
#
# ---- 给不熟悉 shell 的读者：这份脚本用到的语法，一次讲清 --------------------
#
#   $VAR / ${VAR}      取变量的值。${} 是为了和后面的字符分开，如 ${N}x 。
#   "$VAR"             几乎永远要加双引号 —— 不加的话，值里有空格就会被拆成多个参数。
#   ${VAR:-默认值}      VAR 没设置或为空时用「默认值」。这就是下面那几行
#                      ROUNDS=${ROUNDS:-3} 的意思：允许调用者从环境覆盖。
#   $(命令)             命令替换：先跑命令，把它打印的内容替换到这里。
#   $((算术))           算术展开：$(( (i+s) % n )) 就是数学运算，shell 默认只有整数。
#   名字() { ...; }     定义函数。调用时直接写名字，第 1、2 个参数是 $1 $2 …
#   "$@"  与  "$*"     都表示"全部参数"，但用途相反：
#                      "$@" 保持每个参数各自独立 → 拿来【执行】；
#                      "$*" 拼成一整串           → 拿来【打印】。下面 run() 里两个都用到了。
#   cmd1 | cmd2        管道：把 cmd1 的输出喂给 cmd2 的输入。
#   >  文件            把输出写进文件（覆盖）；  >>  是追加。
#   2>/dev/null        把「错误输出」丢掉（1 是正常输出，2 是错误输出）。
#   2>&1               把错误输出并进正常输出，这样管道和重定向才收得到它。
#   cmd &              让 cmd 在后台跑，脚本不等它。  $!  是它的进程号。
#   [ "$a" -ge 5 ]     条件测试。数值比较用 -eq -ne -lt -le -gt -ge，
#                      字符串相等用 = ，注意两边都要留空格。
#   A || { B; }        A 失败（返回非 0）时才执行 B。  A && B 则相反。
#   set -u             用到没定义过的变量就立刻报错退出，防手滑打错变量名。
#
# =============================================================================
set -u

# ---- 可调参数：都用 ${VAR:-默认} 写，于是能从命令行环境覆盖 ----
#      例如  ROUNDS=1 SIZES="4096" bash run_all.sh
ROUNDS=${ROUNDS:-3}                       # 主实验重复几轮
SIZES=${SIZES:-"1024 2048 4096 8192"}     # 要扫的矩阵边长（空格分隔的一串）
KERNELS=${KERNELS:-"k0 k1 k2 k3 k3c k4 k5 k6"}   # k0 = cuBLAS 基准，主表的分母
SRC=${SRC:-sgemm_all.cu}                  # 唯一源文件（K0~K6 都在里面）
LIBS=${LIBS:--lcublas}                    # k0 要用 cuBLAS，所以默认就链上
SKIP_RACECHECK=${SKIP_RACECHECK:-0}

# date +格式 生成时间戳，$( ) 把它取出来拼进文件名 → 每次运行各自一份输出
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="all_${STAMP}.txt"                    # ${STAMP} 的花括号是为了和后面的 .txt 分开
CLK="all_${STAMP}_clocks.csv"

# ---- 三个小函数 ----
# "$@" = 调用时传进来的全部参数（保留原有的分词）；tee 同时写屏幕和文件，-a 表示追加
log()  { echo "$@" | tee -a "$OUT"; }
rule() { log "-------------------------------------------------------------"; }
# run：先把命令行原样记进日志（方便复现），再真正执行它。函数体是两条命令，用 ; 隔开：
#   echo "\$ $*" >>"$OUT"     ← 记日志的是这半句。\$ 输出字面的 $ 符号（模仿终端提示符），
#                                $* 把全部参数拼成一行；>> 只写文件、不上屏幕
#   "$@" 2>&1 | tee -a "$OUT" ← 这半句才是真执行。"$@" 保持每个参数独立
#                                （$* 会拼成一整串，不能拿来执行）；2>&1 让报错也进日志
run()  { echo "\$ $*" >>"$OUT"; "$@" 2>&1 | tee -a "$OUT"; }

# 把 kernel 列表旋转 $1 位（第 r 轮转 r-1 位）
rotate() {
  # local 表示只在函数内有效；(...) 把字符串按空格切成数组；${#list[@]} 是数组长度
  local list=($KERNELS) n=${#list[@]} s=$1 i out=""
  # for ((...)) 是 C 风格循环；${list[下标]} 取数组元素，下标用 $(( )) 算出来
  for ((i = 0; i < n; i++)); do out+="${list[$(((i + s) % n))]} "; done
  echo "$out"                             # 函数「返回值」就是它打印的内容，由 $( ) 接住
}

# K1/K2 在大尺寸上慢得多，按尺寸给不同的迭代次数（仍取中位数，样本够用）
# case 语法：case 值 in 模式) 命令 ;; 模式) 命令 ;; esac —— esac 是 case 倒过来写
iters_for() {
  case "$1" in
    k1) if   [ "$2" -ge 8192 ]; then echo 3      # -ge 是「大于等于」
        elif [ "$2" -ge 4096 ]; then echo 5
        else echo 11; fi ;;                      # if 用 fi 收尾，每个分支用 ;; 结束
    k2) if   [ "$2" -ge 8192 ]; then echo 5
        else echo 11; fi ;;
    *)  echo 11 ;;                               # * 是「以上都不匹配」，相当于 default
  esac
}

# ---------------------------------------------------------------- 0 环境快照
log "============================================================="
log " 第 3~6 章统一基准 · K0（cuBLAS）+ K1–K6   $(date -Is)"
log " 主机 $(hostname)   $(uname -sr)"
log "============================================================="
run nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,persistence_mode --format=csv
rule
log "卡上还有谁在跑（下面应该是空的，只有你自己）："
run nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# ---------------------------------------------------------------- 1 编译
rule
log "编译 + 寄存器 / 栈帧检查"
rule
# ||  前面失败才执行后面的 { }；{ } 是把多条命令打包成一组（注意最后一条也要有分号）
run nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v "$SRC" -o sgemm_all $LIBS || {
  log "!! 编译失败，停在这里"; exit 1; }
log ""
log ">>> 每个 kernel 必须都是 0 bytes stack frame / 0 bytes spill stores。"
log ">>> 表里应该有七个（k1~k6 加 k3c）—— k0 走 cuBLAS，不是我们的 kernel，不出现在这里。"
log ">>> 任何一个不是，下面的耗时都不用看了 —— 那不是在比 kernel，是在比 local memory。"
log ">>> 另外记下每个 kernel 的 registers 数，K5/K6 的 acc[8][4] 是最可能出事的地方。"

# ---------------------------------------------------------------- 2 时钟采样
# ( ... ) 开一个子 shell，末尾的 & 让它在后台跑；while : 是无限循环（: 永远为真）
# 整个子 shell 的输出用 > 写进 CSV 文件，脚本主体继续往下走
( while :; do
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
               --format=csv,noheader,nounits
    sleep 1
  done ) > "$CLK" 2>/dev/null &
CLK_PID=$!                                 # $! = 刚刚那个后台进程的进程号，留着好杀掉
# trap 注册「退出时要做的事」：万一脚本中途被 Ctrl-C，也不会留下采样进程在后台
trap 'kill $CLK_PID 2>/dev/null' EXIT

# ---------------------------------------------------------------- 3 正确性
rule
log "小规模正确性：每个 kernel 跑一次，看校验行"
rule
# for k in 一串词 —— 按空格逐个取出。$KERNELS 这里故意不加引号，就是要让它被拆开
for k in $KERNELS; do
  log "--- $k @ 512 ---"
  run ./sgemm_all "$k" 512 2
done

# ---------------------------------------------------------------- 4 主实验
log ""
log "============================================================="
log " 主实验：${ROUNDS} 轮 × 尺寸 × kernel"
log " 尺寸: $SIZES"
log " kernel 顺序每轮轮转，避免热漂移固定惩罚排在最后的那个"
log " 注意：N=4096 这一档就是主基准 —— 不再有单独的一批"
log " 注意：K1 在 4096/8192 上自动降到 5/3 次迭代（它单次就要 0.2/1.6 秒）"
log "============================================================="
for r in $(seq 1 "$ROUNDS"); do            # seq 1 3 打印 1 2 3，$( ) 接住当成列表
  ORDER=$(rotate $((r - 1)))               # 第 r 轮把 kernel 顺序转 r-1 位
  log ""
  log "##################### 第 $r 轮   顺序: $ORDER #####################"
  for n in $SIZES; do
    for k in $ORDER; do
      it=$(iters_for "$k" "$n")            # 调函数并接住它打印的数字
      log ""
      log "--- round $r · N=$n · $k · ITERS=$it ---"
      run ./sgemm_all "$k" "$n" "$it"
    done
  done
done

# ---------------------------------------------------------------- 5 边界
log ""
log "============================================================="
log " 边界测试"
log "   4097 = 32×128+1  →  K3c/K5 的 grid.x 多出一列 block；K1/K2 走 if 边界分支"
log "   4100             →  K6 专用（向量化要求 N%4==0），K5 同尺寸作对照"
log "============================================================="
for k in k0 k1 k2 k3 k3c k4 k5; do
  it=$(iters_for "$k" 4097)
  log ""; log "--- 4097 · $k ---"; run ./sgemm_all "$k" 4097 "$it"
done
log ""; log "--- 4100 · k6 ---"; run ./sgemm_all k6 4100
log ""; log "--- 4100 · k5（k6 的同尺寸对照）---"; run ./sgemm_all k5 4100

# ---------------------------------------------------------------- 6 racecheck
# [ ... ] 是条件测试，&& 表示两个条件都要成立
# command -v xxx 用来查「系统里有没有 xxx 这个命令」，输出丢掉只看成败
if [ "$SKIP_RACECHECK" = "0" ] && command -v compute-sanitizer >/dev/null 2>&1; then
  log ""
  log "============================================================="
  log " 竞态检查 —— 放在计时之后，不干扰基准"
  log " 规模必须小：N=128 且只发射 1 次，否则慢到像死机"
  log " K1/K2 没有 shared，本来就不可能有竞态，跑一遍只是走完流程"
  log "============================================================="
  for k in $KERNELS; do
    # k0 是 cuBLAS 的闭源 kernel，查它的竞态没有意义（也不是我们能改的）
    [ "$k" = "k0" ] && continue
    log ""; log "--- racecheck · $k ---"
    run compute-sanitizer --tool racecheck ./sgemm_all "$k" 128 1
  done
  log ""
  log ">>> 每一段结尾都要看到 RACECHECK SUMMARY: 0 hazards displayed"
fi

# ---------------------------------------------------------------- 7 收尾
kill $CLK_PID 2>/dev/null; trap - EXIT; sleep 1   # 停掉采样；trap - EXIT 撤销上面注册的清理

log ""
log "============================================================="
log " 全程时钟 / 温度 / 功耗统计"
log "============================================================="
# awk 逐行处理文本：-F', ' 指定用「逗号加空格」切分字段，$1 $2 … 就是切出来的各列
# NF 是本行字段数，NR 是当前行号；{ } 里的代码对每一行执行一次，END{ } 在最后执行一次
awk -F', ' 'NF>=6 {
        c[NR]=$2+0; t[NR]=$4+0; p[NR]=$5+0;        # +0 是强制当数字用（否则是字符串）
        if (c[NR]>maxc || NR==1) maxc=c[NR];
        if (c[NR]<minc || NR==1) minc=c[NR];
        if (t[NR]>maxt) maxt=t[NR];
        if (p[NR]>maxp) maxp=p[NR];
        sumc+=c[NR];
     }
     END {
        if (NR==0) { print "  （没采到数据，检查 nvidia-smi 是否可用）"; exit }
        printf "  样本      %d 条（约 %d 秒）\n", NR, NR;
        printf "  SM 时钟   最低 %d   平均 %d   最高 %d MHz\n", minc, sumc/NR, maxc;
        printf "  波动      %.1f%%   ", (maxc-minc)*100.0/maxc;
        if ((maxc-minc)*100.0/maxc > 15) print "<<< 超过 15%，这批数据要打问号";
        else print "（低于 15% 算稳定）";
        printf "  峰值温度  %d C      峰值功耗 %.0f W\n", maxt, maxp;
     }' "$CLK" | tee -a "$OUT"
# ⚠ 这段统计有缺陷：没有先按 GPU 利用率（第 6 列）过滤就算波动，
#   于是只要采样窗口里有一次空闲降频，就会误报。应改成 NF>=6 && $6>50 再统计。

log ""
log "============================================================="
log " 完成"
log "   原始输出： $OUT"
log "   时钟记录： $CLK"
log ""
log " 贴回导学时，至少带上这四样："
log "   ① 编译那段的七行 registers / stack frame"
log "   ② 主实验里 N=4096 的全部 ${ROUNDS} 轮 × 7 个 kernel"
log "   ③ 边界那 8 行"
log "   ④ 最后这段时钟统计"
log "============================================================="
