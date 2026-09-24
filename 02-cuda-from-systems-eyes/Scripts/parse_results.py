#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# parse_results.py —— 把 run_all.sh / sweep_shapes.sh 的文本输出解析成 CSV
#
#   用法：
#       python3 parse_results.py all_20260921_210931.txt
#       python3 parse_results.py all_*.txt shapes_*.txt        # 可以一次给多个
#       python3 parse_results.py --outdir Results all_*.txt
#
#   产出两份（放在 Results/ 下，文件名跟着输入的时间戳走）：
#       bench_<stamp>.csv          每一轮每个 kernel 一行（原始，不做任何聚合）
#       bench_<stamp>_median.csv   同一 (形状, kernel) 跨轮取中位数，一行一个
#
#   为什么要两份：
#       原始表是证据，中位表是结论。报告主表引的是中位表，
#       但审稿人问"你这 3 轮抖了多少"时，答案只在原始表里。
#       中位表额外带一列 ms_spread_pct = (max-min)/median，跨批漂移一眼可见。
#
#   设计原则：解析器只做搬运，不做任何再计算 ——
#       CSV 里每个数都必须能在原文里逐字找到。
#       唯一的例外是 _median.csv 的中位数和 spread，那是聚合不是推导。
# =============================================================================
import csv
import glob
import io
import os
import re
import statistics
import sys

# ---------------------------------------------------------------- 正则表
# 说明：程序输出里用的是全角字符（×、→、·），正则里要原样写。
#       所有数字都写成 ([-\d.]+)，负号留着是防止将来出现负的漂移值。
RE_ROUND_SQ = re.compile(r'^--- round (\d+) · N=(\d+) · (\S+) · ITERS=(\d+) ---')
RE_ROUND_SH = re.compile(r'^--- round (\d+) · (\S+) · (\S+) · ITERS=(\d+) ---')
RE_SHAPEHDR = re.compile(r'^=+ (\S+) : M=(\d+) N=(\d+) K=(\d+) =+')
# 任何以 --- 或 === 开头、又不是 round 标题的行，都算一个块的结束。
# 没有这条，racecheck / 正确性检查那些块的输出会被当成上一个测量块的续写，
# 把最后一轮 k6 的 8192 数据覆盖成 N=128 —— 第一版就是这么错的。
RE_SEP = re.compile(r'^(-{3,}|={3,})')

RE_GPU = re.compile(r'^GPU: (.+?)\s+sm_(\d+)\s+(\d+) SM\s+L2 ([\d.]+) MB\s+时钟 ([\d.]+) GHz')
RE_SHAPE = re.compile(r'^\[(\S+)\] 矩阵 (\d+)×(\d+)×(\d+)')
RE_GRID = re.compile(r'block (\d+)×(\d+)\s+grid (\d+)×(\d+)')
RE_SMEM = re.compile(r'^shared/block = (\d+) B')
RE_OCC = re.compile(r'occupancy: (\d+) block × (\d+) 线程 = (\d+) / (\d+) = ([\d.]+)%')
RE_FOOT = re.compile(r'^footprint = A\+B\+C = ([\d.]+) MB')
RE_CHECK = re.compile(r'^(✓|✗|!)')
RE_TIME = re.compile(r'^中位耗时 ([\d.]+) ms\s+([\d.]+) GFLOP/s = FP32 峰值的 ([\d.]+)%')
RE_BYTES = re.compile(r'^字节账: 算术强度 ([\d.]+) FLOP/byte → roofline 上限 ([\d.]+) GFLOP/s，达成率 ([\d.]+)%')
RE_COUNT = re.compile(r'^条数账: load/FMA ([\d.]+)（32 路 ([\d.]+) \+ 广播 ([\d.]+) \+ 全局 ([\d.]+)）')
RE_BEAT = re.compile(r'节拍 ([\d.]+) 周期/warp圈 → 每条 load ([\d.]+) 周期'
                     r'\s+模型预测 ([\d.]+)(?:（(.+?)）)?')
RE_REQ = re.compile(r'^请求账: 内存请求 ([\d.]+) 个/FMA')

# ptxas 那一段：先看到 entry function 拿到 kernel 名，再看到 Used N registers
RE_PTX_FN = re.compile(r"Compiling entry function '_Z\d+sgemm_(\w+?)iiif")
# 注意 smem 那一截是可选的：K1/K2 不用 shared，ptxas 就直接不打 "N bytes smem"。
# 第一版把它写成必需，结果 K1/K2 的 regs 整列是空的。
RE_PTX_REG = re.compile(r'Used (\d+) registers, used (\d+) barriers'
                        r'(?:, (\d+) bytes smem)?')
RE_PTX_STACK = re.compile(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads')

RE_STAMP = re.compile(r'_(\d{8}_\d{6})')

# CSV 列顺序：先"是什么"，再"跑多快"，最后"三本账"和编译期属性
COLS = ['src_file', 'stamp', 'gpu', 'sm_count', 'clock_ghz',
        'round', 'shape', 'M', 'N', 'K', 'kernel', 'iters',
        'ms', 'gflops', 'pct_peak',
        'ai', 'roofline_gflops', 'roofline_pct',
        'load_per_fma', 'dist', 'bcast', 'ldg', 'req_per_fma',
        'beat', 'cyc_per_load', 'model_pred', 'model_note',
        'regs', 'barriers', 'smem_bytes', 'spill_bytes',
        'occ_pct', 'blocks_per_sm', 'threads_per_block',
        'block_x', 'block_y', 'grid_x', 'grid_y',
        'footprint_mb', 'verified']


def parse_file(path):
    """把一份 all_*.txt / shapes_*.txt 解析成 [dict, ...]。"""
    text = io.open(path, encoding='utf-8', errors='replace').read()
    lines = text.split('\n')

    m = RE_STAMP.search(os.path.basename(path))
    stamp = m.group(1) if m else ''

    # ---- 第一遍：ptxas 段，建 kernel -> 编译期属性 的表 ----
    ptx = {}
    cur_fn = None
    for ln in lines:
        m = RE_PTX_FN.search(ln)
        if m:
            cur_fn = m.group(1)
            ptx.setdefault(cur_fn, {})
            continue
        if cur_fn is None:
            continue
        m = RE_PTX_STACK.search(ln)
        if m:
            # stack frame / spill stores / spill loads 三个加起来，非 0 就是出事了
            ptx[cur_fn]['spill_bytes'] = int(m.group(1)) + int(m.group(2)) + int(m.group(3))
            continue
        m = RE_PTX_REG.search(ln)
        if m:
            ptx[cur_fn]['regs'] = int(m.group(1))
            ptx[cur_fn]['barriers'] = int(m.group(2))
            ptx[cur_fn]['smem_bytes'] = int(m.group(3)) if m.group(3) else 0
            cur_fn = None

    # ---- 第二遍：逐个测量块 ----
    rows = []
    gpu = {}
    shape_name = 'square'          # run_all.sh 只跑方阵，没有形状名
    cur = None

    def flush():
        if cur is not None and cur.get('ms') is not None:
            rows.append(cur)

    for ln in lines:
        m = RE_SHAPEHDR.match(ln)
        if m:                                  # sweep_shapes.sh 的形状分段标题
            shape_name = m.group(1)
            continue

        m = RE_ROUND_SQ.match(ln) or RE_ROUND_SH.match(ln)
        if m:
            flush()
            rnd, second, kern, iters = m.groups()
            cur = dict.fromkeys(COLS)
            cur.update(src_file=os.path.basename(path), stamp=stamp,
                       round=int(rnd), kernel=kern, iters=int(iters),
                       shape=(shape_name if not second.startswith('N=') else 'square'),
                       verified=0)
            cur.update(ptx.get(kern, {}))
            cur.update(gpu)
            continue

        if RE_SEP.match(ln):
            # 块结束。正确性检查（--- k1 @ 512 ---）和竞态检查（--- racecheck · k1 ---）
            # 的输出格式和测量块一模一样，不在这里切断就会被吸进上一行数据里。
            flush()
            cur = None
            continue

        if cur is None:
            # 还没进到测量块，但 GPU 那行在正确性检查里就出现了，先存着
            m = RE_GPU.match(ln)
            if m:
                gpu = dict(gpu=m.group(1).strip(), sm_count=int(m.group(3)),
                           clock_ghz=float(m.group(5)))
            continue

        m = RE_GPU.match(ln)
        if m:
            gpu = dict(gpu=m.group(1).strip(), sm_count=int(m.group(3)),
                       clock_ghz=float(m.group(5)))
            cur.update(gpu)
            continue

        m = RE_SHAPE.match(ln)
        if m:
            cur.update(M=int(m.group(2)), N=int(m.group(3)), K=int(m.group(4)))
            g = RE_GRID.search(ln)
            if g:
                cur.update(block_x=int(g.group(1)), block_y=int(g.group(2)),
                           grid_x=int(g.group(3)), grid_y=int(g.group(4)))
            continue

        m = RE_SMEM.match(ln)
        if m:
            # 运行时报的 shared/block 和 ptxas 报的 smem 应该一致；以运行时为准
            cur['smem_bytes'] = int(m.group(1))
        m = RE_OCC.search(ln)
        if m:
            cur.update(blocks_per_sm=int(m.group(1)), threads_per_block=int(m.group(2)),
                       occ_pct=float(m.group(5)))
            continue

        m = RE_FOOT.match(ln)
        if m:
            cur['footprint_mb'] = float(m.group(1))
            continue

        if RE_CHECK.match(ln):
            cur['verified'] = 1 if ln.startswith('✓') else 0
            continue

        m = RE_TIME.match(ln)
        if m:
            cur.update(ms=float(m.group(1)), gflops=float(m.group(2)),
                       pct_peak=float(m.group(3)))
            continue

        m = RE_BYTES.match(ln)
        if m:
            cur.update(ai=float(m.group(1)), roofline_gflops=float(m.group(2)),
                       roofline_pct=float(m.group(3)))
            continue

        m = RE_REQ.match(ln)
        if m:
            # 只有 K1/K2 会打这一行（没有 shared 的特例），其余 kernel 这列就是空的
            cur['req_per_fma'] = float(m.group(1))
            continue

        m = RE_COUNT.match(ln)
        if m:
            cur.update(load_per_fma=float(m.group(1)), dist=float(m.group(2)),
                       bcast=float(m.group(3)), ldg=float(m.group(4)))
            continue

        m = RE_BEAT.search(ln)
        if m:
            cur.update(beat=float(m.group(1)), cyc_per_load=float(m.group(2)),
                       model_pred=float(m.group(3)))
            # K6 那行后面带一句"128 位 load 单价未标定，这个数只当下界"——
            # 是限定条件，不能丢，丢了 0.13 就会被当成一个真预测值去和 0.50 比。
            if m.group(4):
                cur['model_note'] = m.group(4)
            continue

    flush()
    return stamp, rows


def median_rows(rows):
    """同一 (形状, M, N, K, kernel) 跨轮聚合。ms 取中位，其余取第一轮的值。"""
    buckets = {}
    for r in rows:
        key = (r['shape'], r['M'], r['N'], r['K'], r['kernel'])
        buckets.setdefault(key, []).append(r)

    out = []
    for key, group in buckets.items():
        group.sort(key=lambda r: r['round'])
        base = dict(group[0])
        ms = [r['ms'] for r in group if r['ms'] is not None]
        med = statistics.median(ms)
        base['round'] = 'med(%d)' % len(group)
        base['ms'] = round(med, 4)
        # 跨轮漂移：(max-min)/median。报告里"绝对值 ±4.5%、比值 ±0.5%"那句话的出处。
        base['ms_spread_pct'] = round((max(ms) - min(ms)) / med * 100.0, 2) if med else ''
        # GFLOP/s 不取各轮的中位数，而是由中位耗时反算 —— 保证 ms 和 gflops 自洽
        if base['M'] and base['N'] and base['K'] and med:
            base['gflops'] = round(2.0 * base['M'] * base['N'] * base['K'] / (med / 1000.0) / 1e9, 1)
        base['rounds'] = len(group)
        out.append(base)

    out.sort(key=lambda r: (r['shape'], r['N'] or 0, r['M'] or 0,
                            ['k0', 'k1', 'k2', 'k3', 'k3c', 'k4', 'k5', 'k6'].index(r['kernel'])
                            if r['kernel'] in ['k0', 'k1', 'k2', 'k3', 'k3c', 'k4', 'k5', 'k6'] else 99))
    return out


def write_csv(path, rows, cols):
    with io.open(path, 'w', encoding='utf-8-sig', newline='') as f:
        # utf-8-sig：带 BOM，Excel 双击打开才不会把中文列（gpu 名）显示成乱码
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r[k]) for k in cols})


def main():
    args = sys.argv[1:]
    outdir = 'Results'
    if '--outdir' in args:
        i = args.index('--outdir')
        outdir = args[i + 1]
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__ or 'usage: parse_results.py [--outdir DIR] <all_*.txt> ...')

    # Windows 的 shell 不展开通配符，自己来
    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(a)) or [a])

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    for path in paths:
        if not os.path.exists(path):
            print('  !! 找不到 %s，跳过' % path)
            continue
        stamp, rows = parse_file(path)
        if not rows:
            print('  !! %s 里没解析出任何测量块，跳过' % path)
            continue

        tag = stamp or os.path.splitext(os.path.basename(path))[0]
        raw = os.path.join(outdir, 'bench_%s.csv' % tag)
        med = os.path.join(outdir, 'bench_%s_median.csv' % tag)
        write_csv(raw, rows, COLS)
        mrows = median_rows(rows)
        write_csv(med, mrows, COLS + ['ms_spread_pct', 'rounds'])

        print('%s' % path)
        print('  -> %-34s %d 行' % (raw, len(rows)))
        print('  -> %-34s %d 行' % (med, len(mrows)))

        # 自检：解析器最容易出的错是"静默漏字段"，所以把空值率打出来
        holes = [c for c in COLS
                 if sum(1 for r in rows if r.get(c) in (None, '')) == len(rows)]
        if holes:
            print('  ⚠ 整列为空：%s' % ', '.join(holes))
        bad = [r for r in rows if not r['verified']]
        if bad:
            print('  ⚠ 有 %d 行没看到 ✓ 校验通过' % len(bad))
        spill = [r for r in mrows if r.get('spill_bytes')]
        if spill:
            print('  ⚠ 有 spill：%s' % ', '.join(sorted({r['kernel'] for r in spill})))


if __name__ == '__main__':
    main()
