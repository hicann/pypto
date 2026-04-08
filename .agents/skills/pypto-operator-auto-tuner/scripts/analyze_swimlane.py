#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Analyze swimlane data for PyPTO operator performance tuning.

Extracts per-leafHash statistics from three linked data sources:
  - merged_swimlane.json: task durations and core types (via tid metadata)
  - dyn_topo.txt: psgId (= hashorder) and rootHash mapping
  - program.json: compute operations (from graphtype=3 leaf functions)

Usage:
    python3 analyze_swimlane.py <output_dir> [--outer-loops N]

Output columns:
  #        - rank by total time
  leafHash - leaf function hash
  cnt      - total task count for this leafHash
  t/iter   - cnt / outer_loops, tasks per outer-loop iteration
  min/max/avg/total(us) - duration statistics
  core     - AIC or AIV (from swimlane tid metadata)
  psgId    - subgraph id (= hashorder for set_pass_options)
  root_name - root function name
  compute_ops - compute opcodes (excludes COPY_IN/OUT, SYNC, PHASE)

The t/iter column guides merge granularity:
  - t/iter=1: no natural merge opportunity within one iteration
  - t/iter=2: can try merge granularity 2
  - t/iter=4+: can try 2/4/8

outer_loops is auto-detected as GCD of all cnt values.
Override with --outer-loops if needed (analyze code loop nesting to confirm).
"""

import logging
import argparse
import json
import re
import csv
import math
from functools import reduce

logging.basicConfig(level=logging.INFO, format='%(message)s')


def gcd_list(nums):
    return reduce(math.gcd, [int(n) for n in nums])


def main():
    parser = argparse.ArgumentParser(
        description="Analyze swimlane data: leafHash stats, core type, psgId, compute ops, t/iter")
    parser.add_argument("output_dir", help="Output directory with swimlane data")
    parser.add_argument("--outer-loops", type=int, default=None,
                        help="Outer loop iteration count (auto=GCD of all cnt if not set)")
    args = parser.parse_args()

    base_dir = args.output_dir.rstrip("/")

    with open(f"{base_dir}/merged_swimlane.json", "r") as f:
        data = json.load(f)
    events = data.get("traceEvents", data)

    tid_core = {}
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            name = ev.get("args", {}).get("name", "")
            tid = ev.get("tid", 0)
            if "AIC" in name:
                tid_core[tid] = "AIC"
            elif "AIV" in name:
                tid_core[tid] = "AIV"
            else:
                tid_core[tid] = name

    leaf_stats = {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        hint = ev.get("args", {}).get("event-hint", "")
        m = re.search(r"leafHash:(\d+)", hint)
        if not m:
            continue
        lh = m.group(1)
        if lh == "0":
            continue
        dur = ev.get("dur", 0)
        ct = tid_core.get(ev.get("tid", 0), "?")
        if lh not in leaf_stats:
            leaf_stats[lh] = {"count": 0, "durs": [], "core": ct}
        leaf_stats[lh]["count"] += 1
        leaf_stats[lh]["durs"].append(dur)

    leaf_info = {}
    with open(f"{base_dir}/dyn_topo.txt", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 9:
                continue
            func_hash = row[6].strip()
            psg_id = row[8].strip()
            root_hash = row[3].strip()
            if func_hash and func_hash != "0":
                if func_hash not in leaf_info:
                    leaf_info[func_hash] = {"psgId": psg_id, "rootHash": root_hash}

    with open(f"{base_dir}/program.json", "r") as f:
        prog = json.load(f)

    root_name = {}
    for func in prog["functions"]:
        if func.get("graphtype") == 2:
            h = str(func.get("hash", ""))
            magic = func.get("func_magicname", func.get("funcmagic", ""))
            root_name[h] = magic.replace("TENSOR_LOOP_", "")

    leaf_ops = {}
    skip_ops = {"COPY_IN", "COPY_OUT", "SYNC_SRC", "SYNC_DST", "PHASE1", "PHASE2", "PHASE3"}
    for func in prog["functions"]:
        if func.get("graphtype") == 3:
            h = str(func.get("hash", ""))
            opcodes = [op.get("opcode", "?") for op in func.get("operations", [])]
            compute = [o for o in opcodes if o not in skip_ops]
            leaf_ops[h] = "+".join(compute) if compute else "(pure copy)"

    all_counts = [st["count"] for st in leaf_stats.values()]
    auto_outer = gcd_list(all_counts) if all_counts else 1
    outer_loops = args.outer_loops if args.outer_loops else auto_outer

    logging.info(f"{'#':<3} {'leafHash':<22} {'cnt':>4} {'t/iter':>6} "
          f"{'min(us)':>9} {'max(us)':>9} {'avg(us)':>9} {'total(us)':>10} "
          f"{'core':>4} {'psgId':>5} {'root_name':<42} {'compute_ops'}")
    logging.info("-" * 230)

    sorted_items = sorted(leaf_stats.items(), key=lambda x: -sum(x[1]["durs"]))
    for i, (lh, st) in enumerate(sorted_items, 1):
        durs = st["durs"]
        info = leaf_info.get(lh, {})
        psg = info.get("psgId", "?")
        ct = st["core"]
        rh = info.get("rootHash", "")
        rn = root_name.get(rh, "?")
        if len(rn) > 40:
            rn = rn[:37] + "..."
        ops = leaf_ops.get(lh, "?")
        if len(ops) > 100:
            ops = ops[:97] + "..."
        t_per_iter = st["count"] / outer_loops
        logging.info(f"{i:<3} {lh:<22} {st['count']:>4} {t_per_iter:>6.1f} "
              f"{min(durs):>9.2f} {max(durs):>9.2f} {sum(durs)/len(durs):>9.2f} {sum(durs):>10.2f} "
              f"{ct:>4} {psg:>5} {rn:<42} {ops}")

    logging.info(f"\nouter_loops={outer_loops}" +
          (" (auto, GCD of counts)" if not args.outer_loops else " (user-specified)"))

    logging.info(f"\n{'=' * 80}")
    logging.info("Merge Tuning Guide (psgId = hashorder)")
    logging.info(f"{'=' * 80}")

    aic = [(lh, st, leaf_info.get(lh, {})) for lh, st in sorted_items if st["core"] == "AIC"]
    aiv = [(lh, st, leaf_info.get(lh, {})) for lh, st in sorted_items if st["core"] == "AIV"]

    if aic:
        logging.info(f"\n[AIC] cube_l1_reuse_setting / cube_nbuffer_setting:")
        for _, st, info in aic:
            psg = info.get("psgId", "?")
            tpi = st["count"] / outer_loops
            avg = sum(st["durs"]) / len(st["durs"])
            line = f"  psgId={psg}: cnt={st['count']}, t/iter={tpi:.0f}, avg={avg:.2f}us"
            if tpi > 1:
                vals = [str(v) for v in [2, 4, 8, 16] if v <= tpi * 4]
                line += f"  → try {{{psg}: {'/'.join(vals)}}}"
            logging.info(line)

    if aiv:
        logging.info(f"\n[AIV] vec_nbuffer_setting:")
        for _, st, info in aiv:
            psg = info.get("psgId", "?")
            tpi = st["count"] / outer_loops
            avg = sum(st["durs"]) / len(st["durs"])
            line = f"  psgId={psg}: cnt={st['count']}, t/iter={tpi:.0f}, avg={avg:.2f}us"
            if tpi > 1:
                vals = [str(v) for v in [2, 4, 8, 16] if v <= tpi * 4]
                line += f"  → try {{{psg}: {'/'.join(vals)}}}"
            logging.info(line)


if __name__ == "__main__":
    main()
