#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import json
import re
import argparse
import os
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def parse_log_file(log_file_path):
    all_blocks_data = []
    current_block_aicpu = []
    current_block_aicore = []
    in_perf_trace_block = False
    block_start_line = 0

    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if "Begin dump machine perf trace:" in line:
                if in_perf_trace_block:
                    print(f"Error: Nested perf trace block at line {line_num}. Block started at line {block_start_line} is not properly closed.")
                    return {"error": f"Nested perf trace block at line {line_num}"}

                in_perf_trace_block = True
                block_start_line = line_num
                current_block_aicpu = []
                current_block_aicore = []
                print(f"Found perf trace block start at line {line_num}")
                continue

            if "Finish dump machine perf trace." in line:
                if not in_perf_trace_block:
                    print(f"Error: Finish without matching begin at line {line_num}")
                    return {"error": f"Finish without matching begin at line {line_num}"}

                in_perf_trace_block = False
                if current_block_aicpu and current_block_aicore:
                    block_json_str = ''.join(current_block_aicpu) + ',' + ''.join(current_block_aicore)
                    block_json_str = block_json_str.replace(",]", "]")
                    all_blocks_data.append(block_json_str)
                    print(f"Successfully parsed performance trace block from line {block_start_line} to {line_num}")
                else:
                    print(f"Warning: Empty performance trace block from line {block_start_line} to {line_num}")

                current_block_aicpu = []
                current_block_aicore = []
                continue

            if in_perf_trace_block:
                if "tile_fwk aicpu prof:" in line:
                    match = re.search(r'tile_fwk aicpu prof:(.*)', line)
                    if match:
                        content = match.group(1).strip()
                        if content.endswith('"'):
                            content = content[:-1]
                        current_block_aicpu.append(content)
                elif "tile_fwk aicore prof:" in line:
                    match = re.search(r'tile_fwk aicore prof:(.*)', line)
                    if match:
                        content = match.group(1).strip()
                        if content.endswith('"'):
                            content = content[:-1]
                        current_block_aicore.append(content)
            else:
                if "tile_fwk aicpu prof:" in line or "tile_fwk aicore prof:" in line:
                    print(f"Warning: Ignoring prof data outside of perf trace block at line {line_num}")

    if in_perf_trace_block:
        print(f"Error: Unclosed perf trace block started at line {block_start_line}. Discarding incomplete block data.")
        return {"error": f"Unclosed perf trace block started at line {block_start_line}"}

    if all_blocks_data:
        full_json_str = '[' + ','.join(all_blocks_data) + ']'
        try:
            parsed_data = json.loads(full_json_str)
            print(f"Successfully parsed {len(all_blocks_data)} performance trace blocks")
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw JSON string: {full_json_str[:500]}...")  # Print first 500 chars for debugging
            return {"error": str(e), "raw_blocks": all_blocks_data}
    else:
        print("No valid performance trace blocks found in log file")
        return []


def save_json(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        if isinstance(data, dict) and "raw_aicpu" in data:
            file.write('[' + data["raw_aicpu"] + ',' + data["raw_aicore"] + ']')
        else:
            json.dump(data, file, indent=2, ensure_ascii=False)


def convert_to_perfetto_format(input_json: List[Dict]) -> List[Dict]:
    trace_events = []
    thread_id = 0
    trace_events_head = {
        "args": {
            "name": "AICPU View",
            "type": "aicpu"
        },
        "cat": "__metadata",
        "name": "process_name",
        "ph": "M",
        "pid": 0
    }
    trace_events.append(trace_events_head)
    for block in input_json:
        block_idx = block.get("blockIdx", 0)
        core_type = block.get("coreType", "UNKNOWN")
        freq = block.get("freq", 0)
        thread_name = f"{core_type}-{block_idx}"
        trace_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": thread_id,
            "args": {
                "name": thread_name
            }
        })

        tasks = block.get("tasks", [])
        sorted_tasks = sorted(tasks, key=lambda x: x.get("end", 0))

        # 处理相同end_time的情况
        adjusted_tasks = []
        prev_end = None
        for task in sorted_tasks:
            task_copy = task.copy()
            current_end = task_copy.get("end", 0)

            # 如果当前end与上一个相同，则递增1
            if prev_end is not None and current_end == prev_end:
                task_copy["end"] = prev_end + 1
                current_end = prev_end + 1

            adjusted_tasks.append(task_copy)
            prev_end = current_end

        prev_end = None
        for task in adjusted_tasks:
            task_name = task.get("name", "UNKNOWN")
            end_time = task.get("end", 0)
            if task_name.startswith("BEGIN"):
                start_time = end_time - 1
            else:
                start_time = prev_end
            ts = start_time / freq
            dur = (end_time - start_time) / freq
            if dur <= 0:
                dur = 1 / freq
            perfetto_event = {
                "name": f"{task_name}",
                "cat": core_type,
                "ph": "X",
                "ts": ts,
                "dur": dur,
                "pid": 0,
                "tid": thread_id,
                "freq": freq
            }
            trace_events.append(perfetto_event)
            prev_end = end_time
        thread_id += 1
    trace_events_pypto = {"traceEvents": trace_events}
    return trace_events_pypto


def parse_log_command(input_file, output_file):
    parsed_data = parse_log_file(input_file)
    save_json(parsed_data, output_file)
    print(f"Parsing completed, result saved to: {output_file}")


def merge_aicpu_aicore_swim_lane(aicpu_perfetto_data, input_kernel_file):
    if input_kernel_file is not None and os.path.exists(input_kernel_file):
        with open(input_kernel_file, 'r', encoding='utf-8') as f:
            aicore_perfetto_data = json.load(f)
            # kernel swim lane + aicpu swim lane
            merged_trace_events = aicpu_perfetto_data["traceEvents"] + aicore_perfetto_data["traceEvents"]
            merged_data = {
                'traceEvents': merged_trace_events
            }
        with open(input_kernel_file, 'w', encoding='utf-8') as fw:
            json.dump(merged_data, fw, indent=2)


def gen_perfetto_command(input_file, output_file, input_kernel_file=None):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        perfetto_data = convert_to_perfetto_format(input_data)
        merge_aicpu_aicore_swim_lane(perfetto_data, input_kernel_file)       
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(perfetto_data, f, ensure_ascii=False, indent=2)
        print(f"Success to generate perfetto file: {output_file}")

        complete_events = [e for e in perfetto_data["traceEvents"] if e.get('ph') == 'X']
        metadata_events = [e for e in perfetto_data["traceEvents"] if e.get('ph') == 'M']
        print(f"Process {len(complete_events)} task events, {len(metadata_events)} meta data events")

        print("\ninfo: upload this json file to https://ui.perfetto.dev/")
    except FileNotFoundError:
        print(f"error: cannot find input file {input_file}")
    except json.JSONDecodeError:
        print(f"error: input file {input_file} is not valid json format")
    except Exception as e:
        print(f"process exception info: {str(e)}")


def gen_perfetto_example():
    sample_data = [
        {"blockIdx": 0, "coreType": "AICPU-SCHED", "freq": 50, "tasks": [
            {"name": "BEGIN", "end": 5236903326282},
            {"name": "ALLOC_THREAD_ID", "end": 5236903326381},
            {"name": "INIT", "end": 5236903329385},
            {"name": "HAND_SHAKE", "end": 5236903330212},
            {"name": "WAIT_ALL_TASK_FIN", "end": 5236903331821},
            {"name": "SEND_STOP", "end": 5236903332087},
            {"name": "EXIT", "end": 5236903332854}
        ]},
        {"blockIdx": 1, "coreType": "AICPU-SCHED", "freq": 50, "tasks": [
            {"name": "BEGIN", "end": 5236903326282},
            {"name": "ALLOC_THREAD_ID", "end": 5236903326383},
            {"name": "INIT", "end": 5236903329389},
            {"name": "HAND_SHAKE", "end": 5236903330219},
            {"name": "WAIT_SEND_FIRST_TASK", "end": 5236903331446},
            {"name": "WAIT_ALL_TASK_FIN", "end": 5236903331829},
            {"name": "SEND_STOP", "end": 5236903332102},
            {"name": "EXIT", "end": 5236903333765}
        ]}
    ]

    result = convert_to_perfetto_format(sample_data)
    with open('perfetto_output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("example have saved to perfetto_output.json")
    print("You can check it by upload this file to https://ui.perfetto.dev/")


def load_json(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def to_us(cycles: float, freq: float) -> float:
    return safe_div(cycles, freq)


def display_width(text: str) -> int:
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
    return width


def pad_cell(text: str, width: int) -> str:
    pad = max(width - display_width(text), 0)
    return text + (" " * pad)


def render_table_lines(headers: List[str], rows: List[List[str]]) -> List[str]:
    line_rows: List[List[str]] = [[str(x) for x in headers]]
    line_rows.extend([[str(x) for x in row] for row in rows])

    widths = [display_width(h) for h in headers]
    for row in line_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], display_width(cell))

    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(pad_cell(cell, widths[i]) for i, cell in enumerate(row)) + " |"

    def fmt_border() -> str:
        return "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"

    lines = [fmt_border(), fmt_row(line_rows[0]), fmt_border()]
    for row in line_rows[1:]:
        lines.append(fmt_row(row))
    lines.append(fmt_border())
    return lines


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    for line in render_table_lines(headers, rows):
        print(line)


def print_section(title: str, total_width: Optional[int] = None) -> None:
    if total_width is None:
        term_cols = shutil.get_terminal_size(fallback=(100, 20)).columns
        total_width = max(60, term_cols - 2)
    else:
        total_width = max(total_width, 20)
    text = f" {title} "
    text_width = display_width(text)
    if text_width >= total_width:
        print(f"\n{text}")
        return
    left = (total_width - text_width) // 2
    right = total_width - text_width - left
    print("\n" + ("=" * left) + text + ("=" * right))


def parse_task_name(name: str) -> Tuple[str, Optional[int], Optional[int]]:
    m = re.match(r"^([A-Z0-9_]+?)(?:_(\d+))?(?:\((\d+)\))?$", str(name))
    if not m:
        return str(name), None, None
    base = m.group(1)
    round_id = int(m.group(2)) if m.group(2) is not None else None
    idx = int(m.group(3)) if m.group(3) is not None else None
    return base, round_id, idx


def collect_round_ids(aicpu_dev_pref: List[Dict[str, Any]]) -> List[Optional[int]]:
    round_ids = set()
    for core in aicpu_dev_pref:
        for task in core.get("tasks", []):
            _, round_id, _ = parse_task_name(task.get("name", ""))
            if round_id is not None:
                round_ids.add(round_id)
    if not round_ids:
        return [None]
    return sorted(round_ids)


def get_task_cycle(
    tasks: List[Dict[str, Any]],
    task_name: str,
    idx: Optional[int] = None,
    round_id: Optional[int] = None,
) -> Optional[float]:
    for task in tasks:
        base, task_round, num = parse_task_name(task.get("name", ""))
        if base != task_name:
            continue
        if round_id is not None and task_round != round_id:
            continue
        if idx is not None and num != idx:
            continue
        return float(task.get("end", 0))
    return None


def get_task_cycle_map(
    tasks: List[Dict[str, Any]],
    task_name: str,
    round_id: Optional[int] = None,
) -> Dict[int, float]:
    cycle_map: Dict[int, float] = {}
    for task in tasks:
        base, task_round, num = parse_task_name(task.get("name", ""))
        if base != task_name:
            continue
        if round_id is not None and task_round != round_id:
            continue
        if num is None:
            continue
        cycle_map[num] = float(task.get("end", 0))
    return cycle_map


def calc_duration_from_ends(start_end: Optional[float], end_end: Optional[float]) -> Optional[float]:
    if start_end is None or end_end is None:
        return None
    return end_end - start_end


@dataclass(frozen=True)
class TaskPoint:
    name: str
    idx: Optional[int] = None


def get_task_duration(
    tasks: List[Dict[str, Any]],
    start: TaskPoint,
    end: TaskPoint,
    round_id: Optional[int] = None,
) -> Optional[float]:
    start_end = get_task_cycle(tasks, start.name, start.idx, round_id)
    end_end = get_task_cycle(tasks, end.name, end.idx, round_id)
    return calc_duration_from_ends(start_end, end_end)


def format_us(v: Optional[float], freq: float) -> str:
    if v is None:
        return "-"
    return f"{to_us(v, freq):.2f}"


def calc_avg_aicore_exit_wait_us(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> Optional[float]:
    aicore_exec_rows = collect_aicore_exec_rows(aicpu_dev_pref, round_id)
    wait_us_values: List[float] = []
    for row in aicore_exec_rows:
        exit_wait = row.get("exit_wait")
        freq = float(row.get("freq", 0)) or 1.0
        if exit_wait is not None and exit_wait > 0:
            wait_us_values.append(to_us(float(exit_wait), freq))
    if not wait_us_values:
        return None
    return sum(wait_us_values) / len(wait_us_values)


def format_sched_post_process(post_dur_cycles: Optional[float], sched_freq: float) -> str:
    if post_dur_cycles is None:
        return "-"
    return f"{to_us(post_dur_cycles, sched_freq):.2f}"


def collect_aicore_exec_rows(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for core in aicpu_dev_pref:
        core_type = str(core.get("coreType", ""))
        if not (core_type.startswith("SCHED") and ("-AIC" in core_type or "-AIV" in core_type)):
            continue
        tasks = core.get("tasks", [])
        begin = get_task_cycle(tasks, "BEGIN", None, round_id)
        wait_first_map = get_task_cycle_map(tasks, "DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK", round_id)
        all_exec_map = get_task_cycle_map(tasks, "DEV_TASK_ALL_CALLOP_TASK_EXEC", round_id)
        wait_first = wait_first_map.get(0)
        all_exec = all_exec_map.get(0)
        wait_exit_notify = get_task_cycle(tasks, "WAIT_EXIT_NOTIFY", None, round_id)
        if not all_exec_map:
            continue

        callop_exec = calc_duration_from_ends(wait_first, all_exec)
        exit_wait = calc_duration_from_ends(all_exec, wait_exit_notify)
        begin_to_exit = calc_duration_from_ends(begin, wait_exit_notify)
        begin_to_wait_first = calc_duration_from_ends(begin, wait_first)
        rows.append(
            {
                "core_type": core_type,
                "block_idx": int(core.get("blockIdx", -1)),
                "freq": float(core.get("freq", 0)) or 1.0,
                "wait_first": wait_first,
                "all_exec": all_exec,
                "wait_first_map": wait_first_map,
                "all_exec_map": all_exec_map,
                "begin_to_wait_first": begin_to_wait_first,
                "callop_exec": callop_exec,
                "exit_wait": exit_wait,
                "begin_to_exit": begin_to_exit,
            }
        )
    rows.sort(key=lambda x: x["block_idx"])
    return rows


def calc_aicore_timing_summary(aicore_exec_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not aicore_exec_rows:
        return "-", "-"

    stitch_ids = set()
    for row in aicore_exec_rows:
        stitch_ids.update(int(x) for x in row.get("all_exec_map", {}).keys())
        stitch_ids.update(int(x) for x in row.get("wait_first_map", {}).keys())
    if not stitch_ids:
        stitch_ids.add(0)

    e2e_per_stitch_us: List[Tuple[int, float]] = []
    for stitch_id in sorted(stitch_ids):
        wait_first_us: List[float] = []
        all_exec_us: List[float] = []
        for row in aicore_exec_rows:
            freq = float(row.get("freq", 1.0)) or 1.0
            wait_first = row.get("wait_first_map", {}).get(stitch_id)
            all_exec = row.get("all_exec_map", {}).get(stitch_id)
            if wait_first is None or all_exec is None or all_exec <= wait_first:
                continue
            wait_first_us.append(to_us(wait_first, freq))
            all_exec_us.append(to_us(all_exec, freq))
        if wait_first_us and all_exec_us:
            e2e_per_stitch_us.append((stitch_id, max(all_exec_us) - min(wait_first_us)))

    begin_to_exit_values: List[float] = []
    for row in aicore_exec_rows:
        begin_to_exit = row.get("begin_to_exit")
        if begin_to_exit is not None and begin_to_exit > 0:
            freq = float(row.get("freq", 1.0)) or 1.0
            begin_to_exit_values.append(to_us(begin_to_exit, freq))

    e2e_time = "-"
    total_runtime_max = "-"
    if e2e_per_stitch_us:
        e2e_time = f"{sum(x[1] for x in e2e_per_stitch_us):.2f}"
    if begin_to_exit_values:
        total_runtime_max = f"{max(begin_to_exit_values):.2f}"
    return e2e_time, total_runtime_max


def build_ctrl_row(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> Optional[List[str]]:
    ctrl = next((x for x in aicpu_dev_pref if str(x.get("coreType")) == "AICPU-CTRL"), None)
    if ctrl is None:
        return None
    tasks = ctrl.get("tasks", [])
    freq = float(ctrl.get("freq", 0)) or 1.0
    block_idx = int(ctrl.get("blockIdx", 0))
    build_dur = get_task_duration(tasks, TaskPoint("BEGIN"), TaskPoint("DEV_TASK_BUILD", 0), round_id)
    ctrl_post_dur = get_task_duration(tasks, TaskPoint("DEV_TASK_BUILD", 0), TaskPoint("EXIT"), round_id)
    ctrl_total_dur = get_task_duration(tasks, TaskPoint("BEGIN"), TaskPoint("EXIT"), round_id)
    return [
        f"AICPU-CTRL-{block_idx}",
        format_us(build_dur, freq),
        "-",
        "-",
        "-",
        "-",
        format_us(ctrl_post_dur, freq),
        "-",
        format_us(ctrl_total_dur, freq),
    ]


def build_sched_rows(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> List[List[str]]:
    rows: List[List[str]] = []
    scheds = [x for x in aicpu_dev_pref if str(x.get("coreType")) == "AICPU-SCHED"]
    for s in sorted(scheds, key=lambda x: int(x.get("blockIdx", 0))):
        block_idx = int(s.get("blockIdx", -1))
        tasks = s.get("tasks", [])
        freq = float(s.get("freq", 0)) or 1.0
        alloc_dur = get_task_duration(tasks, TaskPoint("BEGIN"), TaskPoint("ALLOC_THREAD_ID"), round_id)
        init_dur = get_task_duration(tasks, TaskPoint("ALLOC_THREAD_ID"), TaskPoint("INIT"), round_id)
        handshake_dur = get_task_duration(tasks, TaskPoint("INIT"), TaskPoint("CORE_HAND_SHAKE"), round_id)
        dev_task_rcv = get_task_duration(tasks, TaskPoint("CORE_HAND_SHAKE"), TaskPoint("DEV_TASK_RCV", 0), round_id)
        post_dur = get_task_duration(tasks, TaskPoint("DEV_TASK_SCHED_EXEC", 0), TaskPoint("EXIT"), round_id)
        sched_total_dur = get_task_duration(tasks, TaskPoint("BEGIN"), TaskPoint("WAIT_CORE_EXIT"), round_id)
        rows.append(
            [
                f"AICPU-SCHED-{block_idx}",
                "-",
                format_us(alloc_dur, freq),
                format_us(init_dur, freq),
                format_us(handshake_dur, freq),
                format_us(dev_task_rcv, freq),
                format_sched_post_process(post_dur, freq),
                "-",
                format_us(sched_total_dur, freq),
            ]
        )
    return rows


def build_aicore_row(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> List[str]:
    aicore_exec_rows = collect_aicore_exec_rows(aicpu_dev_pref, round_id)
    aicore_post_process_us = calc_avg_aicore_exit_wait_us(aicpu_dev_pref, round_id)
    if not aicore_exec_rows:
        return ["AICore", "-", "-", "-", "-", "-", "-", "-", "-"]
    e2e_time, total_runtime_max = calc_aicore_timing_summary(aicore_exec_rows)
    return [
        "AICore",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-" if aicore_post_process_us is None else f"{aicore_post_process_us:.2f}",
        e2e_time,
        total_runtime_max,
    ]


def build_round_combined_rows(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> List[List[str]]:
    rows: List[List[str]] = []
    ctrl_row = build_ctrl_row(aicpu_dev_pref, round_id)
    if ctrl_row is not None:
        rows.append(ctrl_row)
    rows.extend(build_sched_rows(aicpu_dev_pref, round_id))
    rows.append(build_aicore_row(aicpu_dev_pref, round_id))
    return rows


def analyze_output_command(output_dir_arg: Optional[str]) -> None:
    if output_dir_arg:
        input_path = Path(output_dir_arg)
    else:
        print("Error: analyze requires an input json path")
        return

    if not input_path.exists():
        print(f"Error: path does not exist: {input_path}")
        return

    aicpu_pref_file = input_path
    analyze_target = str(input_path)

    if not aicpu_pref_file.exists():
        print(f"Error: {aicpu_pref_file} does not exist")
        return

    print(f"Analyzing input: {analyze_target}")
    aicpu_dev_pref = load_json(aicpu_pref_file)
    if not isinstance(aicpu_dev_pref, list):
        print("Error: invalid aicpu_dev_pref.json format, expected list")
        return

    rounds = collect_round_ids(aicpu_dev_pref)
    for round_id in rounds:
        display_round = 1 if round_id is None else (round_id + 1)
        round_name = f"round{display_round}"
        headers = [
            "Compute Units",
            "DEV_TASK_BUILD(us)",
            "ALLOC_THREAD_ID(us)",
            "INIT(us)",
            "CORE_HAND_SHAKE(us)",
            "DEV_TASK_RCV(us)",
            "Post-process(us)",
            "End-to-End time(us)",
            "Total run time(us)",
        ]
        rows = build_round_combined_rows(aicpu_dev_pref, round_id)
        table_lines = render_table_lines(headers, rows)
        table_width = max(display_width(line) for line in table_lines) if table_lines else None
        print_section(round_name, table_width)
        print_table(headers, rows)
    print()


def main():
    parser = argparse.ArgumentParser(description='Performance data processing tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # parse_log 子命令
    parse_parser = subparsers.add_parser('parse_log', help='Parse device log and generate performance JSON')
    parse_parser.add_argument('input_file', help='Path to input log file')
    parse_parser.add_argument('output_file', help='Path to output JSON file')

    # gen_perfetto 子命令
    perfetto_parser = subparsers.add_parser('gen_perfetto', help='Convert performance JSON to Perfetto format')
    perfetto_parser.add_argument('input_file', help='Input JSON file path')
    perfetto_parser.add_argument('output_file', help='Output Perfetto JSON file path')
    perfetto_parser.add_argument('kernel_file', help='aicore kernel Perfetto JSON file path', default="", nargs='?')

    # gen_perfetto_example 子命令
    example_parser = subparsers.add_parser('gen_perfetto_example', help='Generate example Perfetto data')
    # analyze 子命令
    analyze_parser = subparsers.add_parser('analyze', help='Analyze perf json by output dir or json file path')
    analyze_parser.add_argument(
        'output_dir',
        nargs='?',
        help='Output directory or perf json file path; latest output_* if omitted',
    )
    args = parser.parse_args()

    if args.command == 'parse_log':
        parse_log_command(args.input_file, args.output_file)
    elif args.command == 'gen_perfetto':
        gen_perfetto_command(args.input_file, args.output_file, args.kernel_file)
    elif args.command == 'gen_perfetto_example':
        gen_perfetto_example()
    elif args.command == 'analyze':
        analyze_output_command(args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
