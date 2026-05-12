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
    sorted_tasks: Optional[List[Dict[str, Any]]] = None,
) -> Optional[float]:
    tasks_by_end = sorted_tasks if sorted_tasks is not None else sort_tasks_by_end(tasks)
    for task in tasks_by_end:
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
    sorted_tasks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[int, float]:
    cycle_map: Dict[int, float] = {}
    tasks_by_end = sorted_tasks if sorted_tasks is not None else sort_tasks_by_end(tasks)
    for task in tasks_by_end:
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


def sort_tasks_by_end(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(tasks, key=lambda x: float(x.get("end", 0)))


def normalize_cores_tasks_by_end(aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    for core in aicpu_dev_pref:
        tasks = core.get("tasks", [])
        if isinstance(tasks, list):
            core["tasks"] = sort_tasks_by_end(tasks)


def get_event_duration(
    tasks: List[Dict[str, Any]],
    event_name: str,
    round_id: Optional[int] = None,
    idx: Optional[int] = None,
) -> Optional[float]:
    prev_end: Optional[float] = None
    for task in sort_tasks_by_end(tasks):
        base, task_round, task_idx = parse_task_name(task.get("name", ""))
        if round_id is not None and task_round != round_id:
            continue

        task_end = float(task.get("end", 0))
        event_dur = None if prev_end is None else (task_end - prev_end)

        if base == event_name and (idx is None or task_idx == idx):
            return event_dur

        prev_end = task_end
    return None


def calc_event_duration_sum(
    tasks: List[Dict[str, Any]],
    event_name: str,
    round_id: Optional[int] = None,
) -> Optional[float]:
    total = 0.0
    found = False
    prev_end: Optional[float] = None
    for task in sort_tasks_by_end(tasks):
        base, task_round, _ = parse_task_name(task.get("name", ""))
        if round_id is not None and task_round != round_id:
            continue

        task_end = float(task.get("end", 0))
        event_dur = None if prev_end is None else (task_end - prev_end)

        if base == event_name and event_dur is not None:
            found = True
            total += max(event_dur, 0.0)

        prev_end = task_end
    return total if found else None


def calc_sched_post_process_sum(
    tasks: List[Dict[str, Any]],
    round_id: Optional[int],
) -> Optional[float]:
    post_events = [
        ("DEV_TASK_SYNC_CORE_STOP", 0),
        ("DEV_TASK_RSP", 0),
        ("WAIT_ALL_DEV_TASK_FINISH", None),
        ("WAIT_CORE_EXIT", None),
        ("EXIT", None),
    ]

    total = 0.0
    found = False
    for event_name, idx in post_events:
        dur = get_event_duration(tasks, event_name, round_id, idx)
        if dur is None:
            continue
        found = True
        total += max(dur, 0.0)
    return total if found else None


def calc_total_runtime_sum(
    tasks: List[Dict[str, Any]],
    round_id: Optional[int] = None,
) -> Optional[float]:
    prev_end: Optional[float] = None
    total = 0.0
    found = False
    for task in sort_tasks_by_end(tasks):
        _, task_round, _ = parse_task_name(task.get("name", ""))
        if round_id is not None and task_round != round_id:
            continue
        task_end = float(task.get("end", 0))
        if prev_end is not None:
            total += max(task_end - prev_end, 0.0)
        prev_end = task_end
        found = True
    return total if found else None


def calc_runtime_in_window(
    tasks: List[Dict[str, Any]],
    round_id: Optional[int],
    start_event: str,
    end_events: List[str],
) -> Optional[float]:
    sorted_tasks = sort_tasks_by_end(tasks)
    start_cycle = get_task_cycle(tasks, start_event, None, round_id, sorted_tasks)
    if start_cycle is None:
        return None

    end_candidates: List[float] = []
    for end_event in end_events:
        for task in sorted_tasks:
            base, task_round, _ = parse_task_name(task.get("name", ""))
            if base != end_event:
                continue
            if round_id is not None and task_round != round_id:
                continue
            task_end = float(task.get("end", 0))
            if task_end >= float(start_cycle):
                end_candidates.append(task_end)
                break
    if not end_candidates:
        return None
    return max(max(end_candidates) - float(start_cycle), 0.0)


def get_end_bounds(
    tasks: List[Dict[str, Any]],
    round_id: Optional[int] = None,
    sorted_tasks: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    tasks_by_end = sorted_tasks if sorted_tasks is not None else sort_tasks_by_end(tasks)
    min_end: Optional[float] = None
    max_end: Optional[float] = None
    for task in tasks_by_end:
        _, task_round, _ = parse_task_name(task.get("name", ""))
        if round_id is not None and task_round != round_id:
            continue
        task_end = float(task.get("end", 0))
        if min_end is None:
            min_end = task_end
        max_end = task_end
    return min_end, max_end


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


def calc_aicore_init_preprocess_us(aicore_exec_rows: List[Dict[str, Any]]) -> Optional[float]:
    min_end_values = [float(row["min_end"]) for row in aicore_exec_rows if row.get("min_end") is not None]
    wait_first_values = [
        float(row["wait_first_cycle"]) for row in aicore_exec_rows if row.get("wait_first_cycle") is not None
    ]
    if not min_end_values or not wait_first_values:
        return None

    freq = float(aicore_exec_rows[0].get("freq", 1.0)) or 1.0
    return to_us(max(min(wait_first_values) - min(min_end_values), 0.0), freq)


def calc_aicore_last_core_exit_wait_us(aicore_exec_rows: List[Dict[str, Any]]) -> Optional[float]:
    all_exec_values = [
        float(row["all_exec_cycle"]) for row in aicore_exec_rows if row.get("all_exec_cycle") is not None
    ]
    max_end_values = [float(row["max_end"]) for row in aicore_exec_rows if row.get("max_end") is not None]
    if not all_exec_values or not max_end_values:
        return None

    freq = float(aicore_exec_rows[0].get("freq", 1.0)) or 1.0
    return to_us(max(max(max_end_values) - max(all_exec_values), 0.0), freq)


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
        sorted_tasks = sort_tasks_by_end(tasks)
        wait_first_map = get_task_cycle_map(tasks, "DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK", round_id, sorted_tasks)
        all_exec_map = get_task_cycle_map(tasks, "DEV_TASK_ALL_LEAF_TASK_EXEC", round_id, sorted_tasks)
        wait_exit_notify = get_task_cycle(tasks, "WAIT_EXIT_NOTIFY", None, round_id, sorted_tasks)
        if not all_exec_map:
            continue

        wait_first_cycle = min(wait_first_map.values()) if wait_first_map else None
        all_exec_cycle = max(all_exec_map.values()) if all_exec_map else None
        init_dur = get_event_duration(tasks, "INIT", round_id)
        rcv_model_dur = get_event_duration(tasks, "DEV_TASK_RCV_MODEL", round_id)
        first_wait_first_idx = min(wait_first_map, key=wait_first_map.get) if wait_first_map else None
        wait_first_dur = get_event_duration(
            tasks, "DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK", round_id, first_wait_first_idx
        )
        exit_wait = calc_duration_from_ends(all_exec_cycle, wait_exit_notify)
        min_end, max_end = get_end_bounds(tasks, round_id, sorted_tasks)
        rows.append(
            {
                "core_type": core_type,
                "block_idx": int(core.get("blockIdx", -1)),
                "freq": float(core.get("freq", 0)) or 1.0,
                "wait_first_map": wait_first_map,
                "all_exec_map": all_exec_map,
                "exit_wait": exit_wait,
                "wait_first_cycle": wait_first_cycle,
                "all_exec_cycle": all_exec_cycle,
                "init_dur": init_dur,
                "rcv_model_dur": rcv_model_dur,
                "wait_first_dur": wait_first_dur,
                "min_end": min_end,
                "max_end": max_end,
            }
        )
    rows.sort(key=lambda x: x["block_idx"])
    return rows


def calc_aicore_timing_summary(aicore_exec_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not aicore_exec_rows:
        return "-", "-"

    total_runtime_freq = 1.0
    total_runtime_min_end: Optional[float] = None
    total_runtime_max_end: Optional[float] = None
    first_wait_cycle: Optional[float] = None
    last_all_exec_cycle: Optional[float] = None
    for row in aicore_exec_rows:
        min_end = row.get("min_end")
        max_end = row.get("max_end")
        if min_end is None or max_end is None:
            continue
        total_runtime_freq = float(row.get("freq", 1.0)) or 1.0
        total_runtime_min_end = min(min_end, total_runtime_min_end) if total_runtime_min_end is not None else min_end
        total_runtime_max_end = max(max_end, total_runtime_max_end) if total_runtime_max_end is not None else max_end
        wait_first_cycle = row.get("wait_first_cycle")
        all_exec_cycle = row.get("all_exec_cycle")
        if wait_first_cycle is not None:
            first_wait_cycle = (
                min(float(wait_first_cycle), first_wait_cycle)
                if first_wait_cycle is not None
                else float(wait_first_cycle)
            )
        if all_exec_cycle is not None:
            last_all_exec_cycle = (
                max(float(all_exec_cycle), last_all_exec_cycle)
                if last_all_exec_cycle is not None
                else float(all_exec_cycle)
            )

    e2e_time = "-"
    total_runtime_e2e = "-"
    if first_wait_cycle is not None and last_all_exec_cycle is not None:
        e2e_cycles = max(last_all_exec_cycle - first_wait_cycle, 0.0)
        e2e_time = f"{to_us(e2e_cycles, total_runtime_freq):.2f}"
    if total_runtime_min_end is not None and total_runtime_max_end is not None:
        total_runtime_e2e_cycles = max(total_runtime_max_end - total_runtime_min_end, 0.0)
        total_runtime_e2e = f"{to_us(total_runtime_e2e_cycles, total_runtime_freq):.2f}"
    return e2e_time, total_runtime_e2e


def build_ctrl_row(aicpu_dev_pref: List[Dict[str, Any]], round_id: Optional[int]) -> Optional[List[str]]:
    ctrl = next((x for x in aicpu_dev_pref if str(x.get("coreType")) == "AICPU-CTRL"), None)
    if ctrl is None:
        return None
    tasks = ctrl.get("tasks", [])
    freq = float(ctrl.get("freq", 0)) or 1.0
    block_idx = int(ctrl.get("blockIdx", 0))
    build_dur = calc_event_duration_sum(tasks, "DEV_TASK_BUILD", round_id)
    ctrl_post_dur = get_event_duration(tasks, "EXIT", round_id)
    ctrl_total_dur = calc_total_runtime_sum(tasks, round_id)
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
        alloc_dur = get_event_duration(tasks, "ALLOC_THREAD_ID", round_id)
        init_dur = get_event_duration(tasks, "INIT", round_id)
        handshake_dur = get_event_duration(tasks, "CORE_HAND_SHAKE", round_id)
        dev_task_rcv = get_event_duration(tasks, "DEV_TASK_RCV", round_id, 0)
        post_dur = calc_sched_post_process_sum(tasks, round_id)
        sched_total_dur = calc_runtime_in_window(
            tasks, round_id, "BEGIN", ["WAIT_CORE_EXIT", "EXIT"]
        )
        if sched_total_dur is None:
            sched_total_dur = calc_total_runtime_sum(tasks, round_id)
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
    aicore_post_process_us = calc_aicore_last_core_exit_wait_us(aicore_exec_rows)
    aicore_init_us = calc_aicore_init_preprocess_us(aicore_exec_rows)
    if not aicore_exec_rows:
        return ["AICore", "-", "-", "-", "-", "-", "-", "-", "-"]
    e2e_time, total_runtime_e2e = calc_aicore_timing_summary(aicore_exec_rows)
    return [
        "AICore",
        "-",
        "-",
        "-" if aicore_init_us is None else f"{aicore_init_us:.2f}",
        "-",
        "-",
        "-" if aicore_post_process_us is None else f"{aicore_post_process_us:.2f}",
        e2e_time,
        total_runtime_e2e,
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
    normalize_cores_tasks_by_end(aicpu_dev_pref)

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
