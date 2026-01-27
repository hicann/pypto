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
from typing import Dict, List, Any


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
            if prev_end is None:
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


def gen_perfetto_command(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        perfetto_data = convert_to_perfetto_format(input_data)
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

    # gen_perfetto_example 子命令
    example_parser = subparsers.add_parser('gen_perfetto_example', help='Generate example Perfetto data')

    args = parser.parse_args()

    if args.command == 'parse_log':
        parse_log_command(args.input_file, args.output_file)
    elif args.command == 'gen_perfetto':
        gen_perfetto_command(args.input_file, args.output_file)
    elif args.command == 'gen_perfetto_example':
        gen_perfetto_example()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
