#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import subprocess
import re


def validate_path(path, path_type="路径"):
    if not os.path.exists(path):
        return False, f"错误：{path_type}不存在: {path}"
    return True, None


def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def print_error_info(output, logger, max_lines=10):
    error_lines = [line for line in output.split('\n') if 'error' in line.lower()]
    if error_lines:
        logger.info("Error 信息:")
        for line in error_lines[:max_lines]:
            logger.info(f"  {line}")


def get_commentable_lines(lines, error_in_t=False):
    commentable_lines = []
    fast_commentable_lines = []
    skip_keywords = ['set_flag', 'wait_flag', 'pipe_barrier']
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        should_skip = (
            not stripped or
            stripped.startswith('//') or
            '{' in stripped or
            '}' in stripped or
            '#' in stripped or
            any(keyword in stripped for keyword in skip_keywords)
        )

        if should_skip:
            continue
        else:
            commentable_lines.append(i)
            is_t_operation = stripped.startswith('T') and '<' in stripped and '>' in stripped
            if is_t_operation:
                fast_commentable_lines.append(i)

    if error_in_t:
        return fast_commentable_lines
    else:
        return commentable_lines


def comment_lines(lines, line_indices):
    lines_to_comment = set(line_indices)
    
    for line_num in sorted(lines_to_comment, reverse=True):
        line_idx = line_num - 1
        lines[line_idx] = '// ' + lines[line_idx]
    
    return lines


def uncomment_lines(lines, line_indices):
    lines_to_uncomment = set(line_indices)
    
    for line_num in sorted(lines_to_uncomment):
        line_idx = line_num - 1
        if lines[line_idx].strip().startswith('//'):
            lines[line_idx] = lines[line_idx][3:]
    
    return lines


def has_error(returncode, output):
    if returncode != 0:
        return True
    
    output_lower = output.lower()
    
    true_error_keywords = [
        ' error',
        'error ',
        'exception',
        'segmentation fault',
        'core dump',
    ]
    
    for keyword in true_error_keywords:
        if keyword in output_lower and "aicore error" not in output_lower:
            raise RuntimeError(f"检测到非 aicore error: {output_lower}")
        
        if keyword in output_lower:
            return True
    
    return False


def run_test(test_cmd, run_dir):
    import shlex
    if isinstance(test_cmd, str):
        test_cmd = shlex.split(test_cmd)
    result = subprocess.run(
        test_cmd,
        cwd=run_dir,
        capture_output=True,
        text=True,
        errors='ignore',
        timeout=1800,
        check=False
    )
    return result.returncode, result.stdout + result.stderr


def comment_special_lines(lines):
    for i, line in enumerate(lines):
        if 'set_flag' in line or 'wait_flag' in line or 'pipe_barrier' in line:
            if not line.strip().startswith('//'):
                lines[i] = '// ' + line
    return lines


def parse_luid(luid_str):
    match = re.search(r'LUid\{(\d+),(\d+),(\d+),(\d+),(\d+)\}', luid_str)
    if match:
        return {
            'deviceTaskId': int(match.group(1)),
            'funcId': int(match.group(2)),
            'rootIndex': int(match.group(3)),
            'opIdx': int(match.group(4)),
            'leafIndex': int(match.group(5))
        }
    return None


def parse_core_idx(event_str, pattern):
    match = re.search(pattern, event_str)
    if match:
        return int(match.group(1))
    return None
