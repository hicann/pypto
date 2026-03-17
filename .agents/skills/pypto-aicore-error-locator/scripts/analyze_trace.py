#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import re
import sys
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    parse_luid,
    parse_core_idx,
    validate_path,
    setup_logging
)

setup_logging()

logger = logging.getLogger(__name__)


def find_trace_log_file(device_log_path):
    logger.info("在 %s 下搜索包含 trace 的日志文件...", device_log_path)
    
    log_files = list(Path(device_log_path).rglob("*.log"))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'trace' in content and ('LActStart' in content or 'LActFinish' in content):
                    logger.info("找到 trace 日志文件: %s", log_file)
                    return str(log_file)
        except OSError as e:
            logger.warning("读取文件失败: %s, 原因: %s", log_file, e)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("处理文件时发生异常: %s, 原因: %s", log_file, e)
    
    logger.info("错误：在 %s 下未找到包含 trace 的日志文件", device_log_path)
    return None


def analyze_trace(log_file):
    lactstart_events = []
    lactfinish_events = []

    logger.info("=" * 80)
    logger.info("分析追踪日志")
    logger.info("=" * 80)
    logger.info("日志文件: %s", log_file)
    logger.info("")

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'trace' in line and 'LActStart' in line:
                luid_match = re.search(r'LUid\{[^}]+\}', line)
                lactstart_match = re.search(r'LActStart\{[^}]+\}', line)
                
                if luid_match and lactstart_match:
                    luid = parse_luid(luid_match.group(0))
                    core_idx = parse_core_idx(lactstart_match.group(0), r'LActStart\{(\d+)\}')
                    
                    if luid and core_idx is not None:
                        lactstart_events.append({
                            'luid': luid,
                            'coreIdx': core_idx
                        })
            
            elif 'trace' in line and 'LActFinish' in line:
                luid_match = re.search(r'LUid\{[^}]+\}', line)
                lactfinish_match = re.search(r'LActFinish\{[^}]+\}', line)
                
                if luid_match and lactfinish_match:
                    luid = parse_luid(luid_match.group(0))
                    core_idx = parse_core_idx(lactfinish_match.group(0), r'LActFinish\{(\d+)\}')
                    
                    if luid and core_idx is not None:
                        lactfinish_events.append({
                            'luid': luid,
                            'coreIdx': core_idx
                        })

    lactstart_core_idxs = sorted(set(event['coreIdx'] for event in lactstart_events))
    lactfinish_core_idxs = sorted(set(event['coreIdx'] for event in lactfinish_events))

    logger.info("LActStart 事件数量: %d", len(lactstart_events))
    logger.info("LActStart coreIdxs: %s", lactstart_core_idxs)
    logger.info("LActFinish 事件数量: %d", len(lactfinish_events))
    logger.info("LActFinish coreIdxs: %s", lactfinish_core_idxs)

    missing_core_idxs = [idx for idx in lactstart_core_idxs if idx not in lactfinish_core_idxs]
    
    logger.info("\n缺失的 coreIdxs: %s", missing_core_idxs)

    missing_leaf_indices = []
    for event in lactstart_events:
        if event['coreIdx'] in missing_core_idxs:
            leaf_idx = event['luid']['leafIndex']
            if leaf_idx not in missing_leaf_indices:
                missing_leaf_indices.append(leaf_idx)

    logger.info("对应的 leafIndices: %s", missing_leaf_indices)
    logger.info("")

    return missing_leaf_indices


def find_cce_file(kernel_aicore_dir, leaf_index):
    logger.info("定位问题 CCE 文件 (leafIndex: %d)", leaf_index)
    logger.info("kernel_aicore 目录: %s", kernel_aicore_dir)
    logger.info("")
    
    record_files = list(Path(kernel_aicore_dir).glob("**/sub_func_*_call_*.h"))
    
    logger.info("找到 %d 个 Record 文件", len(record_files))
    
    target_record_file = None
    core_type = None
    func_name = None
    
    for record_file in record_files:
        with open(record_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if f'case {leaf_index}:' in content:
                logger.info("\n在文件中找到 leafIndex %d: %s", leaf_index, record_file)
                target_record_file = record_file
                
                match = re.search(r'sub_func_(\w+)_call_\d+\.h', record_file.name)
                if match:
                    core_type = match.group(1)
                    logger.info("Core type: %s", core_type)
                
                pattern = rf'case {leaf_index}:\s*\{{\s*(\w+)\('
                match = re.search(pattern, content)
                if match:
                    func_name = match.group(1)
                    logger.info("Function name: %s", func_name)
                break
    
    if not target_record_file:
        logger.info("\n错误：无法在任何 Record 文件中找到 leafIndex %d", leaf_index)
        return None
    
    if func_name:
        match = re.search(r'_(\d+)_(\d+)_(\d+)$', func_name)
        if match:
            cce_id = match.group(1)
            id_val = match.group(2)
            func_hash = match.group(3)
            cce_pre_name = func_name[:func_name.rfind(f'_{cce_id}_{id_val}_{func_hash}')]
            
            logger.info("\n从函数名提取信息:")
            logger.info("  函数名: %s", func_name)
            logger.info("  CCE_pre_name: %s", cce_pre_name)
            logger.info("  CCE_ID: %s", cce_id)
            logger.info("  ID: %s", id_val)
            logger.info("  func_hash: %s", func_hash)
            
            cce_pattern = f"{cce_pre_name}_{cce_id}_*_{id_val}_{core_type}.cpp"
            cce_files = list(Path(kernel_aicore_dir).glob(f"**/{cce_pattern}"))
            
            logger.info("\n搜索 CCE 文件模式: %s", cce_pattern)
            logger.info("找到 %d 个匹配文件", len(cce_files))
            
            if cce_files:
                logger.info("\n找到 CCE 文件: %s", cce_files[0])
                return str(cce_files[0])
            else:
                logger.info("\n错误：无法查找到匹配模式的 CCE 文件: %s", cce_pattern)
                return None
        else:
            logger.info("\n错误：无法解析函数名格式: %s", func_name)
            return None
    
    return None


def find_all_cce_files(kernel_aicore_dir, missing_leaf_indices):
    problem_cce_files = []
    for leaf_index in missing_leaf_indices:
        logger.info("=" * 80)
        cce_file = find_cce_file(kernel_aicore_dir, leaf_index)
        if cce_file:
            problem_cce_files.append(cce_file)
    
    return problem_cce_files


def print_usage():
    logger.info("用法: python3 analyze_trace.py <device_log_path> <kernel_aicore_dir>")
    logger.info("")
    logger.info("参数说明:")
    logger.info("  device_log_path: device log 落盘路径")
    logger.info("  kernel_aicore_dir: kernel_aicore 目录路径")


def main():
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    device_log_path = sys.argv[1]
    kernel_aicore_dir = sys.argv[2]
    
    device_log_path = os.path.abspath(device_log_path)
    kernel_aicore_dir = os.path.abspath(kernel_aicore_dir)
    
    valid, error_msg = validate_path(device_log_path, "device log 路径")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)
    
    valid, error_msg = validate_path(kernel_aicore_dir, "kernel_aicore 目录")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)
    
    log_file = find_trace_log_file(device_log_path)
    if not log_file:
        sys.exit(1)
    
    logger.info("")
    missing_leaf_indices = analyze_trace(log_file)
    
    if not missing_leaf_indices:
        logger.info("=" * 80)
        logger.info("结果：没有发现缺失的 leaf index")
        logger.info("所有任务已成功完成，无需定位问题 CCE 文件")
        logger.info("=" * 80)
        return
    
    problem_cce_files = find_all_cce_files(kernel_aicore_dir, missing_leaf_indices)
    
    logger.info("=" * 80)
    logger.info("定位结果")
    logger.info("=" * 80)
    
    if not problem_cce_files:
        logger.info("\n未找到问题 CCE 文件")
        logger.info("=" * 80)
        return
    
    logger.info("\n找到 %d 个问题 CCE 文件:", len(problem_cce_files))
    for i, cce_file in enumerate(problem_cce_files, 1):
        logger.info("  %d. %s", i, cce_file)
    
    logger.info("=" * 80)
    for cce_file in problem_cce_files:
        logger.info(cce_file)


if __name__ == "__main__":
    main()
