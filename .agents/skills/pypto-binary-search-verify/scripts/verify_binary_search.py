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
PyPTO 二分调试通用对比脚本
按照技能文档自动检测检查点并对比 jit 和 golden 的中间结果
"""
import os
import sys
import glob
import argparse
import logging
from pathlib import Path
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def find_latest_output_dir(work_dir="."):
    """找到最新的 output 目录"""
    output_base = os.path.join(work_dir, "output")
    if not os.path.exists(output_base):
        return None

    output_dirs = [d for d in os.listdir(output_base) if d.startswith("output_")]
    if not output_dirs:
        return None

    # 按修改时间排序，取最新的
    output_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(output_base, x)), reverse=True)
    return output_dirs[0]


def scan_checkpoints(work_dir="."):
    """扫描所有检查点文件"""
    latest_dir = find_latest_output_dir(work_dir)
    if not latest_dir:
        return None, []

    tensor_dir = os.path.join(work_dir, "output", latest_dir, "tensor")
    if not os.path.exists(tensor_dir):
        return latest_dir, []

    # 查找所有 .data 文件
    data_files = glob.glob(os.path.join(tensor_dir, "*.data"))

    # 提取检查点名称（去掉后缀的数字和 .data）
    checkpoints = set()
    for data_file in data_files:
        basename = os.path.basename(data_file)
        # 格式: checkpoint_name_number.data
        # 提取 checkpoint_name
        parts = basename.rsplit('_', 1)
        if len(parts) == 2 and parts[1].replace('.data', '').isdigit():
            checkpoint_name = parts[0]
            checkpoints.add(checkpoint_name)

    return latest_dir, sorted(list(checkpoints))


def read_jit_data(filename):
    """读取 jit 生成的数据文件"""
    data = np.fromfile(filename, dtype=np.float32)
    return data


def compare_with_golden(jit_data, golden_data, name, rtol=1e-3, atol=1e-3, verbose=True):
    """对比 jit 结果与 golden 结果"""
    min_size = min(jit_data.shape[0], golden_data.shape[0])
    jit_data_to_compare = jit_data[:min_size]
    golden_data_to_compare = golden_data[:min_size]

    if verbose:
        logger.info(f"  对比范围: 前 {min_size} 个元素 (jit={jit_data.shape[0]}, golden={golden_data.shape[0]})")

    diff = np.max(np.abs(jit_data_to_compare - golden_data_to_compare))
    max_val = np.max(np.abs(golden_data_to_compare))
    relative_error = diff / (max_val + 1e-10)

    match = relative_error < rtol and diff < atol

    status = "✓ PASS" if match else "✗ FAIL"
    logger.info(f"\n{name}: {status}")
    logger.info(f"  Max diff: {diff:.6f}")
    logger.info(f"  Max val: {max_val:.6f}")
    logger.info(f"  Relative error: {relative_error:.6f}")
    logger.info(f"  Tolerance: rtol={rtol}, atol={atol}")

    if verbose and not match:
        logger.info(f"  前10个元素对比:")
        for i in range(min(10, min_size)):
            jit_val = jit_data_to_compare[i]
            golden_val = golden_data_to_compare[i]
            diff_val = abs(jit_val - golden_val)
            logger.info(f"    [{i}] jit={jit_val:.6f}, golden={golden_val:.6f}, diff={diff_val:.6f}")

    return match


def analyze_results(results):
    """分析对比结果，给出二分建议"""
    logger.info("\n" + "=" * 80)
    logger.info("步骤 5：根据结果继续二分")
    logger.info("=" * 80)

    first_fail_idx = -1
    for idx, (name, match) in enumerate(results):
        if not match:
            first_fail_idx = idx
            break

    if first_fail_idx == -1:
        logger.info("✓ 所有检查点都匹配")
        logger.info("→ 问题可能在：检查点之后的操作")
        return

    fail_name = results[first_fail_idx][0]

    if first_fail_idx == 0:
        logger.info(f"✗ 第一个检查点 ({fail_name}) 就不匹配")
        logger.info(f"→ 问题可能在：输入数据或第一个计算步骤")
        logger.info(f"→ 建议：检查输入数据是否正确，或在更早的位置插入检查点")
    else:
        prev_name = results[first_fail_idx - 1][0]
        logger.info(f"✗ 检查点 {prev_name} 匹配，但 {fail_name} 不匹配")
        logger.info(f"→ 问题位置：{prev_name} 和 {fail_name} 之间的操作")
        logger.info(f"→ 建议：在这两个检查点之间插入新的检查点，进一步定位问题")


def main():
    parser = argparse.ArgumentParser(description='PyPTO 二分调试对比工具')
    parser.add_argument('--work-dir', '-w', default='.',
                        help='工作目录（默认为当前目录）')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='指定 output 目录名（不指定则自动检测最新的）')
    parser.add_argument('--rtol', type=float, default=1e-3,
                        help='相对误差容忍度（默认 1e-3）')
    parser.add_argument('--atol', type=float, default=1e-3,
                        help='绝对误差容忍度（默认 1e-3）')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细的元素级对比')
    parser.add_argument('--list', '-l', action='store_true',
                        help='只列出检查点，不进行对比')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PyPTO 二分调试对比工具")
    logger.info("=" * 80)

    if args.output_dir:
        latest_dir = args.output_dir
    else:
        latest_dir = find_latest_output_dir(args.work_dir)

    if not latest_dir:
        logger.error("✗ 未找到 output 目录")
        sys.exit(1)

    latest_dir_full = latest_dir
    checkpoints = []

    # 扫描指定目录
    tensor_dir = os.path.join(args.work_dir, "output", latest_dir_full, "tensor")
    if os.path.exists(tensor_dir):
        data_files = glob.glob(os.path.join(tensor_dir, "*.data"))
        checkpoints = set()
        for data_file in data_files:
            basename = os.path.basename(data_file)
            parts = basename.rsplit('_', 1)
            if len(parts) == 2 and parts[1].replace('.data', '').isdigit():
                checkpoint_name = parts[0]
                checkpoints.add(checkpoint_name)
        checkpoints = sorted(list(checkpoints))

    if not checkpoints:
        logger.error(f"✗ 未在 {tensor_dir} 找到检查点文件")
        sys.exit(1)

    logger.info(f"✓ 找到 output 目录: {latest_dir_full}")
    logger.info(f"✓ 找到 {len(checkpoints)} 个检查点: {checkpoints}")

    if args.list:
        logger.info("\n检查点列表:")
        for idx, ckpt in enumerate(checkpoints, 1):
            logger.info(f"  {idx}. {ckpt}")
        sys.exit(0)

    logger.info("\n" + "=" * 80)
    logger.info("步骤 4：对比 jit 和 golden 数据")
    logger.info("=" * 80)

    results = []

    for checkpoint_name in checkpoints:
        # 查找 jit 文件
        jit_pattern = os.path.join(tensor_dir, f"{checkpoint_name}_*.data")
        jit_files = sorted(glob.glob(jit_pattern))

        if not jit_files:
            logger.warning(f"\n{checkpoint_name}: ✗ 未找到 jit 文件")
            results.append((checkpoint_name, False))
            continue

        golden_pattern = os.path.join(args.work_dir, f"golden_{checkpoint_name}.bin")
        golden_files = glob.glob(golden_pattern)

        if not golden_files:
            logger.warning(f"\n{checkpoint_name}: ✗ 未找到 golden 文件 ({golden_pattern})")
            results.append((checkpoint_name, False))
            continue

        jit_file = jit_files[0]
        golden_file = golden_files[0]

        logger.info(f"\n使用文件:")
        logger.info(f"  jit: {os.path.basename(jit_file)}")
        logger.info(f"  golden: {os.path.basename(golden_file)}")

        # 读取数据
        jit_data = read_jit_data(jit_file)
        golden_data = np.fromfile(golden_file, dtype=np.float32)

        # 对比
        match = compare_with_golden(jit_data, golden_data, checkpoint_name,
                                   rtol=args.rtol, atol=args.atol, verbose=args.verbose)
        results.append((checkpoint_name, match))

    # 分析结果
    analyze_results(results)

    # 返回码
    all_match = all(match for _, match in results)
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
