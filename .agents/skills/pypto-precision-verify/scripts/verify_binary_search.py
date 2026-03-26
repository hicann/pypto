#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#udi INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO 精度检查点文件对比工具
自动检测检查点并对比 jit 和 golden 的中间结果
"""
import os
import sys
import glob
import argparse
import logging
import re
import numpy as np

# PyPTO 数据类型映射
DTYPE_MAP = {
    1: ('int8', np.int8, 1),
    2: ('int16', np.int16, 2),
    3: ('int32', np.int32, 4),
    4: ('int64', np.int64, 8),
    5: ('fp8', np.uint8, 1),
    6: ('fp16', np.float16, 2),
    7: ('fp32', np.float32, 4),
    8: ('bf16', np.uint16, 2),
}

# 容差标准映射表
TOLERANCE_MAP = {
    1: (1e-3, 1e-3),
    2: (1e-4, 1e-4),
    3: (1e-5, 1e-5),
    4: (1e-5, 1e-5),
    5: (1e-1, 1e-2),
    6: (1e-2, 1e-3),
    7: (1e-4, 1e-5),
    8: (5e-2, 5e-3),
}

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def convert_bf16_to_fp32(data_bf16):
    """BF16 转 FP32"""
    data_fp32 = data_bf16.astype(np.uint32) << 16
    return data_fp32.view(np.float32)


def find_latest_output_dir(work_dir="."):
    """找到最新的 output 目录"""
    output_base = os.path.join(work_dir, "output")
    if not os.path.exists(output_base):
        return None

    output_dirs = [d for d in os.listdir(output_base) if d.startswith("output_")]
    if not output_dirs:
        return None

    output_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(output_base, x)), reverse=True)
    return output_dirs[0]


def scan_checkpoints_from_dir(tensor_dir):
    """从指定目录扫描所有检查点文件"""
    if not os.path.exists(tensor_dir):
        return []

    data_files = glob.glob(os.path.join(tensor_dir, "*.data"))
    checkpoints = set()

    for data_file in data_files:
        basename = os.path.basename(data_file)
        parts = basename.rsplit('_', 1)
        if len(parts) == 2 and parts[1].replace('.data', '').isdigit():
            checkpoint_name = parts[0]
            checkpoints.add(checkpoint_name)

    def get_sort_key(name):
        match = re.match(r'^(\d+)', name)
        return int(match.group(1)) if match else float('inf')

    return sorted(list(checkpoints), key=get_sort_key)


def get_tolerance_by_dtype(dtype, custom_rtol=None, custom_atol=None):
    """根据数据类型返回对应的容差标准"""
    if custom_rtol is not None and custom_atol is not None:
        return custom_rtol, custom_atol
    return TOLERANCE_MAP.get(dtype, (1e-3, 1e-3))


def read_jit_data(filename):
    """读取 jit 生成的数据文件，自动检测数据类型"""
    csv_file = filename.replace('.data', '.csv')

    dtype = None
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            for line in f:
                if line.startswith('dtype,'):
                    dtype = int(line.split(',')[1].strip())
                    break

    data_bytes = np.fromfile(filename, dtype=np.uint8)

    if dtype is None:
        file_size = len(data_bytes)
        if file_size % 2 == 0:
            data_bf16 = np.frombuffer(data_bytes.tobytes(), dtype=np.uint16)
            data = convert_bf16_to_fp32(data_bf16)
            logger.warning(f"  警告: 未找到 CSV 文件，假设为 BF16 格式")
            return data, 8
        else:
            return np.fromfile(filename, dtype=np.float32), 7

    if dtype not in DTYPE_MAP:
        logger.warning(f"  警告: 未知的数据类型 {dtype}, 尝试作为 FP32 读取")
        return np.fromfile(filename, dtype=np.float32), dtype

    type_name, np_dtype, bytes_per_element = DTYPE_MAP[dtype]

    if type_name == 'bf16':
        data_bf16 = np.frombuffer(data_bytes.tobytes(), dtype=np.uint16)
        data = convert_bf16_to_fp32(data_bf16)
        return data, dtype
    elif type_name == 'fp16':
        data_fp16 = np.frombuffer(data_bytes.tobytes(), dtype=np.float16)
        return data_fp16.astype(np.float32), dtype
    elif type_name == 'fp32':
        return np.fromfile(filename, dtype=np.float32), dtype
    elif type_name in ['int32', 'int64']:
        data_int = np.fromfile(filename, dtype=np_dtype)
        return data_int.astype(np.float32), dtype
    elif type_name in ['int8', 'int16']:
        data_int = np.fromfile(filename, dtype=np_dtype)
        return data_int.astype(np.float32), dtype
    else:
        logger.warning(f"  警告: 数据类型 {type_name} (dtype={dtype}) 暂不支持，尝试作为 FP32 读取")
        return np.fromfile(filename, dtype=np.float32), dtype


def read_golden_data(filename, dtype):
    """读取 golden 数据文件"""
    if dtype not in DTYPE_MAP:
        golden_data = np.fromfile(filename, dtype=np.float32)
    else:
        _, np_dtype, _ = DTYPE_MAP[dtype]
        golden_data = np.fromfile(filename, dtype=np_dtype)

    if dtype == 8:
        golden_data = convert_bf16_to_fp32(golden_data)

    return golden_data


def compare_with_golden(jit_data, golden_data, name, dtype=None, verbose=True, custom_rtol=None, custom_atol=None):
    """对比 jit 结果与 golden 结果"""
    min_size = min(jit_data.shape[0], golden_data.shape[0])
    jit_data_to_compare = jit_data[:min_size]
    golden_data_to_compare = golden_data[:min_size]

    if dtype is not None:
        rtol, atol = get_tolerance_by_dtype(dtype, custom_rtol, custom_atol)
    else:
        rtol, atol = 1e-3, 1e-3

    if verbose:
        logger.info(f"  对比范围: 前 {min_size} 个元素 (jit={jit_data.shape[0]}, golden={golden_data.shape[0]})")

    close_mask = np.isclose(jit_data_to_compare, golden_data_to_compare, rtol=rtol, atol=atol)
    mismatch_count = (~close_mask).sum()
    total_count = min_size

    diff = np.abs(jit_data_to_compare - golden_data_to_compare)
    max_diff = np.max(diff)
    max_val = np.max(np.abs(golden_data_to_compare))
    relative_error = max_diff / (max_val + 1e-10)

    actual_rtol = relative_error
    actual_atol = max_diff

    threshold = total_count * max(rtol, atol)
    match = mismatch_count < threshold

    status = "✓ PASS" if match else "✗ FAIL"
    logger.info(f"\n{name}: {status}")
    logger.info(f"  Max diff: {max_diff:.6f}")
    logger.info(f"  Max val: {max_val:.6f}")
    logger.info(f"  Relative error: {relative_error:.6f}")
    logger.info(f"  Mismatch count: {mismatch_count}/{total_count} ({mismatch_count/total_count*100:.2f}%)")
    logger.info(f"  Tolerance: rtol={rtol}, atol={atol} (dtype={dtype})")
    logger.info(f"  Actual: rtol={actual_rtol:.6f}, atol={actual_atol:.6f}")

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


def setup_logging(work_dir, golden_dir):
    """配置日志输出"""
    log_formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    operator_dir = os.path.dirname(os.path.abspath(golden_dir)) if golden_dir else os.path.abspath(work_dir)
    operator_name = os.path.basename(operator_dir)
    log_file = os.path.join(operator_dir, f'{operator_name}_verify_result.log')

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)

    logger.handlers = [console_handler, file_handler]
    return operator_name


def main():
    parser = argparse.ArgumentParser(description='PyPTO 精度检查点文件对比工具')
    parser.add_argument('--work-dir', '-w', default='.', help='工作目录（默认为当前目录）')
    parser.add_argument('--output-dir', '-o', default=None, help='指定 output 目录名（不指定则自动检测最新的）')
    parser.add_argument('--golden-dir', '-g', default=None, help='指定 golden 文件所在目录（默认从工作目录查找）')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细的元素级对比')
    parser.add_argument('--list', '-l', action='store_true', help='只列出检查点，不进行对比')
    parser.add_argument('--rtol', type=float, default=None, help='自定义相对容差（用于量化场景，推荐值：0.0078125）')
    parser.add_argument('--atol', type=float, default=None, help='自定义绝对容差（用于量化场景，推荐值：0.0001）')

    args = parser.parse_args()

    operator_name = setup_logging(args.work_dir, args.golden_dir)

    logger.info("=" * 80)
    logger.info("PyPTO 精度检查点文件对比工具")
    logger.info("=" * 80)

    latest_dir = args.output_dir if args.output_dir else find_latest_output_dir(args.work_dir)

    if not latest_dir:
        logger.error("✗ 未找到 output 目录")
        sys.exit(1)

    golden_base_dir = args.golden_dir if args.golden_dir else args.work_dir
    tensor_dir = os.path.join(args.work_dir, "output", latest_dir, "tensor")
    checkpoints = scan_checkpoints_from_dir(tensor_dir)

    if not checkpoints:
        logger.error(f"✗ 未在 {tensor_dir} 找到检查点文件")
        sys.exit(1)

    logger.info(f"✓ 找到 output 目录: {latest_dir}")
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
        jit_pattern = os.path.join(tensor_dir, f"{checkpoint_name}_*.data")
        jit_files = sorted(glob.glob(jit_pattern))

        if not jit_files:
            logger.warning(f"\n{checkpoint_name}: ✗ 未找到 jit 文件")
            results.append((checkpoint_name, False))
            continue

        golden_pattern = os.path.join(golden_base_dir, f"golden_{checkpoint_name}.bin")
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

        jit_data, dtype = read_jit_data(jit_file)
        golden_data = read_golden_data(golden_file, dtype)

        match = compare_with_golden(jit_data, golden_data, checkpoint_name,
                                   dtype=dtype, verbose=args.verbose,
                                   custom_rtol=args.rtol, custom_atol=args.atol)
        results.append((checkpoint_name, match))

    analyze_results(results)

    all_match = all(match for _, match in results)
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
