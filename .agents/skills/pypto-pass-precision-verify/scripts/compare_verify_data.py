#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Compare Verify Data Tool
对比pypto上板输出与精度工具保存的Pass数据

用途：
- 定位Pass精度问题时，对比前端输出与精度工具保存的数据
- 确认Pass输出是否与上板数据一致
- 二分定位问题Pass

支持的格式：
- .pt文件：torch.save保存的格式（支持多个tensor）
- .data文件：二进制流格式（numpy tofile，单个tensor，与精度工具统一）

示例用法：
    # 对比 .pt 格式输出
    python3 tools/verifier/compare_verify_data.py \
        --pypto-output pypto_output.pt \
        --verify-data ./output/output_xxx/verify_xxx/tensor.data
    
    # 对比 .data 格式输出（推荐，与精度工具统一）
    python3 tools/verifier/compare_verify_data.py \
        --pypto-output pypto_output.data \
        --verify-data ./output/output_xxx/verify_xxx/tensor.data

前端保存示例：
    # 在Python代码中保存tensor为.data文件（推荐）
    output_tensor.numpy().tofile("output.data")
"""

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_tensor_data(file_path: str, dtype: str = "auto") -> dict:
    """
    加载tensor数据文件（支持.pt和.data格式）
    
    Args:
        file_path: 数据文件路径（.pt或.data）
        dtype: 数据类型，"auto"自动推断，或指定"float16"/"float32"/"int32"等
    
    Returns:
        dict: {tensor_name: numpy_array}
    
    说明:
        - .pt文件：torch.save保存的格式，支持多个tensor
        - .data文件：二进制流格式（numpy tofile），单个tensor
    """
    file_ext = Path(file_path).suffix.lower()
    
    # .pt 文件格式
    if file_ext == ".pt":
        data = torch.load(file_path)
        
        # 处理不同的保存格式
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.numpy()
                elif isinstance(value, np.ndarray):
                    result[key] = value
            return result
        elif isinstance(data, torch.Tensor):
            return {"output": data.numpy()}
        elif isinstance(data, np.ndarray):
            return {"output": data}
        else:
            raise ValueError(f"不支持的.pt文件格式: {type(data)}")
    
    # .data 文件格式（二进制流）
    elif file_ext == ".data":
        # 使用 load_verify_data 的逻辑加载
        data = _load_binary_data(file_path, dtype)
        return {"output": data}
    
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .pt 和 .data")


def _load_binary_data(data_file: str, dtype: str = "auto") -> np.ndarray:
    """
    加载二进制流格式的.data文件
    
    Args:
        data_file: .data文件路径
        dtype: 数据类型，"auto"自动推断
    
    Returns:
        np.ndarray: tensor数据
    """
    file_size = os.path.getsize(data_file)
    
    # 如果指定了dtype，直接使用
    if dtype != "auto":
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
            "int32": np.int32,
            "int16": np.int16,
            "int8": np.int8,
        }
        target_dtype = dtype_map.get(dtype)
        if target_dtype:
            return np.fromfile(data_file, dtype=target_dtype).astype(np.float32)
    
    # 自动推断dtype
    dtype_candidates = []
    
    if file_size % 4 == 0:
        dtype_candidates.extend([np.float32, np.int32])
    if file_size % 2 == 0:
        dtype_candidates.extend([np.float16, np.int16])
    dtype_candidates.append(np.int8)
    
    for dtype in dtype_candidates:
        try:
            data = np.fromfile(data_file, dtype=dtype)
            
            if dtype in [np.float16, np.float32]:
                nan_count = np.sum(np.isnan(data))
                inf_count = np.sum(np.isinf(data))
                total = data.size
                
                if (nan_count + inf_count) / total > 0.5:
                    continue
                
                valid_data = data[~np.isnan(data) & ~np.isinf(data)]
                if len(valid_data) > 0:
                    if dtype == np.float16:
                        if np.any(np.abs(valid_data) > 65504):
                            continue
                    
                    return data.astype(np.float32) if dtype != np.float32 else data
            
            elif dtype in [np.int32, np.int16, np.int8]:
                if np.any(data != 0):
                    return data.astype(np.float32)
                    
        except (ValueError, IOError, OSError) as e:
            logger.debug("跳过 dtype %s: %s", dtype, e)
            continue
    
    # 默认使用float32
    return np.fromfile(data_file, dtype=np.float32)


def load_verify_data(data_file: str) -> np.ndarray:
    """
    加载精度工具保存的.data文件（二进制流格式）
    
    Args:
        data_file: .data文件路径
    
    Returns:
        np.ndarray: tensor数据
    """
    return _load_binary_data(data_file, dtype="auto")


def compare_tensors(pypto_data: np.ndarray, verify_data: np.ndarray, 
                    tol: float = 1e-5, rtol: float = 1e-3) -> dict:
    """
    对比两个tensor数据
    
    Args:
        pypto_data: pypto输出的numpy数组
        verify_data: 精度工具保存的numpy数组
        tol: 绝对误差容忍度
        rtol: 相对误差容忍度
    
    Returns:
        dict: 对比结果
    """
    # 确保shape一致
    if pypto_data.shape != verify_data.shape:
        # 尝试reshape
        if pypto_data.size == verify_data.size:
            verify_data = verify_data.reshape(pypto_data.shape)
        else:
            return {
                "match": False,
                "reason": f"shape不一致: pypto={pypto_data.shape}, verify={verify_data.shape}",
                "max_diff": None,
                "mean_diff": None
            }
    
    # 计算差异
    diff = np.abs(pypto_data - verify_data)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # 判断是否一致
    is_close = np.allclose(pypto_data, verify_data, atol=tol, rtol=rtol)
    
    return {
        "match": is_close,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "diff_ratio": float(max_diff / (np.max(np.abs(pypto_data)) + 1e-10))
    }


def find_verify_files(verify_path: str, pass_name: Optional[str] = None) -> list:
    """
    在验证目录中查找.data文件（支持子目录）
    
    Args:
        verify_path: 验证数据目录路径
        pass_name: 可选的Pass名称过滤
    
    Returns:
        list: .data文件路径列表
    """
    verify_dir = Path(verify_path)
    
    if not verify_dir.exists():
        raise ValueError(f"验证目录不存在: {verify_path}")
    
    # 递归查找所有.data文件
    files = list(verify_dir.rglob("*.data"))
    
    # 如果指定Pass名称，过滤
    if pass_name:
        files = [f for f in files if pass_name in str(f) or pass_name in f.name]
    
    return sorted(files)


def parse_tensor_name(filename: str) -> dict:
    """
    解析.data文件名
    
    支持格式:
    1. tensor~TENSOR_xxx~PassName~0@0@0~0~xxxx.data
    2. 4~~0~5~10000~ADD~24~25~1776860082047310.data (新格式: tensor_id~~?~?~op_id~OP_NAME~...~timestamp)
    3. tensor_Incast_19.data (中间tensor格式)
    
    Returns:
        dict: {kernel, pass_name, tensor_id, hash, op_name}
    """
    name = Path(filename).stem  # 去掉.data后缀
    
    # 格式2: 数字开头的新格式
    if name[0].isdigit() and "~~" in name:
        parts = name.split("~")
        # 格式: 4~~0~5~10000~ADD~24~25~timestamp
        # parts[0] = "4", parts[1] = "", parts[2] = "0", parts[3] = "5", parts[4] = "10000", parts[5] = "ADD", ...
        if len(parts) >= 6:
            return {
                "tensor_id": parts[0] + "~~" + parts[2] + "~" + parts[3] if len(parts) > 3 else parts[0],
                "op_id": parts[4] if len(parts) > 4 else "",
                "op_name": parts[5] if len(parts) > 5 else "",
                "pass_name": parts[5] if len(parts) > 5 else "",  # op_name as pass_name
                "hash": parts[-1] if len(parts) > 5 else ""
            }
    
    # 格式1: tensor~ 开头的旧格式
    if name.startswith("tensor~"):
        parts = name.split("~")
        if len(parts) >= 4:
            return {
                "kernel": parts[1] if len(parts) > 1 else "",
                "pass_name": parts[2] if len(parts) > 2 else "",
                "tensor_id": parts[3] if len(parts) > 3 else "",
                "hash": parts[-1] if len(parts) > 4 else ""
            }
    
    # 格式3: tensor_Incast_19 格式
    if name.startswith("tensor_"):
        parts = name.split("_")
        if len(parts) >= 2:
            return {
                "kernel": parts[1] if len(parts) > 1 else "",
                "pass_name": parts[1] if len(parts) > 1 else "",
                "tensor_id": parts[2] if len(parts) > 2 else ""
            }
    
    return {"raw_name": name}


def main():
    parser = argparse.ArgumentParser(
        description="对比pypto上板输出与精度工具保存的Pass数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对比单个文件（.pt格式）
  python3 compare_verify_data.py \
      --pypto-output output.pt \
      --verify-data ./verify_xxx/tensor~TENSOR_xxx~SplitK~xxx.data

  # 对比单个文件（.data格式，二进制流）
  python3 compare_verify_data.py \
      --pypto-output output.data \
      --verify-data ./verify_xxx/tensor.data

  # 对比单个文件（指定dtype）
  python3 compare_verify_data.py \
      --pypto-output output.data \
      --dtype float32 \
      --verify-data ./verify_xxx/tensor.data

  # 对比指定Pass的所有数据
  python3 compare_verify_data.py \
      --pypto-output output.data \
      --verify-path ./verify_xxx \
      --pass-name SplitK

  # 列出验证目录中的所有数据文件
  python3 compare_verify_data.py \
      --verify-path ./verify_xxx \
      --list-files
        
前端保存示例（使用二进制流格式，与精度工具统一）:
  # Python代码中保存tensor为.data文件
  output_tensor.numpy().tofile("output.data")
        
  # 加载.data文件
  data = np.fromfile("output.data", dtype=np.float32)
        """
    )
    
    parser.add_argument("--pypto-output", help="pypto保存的数据文件路径（支持.pt和.data格式）")
    parser.add_argument("--dtype", default="auto", 
                        choices=["auto", "float16", "float32", "int32", "int16", "int8"],
                        help="数据类型（仅对.data文件有效，默认auto自动推断）")
    parser.add_argument("--verify-data", help="精度工具保存的.data文件路径")
    parser.add_argument("--verify-path", help="验证数据目录路径")
    parser.add_argument("--pass-name", help="过滤指定Pass名称的数据文件")
    parser.add_argument("--list-files", action="store_true", help="仅列出验证目录中的数据文件")
    parser.add_argument("--tol", type=float, default=1e-5, help="绝对误差容忍度（默认1e-5）")
    parser.add_argument("--rtol", type=float, default=1e-3, help="相对误差容忍度（默认1e-3）")
    
    args = parser.parse_args()
    
    # 列出文件模式
    if args.list_files and args.verify_path:
        files = find_verify_files(args.verify_path, args.pass_name)
        logger.info(f"=== 验证目录: {args.verify_path} ===")
        logger.info(f"找到 {len(files)} 个.data文件:")
        for f in files:
            info = parse_tensor_name(f.name)
            logger.info(f"  {f.name}")
            logger.info(f"    Pass: {info.get('pass_name', 'N/A')}")
            logger.info(f"    Kernel: {info.get('kernel', 'N/A')}")
        return 0
    
    # 必须提供对比数据
    if not args.pypto_output:
        logger.error("错误: 必须提供 --pypto-output 参数")
        return 1
    
    if not args.verify_data and not args.verify_path:
        logger.error("错误: 必须提供 --verify-data 或 --verify-path 参数")
        return 1
    
    logger.info(f"=== 加载pypto输出: {args.pypto_output} ===")
    pypto_data = load_tensor_data(args.pypto_output, args.dtype)
    logger.info(f"  包含 {len(pypto_data)} 个tensor:")
    for name, arr in pypto_data.items():
        logger.info(f"    {name}: shape={arr.shape}, dtype={arr.dtype}")
    
    # 获取要对比的文件列表
    if args.verify_data:
        verify_files = [args.verify_data]
    else:
        verify_files = find_verify_files(args.verify_path, args.pass_name)
    
    logger.info(f"\n=== 对比验证数据 ===")
    logger.info(f"对比文件数: {len(verify_files)}")
    
    results = []
    for verify_file in verify_files:
        logger.info(f"\n--- {Path(verify_file).name} ---")
        
        info = parse_tensor_name(verify_file)
        logger.info(f"  Pass: {info.get('pass_name', 'N/A')}")
        
        try:
            verify_data = load_verify_data(verify_file)
            logger.info(f"  Shape: {verify_data.shape}, Size: {verify_data.size}")
            
            pypto_tensor = None
            for name, arr in pypto_data.items():
                if arr.size == verify_data.size:
                    pypto_tensor = arr
                    logger.info(f"  匹配pypto tensor: {name}")
                    break
            
            if pypto_tensor is None:
                logger.warning(f"  [警告] 未找到size匹配的pypto tensor")
                results.append({
                    "file": verify_file,
                    "match": False,
                    "reason": "size不匹配"
                })
                continue
            
            cmp_result = compare_tensors(pypto_tensor, verify_data, args.tol, args.rtol)
            
            logger.info(f"  结果: {'✓ 一致' if cmp_result['match'] else '× 不一致'}")
            if not cmp_result['match']:
                logger.info(f"    最大差异: {cmp_result['max_diff']:.6e}")
                logger.info(f"    平均差异: {cmp_result['mean_diff']:.6e}")
                logger.info(f"    差异比例: {cmp_result['diff_ratio']:.6e}")
            
            results.append({
                "file": verify_file,
                "pass": info.get('pass_name', ''),
                **cmp_result
            })
            
        except Exception as e:
            logger.error(f"  [错误] {e}")
            results.append({
                "file": verify_file,
                "match": False,
                "reason": str(e)
            })
    
    # 输出汇总
    logger.info(f"\n=== 对比汇总 ===")
    matched_count = sum(1 for r in results if r.get('match', False))
    logger.info(f"一致文件: {matched_count}/{len(results)}")
    
    mismatched_passes = [r for r in results if not r.get('match', False) and r.get('pass')]
    if mismatched_passes:
        logger.info(f"\n不一致的Pass:")
        for r in mismatched_passes:
            logger.info(f"  - {r['pass']}: {r['file']}")
            if r.get('max_diff'):
                logger.info(f"    最大差异: {r['max_diff']:.6e}")
    
    if matched_count == len(results):
        logger.info("\n结论: 所有Pass输出与上板数据一致 ✓")
    else:
        logger.info(f"\n结论: 存在 {len(mismatched_passes)} 个Pass输出与上板数据不一致 ×")
        if mismatched_passes:
            logger.info("建议: 问题在这些不一致的Pass中")
    
    return 0


if __name__ == "__main__":
    exit(main())