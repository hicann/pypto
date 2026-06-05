# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from __future__ import annotations

import ast
import re

from ..ast_helpers import _has_test_level_markers
from ..core import CheckContext, Finding, register
from ..utils import _check_npu_available


@register("OL19")
def check_ol19(ctx: CheckContext) -> Finding:
    """test 必须使用 assert_allclose 或 detailed_tensor_compare 做精度比对。"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL19", "SKIP", f"{test_file} 不存在")
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL19", "SKIP", f"{test_file} 无法解析")
    # 接受任一标准比对工具: numpy assert_allclose 或 verifier 的 detailed_tensor_compare
    _compare_helpers = ("assert_allclose", "detailed_tensor_compare")
    has_compare = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_str = ast.dump(node.func)
            if any(h in call_str for h in _compare_helpers):
                has_compare = True
                break
    if not has_compare:
        return ctx.make_finding("OL19", "FAIL",
            "未找到 assert_allclose / detailed_tensor_compare 调用，禁止手写 assert max_diff",
            file=test_file)
    return ctx.make_finding("OL19", "PASS",
        "使用了 assert_allclose / detailed_tensor_compare", file=test_file)


@register("OL20")
def check_ol20(ctx: CheckContext) -> Finding:
    """test 必须处理 TILE_FWK_DEVICE_ID 并调用 set_device"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL20", "SKIP", f"{test_file} 不存在")
    has_device_id = "TILE_FWK_DEVICE_ID" in source
    has_set_device = "set_device" in source
    if has_device_id and has_set_device:
        return ctx.make_finding("OL20", "PASS",
            "找到 TILE_FWK_DEVICE_ID 处理和 set_device 调用", file=test_file)
    missing = []
    if not has_device_id:
        missing.append("TILE_FWK_DEVICE_ID 环境变量处理")
    if not has_set_device:
        missing.append("set_device 调用")
    return ctx.make_finding("OL20", "FAIL",
        f"缺少: {', '.join(missing)}", file=test_file)


@register("OL21")
def check_ol21(ctx: CheckContext) -> Finding:
    """test 必须有 Level 0 和 Level 1 两级测试函数"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL21", "SKIP", f"{test_file} 不存在")
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL21", "SKIP", f"{test_file} 不存在或无法解析")
    has_level0, has_level1 = _has_test_level_markers(tree, source)
    missing = []
    if not has_level0:
        missing.append("level0")
    if not has_level1:
        missing.append("level1")
    if missing:
        return ctx.make_finding("OL21", "FAIL",
            f"缺少测试级别: {', '.join(missing)}", file=test_file)
    return ctx.make_finding("OL21", "PASS",
        "包含 Level 0 和 Level 1 测试", file=test_file)


@register("OL22")
def check_ol22(ctx: CheckContext) -> Finding:
    """test 应设置 torch.manual_seed 保证可复现"""
    test_file = f"test_{ctx.op_name}.py"
    source = ctx.read_file(test_file)
    if not source:
        return ctx.make_finding("OL22", "SKIP", f"{test_file} 不存在")
    if "manual_seed" in source:
        return ctx.make_finding("OL22", "PASS",
            "找到 manual_seed 设置", file=test_file)
    return ctx.make_finding("OL22", "WARN",
        "未设置 torch.manual_seed，建议显式设置以保证测试可复现",
        file=test_file)


@register("OL42")
def check_ol42(ctx: CheckContext) -> Finding:
    """NPU 环境下 test 不得硬编码 sim 模式"""
    test_file = f"test_{ctx.op_name}.py"
    content = ctx.read_file(test_file)
    if not content:
        return ctx.make_finding("OL42", "SKIP", f"{test_file} 不存在")

    # 检测是否有 NPU 环境
    if not _check_npu_available():
        return ctx.make_finding("OL42", "SKIP",
            "未检测到 NPU 环境（npu-smi 不可用），跳过 sim 模式检查")

    # 在有 NPU 的环境下，检查是否硬编码了 sim 模式
    problems = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        # 跳过注释行
        if stripped.startswith("#"):
            continue
        # 检查 default="sim" 或 default='sim' (argparse default)
        if re.search(r"""default\s*=\s*['"]sim['"]""", line):
            problems.append(f"L{i}: argparse default 设置为 sim")
        # 检查 run_mode="sim" 或 run_mode='sim' 的硬编码赋值
        elif re.search(r"""run_mode\s*=\s*['"]sim['"]""", line):
            problems.append(f"L{i}: run_mode 硬编码为 sim")

    if problems:
        return ctx.make_finding("OL42", "FAIL",
            f"NPU 环境下不应使用 sim 模式: {'; '.join(problems)}",
            file=f"test_{ctx.op_name}.py")
    return ctx.make_finding("OL42", "PASS",
        "未发现 sim 模式硬编码", file=f"test_{ctx.op_name}.py")
