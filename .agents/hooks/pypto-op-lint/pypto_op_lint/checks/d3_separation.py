# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from __future__ import annotations

import ast

from ..ast_helpers import _get_jit_functions
from ..core import CheckContext, Finding, register


@register("OL15")
def check_ol15(ctx: CheckContext) -> Finding:
    """golden 文件禁止 import pypto"""
    golden_file = f"{ctx.op_name}_golden.py"
    tree = ctx.parse_file(golden_file)
    if tree is None:
        return ctx.make_finding("OL15", "SKIP", f"{golden_file} 不存在或无法解析")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pypto" or alias.name.startswith("pypto."):
                    return ctx.make_finding("OL15", "FAIL",
                        "golden 文件禁止 import pypto",
                        file=golden_file, line=node.lineno)
        if (isinstance(node, ast.ImportFrom)
                and node.module and node.module.startswith("pypto")):
            return ctx.make_finding("OL15", "FAIL",
                "golden 文件禁止 from pypto import ...",
                file=golden_file, line=node.lineno)
    return ctx.make_finding("OL15", "PASS",
        "golden 文件未导入 pypto", file=golden_file)


@register("OL16")
def check_ol16(ctx: CheckContext) -> Finding:
    """impl 文件不应导入 golden 模块"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL16", "SKIP", f"{impl_file} 不存在或无法解析")
    golden_module = f"{ctx.op_name}_golden"
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module == golden_module:
            return ctx.make_finding("OL16", "FAIL",
                f"impl 文件不应导入 {golden_module}",
                file=impl_file, line=node.lineno)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == golden_module:
                    return ctx.make_finding("OL16", "FAIL",
                        f"impl 文件不应导入 {golden_module}",
                        file=impl_file, line=node.lineno)
    return ctx.make_finding("OL16", "PASS",
        "impl 文件未导入 golden", file=impl_file)


@register("OL17")
def check_ol17(ctx: CheckContext) -> Finding:
    """test 文件不应包含 kernel 实现代码"""
    test_file = f"test_{ctx.op_name}.py"
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL17", "SKIP", f"{test_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(test_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if jit_funcs:
        func = jit_funcs[0]
        return ctx.make_finding("OL17", "FAIL",
            f"test 文件包含 @pypto.frontend.jit 装饰的函数: {func.name}",
            file=test_file, line=func.lineno)
    return ctx.make_finding("OL17", "PASS",
        "test 文件未包含 kernel 实现", file=test_file)


@register("OL18")
def check_ol18(ctx: CheckContext) -> Finding:
    """test 文件必须从 impl 和 golden 分别导入"""
    test_file = f"test_{ctx.op_name}.py"
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL18", "SKIP", f"{test_file} 不存在或无法解析")
    impl_module = f"{ctx.op_name}_impl"
    golden_module = f"{ctx.op_name}_golden"
    has_impl = False
    has_golden = False

    def _matches_module(module_name: str | None, expected: str) -> bool:
        if module_name is None:
            return False
        return module_name == expected or module_name.endswith(f".{expected}")

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _matches_module(node.module, impl_module):
                has_impl = True
            if _matches_module(node.module, golden_module):
                has_golden = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _matches_module(alias.name, impl_module):
                    has_impl = True
                if _matches_module(alias.name, golden_module):
                    has_golden = True
    missing = []
    if not has_impl:
        missing.append(impl_module)
    if not has_golden:
        missing.append(golden_module)
    if missing:
        return ctx.make_finding("OL18", "FAIL",
            f"test 文件缺少导入: {', '.join(missing)}", file=test_file)
    return ctx.make_finding("OL18", "PASS",
        "test 文件正确导入了 impl 和 golden", file=test_file)
