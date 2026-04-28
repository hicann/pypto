# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from __future__ import annotations

import ast

from ..ast_helpers import (
    _extract_symbolic_dynamic_aliases,
    _get_jit_functions,
    _get_primary_jit_functions,
    _has_loop_structure,
    _is_fp32_only_call,
    _is_non_tensor_annotation,
    _is_pypto_tensor_annotation,
    _shape_has_dynamic,
)
from ..core import CheckContext, Finding, register
from ..utils import _syntax_error_finding


@register("OL01")
def check_ol01(ctx: CheckContext) -> Finding:
    """kernel 函数必须有 @pypto.frontend.jit 装饰器"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL01", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if jit_funcs:
        func = jit_funcs[0]
        return ctx.make_finding("OL01", "PASS",
            f"找到 @pypto.frontend.jit 装饰的函数: {func.name}",
            file=impl_file, line=func.lineno)
    return ctx.make_finding("OL01", "FAIL",
        f"[S0 致命] {impl_file} 中未找到任何 @pypto.frontend.jit 装饰的函数。"
        f"这是原则性错误——缺少 jit 装饰器的文件不构成有效的 kernel 实现，"
        f"无法编译、无法在 NPU 上执行。"
        f"禁止在此文件上做局部修补或变通处理，"
        f"必须删除当前 {impl_file} 并基于 SPEC.md / DESIGN.md 从零重新生成。",
        file=impl_file)


@register("OL02")
def check_ol02(ctx: CheckContext) -> Finding:
    """输出写回必须用 [:]/move()/assemble()，禁止 out = expr"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL02", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL02", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        param_names = {arg.arg for arg in func.args.args}
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in param_names:
                        return ctx.make_finding("OL02", "FAIL",
                            f"禁止 `{target.id} = expr` 写回，应使用 "
                            f"`{target.id}[:] = ...` 或 `{target.id}.move(...)`",
                            file=impl_file, line=node.lineno)
            if isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id in param_names:
                    return ctx.make_finding("OL02", "FAIL",
                        f"禁止 `{node.target.id} += expr` 写回，应使用 "
                        f"`{node.target.id}[:] = {node.target.id} + ...`",
                        file=impl_file, line=node.lineno)
    return ctx.make_finding("OL02", "PASS", "输出写回方式正确", file=impl_file)


@register("OL03")
def check_ol03(ctx: CheckContext) -> Finding:
    """kernel 函数不能有 return 语句"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL03", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL03", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                return ctx.make_finding("OL03", "FAIL",
                    f"jit 函数 {func.name} 内存在 return 语句",
                    file=impl_file, line=node.lineno)
    return ctx.make_finding("OL03", "PASS", "jit 函数无 return 语句", file=impl_file)


@register("OL04")
def check_ol04(ctx: CheckContext) -> Finding:
    """必须调用 set_vec_tile_shapes 或 set_cube_tile_shapes"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL04", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    for func in _get_jit_functions(tree, aliases):
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                call_str = ast.dump(node.func)
                if "set_vec_tile_shapes" in call_str or "set_cube_tile_shapes" in call_str:
                    return ctx.make_finding("OL04", "PASS",
                        "找到 tile shapes 配置调用",
                        file=impl_file, line=node.lineno)
    return ctx.make_finding("OL04", "FAIL",
        "jit 函数体内未找到 set_vec_tile_shapes 或 set_cube_tile_shapes 调用"
        "（注意：必须在 @jit 装饰的函数内部调用）", file=impl_file)


@register("OL05")
def check_ol05(ctx: CheckContext) -> Finding:
    """kernel 张量参数必须有 pypto.Tensor 类型注解"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL05", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL05", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        for arg in func.args.args:
            if arg.annotation is None:
                return ctx.make_finding("OL05", "FAIL",
                    f"jit 函数参数 `{arg.arg}` 缺少类型注解",
                    file=impl_file, line=func.lineno)
            # 非张量参数（int/float 等标量）跳过 Tensor 注解检查
            if _is_non_tensor_annotation(arg.annotation, aliases):
                continue
            if not _is_pypto_tensor_annotation(arg.annotation, aliases):
                return ctx.make_finding("OL05", "FAIL",
                    f"jit 函数张量参数 `{arg.arg}` 注解必须为 pypto.Tensor",
                    file=impl_file, line=func.lineno)
    return ctx.make_finding("OL05", "PASS",
        "jit 函数张量参数均有 pypto.Tensor 类型注解", file=impl_file)


@register("OL06")
def check_ol06(ctx: CheckContext) -> Finding:
    """kernel 内禁用 Python 原生 min()/max()"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL06", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    for func in _get_jit_functions(tree, aliases):
        for node in ast.walk(func):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("min", "max")):
                return ctx.make_finding("OL06", "FAIL",
                    f"jit 函数内使用了 Python 原生 {node.func.id}()，"
                    "应使用 pypto 等价函数",
                    file=impl_file, line=node.lineno)
    return ctx.make_finding("OL06", "PASS", "未使用原生 min/max", file=impl_file)


@register("OL07")
def check_ol07(ctx: CheckContext) -> Finding:
    """必须 import pypto"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL07", "SKIP", f"{impl_file} 不存在或无法解析")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pypto" or alias.name.startswith("pypto."):
                    return ctx.make_finding("OL07", "PASS", "找到 import pypto",
                        file=impl_file, line=node.lineno)
        if (isinstance(node, ast.ImportFrom)
                and node.module and node.module.startswith("pypto")):
            return ctx.make_finding("OL07", "PASS", "找到 from pypto import ...",
                file=impl_file, line=node.lineno)
    return ctx.make_finding("OL07", "FAIL",
        f"[S0 致命] {impl_file} 中未 import pypto——"
        "这是 PyPTO 算子实现的基础前提，缺少 import 说明该文件不是合法的 kernel 实现。"
        "禁止任何形式的局部修补或绕过。"
        f"必须删除当前 {impl_file} 并基于 DESIGN.md 重新实现。",
        file=impl_file)


@register("OL08")
def check_ol08(ctx: CheckContext) -> Finding:
    """wrapper 函数必须导出且以 _wrapper 结尾"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL08", "SKIP", f"{impl_file} 不存在或无法解析")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_wrapper"):
            return ctx.make_finding("OL08", "PASS",
                f"找到 wrapper 函数: {node.name}",
                file=impl_file, line=node.lineno)
    return ctx.make_finding("OL08", "FAIL",
        "未找到以 _wrapper 结尾的模块级函数", file=impl_file)


@register("OL23")
def check_ol23(ctx: CheckContext) -> Finding:
    impl_file = f"{ctx.op_name}_impl.py"
    syntax_error = _syntax_error_finding(ctx, "OL23", impl_file)
    if syntax_error:
        return syntax_error
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL23", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_primary_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL23", "SKIP", "无 jit 函数")
    if any(_has_loop_structure(func) for func in jit_funcs):
        return ctx.make_finding("OL23", "PASS",
            "检测到 loop 相关结构", file=impl_file)
    return ctx.make_finding("OL23", "WARN",
        "未检测到 loop 相关结构；若该算子需要分块或迭代，请确认设计已说明无需 loop",
        file=impl_file)


@register("OL25")
def check_ol25(ctx: CheckContext) -> Finding:
    """Tensor 注解完整性检查：无参数或缺少 dtype 时告警。"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL25", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL25", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        for arg in func.args.args:
            ann = arg.annotation
            if ann is None:
                continue
            if _is_non_tensor_annotation(ann, aliases):
                continue
            if not _is_pypto_tensor_annotation(ann, aliases):
                continue
            if not isinstance(ann, ast.Call):
                continue
            if not ann.args:
                return ctx.make_finding("OL25", "FAIL",
                    f"参数 `{arg.arg}` 使用 pypto.Tensor()（无参数形式）；"
                    "动态轴必须显式标 pypto.DYNAMIC，静态轴写常量整数，"
                    "禁止使用空注解。例：pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)",
                    file=impl_file, line=ann.lineno)
            if len(ann.args) == 1:
                # 第二种空注解形态：pypto.Tensor([], dtype) — 同样禁止。
                shape_arg = ann.args[0]
                if isinstance(shape_arg, ast.List) and not shape_arg.elts:
                    return ctx.make_finding("OL25", "FAIL",
                        f"参数 `{arg.arg}` 使用 pypto.Tensor([], dtype) 空 shape 注解；"
                        "动态轴必须显式标 pypto.DYNAMIC，静态轴写常量整数。",
                        file=impl_file, line=ann.lineno)
                return ctx.make_finding("OL25", "WARN",
                    f"参数 `{arg.arg}` 只声明了 shape，缺少 dtype。"
                    "建议写成 pypto.Tensor([shape], dtype)",
                    file=impl_file, line=ann.lineno)
    return ctx.make_finding("OL25", "PASS",
        "JIT Tensor 注解均包含 shape 与 dtype", file=impl_file)


@register("OL26")
def check_ol26(ctx: CheckContext) -> Finding:
    """JIT 函数中张量参数必须在非张量参数之前"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL26", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL26", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        seen_non_tensor = False
        for arg in func.args.args:
            ann = arg.annotation
            if ann is None:
                continue
            if _is_non_tensor_annotation(ann, aliases):
                seen_non_tensor = True
            elif _is_pypto_tensor_annotation(ann, aliases):
                if seen_non_tensor:
                    return ctx.make_finding("OL26", "FAIL",
                        f"jit 函数 {func.name} 中张量参数 `{arg.arg}` 出现在非张量参数之后，"
                        "JIT 要求张量参数在前、非张量参数在后",
                        file=impl_file, line=func.lineno)
    return ctx.make_finding("OL26", "PASS",
        "jit 函数参数顺序正确（张量在前、标量在后）", file=impl_file)


@register("OL28")
def check_ol28(ctx: CheckContext) -> Finding:
    """sigmoid/softmax/sin/cos 仅支持 DT_FP32，非 FP32 dtype 时警告"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL28", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL28", "SKIP", "无 jit 函数")
    for func in jit_funcs:
        used_fp32_only = set()
        for node in ast.walk(func):
            if _is_fp32_only_call(node, aliases):
                used_fp32_only.add(node.func.attr)
        if not used_fp32_only:
            continue
        for arg in func.args.args:
            ann = arg.annotation
            if not isinstance(ann, ast.Call) or not _is_pypto_tensor_annotation(ann, aliases):
                continue
            if len(ann.args) >= 2:
                dtype_str = ast.dump(ann.args[1])
                if "DT_FP32" not in dtype_str:
                    return ctx.make_finding("OL28", "WARN",
                        f"jit 函数使用了仅支持 DT_FP32 的 API "
                        f"({', '.join(sorted(used_fp32_only))})，"
                        f"但参数 `{arg.arg}` 的 dtype 不是 DT_FP32，"
                        "请确认已正确处理 dtype 转换（cast）",
                        file=impl_file, line=func.lineno)
    return ctx.make_finding("OL28", "PASS",
        "FP32-only API 与 dtype 注解一致", file=impl_file)


@register("OL29")
def check_ol29(ctx: CheckContext) -> Finding:
    """Tensor 注解的 shape 中应声明 pypto.DYNAMIC/pypto.DYN 维度"""
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL29", "SKIP", f"{impl_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_primary_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL29", "SKIP", "无 jit 函数")
    tensor_count = 0
    has_dynamic = False
    dynamic_aliases = _extract_symbolic_dynamic_aliases(tree, aliases)
    for func in jit_funcs:
        for arg in func.args.args:
            ann = arg.annotation
            if not isinstance(ann, ast.Call) or not _is_pypto_tensor_annotation(ann, aliases):
                continue
            tensor_count += 1
            if ann.args:
                if _shape_has_dynamic(ann.args[0], dynamic_aliases, aliases):
                    has_dynamic = True
    if tensor_count == 0:
        return ctx.make_finding("OL29", "SKIP", "无 Tensor 注解")
    if has_dynamic:
        return ctx.make_finding("OL29", "PASS",
            "Tensor 注解中包含 DYNAMIC 维度声明", file=impl_file)
    return ctx.make_finding("OL29", "WARN",
        "所有 Tensor 注解的 shape 中均未声明 pypto.DYNAMIC/pypto.DYN，"
        "若有输入维度在运行时可变，必须标记为 DYNAMIC 以避免重编译",
        file=impl_file)
