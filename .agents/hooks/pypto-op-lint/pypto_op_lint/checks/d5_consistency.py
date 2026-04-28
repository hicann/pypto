# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from __future__ import annotations

import ast
import os
import re

from ..ast_helpers import (
    _extract_jit_assigned_names,
    _extract_shapes_from_test_ast,
    _extract_symbolic_dynamic_aliases,
    _get_func_param_count,
    _get_jit_functions,
    _get_primary_jit_functions,
    _has_loop_structure,
    _is_pypto_tensor_annotation,
    _shape_has_dynamic,
)
from ..core import API_REPORT_FILE, DESIGN_FILE, SPEC_FILE, STRICT_ENV, CheckContext, Finding, register
from ..utils import (
    _extract_design_identifiers,
    _extract_shapes_from_text,
    _extract_spec_dtypes_from_meta,
    _extract_test_dtypes,
    _extract_tolerance,
    _load_doc_meta,
    _load_tolerance_schema,
    _parse_front_matter,
    _syntax_error_finding,
    _validate_doc_schema,
)


@register("OL30")
def check_ol30(ctx: CheckContext) -> Finding:
    """SPEC.md front matter 的 supported_dtypes 必须在测试文件中覆盖。"""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL30", "SKIP", f"{SPEC_FILE} 不存在")
    spec_meta = _load_doc_meta(ctx, SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL30", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    test_file = f"test_{ctx.op_name}.py"
    test_source = ctx.read_file(test_file)
    if not test_source:
        return ctx.make_finding("OL30", "SKIP", f"{test_file} 不存在")

    spec_dtypes = _extract_spec_dtypes_from_meta(spec_meta)
    if not spec_dtypes:
        return ctx.make_finding("OL30", "FAIL",
            f"{SPEC_FILE} front matter 中未声明 supported_dtypes",
            file=SPEC_FILE)

    test_dtypes = _extract_test_dtypes(test_source)
    missing = spec_dtypes - test_dtypes
    if missing:
        canonical_names = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        missing_names = sorted(canonical_names.get(d, d) for d in missing)
        return ctx.make_finding("OL30", "FAIL",
            f"{SPEC_FILE} 声明支持的 dtype ({', '.join(missing_names)}) "
            "在测试文件中未覆盖；请补充对应 dtype 的测试用例",
            file=test_file)
    return ctx.make_finding("OL30", "PASS", "spec dtype 覆盖与 test 一致")


@register("OL31")
def check_ol31(ctx: CheckContext) -> Finding:
    """DESIGN.md front matter dynamic_axes 与 impl 动态注解一致。"""
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL31", "SKIP", f"{DESIGN_FILE} 不存在")
    design_meta = _load_doc_meta(ctx, DESIGN_FILE)
    schema_errors = _validate_doc_schema("DESIGN", design_meta)
    if schema_errors:
        return ctx.make_finding("OL31", "FAIL",
            f"{DESIGN_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=DESIGN_FILE)

    impl_file = f"{ctx.op_name}_impl.py"
    if not ctx.read_file(impl_file):
        return ctx.make_finding("OL31", "SKIP", f"{impl_file} 不存在")
    syntax_error = _syntax_error_finding(ctx, "OL31", impl_file)
    if syntax_error:
        return syntax_error
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL31", "SKIP", f"{impl_file} 无法解析")
    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_primary_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL31", "SKIP", "无 jit 函数")

    dynamic_axes = design_meta.get("dynamic_axes", [])
    if not isinstance(dynamic_axes, list) or not dynamic_axes:
        return ctx.make_finding("OL31", "PASS",
            f"{DESIGN_FILE} front matter 未声明动态轴，无需检查")

    dynamic_aliases = _extract_symbolic_dynamic_aliases(tree, aliases)
    has_dynamic_in_impl = False
    for func in jit_funcs:
        for arg in func.args.args:
            ann = arg.annotation
            if not isinstance(ann, ast.Call) or not _is_pypto_tensor_annotation(ann, aliases):
                continue
            # pypto.Tensor() / pypto.Tensor([]) 空注解由 OL25 统一拦截，此处不重复处理。
            if not ann.args:
                continue
            if _shape_has_dynamic(ann.args[0], dynamic_aliases, aliases):
                has_dynamic_in_impl = True
                break
        if has_dynamic_in_impl:
            break

    if not has_dynamic_in_impl:
        return ctx.make_finding("OL31", "FAIL",
            f"{DESIGN_FILE} front matter 声明了动态轴，但 impl 的 Tensor 注解中未使用 "
            "pypto.DYNAMIC/pypto.DYN，动态轴必须在类型注解中显式标记",
            file=impl_file)
    return ctx.make_finding("OL31", "PASS",
        "design 动态轴声明与 impl 注解一致", file=impl_file)


@register("OL43")
def check_ol43(ctx: CheckContext) -> Finding:
    """DESIGN 声明动态轴时 impl 必须包含 pypto.loop 调用"""
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL43", "SKIP", f"{DESIGN_FILE} 不存在")

    design_meta = _load_doc_meta(ctx, DESIGN_FILE)
    dynamic_axes = design_meta.get("dynamic_axes") or design_meta.get("dynamic_axis")

    # 若 front matter 无声明，在正文中搜索动态轴相关关键词
    if not dynamic_axes:
        content = ctx.read_file(DESIGN_FILE)
        has_dynamic_keyword = bool(
            re.search(r'(?:pypto\.DYNAMIC|pypto\.DYN|动态轴|dynamic.{0,10}axis)', content, re.IGNORECASE)
        )
        if not has_dynamic_keyword:
            return ctx.make_finding("OL43", "SKIP",
                "DESIGN.md 未声明动态轴，无需检查 pypto.loop")

    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL43", "SKIP", f"{impl_file} 不存在或无法解析")

    aliases = ctx.pypto_aliases(impl_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if not jit_funcs:
        return ctx.make_finding("OL43", "SKIP", "未找到 JIT 函数")

    # 在 impl 中搜索 pypto.loop 调用
    impl_source = ctx.read_file(impl_file) or ""
    has_pypto_loop = bool(re.search(r'pypto\.loop\s*\(', impl_source))
    if not has_pypto_loop:
        has_pypto_loop = _has_loop_structure(jit_funcs[0])

    if has_pypto_loop:
        return ctx.make_finding("OL43", "PASS",
            "DESIGN 声明动态轴，impl 中存在 loop 结构", file=f"{ctx.op_name}_impl.py")

    return ctx.make_finding("OL43", "FAIL",
        f"DESIGN.md 声明了动态轴，但 {ctx.op_name}_impl.py 中未找到 "
        f"pypto.loop / pypto.lang.loop 调用。建议补充 pypto.loop 以覆盖动态轴，"
        f"可对现有 impl 做局部修补。",
        file=f"{ctx.op_name}_impl.py")


@register("OL32")
def check_ol32(ctx: CheckContext) -> Finding:
    """SPEC.md front matter tolerance 须与 test 文件一致（支持 tolerance_schema 多模式）。"""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL32", "SKIP", f"{SPEC_FILE} 不存在")
    spec_meta = _load_doc_meta(ctx, SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL32", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    test_file = f"test_{ctx.op_name}.py"
    test_source = ctx.read_file(test_file)
    if not test_source:
        return ctx.make_finding("OL32", "SKIP", f"{test_file} 不存在")

    tolerance_meta = spec_meta.get("tolerance", {})
    if not isinstance(tolerance_meta, dict):
        return ctx.make_finding("OL32", "FAIL",
            f"{SPEC_FILE} front matter 中 tolerance 必须是 dict",
            file=SPEC_FILE)

    # 根据 tolerance_schema.oneOf 确定当前使用的模式及其 required 字段
    schemas = _load_tolerance_schema()
    matched_keys: list[str] = []
    if schemas:
        for schema in schemas:
            required = schema.get("required", [])
            if all(k in tolerance_meta for k in required):
                matched_keys = required
                break
        if not matched_keys:
            modes = [f"{s.get('mode', '?')}({', '.join(s.get('required', []))})" for s in schemas]
            return ctx.make_finding("OL32", "FAIL",
                f"{SPEC_FILE} tolerance 不匹配任何已知模式: {' | '.join(modes)}",
                file=SPEC_FILE)
    else:
        # 无 schema 定义时回退到标准模式
        matched_keys = ["atol", "rtol"]

    mismatches = []
    for key in matched_keys:
        if key not in tolerance_meta:
            continue
        try:
            spec_vals = [float(tolerance_meta[key])]
        except (ValueError, TypeError):
            continue
        test_vals = _extract_tolerance(test_source, key)
        if not spec_vals or not test_vals:
            continue
        spec_strictest = min(spec_vals)
        test_loosest = max(test_vals)
        if test_loosest > spec_strictest * 3:
            mismatches.append(
                f"{key}: spec 最严={spec_strictest}, test 最松={test_loosest}")
    if mismatches:
        return ctx.make_finding("OL32", "WARN",
            f"{SPEC_FILE} 与 test 的精度容差差距较大: {'; '.join(mismatches)}",
            file=test_file)
    return ctx.make_finding("OL32", "PASS", "spec 精度容差与 test 一致")


@register("OL33")
def check_ol33(ctx: CheckContext) -> Finding:
    """golden 函数签名须与 impl wrapper 函数兼容"""
    golden_file = f"{ctx.op_name}_golden.py"
    impl_file = f"{ctx.op_name}_impl.py"
    syntax_error = _syntax_error_finding(ctx, "OL33", impl_file)
    if syntax_error:
        return syntax_error
    golden_tree = ctx.parse_file(golden_file)
    impl_tree = ctx.parse_file(impl_file)
    if golden_tree is None:
        return ctx.make_finding("OL33", "SKIP", f"{golden_file} 不存在或无法解析")
    if impl_tree is None:
        return ctx.make_finding("OL33", "SKIP", f"{impl_file} 不存在或无法解析")
    golden_func = f"{ctx.op_name}_golden"
    wrapper_func = f"{ctx.op_name}_wrapper"
    golden_count = _get_func_param_count(golden_tree, golden_func)
    wrapper_count = _get_func_param_count(impl_tree, wrapper_func)
    if golden_count is None:
        return ctx.make_finding("OL33", "SKIP", f"未找到 {golden_func} 函数")
    if wrapper_count is None:
        return ctx.make_finding("OL33", "SKIP", f"未找到 {wrapper_func} 函数")
    if golden_count != wrapper_count:
        return ctx.make_finding("OL33", "WARN",
            f"{golden_func} 需要 {golden_count} 个必需参数，"
            f"但 {wrapper_func} 需要 {wrapper_count} 个必需参数，接口可能不兼容",
            file=impl_file)
    return ctx.make_finding("OL33", "PASS",
        f"golden ({golden_count} 参数) 与 wrapper ({wrapper_count} 参数) 签名兼容")


@register("OL34")
def check_ol34(ctx: CheckContext) -> Finding:
    """SPEC.md front matter p0_shapes 应在 test 文件中覆盖。"""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL34", "SKIP", f"{SPEC_FILE} 不存在")
    spec_meta = _load_doc_meta(ctx, SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL34", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    test_file = f"test_{ctx.op_name}.py"
    test_source = ctx.read_file(test_file)
    if not test_source:
        return ctx.make_finding("OL34", "SKIP", f"{test_file} 不存在")
    test_tree = ctx.parse_file(test_file)
    if test_tree is None:
        return ctx.make_finding("OL34", "SKIP", f"{test_file} 不存在或无法解析")
    spec_p0_shapes: set[tuple[int, ...]] = set()
    raw_shapes = spec_meta.get("p0_shapes", [])
    if not isinstance(raw_shapes, list):
        return ctx.make_finding("OL34", "FAIL",
            f"{SPEC_FILE} front matter 中 p0_shapes 必须是 list",
            file=SPEC_FILE)
    for item in raw_shapes:
        # Support both list shapes and dict shapes (e.g. {"query": [1,64,128], ...})
        if isinstance(item, dict):
            for v in item.values():
                if isinstance(v, (list, tuple)):
                    try:
                        dims = tuple(int(x) for x in v)
                        if len(dims) >= 2:
                            spec_p0_shapes.add(dims)
                    except (ValueError, TypeError):
                        pass
            continue
        if not isinstance(item, (list, tuple)):
            continue
        # 尝试作为 flat shape 解析（如 [1024, 128]）
        try:
            dims = tuple(int(x) for x in item)
            if len(dims) >= 2:
                spec_p0_shapes.add(dims)
            continue
        except (ValueError, TypeError):
            pass
        # 多输入算子：item 是一组 shape（如 [[1,1,1,64], [1,1,128,64], ...]），
        # 展开子项分别解析
        for sub in item:
            if not isinstance(sub, (list, tuple)):
                continue
            try:
                dims = tuple(int(x) for x in sub)
                if len(dims) >= 2:
                    spec_p0_shapes.add(dims)
            except (ValueError, TypeError):
                continue

    if not spec_p0_shapes:
        return ctx.make_finding("OL34", "FAIL",
            f"{SPEC_FILE} front matter 中未找到有效 p0_shapes",
            file=SPEC_FILE)
    test_shapes = _extract_shapes_from_test_ast(test_tree) | _extract_shapes_from_text(test_source)
    missing = spec_p0_shapes - test_shapes
    if missing:
        missing_str = ", ".join(str(list(s)) for s in sorted(missing))
        return ctx.make_finding("OL34", "WARN",
            f"{SPEC_FILE} 中 P0 配置的 shape {missing_str} 在测试文件中未覆盖",
            file=test_file)
    return ctx.make_finding("OL34", "PASS",
        "spec P0 配置 shape 在 test 中均有覆盖")


@register("OL39")
def check_ol39(ctx: CheckContext) -> Finding:
    """strict 模式下，三个文档必须包含 front matter。"""
    if os.environ.get(STRICT_ENV, "1") != "1":
        return ctx.make_finding("OL39", "SKIP", "strict 模式关闭")
    for filename in (SPEC_FILE, DESIGN_FILE, API_REPORT_FILE):
        content = ctx.read_file(filename)
        if not content:
            return ctx.make_finding("OL39", "FAIL", f"{filename} 不存在", file=filename)
        meta, _ = _parse_front_matter(content)
        if not meta:
            return ctx.make_finding("OL39", "FAIL",
                f"{filename} 缺少 front matter（必须以 --- 开头）",
                file=filename)
    return ctx.make_finding("OL39", "PASS", "front matter 完整")


@register("OL40")
def check_ol40(ctx: CheckContext) -> Finding:
    """strict 模式下，三个文档 front matter 必填字段必须完整。"""
    if os.environ.get(STRICT_ENV, "1") != "1":
        return ctx.make_finding("OL40", "SKIP", "strict 模式关闭")
    docs = ((SPEC_FILE, "SPEC"), (DESIGN_FILE, "DESIGN"), (API_REPORT_FILE, "API_REPORT"))
    for filename, doc_type in docs:
        content = ctx.read_file(filename)
        if not content:
            return ctx.make_finding("OL40", "FAIL", f"{filename} 不存在", file=filename)
        meta, _ = _parse_front_matter(content)
        errors = _validate_doc_schema(doc_type, meta)
        if errors:
            return ctx.make_finding("OL40", "FAIL",
                f"{filename} front matter schema 非法: {'; '.join(errors)}",
                file=filename)
    return ctx.make_finding("OL40", "PASS", "front matter schema 完整")


@register("OL41")
def check_ol41(ctx: CheckContext) -> Finding:
    """禁止将 lint/门禁输出文本污染到代码工件。"""
    suspicious_tokens = (
        "[pypto-op-lint]",
        "交付门禁阻断",
        "以下规则违规",
        "fix_hints:",
        "blocking_rules:",
        "docs_ref:",
    )
    targets = [
        f"{ctx.op_name}_impl.py",
        f"{ctx.op_name}_golden.py",
        f"test_{ctx.op_name}.py",
        "README.md",
    ]
    for filename in targets:
        source = ctx.read_file(filename)
        if not source:
            continue
        lower = source.lower()
        for token in suspicious_tokens:
            if token.lower() in lower:
                return ctx.make_finding(
                    "OL41",
                    "FAIL",
                    f"{filename} 检测到 lint 输出污染片段: {token}",
                    file=filename,
                )
    return ctx.make_finding("OL41", "PASS", "未检测到 lint 输出污染")


@register("OL37")
def check_ol37(ctx: CheckContext) -> Finding:
    """design 与 impl 的关键命名可追溯性检查（信息提示）"""
    design_content = ctx.read_file(DESIGN_FILE)
    if not design_content:
        return ctx.make_finding("OL37", "SKIP", f"{DESIGN_FILE} 不存在")
    impl_file = f"{ctx.op_name}_impl.py"
    tree = ctx.parse_file(impl_file)
    if tree is None:
        return ctx.make_finding("OL37", "SKIP", f"{impl_file} 不存在或无法解析")

    design_names = _extract_design_identifiers(design_content)
    aliases = ctx.pypto_aliases(impl_file)
    impl_names = _extract_jit_assigned_names(tree, aliases)
    impl_blacklist = {
        "i", "j", "k", "n", "m", "x", "y", "z", "tmp", "temp", "result", "out",
        "input", "output",
    }
    impl_names = {n for n in impl_names if n not in impl_blacklist}

    # design 中缺少可对齐变量名时直接跳过，避免误报
    if len(design_names) < 3:
        return ctx.make_finding(
            "OL37",
            "SKIP",
            f"{DESIGN_FILE} 中可用于对齐的代码变量名不足（<3），跳过可追溯性检查",
        )

    if not impl_names:
        return ctx.make_finding("OL37", "SKIP", "未检测到 jit 内局部变量赋值")

    overlap = sorted(design_names & impl_names)
    # 命中 >=2 视为具备可追溯性；该规则为 S3 信息提示，避免因 design 关键名
    # 抽取范围较大导致“有重合却仍提示不重合”的假阳性。
    if len(overlap) >= 2:
        return ctx.make_finding(
            "OL37",
            "PASS",
            f"design/impl 命名可追溯性良好（命中 {len(overlap)} 个：{', '.join(overlap[:5])}）",
            file=impl_file,
        )

    sample_impl = ", ".join(sorted(list(impl_names))[:5])
    return ctx.make_finding(
        "OL37",
        "INFO",
        "design 与 impl 的关键命名重合较少，建议对齐中间变量命名以提升可追溯性；"
        f"当前 impl 示例变量：{sample_impl}",
        file=impl_file,
    )
