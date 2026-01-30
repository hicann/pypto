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
import sys
import enum
from typing import List, Union, Dict, Optional
from functools import wraps

from . import pypto_impl


class CompStage(enum.Enum):
    CODEGEN = 1
    HOST = 2
    FUNCTION = 3


def set_print_options(*,
                     edgeitems: Optional[int] = 3,
                     precision: Optional[int] = 4,
                     threshold: Optional[int] = 10,
                     linewidth: Optional[int] = 10,
                     ) -> None:
    """
    Set tensor print options.

    Parameters
    ----------
    edge_items : int
        Print max items in tensor head and tail.

    precision : int
        Print precision.

    threshold : int
        Threshold to use.

    linewidth : int
        Max line width.
    """
    pypto_impl.SetPrintOptions(edgeitems, precision, threshold, linewidth)


def set_pass_options(*,
                     pg_skip_partition: Optional[bool] = None,
                     pg_upper_bound: Optional[int] = None,
                     pg_lower_bound: Optional[int] = None,
                     pg_parallel_lower_bound: Optional[int] = None,
                     mg_vec_parallel_lb: Optional[int] = None,
                     vec_nbuffer_mode: Optional[int] = None,
                     vec_nbuffer_setting: Optional[Dict[int, int]] = None,
                     cube_l1_reuse_mode: Optional[int] = None,
                     cube_l1_reuse_setting: Optional[Dict[int, int]] = None,
                     cube_nbuffer_mode: Optional[int] = None,
                     cube_nbuffer_setting: Optional[Dict[int, int]] = None,
                     mg_copyin_upper_bound: Optional[int] = None,
                     sg_set_scope: Optional[int] = None,
                     ) -> None:
    """
    Set pass options.

    Parameters
    ---------
    pg_skip_partition : bool
        Whether to skip the subgraph partitioning process.

    pg_upper_bound : int
        Merged graph parameter, used to configure
        the upper bound of subgraph size.

    pg_lower_bound : int
        Merged graph parameter, used to configure
        the lower bound of subgraph size.

    pg_parallel_lower_bound : int
        Merged graph parameter, used to configure
        the minimum parallelism of subgraphs with the same structure.

    mg_vec_parallel_lb : int
        Merged graph parameter, used to configure
        the minimum parallelism of AIV subgraphs with the same structure.

    vec_nbuffer_mode : int
        Merged graph parameter, used to configure
        the merging strategy for AIV subgraphs with the same structure.

    vec_nbuffer_setting : Dict[int, int]
        Merged graph parameter, used to configure
        the merging quantity of AIV subgraphs with the same structure.

    cube_l1_reuse_mode : int
        Merged graph parameter, used to configure
        the merging strategy for subgraphs with the same structure
        and repeated transfer of the same GM data.

    cube_l1_reuse_setting : Dict[int, int]
        Merged graph parameter, used to configure
        the merging quantity of subgraphs with the same structure
        and repeated transfer of the same GM data.

    cube_nbuffer_mode : int
        Merged graph parameter, used to configure
        the merging strategy for AIC subgraphs with the same structure.

    cube_nbuffer_setting : Dict[int, int]
        Merged graph parameter, used to configure
        the merging quantity of AIC subgraphs with the same structure.

    mg_copyin_upper_bound : int
        Merged graph parameter, used to configure the merged graph size.
    """
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(pass_options=options_dict)


def get_pass_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get pass options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All pass options
    """
    scope = get_current_scope()
    return scope.get_pass_options()



def set_host_options(*, compile_stage: Optional[CompStage] = None) -> None:
    """
    Set host options.

    Parameters
    ---------
    compile_stage : CompStage
        Control the compilation phase.
    """
    options_dict = {k: v.value for k, v in locals().items() if v is not None}
    set_options(host_options=options_dict)


def get_host_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get host options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All host options
    """
    scope = get_current_scope()
    return scope.get_host_options()


def set_codegen_options(*, support_dynamic_aligned: Optional[bool] = None) -> None:
    """
    Set codegen options.

    Parameters
    ---------
    support_dynamic_aligned : bool
        Whether to support dynamic shape which is aligned.

    """
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(codegen_options=options_dict)


def get_codegen_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get codegen options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All codegen options
    """
    scope = get_current_scope()
    return scope.get_codegen_options()


def set_runtime_options(*,
                        device_sched_mode: Optional[int] = None,
                        stitch_function_inner_memory: Optional[int] = None,
                        stitch_function_outcast_memory: Optional[int] = None,
                        stitch_function_num_initial: Optional[int] = None,
                        stitch_function_num_step: Optional[int] = None,
                        stitch_function_size: int = None,
                        stitch_cfgcache_size: Optional[int] = None,
                        run_mode: Optional[int] = None,
                        valid_shape_optimize: Optional[int] = None
                        ) -> None:
    """
    Set runtime options.

    Parameters
    ---------
    device_sched_mode : int
        Set the scheduling mode of the computation subgraph.

    stitch_function_inner_memory : int
        Parameter for controlling the size of the non-outcast memory pool
        allocated to the root function, where the memory pool size is
        max_root_nonoutcast_workspace *.

    stitch_function_outcast_memory : int
        Used to evaluate the size of workspace memory required by
        an operator during runtime when compiling the operator.

    stitch_function_num_initial : int
        The amount of computation tasks for the first stitch task submitted to
        the scheduling AICPU for processing, controlled in the ctrlflow AICPU
        during machine runtime.

    stitch_function_num_step : int
        The computation amount of the processing loop for non-initial
        stitch tasks, controlled in the ctrlflow AICPU during machine runtime.

    stitch_function_size: int
        The maximum Callop computation amount per loop for stitch tasks,
        controlled in the ctrlflow AICPU during machine runtime.

    stitch_cfgcache_size: int
        The size of the control flow cache, in bytes.

    valid_shape_optimize: int
        Dynamic validShape compilation optimization option.
    """
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(runtime_options=options_dict)


def get_runtime_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get runtime options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All runtime options
    """
    scope = get_current_scope()
    return scope.get_runtime_options()


def set_verify_options(*,
                       enable_pass_verify: Optional[bool] = None,
                       pass_verify_save_tensor: Optional[bool] = None,
                       pass_verify_save_tensor_dir: Optional[str] = None,
                       pass_verify_pass_filter: Optional[List[str]] = None,
                       pass_verify_error_tol: Optional[List[float]] = None,
                       ) -> None:
    """
    Set verify options.

    Parameters
    ---------
    enable_pass_verify : bool
        Whether to verify pass.

    pass_verify_save_tensor : bool
        Whether to dump the tensor.

    pass_verify_save_tensor_dir : str
        Pass verify tensor save path.

    pass_verify_pass_filter : List
        Filting pass to verify.

    pass_verify_error_tol : List
        Customize atol and rtol.
    """
    if pass_verify_pass_filter == []:
        pass_verify_pass_filter = None
    if pass_verify_error_tol is None or len(pass_verify_error_tol) != 2:
        pass_verify_error_tol = [1e-3, 1.5e-3]
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(verify_options=options_dict)


def get_verify_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get verify options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All verify options
    """
    scope = get_current_scope()
    return scope.get_verify_options()


def set_debug_options(*,
                      compile_debug_mode: Optional[int] = None,
                      runtime_debug_mode: Optional[int] = None
                      ) -> None:
    """
    Set debug options.

    Parameters
    ---------
    compile_debug_mode : int
        Whether to enable debug mode during compilation stage.

    runtime_debug_mode : int
        Whether to enable debug mode during execution stage.
    """
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(debug_options=options_dict)


def get_debug_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get debug options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All verify options
    """
    scope = get_current_scope()
    return scope.get_debug_options()


def set_semantic_label(label: str) -> None:
    """
    Set the semantic label object.

    Parameters
    ---------
    label: str
        Semantic label.
        Note: label will be attached to subsequent operations

    """
    frame = sys._getframe(1)
    pypto_impl.SetSemanticLabel(label, frame.f_code.co_filename, frame.f_lineno)


def reset_options() -> None:
    """
        Reset all configuration items to their default values.
    """
    pypto_impl.ResetOptions()


class _Options:
    """Configuration options class, supports context manager and decorator modes"""
    INIT_FIELDS = [
        "name", "codegen_options", "host_options", "pass_options",
        "runtime_options", "verify_options", "debug_options",
        "vec_tile_shapes", "cube_tile_shapes", "matrix_size",
        "operation_options"
    ]

    PREFIX_MAP = {
        "codegen_options": "codegen.",
        "host_options": "host.",
        "pass_options": "pass.",
        "runtime_options": "runtime.",
        "verify_options": "verify.",
        "debug_options": "debug.",
        "operation_options": "operation."
    }

    def __init__(self, **kwargs):
        for field in self.INIT_FIELDS:
            setattr(self, field, kwargs.get(field, None))

    def prepare_options(self):
        """Convert configuration to target format"""
        opts = {}

        for attr, prefix in self.PREFIX_MAP.items():
            value = getattr(self, attr)
            if isinstance(value, dict):
                opts.update(
                    {f"{prefix}{k}": v.value if isinstance(v, enum.Enum) else v for k, v in value.items()})

        if self.vec_tile_shapes is not None:
            opts["vec_tile_shapes"] = self.vec_tile_shapes

        if self.cube_tile_shapes is not None:
            if isinstance(self.cube_tile_shapes, CubeTile):
                opts["cube_tile_shapes"] = self.cube_tile_shapes._impl
            else:
                opts["cube_tile_shapes"] = CubeTile(*self.cube_tile_shapes)._impl

        if self.matrix_size is not None:
            opts["matrix_size"] = self.matrix_size

        return opts

    def __enter__(self):
        """Context manager enter logic"""
        opts = self.prepare_options()
        frame = sys._getframe(1)
        # Use decorator position if available, otherwise use caller position
        filename = getattr(self, 'decorator_filename', frame.f_code.co_filename) or '<unknown>'
        lineno = getattr(self, 'decorator_lineno', frame.f_lineno) or 0

        pypto_impl.BeginScope(self.name, opts, filename, lineno)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit logic"""
        frame = sys._getframe(1)
        pypto_impl.EndScope(frame.f_code.co_filename, frame.f_lineno)

    def __call__(self, func):
        """Decorator mode logic: capture function definition location and wrap"""
        self.decorator_filename = func.__code__.co_filename
        self.decorator_lineno = func.__code__.co_firstlineno

        if not self.name:
            self.name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def options(
    name="",
    codegen_options=None,
    host_options=None,
    pass_options=None,
    runtime_options=None,
    verify_options=None,
    operation_options=None,
    debug_options=None,
    vec_tile_shapes=None,
    cube_tile_shapes=None,
    matrix_size=None,
):
    """
    Create an Options instance. Can be used as decorator or context manager.

    Parameters
    ---------
    name: Scope name
    codegen_options: Code generation options (dict)
    host_options: Host options (dict)
    pass_options: Pass options (dict)
    runtime_options: Runtime options (dict)
    verify_options: Verify options (dict)
    debug_options: Debug options (dict)
    operation_options: Operation options (dict)
    vec_tile_shapes: Vector tile shapes (list)
    cube_tile_shapes: Cube tile shapes (CubeTile instance or list)
    matrix_size: Matrix size (list)

    Returns:
    -------
    Options instance

    Examples:
    -------
    # As decorator
    @pypto.options(pass_options={"cube_l1_reuse_setting": {-1: 4}})
    def func():
        pass

    # As context manager
    with pypto.options(name="test", cube_tile_shapes=[[16, 16], [256, 512, 128], [128, 128], True]):
        pass
    """
    # Automatically collect parameters and pass them with unpacking (eliminate duplicate parameter writing)
    return _Options(**locals())


def get_current_scope():
    """Get current config scope."""
    cpp_scope = pypto_impl.CurrentScope()
    return ConfigScope(cpp_scope)


def get_global_config(key: str):
    """Get global config config."""
    cpp_scope = pypto_impl.GlobalScope()
    py_scope = ConfigScope(cpp_scope)
    return py_scope.get_options_prefix("global." + key)


def set_global_config(key, value):
    """Set global config config."""
    pypto_impl.SetGlobalConfig({"global." + key: value})


def set_options(
    codegen_options=None,
    host_options=None,
    pass_options=None,
    runtime_options=None,
    verify_options=None,
    debug_options=None,
    operation_options=None,
    vec_tile_shapes=None,
    cube_tile_shapes=None,
    matrix_size=None,
):
    """
    Finish the old scope and start a new scope.

    Parameters
    ---------
    codegen_options: Code generation options (dict)
    host_options: Host options (dict)
    pass_options: Pass options (dict)
    runtime_options: Runtime options (dict)
    verify_options: Verify options (dict)
    debug_options: Debug options (dict)
    operation_options: Operation options (dict)
    vec_tile_shapes: Vector tile shapes (list)
    cube_tile_shapes: Cube tile shapes (CubeTile instance or list)
    matrix_size: Matrix size (list)

    Examples:
    ---------
    set_options(pass_options={"cube_l1_reuse_setting": {-1: 4}})
    set_options(cube_tile_shapes=[[16, 16], [256, 512, 128], [128, 128], True])
    """
    temp_opts = options(**locals())
    opts = temp_opts.prepare_options()
    frame = sys._getframe(1)
    pypto_impl.SetScope(opts, frame.f_code.co_filename, frame.f_lineno)


def get_options_tree():
    """Get the tree structure string of configuration options"""
    return pypto_impl.GetOptionsTree()


class CubeTile:
    """CubeTile"""
    def __init__(self, m: List[int], k: List[int], n: List[int], enable_multi_data_load: bool = False,
                        enable_split_k: bool = False):
        """
        CubeTile tile for matmul operation, m[0], k[0], n[0] for L0 Cache, m[1], k[1], n[1] for L1 Cache

        Parameters
        ---------
        m: List[int]
        the value of the tile shape in m dimension.
        The length of the list must be 2.

        k: List[int]
            the value of the tile shape in k dimension
            The length of the list must be 2.

        n: List[int]
            the value of the tile shape in n dimension
            The length of the list must be 2.

        enable_multi_data_load: bool
            whether the process of moving L1 to L0 is multi data load.
            default is false (i.e. not multi data load)

        enable_split_k: bool
            whether the matmul result accumulated in the GM.
            default is false (i.e. not GM ACC)
        """

        if len(m) != 2:
            raise ValueError(f"m must have exactly 2 elements, got {len(m)}")
        if len(n) != 2:
            raise ValueError(f"n must have exactly 2 elements, got {len(n)}")
        if len(k) not in [2, 3]:
            raise ValueError(f"k must have 2 or 3 elements, got {len(k)}")

        k_padded = list(k)
        if len(k_padded) == 2:
            k_padded.append(k_padded[1])  # k[2] = k[1]

        self._impl = pypto_impl.CubeTile(list(m), k_padded, list(n), enable_multi_data_load, enable_split_k)

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def __repr__(self):
        return repr(self._impl)

    def __str__(self):
        return str(self._impl)

    def impl(self) -> pypto_impl.CubeTile:
        return self._impl


class ConfigScope:

    def __init__(self, cpp_config_scope=None):
        self._options = {}

        if cpp_config_scope is not None:
            self._options = cpp_config_scope.GetAllConfig()

    def __repr__(self):
        lines = []
        for key, value in sorted(self._options.items()):
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def get_options_prefix(self, key):
        if key not in self._options:
            raise KeyError(f"Option not found: {key}")
        return self._options[key]

    def get_options(self, prefix):
        prefix = f"{prefix}."
        return {k[len(prefix):]: v for k, v in self._options.items() if k.startswith(prefix)}

    def get_pass_options(self):
        return self.get_options("pass")

    def get_codegen_options(self):
        return self.get_options("codegen")

    def get_host_options(self):
        return self.get_options("host")

    def get_debug_options(self):
        return self.get_options("debug")

    def get_runtime_options(self):
        return self.get_options("runtime")

    def get_verify_options(self):
        return self.get_options("verify")

    def get_operation_options(self):
        return self.get_options("operation")

    def get_vec_tile_shapes(self):
        return self._options.get("vec_tile_shapes")

    def get_cube_tile_shapes(self):
        return self._options.get("cube_tile_shapes")

    def get_matrix_size(self):
        return self._options.get("matrix_size")

    def get_all(self):
        return self._options.copy()

    def has(self, key):
        return key in self._options
