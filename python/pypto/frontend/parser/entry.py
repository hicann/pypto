#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Entry point for PTO Script Parser.

This module provides the main entry points for parsing PTO scripts,
including the parse function and JIT decorator.
"""

import inspect
import os
from typing import Any, Callable, Optional, Union
from enum import IntEnum

import pypto
import torch
from pypto import pypto_impl
from pypto.converter import _torch_dtype_from, _gen_pto_tensor
from pypto.cost_model import _cost_model_run_once_data_from_host
from pypto.frontend.parser.diagnostics import Source
from pypto.frontend.parser.parser import NestedFunctionMarker, Parser
from pypto.runtime import _pto_verify_datas


def _default_globals() -> dict[str, Any]:
    """Get the default global variables for parsing.

    Returns
    -------
    dict[str, Any]
        Dictionary containing default global variables (pto module).
    """
    return {
        "pypto": pypto,
    }


class RunMode(IntEnum):
    NPU = 0
    SIM = 1


class DebugMode(IntEnum):
    OFF = 0
    SWIM = 1
    TENSOR_NODEPEND = 2
    CHECKATTR = 3


def parse(program: Source, extra_vars: Optional[dict[str, Any]] = None) -> Any:
    """Parse a PTO script program.

    This function parses a PTO script source and returns the parsed result,
    typically a pypto.Function object.

    Parameters
    ----------
    program : Source
        The source code to parse.
    extra_vars : Optional[dict[str, Any]], optional
        Additional variables to make available during parsing.
        These are merged with default globals (pto module).

    Returns
    -------
    Any
        The parsed result, typically a pypto.Function object.

    Examples
    --------
    >>> source = Source("def foo(): ...")
    >>> func = parse(source)
    >>> isinstance(func, pypto.Function)
    True
    """
    if extra_vars is None:
        merged_vars = _default_globals()
    else:
        merged_vars = {**_default_globals(), **extra_vars}
    parser = Parser(program, merged_vars)
    parser.parse()
    return parser.execute()


def _pto_to_tensor_data(
    tensors: list[pypto.Tensor],
) -> list[pypto_impl.DeviceTensorData]:
    """Convert PTO tensors to device tensor data for runtime execution.

    This helper function creates DeviceTensorData objects that encapsulate
    the tensor metadata (dtype, pointer, shape) required by the backend runtime.

    Parameters
    ----------
    tensors : list[pypto.Tensor]
        List of PTO tensors to convert.

    Returns
    -------
    list[pypto_impl.DeviceTensorData]
        List of device tensor data objects ready for runtime execution.

    Raises
    ------
    RuntimeError
        If any tensor's ori_shape is not specified.
    """
    datas = []
    for t in tensors:
        if t.ori_shape is None:
            raise RuntimeError("The ori_shape of the tensor is not specified.")
        data = pypto_impl.DeviceTensorData(
            t.dtype,
            t.data_ptr,
            list(t.ori_shape),
        )
        datas.append(data)
    return datas


def _current_stream():
    """Retrieve current NPU stream (0 if torch.npu is not present)"""
    npu = getattr(torch, 'npu', None)
    if npu:
        return npu.current_stream().npu_stream
    else:
        return 0


class JitCallableWrapper:
    """Callable wrapper for pypto.Function that integrates frontend parsing with runtime execution.

    This class wraps a pypto.Function and makes it callable with torch tensors,
    integrating the frontend JIT compilation with the runtime execution mechanism.
    Parsing is deferred until the first __call__ invocation (lazy mode), allowing
    dynamic shape binding and cost model evaluation before compilation.

    The wrapper maintains the original function's metadata (__name__, __doc__) and
    provides transparent execution by handling tensor conversion, workspace allocation,
    and device management automatically. It also supports compilation caching to avoid
    redundant recompilation, with cache key generated from compilation-related options.

    Attributes
    ----------
    _pto_function : Optional[pypto.Function]
        The parsed PTO function (None until first call in lazy mode).
    _original_func : Callable
        The original Python function being wrapped.
    _handler : Optional[int]
        The backend runtime handler (None until first call in lazy mode).
    _is_compiled : bool
        Flag indicating whether the function has been compiled.
    _parser : Optional[Parser]
        Parser instance stored for lazy parsing.
    _codegen_options : Optional[dict[str, Any]]
        Options for code generation.
    _host_options : Optional[dict[str, Any]]
        Options for host configuration.
    _runtime_options : Optional[dict[str, Any]]
        Options for runtime execution (including run_mode: NPU or SIM).
    _pass_options : Optional[dict[str, Any]]
        Options for compiler passes.
    _verify_options : Optional[dict[str, Any]]
        Options for verification.
    _debug_options : Optional[dict[str, Any]]
        Options for debugging (including runtime_debug_mode).
    _captured_locals : Optional[dict[str, Any]]
        Captured local variables from the original function's scope (copied to avoid reference issues).
    _infer_controlflow_shape : Optional[type]
        Class type for inferring control flow shape during compilation (None to use default logic).
    _use_cache : bool
        Whether to use compilation caching. If False, force recompilation on each call.
    _kernel_module_cache : dict[tuple, Any]
        Class-level global cache for KernelModule instances, keyed by compilation cache key.
    kmodule : Any
        KernelModule instance associated with this wrapper (from cache if enabled).
    """

    # Global KernelModule cache
    _kernel_module_cache: dict[tuple, Any] = {}

    _dtype_dict = {
        "torch.int8": pypto.DataType.DT_INT8,
        "torch.int16": pypto.DataType.DT_INT16,
        "torch.short": pypto.DataType.DT_INT16,
        "torch.int32": pypto.DataType.DT_INT32,
        "torch.int": pypto.DataType.DT_INT32,
        "torch.int64": pypto.DataType.DT_INT64,
        "torch.long": pypto.DataType.DT_INT64,

        "torch.float16": pypto.DataType.DT_FP16,
        "torch.half": pypto.DataType.DT_FP16,
        "torch.float32": pypto.DataType.DT_FP32,
        "torch.float": pypto.DataType.DT_FP32,

        "torch.bfloat16": pypto.DataType.DT_BF16,
        "torch.float8_e4m3fn": pypto.DataType.DT_FP8E4M3,
        "torch.float8_e5m2": pypto.DataType.DT_FP8E5M2,
        "torch.float8_e8m0fnu": pypto.DataType.DT_FP8E8M0,

        "torch.uint8": pypto.DataType.DT_UINT8,
        "torch.uint16": pypto.DataType.DT_UINT16,
        "torch.uint32": pypto.DataType.DT_UINT32,
        "torch.uint64": pypto.DataType.DT_UINT64,

        "torch.bool": pypto.DataType.DT_BOOL,
    }

    _format_dict = {
        "ND": pypto.TileOpFormat.TILEOP_ND,
        "NZ": pypto.TileOpFormat.TILEOP_NZ,
    }

    def __init__(
        self,
        pto_function: Optional[pypto.Function],
        original_func: Callable,
        handler: Optional[int],
        codegen_options: Optional[dict[str, Any]] = None,
        host_options: Optional[dict[str, Any]] = None,
        pass_options: Optional[dict[str, Any]] = None,
        runtime_options: Optional[dict[str, Any]] = None,
        verify_options: Optional[dict[str, Any]] = None,
        debug_options: Optional[dict[str, Any]] = None,
        captured_locals: Optional[dict[str, Any]] = None,
        infer_controlflow_shape: Optional[type] = None,
        use_cache: bool = True,
    ):
        """Initialize the JIT callable wrapper with compilation and runtime configurations.

        Parameters
        ----------
        pto_function : Optional[pypto.Function]
            The parsed PTO function (None initially in lazy mode).
        original_func : Callable
            The original Python function to be wrapped and compiled.
        handler : Optional[int]
            The backend runtime handler (None initially in lazy mode).
        codegen_options : Optional[dict[str, Any]], optional
            Options for code generation configuration.
        host_options : Optional[dict[str, Any]], optional
            Options for host environment configuration.
        pass_options : Optional[dict[str, Any]], optional
            Options for compiler pass configuration.
        runtime_options : Optional[dict[str, Any]], optional
            Options for runtime execution (e.g., run_mode: NPU or SIM). Defaults to empty dict if None.
        verify_options : Optional[dict[str, Any]], optional
            Options for verification during compilation.
        debug_options : Optional[dict[str, Any]], optional
            Options for debugging (e.g., runtime_debug_mode).
        captured_locals : Optional[dict[str, Any]], optional
            Local variables captured from the original function's scope (copied to a new dict
            to prevent external modification). Defaults to None.
        infer_controlflow_shape : Optional[type], optional
            Class type used for inferring control flow shape during compilation (None uses default inference logic).
            Defaults to None.
        use_cache : bool, optional
            Whether to use compilation caching. If True, reuse existing KernelModule from global cache; if False,
            force recompilation and update cache. Defaults to True.
        """
        self._pto_function = pto_function
        self._original_func = original_func
        self._handler = handler
        self._is_compiled = pto_function is not None
        self._parser = None  # Store parser for lazy parsing
        self._captured_locals = (
            None if captured_locals is None else dict(captured_locals)
        )
        self._use_cache = use_cache

        # Handling options
        self._codegen_options = (
            None if codegen_options is None else dict(codegen_options)
        )
        self._host_options = None if host_options is None else dict(host_options)
        self._runtime_options = runtime_options or {}
        self._pass_options = None if pass_options is None else dict(pass_options)
        self._verify_options = None if verify_options is None else dict(verify_options)
        self._debug_options = None if debug_options is None else dict(debug_options)
        self._infer_controlflow_shape = infer_controlflow_shape

        self._set_run_mode()
        self.kwargs = None

        # Copy metadata from the original function
        if hasattr(original_func, "__name__"):
            self.__name__ = original_func.__name__
        if hasattr(original_func, "__doc__"):
            self.__doc__ = original_func.__doc__

        # kmodule is created lazily in __call__ with cache key including non_tensor_values
        self.kmodule = None


    def __call__(self, *args, **kwargs):
        """Execute the function with torch tensors and optional non-tensor parameters.

        Parameters
        ----------
        *args : Union[torch.Tensor, Any]
            First N arguments must be torch.Tensor (matching tensor params).
            Remaining arguments are non-tensor values (matching non-tensor params in order).
        **kwargs : Any
            Non-tensor parameters as key=value. Overrides positional non-tensor args.

        Returns
        -------
        Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]]
            Output tensor(s), or None if the kernel has no return value.
        """
        in_tensors, non_tensor_values, input_tensor_defs, output_tensor_defs = (
            self._parse_call_args(args, kwargs)
        )
        self._get_or_create_kmodule(non_tensor_values)
        device = self._resolve_device(in_tensors)
        out_tensors = self._allocate_output_tensors(
            in_tensors, input_tensor_defs, output_tensor_defs, device
        )
        pto_in_tensors = self._convert_tensors_with_metadata(
            in_tensors, input_tensor_defs
        )
        pto_out_tensors = self._convert_tensors_with_metadata(
            out_tensors, output_tensor_defs
        )
        self._execute_kernel(pto_in_tensors, pto_out_tensors)
        if not out_tensors:
            return None
        if len(out_tensors) == 1:
            return out_tensors[0]
        return tuple(out_tensors)


    @property
    def function(self) -> Optional[pypto.Function]:
        """Get the underlying pypto.Function.

        Returns
        -------
        Optional[pypto.Function]
            The compiled PTO function, or None if not yet compiled (lazy mode).
        """
        return self._pto_function

    @property
    def handler(self) -> Optional[int]:
        """Get the runtime handler.

        Returns
        -------
        Optional[int]
            The backend runtime handler, or None if not yet compiled (lazy mode).
        """
        return self._handler


    @staticmethod
    def alloc(size):
        """Allocate NPU int8 memory and return its data pointer"""
        return torch.empty(size, dtype=torch.int8, device='npu').data_ptr()


    @staticmethod
    def get_signature_high_performance(
        func: Callable,
    ) -> tuple[list[pypto.Tensor], list[pypto.Tensor], list[str]]:
        """Quickly extract function signature inputs, outputs, and non-tensor param names.

        Parameters
        ----------
        func : Callable
            The function to analyze.

        Returns
        -------
        input_tensors : list[pypto.Tensor]
            List of input parameter annotations (tensor definitions).
        output_tensors : list[pypto.Tensor]
            List of return annotation (tensor definition, wrapped in list for consistency).
        non_tensor_param_names : list[str]
            Ordered list of non-tensor parameter names (must come after tensor params).
        """
        code = func.__code__
        annotations = func.__annotations__ or {}
        argcount = code.co_argcount
        param_names = code.co_varnames[:argcount]

        input_tensor_list: list[pypto.Tensor] = []
        non_tensor_param_names: list[str] = []
        seen_non_tensor = False

        for param_name in param_names:
            if param_name == "return":
                continue
            ann = annotations.get(param_name)
            if ann is not None and isinstance(ann, pypto.Tensor):
                if seen_non_tensor:
                    raise ValueError(
                        "Non-tensor parameters must come after all tensor parameters. "
                        f"Found tensor parameter '{param_name}' after non-tensor "
                        "parameter(s)."
                    )
                input_tensor_list.append(ann)
            else:
                seen_non_tensor = True
                non_tensor_param_names.append(param_name)

        return_annotation = annotations.get("return")
        output_tensor_list: list[pypto.Tensor] = []
        if return_annotation is not None:
            if isinstance(return_annotation, (list, tuple)):
                output_tensor_list = list(return_annotation)
            else:
                output_tensor_list = [return_annotation]

        return input_tensor_list, output_tensor_list, non_tensor_param_names


    @staticmethod
    def _get_func_nonlocals(func: Callable) -> dict[str, Any]:
        """Extract nonlocal (closure) variables from a function.

        This is a modified version of `inspect.getclosurevars` that specifically
        extracts only the nonlocal variables (captured from enclosing scopes)
        without global or builtin variables. These variables must be made available
        during parsing to properly evaluate the function body.

        Parameters
        ----------
        func : Callable
            The function to extract nonlocal variables from.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their captured values.

        Raises
        ------
        TypeError
            If func is not a Python function.
        """
        if inspect.ismethod(func):
            func = func.__func__

        if not inspect.isfunction(func):
            raise TypeError(f"{func!r} is not a Python function")

        code = func.__code__
        # Nonlocal references are named in co_freevars and resolved
        # by looking them up in __closure__ by positional index
        nonlocal_vars = {}
        if func.__closure__ is not None:
            for var, cell in zip(code.co_freevars, func.__closure__):
                try:
                    nonlocal_vars[var] = cell.cell_contents
                except ValueError as err:
                    # cell_contents may raise ValueError if the cell is empty.
                    if "empty" not in str(err):
                        raise
        return nonlocal_vars


    @staticmethod
    def _convert_tensors_with_metadata(
        torch_tensors: list, tensor_defs: list
    ) -> list:
        """Convert torch tensors to pypto tensors with name and dynamic_axis metadata."""
        pto_tensors = []
        for torch_tensor, tensor_def in zip(torch_tensors, tensor_defs):
            dynamic_axis = [
                i
                for i, dim in enumerate(tensor_def.shape)
                if isinstance(dim, pypto.SymbolicScalar)
            ]
            pto_tensors.append(
                pypto.from_torch(
                    torch_tensor,
                    name=tensor_def.name,
                    dynamic_axis=dynamic_axis if dynamic_axis else None,
                    tensor_format=tensor_def.format,
                    dtype=tensor_def.dtype,
                )
            )
        return pto_tensors


    @staticmethod
    def _resolve_output_shape(
        out_tensor_def: Any,
        in_tensors: list,
        input_tensor_defs: list,
        symbolic_dim_value_map: Optional[dict],
    ) -> tuple[list, Optional[dict]]:
        """Resolve shape for one output tensor. Returns (shape_list, updated_map)."""
        shape_list = []
        for dim in out_tensor_def.shape:
            if isinstance(dim, pypto.SymbolicScalar):
                if symbolic_dim_value_map is None:
                    concrete_shapes = [list(t.shape) for t in in_tensors]
                    symbolic_dim_value_map = Parser.match_input_shapes(
                        concrete_shapes, input_tensor_defs
                    )
                dim_value = symbolic_dim_value_map.get(str(dim))
                if dim_value is None:
                    raise ValueError(
                        f"Dynamic dimension {dim} not found in symbolic_dim_value_map"
                    )
                shape_list.append(dim_value)
            else:
                shape_list.append(dim)
        return shape_list, symbolic_dim_value_map

    def compile(
        self,
        args
    ) -> None:
        """Lazily compile function using PTO tensors for dynamic shape binding & verification.

        Compiles the wrapped function on first call (lazy mode), using input PTO tensors to:
        1. Bind dynamic dimensions from concrete shapes (ori_shape)
        2. Set up verification data
        3. Generate PTO IR via deferred parsing

        Parameters
        ----------
        pto_tensors : list[pypto.Tensor]
            PTO tensors with concrete shapes for dynamic dim binding & verification setup
        """
        # Re-create parser for compilation
        self._parser = self._create_parser()
        self._parser.parse()

        # Initialize backend for compilation

        self._setup_verify_data(args)

        # Set options AFTER OperatorBegin() to match @pypto.jit behavior
        self._set_config_option()

        # Bind dynamic dimensions from concrete inputs
        self._parser.bind_dynamic_dims_to_input_tensors()

        # Execute the deferred parsing (happens on first __call__)
        self._pto_function = self._parser.execute()

        # Reset golden data after compilation, similar to pypto.jit
        _pto_verify_datas.reset()

    def _parse_call_args(
        self, args: tuple, kwargs: dict
    ) -> tuple[list, dict[str, Any], list, list]:
        """Parse *args and **kwargs into in_tensors and non_tensor_values.

        Returns
        -------
        in_tensors : list[torch.Tensor]
        non_tensor_values : dict[str, Any]
        input_tensor_defs : list
        output_tensor_defs : list
        """
        input_tensor_defs, output_tensor_defs, non_tensor_param_names = (
            self.get_signature_high_performance(self._original_func)
        )
        n_tensors = len(input_tensor_defs)
        if len(args) < n_tensors:
            raise RuntimeError(
                f"Expected at least {n_tensors} tensor argument(s), got {len(args)}."
            )
        in_tensors = list(args[:n_tensors])
        non_tensor_from_args = list(args[n_tensors:])

        non_tensor_values = self._merge_non_tensor_params(
            non_tensor_param_names, non_tensor_from_args, kwargs
        )

        if len(non_tensor_from_args) > len(non_tensor_param_names):
            raise RuntimeError(
                f"Too many positional arguments: expected {n_tensors} tensor(s) "
                f"and up to {len(non_tensor_param_names)} non-tensor(s), "
                f"got {len(args)} total."
            )
        extra_kwargs = set(kwargs.keys()) - set(non_tensor_param_names)
        if extra_kwargs:
            raise RuntimeError(
                f"Unknown keyword argument(s): {sorted(extra_kwargs)}. "
                f"Valid non-tensor parameters: {non_tensor_param_names}."
            )
        return in_tensors, non_tensor_values, input_tensor_defs, output_tensor_defs

    def _merge_non_tensor_params(
        self,
        non_tensor_param_names: list[str],
        non_tensor_from_args: list,
        kwargs: dict,
    ) -> dict[str, Any]:
        """Merge positional non-tensor args with kwargs, using func defaults when needed."""
        result: dict[str, Any] = {}
        try:
            sig = inspect.signature(self._original_func)
        except (ValueError, TypeError):
            sig = None
        for i, param_name in enumerate(non_tensor_param_names):
            if param_name in kwargs:
                val = kwargs[param_name]
            elif i < len(non_tensor_from_args):
                val = non_tensor_from_args[i]
            elif sig is not None and param_name in sig.parameters:
                param = sig.parameters[param_name]
                if param.default is not inspect.Parameter.empty:
                    val = param.default
                else:
                    raise RuntimeError(
                        f"Missing required non-tensor argument '{param_name}'."
                    )
            else:
                raise RuntimeError(
                    f"Missing required non-tensor argument '{param_name}'."
                )
            if isinstance(val, torch.Tensor):
                raise RuntimeError(
                    f"Non-tensor parameter '{param_name}' must not be a torch.Tensor. "
                    "Use positional arguments for tensors."
                )
            result[param_name] = val
        return result

    def _get_or_create_kmodule(self, non_tensor_values: dict[str, Any]) -> None:
        """Set self.kwargs and resolve kmodule from cache or create new."""
        self.kwargs = non_tensor_values
        key = self._get_compilation_cache_key(non_tensor_values)
        if (
            self._use_cache
            and key is not None
            and key in JitCallableWrapper._kernel_module_cache
        ):
            self.kmodule = JitCallableWrapper._kernel_module_cache[key]
        else:
            self.kmodule = pypto_impl.KernelModule(self)
            if key is not None:
                JitCallableWrapper._kernel_module_cache[key] = self.kmodule

    def _resolve_device(self, in_tensors: list) -> torch.device:
        """Resolve device from in_tensors or run_mode."""
        if in_tensors:
            device = in_tensors[0].device
            for tensor in in_tensors[1:]:
                if tensor.device != device:
                    raise RuntimeError(
                        f"pypto.frontend.jit requires that all input tensors "
                        f"must be on the same device. Got tensors on devices: "
                        f"{device} and {tensor.device}"
                    )
            return device
        run_mode = self._runtime_options.get("run_mode", None)
        if run_mode == pypto.RunMode.NPU:
            if torch.npu.is_available():
                return torch.device('npu', torch.npu.current_device())
            raise RuntimeError("NPU is not available.")
        if run_mode == pypto.RunMode.SIM:
            return torch.device('cpu')
        raise RuntimeError(f"Invalid run mode: {run_mode}.")

    def _allocate_output_tensors(
        self,
        in_tensors: list,
        input_tensor_defs: list,
        output_tensor_defs: list,
        device: torch.device,
    ) -> list:
        """Allocate output tensors based on output defs and resolved dynamic dims."""
        if self._debug_options is not None:
            debug_mode = self._debug_options.get("runtime_debug_mode", None)
            if debug_mode is not None:
                if debug_mode == DebugMode.CHECKATTR:
                    self._check_input_defs_match_tensors(in_tensors, input_tensor_defs)
        symbolic_dim_value_map = None
        out_tensors = []
        for out_tensor_def in output_tensor_defs:
            shape_list, symbolic_dim_value_map = self._resolve_output_shape(
                out_tensor_def, in_tensors, input_tensor_defs, symbolic_dim_value_map
            )
            shape = tuple(shape_list)
            dtype = _torch_dtype_from(out_tensor_def.dtype)
            out_tensors.append(torch.empty(shape, dtype=dtype, device=device))
        return out_tensors

    def _execute_kernel(
        self,
        pto_in_tensors: list,
        pto_out_tensors: list,
    ) -> None:
        """Run kernel on NPU or CPU (SIM)."""
        if self._runtime_options.get("run_mode", None) == RunMode.NPU:
            pypto_impl.LaunchKernel(
                self, _current_stream(), *pto_in_tensors, *pto_out_tensors
            )
        else:
            with pypto.options("jit_scope"):
                self._set_config_option()
                pypto_impl.DeviceInit()
                self.compile([*pto_in_tensors, *pto_out_tensors])
                self._run_with_cpu([*pto_in_tensors, *pto_out_tensors], [])

    def _check_input_defs_match_tensors(self, in_tensors: list, input_tensor_defs: list[pypto.Tensor]) -> None:
        """Check if the input tensor definitions match the input tensors.
        """
        def get_format(tensor):
            import torch_npu
            if torch_npu.get_npu_format(tensor) == 29:
                return "NZ"
            return "ND"

        # Check the number of input tensors and input tensor definitions
        if len(in_tensors) != len(input_tensor_defs):
            raise RuntimeError(f"There are {len(in_tensors)} input tensor(s), \
                but {len(input_tensor_defs)} input tensor definition(s).")

        def ordinal(n):
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
            if 11 <= n % 100 <= 13:
                suffix = 'th'
            return f"{n}{suffix}"
        idx = 0
        for in_tensor, input_tensor_def in zip(in_tensors, input_tensor_defs):
            idx += 1
            # Check the shape of input tensors and input tensor definitions
            if len(in_tensor.shape) != len(input_tensor_def.shape):
                raise ValueError(f"The number of dimensions of {ordinal(idx)} input tensor {in_tensor.shape} \
                    does not match the number of dimensions of input tensor definition {input_tensor_def.shape}.")
            for i, dim in enumerate(input_tensor_def.shape):
                if isinstance(dim, int) and in_tensor.shape[i] != dim:
                    raise ValueError(f"The shape of {ordinal(idx)} input tensor {in_tensor.shape} \
                        does not match the shape of input tensor definition {input_tensor_def.shape}.")

            # Check the dtype of input tensors and input tensor definitions
            if self._dtype_dict[str(in_tensor.dtype)] != input_tensor_def.dtype:
                raise ValueError(f"The dtype of {ordinal(idx)} input tensor {in_tensor.dtype} \
                    does not match the dtype of input tensor definition {input_tensor_def.dtype}.")

            if in_tensor.device == "npu":
                if self._format_dict[get_format(in_tensor)] != input_tensor_def.format:
                    raise ValueError(f"The format of {ordinal(idx)} input tensor {get_format(in_tensor)} \
                        does not match the format of input tensor definition {input_tensor_def.format}.")

    def _get_compilation_cache_key(
        self,
        non_tensor_values: Optional[dict[str, Any]] = None,
    ) -> Optional[tuple]:
        """Generate a cache key for compiled functions.

        This enables automatic caching of compilation results for factory function patterns.
        Functions with identical source code, input shapes, options, and non-tensor values
        can reuse compilation results, avoiding redundant compilation overhead.

        The cache key is generated based on:
        1. Source code
        2. Compilation options
        3. Closure variables (nonlocals captured by the function)
        4. Non-tensor parameter values (e.g., tiling)

        Parameters
        ----------
        non_tensor_values : Optional[dict[str, Any]], optional
            Non-tensor parameter values from the call. Defaults to empty dict.

        Returns
        -------
        Optional[tuple]
            A hashable cache key, or None if caching is not applicable.
        """
        try:
            # Use the source code as the primary key
            # For factory functions, each call creates a new code object,
            # so we need to use source code string instead
            code_obj = self._original_func.__code__
            # Using __code__ attributes directly for performance instead of source code string
            source_code = (
                code_obj.co_code,
                code_obj.co_consts,
                code_obj.co_names,
                code_obj.co_varnames,
            )

            # Create a hashable representation of options
            def make_hashable(obj):
                """Convert dict/list to hashable tuple representation."""
                if obj is None:
                    return None
                t = type(obj)
                if t is dict:
                    return frozenset((k, make_hashable(v)) for k, v in obj.items())
                if t is list or t is tuple:
                    return tuple(make_hashable(item) for item in obj)
                return str(obj)

            options_hash = (
                make_hashable(self._codegen_options),
                make_hashable(self._host_options),
                make_hashable(self._pass_options),
                make_hashable(self._runtime_options),
                make_hashable(self._verify_options),
                make_hashable(self._debug_options),
            )

            captured_locals_hash = make_hashable(self._captured_locals) if self._captured_locals else None
            non_tensor_hash = make_hashable(non_tensor_values) if non_tensor_values else None

            return (source_code, options_hash, captured_locals_hash, non_tensor_hash)
        except (OSError, TypeError):
            # If we can't generate a cache key (e.g., source not available),
            # disable caching for this function
            return None

    def _set_run_mode(self) -> None:
        """Configure the runtime execution mode (NPU or SIM).

        Determines whether to run on NPU hardware or in simulation mode based on:
        1. Explicit run_mode in runtime_options
        2. Presence of ASCEND_HOME_PATH environment variable (indicates CANN installation)

        If CANN is configured, defaults to NPU mode; otherwise defaults to SIM mode.

        Raises
        ------
        RuntimeError
            If an invalid run_mode is specified (must be RunMode.NPU or RunMode.SIM).
        """
        if self._runtime_options is None:
            self._runtime_options = {}

        run_mode = self._runtime_options.get("run_mode", None)
        if run_mode is not None:
            if run_mode not in [pypto.RunMode.NPU, pypto.RunMode.SIM, 0, 1]:
                raise RuntimeError(
                    "Invalid run mode, run mode must be RunMode.NPU or RunMode.SIM."
                )
            else:
                if isinstance(run_mode, pypto.RunMode):
                    self._runtime_options.update({"run_mode": run_mode.value})
                return

        cann_is_configed: bool = bool(os.environ.get("ASCEND_HOME_PATH"))
        if cann_is_configed:
            self._runtime_options.update({"run_mode": pypto.RunMode.NPU.value})
        else:
            self._runtime_options.update({"run_mode": pypto.RunMode.SIM.value})

    def _create_parser(self) -> Parser:
        """Create and prepare a parser for the wrapped function.

        Extracts the source code and all captured variables (globals and nonlocals)
        from the original function, then creates a Parser instance ready for parsing.

        Returns
        -------
        Parser
            A configured parser instance with source code and captured variables.
        """
        source = Source(self._original_func)
        closure_vars = inspect.getclosurevars(self._original_func)
        captured_vars = {}
        captured_vars.update(closure_vars.builtins)
        captured_vars.update(self._original_func.__globals__)
        captured_vars.update(closure_vars.globals)
        captured_vars.update(closure_vars.nonlocals)
        captured_vars.update(self._get_func_nonlocals(self._original_func))
        if self._captured_locals:
            captured_vars.update(self._captured_locals)
        if self.kwargs:
            captured_vars.update(self.kwargs)
        parser = Parser(source, captured_vars)
        return parser

    def _set_config_option(self) -> None:
        """Apply all configuration options to the PTO backend.

        This method applies the various option dictionaries provided at initialization
        to configure the backend compilation and runtime behavior. Options include:
        - run_mode (NPU or SIM)
        - codegen options (code generation settings)
        - host options (host environment settings)
        - pass options (compiler pass configurations)
        - runtime options (execution settings)
        - verify options (verification settings)
        - debug options (debugging settings)
        """
        self._set_run_mode()
        if self._codegen_options:
            pypto.set_codegen_options(**self._codegen_options)
        if self._host_options:
            pypto.set_host_options(**self._host_options)
        if self._pass_options:
            pypto.set_pass_options(**self._pass_options)
        if self._runtime_options:
            pypto.set_runtime_options(**self._runtime_options)
        if self._verify_options:
            pypto.set_verify_options(**self._verify_options)
        if self._debug_options:
            pypto.set_debug_options(**self._debug_options)


    def _setup_verify_data(
        self,
        pto_tensors
    ) -> None:
        """Set verify input/output/golden data for pass-level verification.

        This mirrors the behavior of pypto.runtime._JIT.compile:
        - Copy current input/output from NPU to Host
        - Use golden data pre-injected via set_verify_golden_data
        - Call SetVerifyData to register all three to the underlying ProgramData
        """
        if not (
            isinstance(self._verify_options, dict)
            and self._verify_options.get("enable_pass_verify")
        ):
            return

        # Copy NPU Tensor to CPU, then convert to pypto.Tensor for constructing DeviceTensorData

        host_pto_tensors, _ = _gen_pto_tensor(pto_tensors)
        host_pto_t_datas = _pto_to_tensor_data(host_pto_tensors)
        for i, dev_tensor in enumerate(_pto_to_tensor_data(pto_tensors)):
            pypto_impl.CopyToHost(dev_tensor, host_pto_t_datas[i])
        pypto_impl.SetVerifyData(
            host_pto_t_datas, [], _pto_verify_datas.get_data())

    def _run(
        self,
        in_tensor_data: list[pypto_impl.DeviceTensorData],
        out_tensor_data: list[pypto_impl.DeviceTensorData],
        device: torch.device,
        ctrl_cache: int = 0
    ) -> None:
        """Execute the compiled kernel on device with workspace allocation.

        This is the core execution method that:
        1. Queries required workspace size from the backend
        2. Allocates workspace memory on the target device
        3. Invokes the backend runtime with input/output tensors and workspace
        4. Checks for runtime errors

        Parameters
        ----------
        in_tensor_data : list[pypto_impl.DeviceTensorData]
            Input tensor metadata for the backend.
        out_tensor_data : list[pypto_impl.DeviceTensorData]
            Output tensor metadata for the backend.
        device : torch.device
            The device to execute on (must be NPU for this method).
        ctrl_cache : int
            Device control flow cache.
        Raises
        ------
        RuntimeError
            If runtime execution fails with an error message from the backend.
        """
        assert self._handler is not None
        workspace_size = pypto_impl.GetWorkSpaceSize(
            self._handler, in_tensor_data, out_tensor_data
        )
        workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
        runtime_error_msg = pypto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            in_tensor_data + out_tensor_data,
            [],  # Mark all output tensors as inplace inputs
            torch.npu.current_stream().npu_stream,
            workspace_tensor.data_ptr(),
            ctrl_cache
        )
        if runtime_error_msg != "":
            raise RuntimeError(runtime_error_msg)

    def _run_with_npu(
        self,
        in_tensors: list[pypto.Tensor],
        out_tensors: list[pypto.Tensor],
        device: torch.device,
    ) -> None:
        """Execute on NPU hardware with automatic device switching.

        Converts PTO tensors to device tensor data and executes on the specified NPU.
        If the target device differs from the current device, temporarily switches
        to the target device and restores the original device after execution.

        Parameters
        ----------
        in_tensors : list[pypto.Tensor]
            Input PTO tensors.
        out_tensors : list[pypto.Tensor]
            Output PTO tensors.
        device : torch.device
            Target NPU device for execution.

        Raises
        ------
        RuntimeError
            If device type is not NPU or if execution fails.
        """
        if device.type == "npu":
            import torch_npu  # pylint: disable=import-outside-toplevel, unused-import

            in_tensor_data = _pto_to_tensor_data(in_tensors)
            out_tensor_data = _pto_to_tensor_data(out_tensors)
            ori_device = torch.npu.current_device()
            if device.index != ori_device:
                torch.npu.set_device(device.index)
                self._run(in_tensor_data, out_tensor_data, device)
                torch.npu.set_device(ori_device)
            else:
                self._run(in_tensor_data, out_tensor_data, device)
        else:
            raise RuntimeError(f"Unsupported device type: {device.type}")

    def _run_with_cpu(
        self, in_tensors: list[pypto.Tensor], out_tensors: list[pypto.Tensor]
    ) -> None:
        """Execute in simulation mode using the cost model interface.

        This method runs the kernel on CPU using the cost model, which simulates
        the kernel behavior without requiring NPU hardware. Useful for development,
        testing, and performance modeling.

        Parameters
        ----------
        in_tensors : list[pypto.Tensor]
            Input PTO tensors.
        out_tensors : list[pypto.Tensor]
            Output PTO tensors.
        """
        _cost_model_run_once_data_from_host(in_tensors, out_tensors)


    def _dispatch_with_run_mode(
        self,
        in_tensors: list[pypto.Tensor],
        out_tensors: list[pypto.Tensor],
        device: torch.device,
    ) -> None:
        """Dispatch kernel execution based on configured run mode (NPU or SIM).

        Routes execution to either NPU hardware or CPU simulation based on the
        run_mode setting. Validates that CANN environment is configured when
        attempting NPU execution.

        Parameters
        ----------
        in_tensors : list[pypto.Tensor]
            Input PTO tensors.
        out_tensors : list[pypto.Tensor]
            Output PTO tensors.
        device : torch.device
            Target device for execution (relevant for NPU mode).

        Raises
        ------
        RuntimeError
            If NPU mode is selected but CANN environment is not configured.
        """
        cann_is_configed = bool(os.environ.get("ASCEND_HOME_PATH"))
        run_mode = pypto.get_runtime_options().get("run_mode", 0)
        if run_mode == 0:  # NPU mode
            if not cann_is_configed:
                raise RuntimeError(
                    "Please source cann environment while run mode is NPU."
                )
            self._run_with_npu(in_tensors, out_tensors, device)
        else:  # SIM mode
            self._run_with_cpu(in_tensors, out_tensors)


def function(
    func: Optional[Callable] = None,
) -> Union[Callable, NestedFunctionMarker]:
    """Decorator to mark a function for inline expansion in PTO kernels.

    Functions decorated with `@pypto.frontend.function` are not compiled as standalone
    kernels. Instead, when called from a JIT-compiled kernel, they are inlined directly
    into the caller's IR. This enables code reuse while maintaining optimal performance
    by avoiding function call overhead.

    Parameters
    ----------
    func : Optional[Callable], optional
        The function to mark for inlining. If None, returns a decorator function.
        This allows both @function and @function() syntax.

    Returns
    -------
    Union[Callable, NestedFunctionMarker]
        A NestedFunctionMarker wrapping the original function, which the parser
        recognizes as eligible for inline expansion.

    Examples
    --------
    >>> @pypto.frontend.function
    ... def helper(x: pypto.Tensor((8,), pypto.DT_FP32)):
    ...     return pypto.add(x, x)
    >>>
    >>> @pypto.frontend.jit
    ... def kernel(a: pypto.Tensor((8,), pypto.DT_FP32)):
    ...     return helper(a)  # helper is inlined here

    Notes
    -----
    - Nested functions must have compatible type signatures with their call sites
    - Parameter names need not match between definition and call
    - Return values from nested functions can be directly used
    - Multiple levels of nesting are supported
    """
    if func is None:

        def decorator(f: Callable) -> NestedFunctionMarker:
            marker = NestedFunctionMarker()
            marker._original_func = f
            marker._func_name = f.__name__
            return marker

        return decorator

    marker = NestedFunctionMarker()
    marker._original_func = func
    marker._func_name = func.__name__
    return marker


def jit(
    func: Optional[Callable] = None,
    *,
    host_options: Optional[dict[str, Any]] = None,
    codegen_options: Optional[dict[str, Any]] = None,
    pass_options: Optional[dict[str, Any]] = None,
    runtime_options: Optional[dict[str, Any]] = None,
    verify_options: Optional[dict[str, Any]] = None,
    debug_options: Optional[dict[str, Any]] = None,
    infer_controlflow_shape: Optional[Any] = None,
    use_cache: bool = True,
) -> Union[Callable, Callable[[Callable], JitCallableWrapper]]:
    """JIT decorator for compiling Python functions to PTO IR.

    This decorator compiles a Python function into PTO's intermediate representation
    at decoration time. The decorated function will be replaced with the compiled
    PTO function.

    Parameters
    ----------
    func : Optional[Callable], optional
        The function to decorate. If None, returns a decorator function.
        This allows both @jit and @jit() syntax.

    host_options : Optional[dict[str, Any]], optional
        Options to configure the host.
    codegen_options : Optional[dict[str, Any]], optional
        Options to configure the codegen.
    pass_options : Optional[dict[str, Any]], optional
        Options to configure the pass.
    runtime_options : Optional[dict[str, Any]], optional
        Options to configure the runtime.
    verify_options : Optional[dict[str, Any]], optional
        Options to configure the verify.
    debug_options : Optional[dict[str, Any]], optional
        Options to configure the debug.
    use_cache : bool, optional
        Whether to use compilation caching. If False, force recompilation
        even if a cached version exists. Defaults to True.

    Returns
    -------
    Union[Callable, Callable[[Callable], Callable]]
        Either the decorated function (if func is provided) or a decorator function.

    Raises
    ------
    TypeError
        If the decorator is applied to a non-function object.

    Examples
    --------
    >>> @jit()
    ... def my_kernel(x: pypto.Tensor([16], "float32")) -> pypto.Tensor([16], "float32"):
    ...     return x + 1
    >>> isinstance(my_kernel, pypto.Function)
    True

    >>> @jit
    ... def my_kernel2(x: pypto.Tensor([16], "float32")) -> pypto.Tensor([16], "float32"):
    ...     return x * 2
    >>> isinstance(my_kernel2, pypto.Function)
    True

    Notes
    -----
    The decorator extracts closure variables (nonlocals and globals) from the
    original function and makes them available during parsing. The resulting
    PTO function preserves the original function's name and docstring.
    """

    def decorator_wrapper(f: Callable) -> JitCallableWrapper:
        if not inspect.isfunction(f):
            raise TypeError("jit decorator can only be used on functions")

        # Create wrapper without compiling - defer to first call
        # This matches the behavior of @pypto.jit and avoids backend initialization
        # during module load time
        captured_locals = None
        frame = inspect.currentframe()
        if frame and frame.f_back:
            captured_locals = dict(frame.f_back.f_locals)
        # Break reference cycle as soon as possible
        del frame

        wrapper = JitCallableWrapper(
            None,
            f,
            None,
            codegen_options=codegen_options,
            host_options=host_options,
            pass_options=pass_options,
            runtime_options=runtime_options,
            verify_options=verify_options,
            debug_options=debug_options,
            captured_locals=captured_locals,
            infer_controlflow_shape=infer_controlflow_shape,
            use_cache=use_cache,
        )
        return wrapper

    if func is None:
        # Called with parentheses: @jit()
        return decorator_wrapper

    # Called without parentheses: @jit
    return decorator_wrapper(func)
