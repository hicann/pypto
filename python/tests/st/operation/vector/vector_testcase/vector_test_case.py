#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Common data descriptions used by vector ST testcase modules."""

import ast
from dataclasses import dataclass
import math
from typing import ClassVar

import torch

import pypto


@dataclass(frozen=True)
class TensorConfig:
    shape: tuple[int, ...]
    dtype: str
    data_min: int | float | str = 0
    data_max: int | float | str = 0

    @classmethod
    def from_test_case(cls, tensor: dict) -> "TensorConfig":
        data_range = tensor.get("data_range", {"min": 0, "max": 0})
        return cls(tuple(tensor["shape"]), tensor["dtype"], data_range["min"], data_range["max"])


@dataclass(frozen=True)
class VectorConfig:
    OPERATION: ClassVar[str] = ""
    case_index: int
    case_name: str
    input_tensors: tuple[TensorConfig, ...]
    output_tensors: tuple[TensorConfig, ...]
    view_shape: tuple[int, ...]
    tile_shape: tuple[int, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    scalar: int | float = 0
    axis: int = 0
    dims: tuple[int, ...] = ()
    keep_dim: bool = False
    correction: int = 1
    descending: tuple[bool, ...] = ()
    count: tuple[int, ...] = ()
    is_largest: tuple[bool, ...] = ()
    perm: tuple[int, ...] = ()
    decimals: int = 0
    first_dim: int = 0
    second_dim: int = 1
    diagonal: int = 0
    pattern_mode: int = 0
    alpha: int | float = 1
    accumulate: bool = False
    num_classes: int = 0
    use_zero_points: bool = False
    flag: int = 0
    x_scalar: int | float = 0
    y_scalar: int | float = 0
    src: int | float = 0
    reduce: str = ""
    base: str | int = "e"
    execution_view_shape: tuple[int, ...] = ()
    iteration_shape: tuple[int, ...] = ()
    input_view_shapes: tuple[tuple[int, ...], ...] = ()
    loop_ranges: tuple[int, ...] = ()
    output_offset_map: tuple[int, ...] = ()
    output_dtype: pypto.DataType = pypto.DT_FP32
    condition_is_packed: bool = False

    @classmethod
    def from_test_case(cls, case: dict) -> "VectorConfig":
        params = {key: literal(value) for key, value in case.get("params", {}).items()}
        scalar = params.get("scalar", 0)
        if (params.get("scalar_type") or "").startswith(("int", "uint")):
            scalar = int(scalar)
        input_shapes = tuple(tuple(tensor["shape"]) for tensor in case["input_tensors"])
        output_shapes = tuple(tuple(tensor["shape"]) for tensor in case["output_tensors"])
        execution_view_shape = cls._execution_view_shape(case, params)
        iteration_shape = cls._iteration_shape(case, input_shapes, execution_view_shape)
        input_view_shapes = tuple(
            tuple(1 if dim == 1 else view for dim, view in zip(shape, execution_view_shape))
            if len(shape) == len(execution_view_shape)
            else shape
            for shape in input_shapes
        )
        output_offset_map = cls._output_offset_map(case, params, len(execution_view_shape))
        return cls(
            case_index=case["case_index"],
            case_name=case["case_name"],
            input_tensors=tuple(TensorConfig.from_test_case(tensor) for tensor in case["input_tensors"]),
            output_tensors=tuple(TensorConfig.from_test_case(tensor) for tensor in case["output_tensors"]),
            view_shape=tuple(case["view_shape"]),
            tile_shape=tuple(case["tile_shape"]),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            scalar=scalar,
            axis=params.get("axis", 0),
            dims=tuple(params.get("dims", params.get("dim", ()))),
            keep_dim=params.get("keepDim", False),
            correction=params.get("correction", 1),
            descending=tuple(params.get("descending", ())),
            count=tuple(params.get("count", ())),
            is_largest=tuple(params.get("islargest", ())),
            perm=tuple(params.get("perm", ())),
            decimals=params.get("decimals", 0),
            first_dim=params.get("first_dim", 0),
            second_dim=params.get("second_dim", 1),
            diagonal=params.get("diagonal", 0),
            pattern_mode=params.get("patternMode", 0),
            alpha=params.get("alpha", 1),
            accumulate=params.get("accumulate", False),
            num_classes=params.get("num_classes", 0),
            use_zero_points=params.get("use_zero_points", False),
            flag=params.get("flag", 0),
            x_scalar=params.get("x_scalar", 0),
            y_scalar=params.get("y_scalar", 0),
            src=params.get("src", 0),
            reduce=params.get("reduce", ""),
            base=params.get("base", "e"),
            execution_view_shape=execution_view_shape,
            iteration_shape=iteration_shape,
            input_view_shapes=input_view_shapes,
            loop_ranges=tuple(math.ceil(dim / view) for dim, view in zip(iteration_shape, execution_view_shape)),
            output_offset_map=output_offset_map,
            output_dtype=PTO_DTYPES[case["output_tensors"][0]["dtype"]],
            condition_is_packed=case["input_tensors"][0]["dtype"] == "uint8",
        )

    @classmethod
    def _execution_view_shape(cls, case: dict, params: dict) -> tuple[int, ...]:
        view_shape = list(case["view_shape"])
        input_shape = case["input_tensors"][0]["shape"]
        if cls.OPERATION in {
            "Gather",
            "GatherElement",
            "GatherMask",
            "IndexAddUB",
            "IndexAdd_",
            "IndexPut_",
            "OneHot",
            "Pack",
            "Permute",
            "UnPack",
        }:
            return tuple(input_shape)
        if cls.OPERATION == "ScatterUpdate":
            return tuple(view_shape)
        if cls.OPERATION == "Where" and case["input_tensors"][0]["dtype"] == "uint8":
            return tuple(case["output_tensors"][0]["shape"])
        axes = []
        if cls.OPERATION in {"CumProd", "CumSum", "Concat"}:
            axes = [params["axis"]]
        elif cls.OPERATION in {"Amax", "Amin", "ArgSort", "Prod", "Sum", "TopK"}:
            axes = params["dims"]
        elif cls.OPERATION == "Var":
            axes = params["dim"]
        elif cls.OPERATION in {"TriL", "TriU"}:
            axes = (-2, -1)
        for axis in axes:
            normalized_axis = axis if axis >= 0 else len(view_shape) + axis
            view_shape[normalized_axis] = input_shape[normalized_axis]
            if cls.OPERATION == "Prod" and len(view_shape) == 3:
                full_view_axis = 1 if normalized_axis == 0 else 0
                view_shape[full_view_axis] = input_shape[full_view_axis]
        return tuple(view_shape)

    @classmethod
    def _iteration_shape(cls, case, input_shapes, execution_view_shape):
        if cls.OPERATION == "Expand":
            return tuple(case["output_tensors"][0]["shape"])
        if cls.OPERATION == "ScatterUpdate":
            index_shape = input_shapes[1]
            return index_shape + input_shapes[0][len(index_shape):]
        rank = len(execution_view_shape)
        return tuple(max(shape[axis] for shape in input_shapes if len(shape) == rank) for axis in range(rank))

    @classmethod
    def _output_offset_map(cls, case, params, rank):
        output_rank = len(case["output_tensors"][0]["shape"])
        if cls.OPERATION in {
            "Gather",
            "GatherElement",
            "GatherMask",
            "IndexAddUB",
            "IndexAdd_",
            "IndexPut_",
            "OneHot",
            "Pack",
            "Permute",
            "ScatterUpdate",
            "UnPack",
        } or (cls.OPERATION == "Where" and case["input_tensors"][0]["dtype"] == "uint8"):
            return (-1,) * output_rank
        mapping = list(range(rank))
        if cls.OPERATION in {"Amax", "Amin", "Prod", "Sum"} and not params["keepDim"]:
            del mapping[params["dims"][0]]
        if cls.OPERATION == "Var" and not params["keepDim"]:
            for axis in sorted(params["dim"], reverse=True):
                del mapping[axis]
        if cls.OPERATION == "Transpose":
            first, second = params["first_dim"], params["second_dim"]
            mapping[first], mapping[second] = mapping[second], mapping[first]
        while len(mapping) < output_rank:
            mapping.append(-1)
        return tuple(mapping)


PTO_DTYPES = {
    "bool": pypto.DT_BOOL,
    "bf16": pypto.DT_BF16,
    "fp16": pypto.DT_FP16,
    "fp32": pypto.DT_FP32,
    "int8": pypto.DT_INT8,
    "int16": pypto.DT_INT16,
    "int32": pypto.DT_INT32,
    "int64": pypto.DT_INT64,
    "uint8": pypto.DT_UINT8,
    "uint16": pypto.DT_UINT16,
    "uint32": pypto.DT_UINT32,
}

TORCH_DTYPES = {
    "bool": torch.bool,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
}


def literal(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
