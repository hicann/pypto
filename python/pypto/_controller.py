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
import inspect
import itertools
import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union, Iterator, overload

from .enum import *  # noqa
from ._utils import to_sym, set_source_location, clear_source_location
from .symbolic_scalar import SymbolicScalar, SymInt
from .tensor import Tensor
from .config import CubeTile, get_current_scope

logging.basicConfig(level=logging.DEBUG)


__all__ = [
    "set_vec_tile_shapes",
    "get_vec_tile_shapes",
    "set_cube_tile_shapes",
    "get_cube_tile_shapes",
    "set_matrix_size",

    "function",
    "loop",
    "loop_unroll",
    "is_loop_begin",
    "is_loop_end",
    "cond",
]


class Controller:
    _loop_idx_generator = itertools.count(0)
    in_function = False

    @classmethod
    def next_loop_idx(cls) -> int:
        return next(cls._loop_idx_generator)

    @classmethod
    def begin_function(cls):
        if cls.in_function:
            raise RuntimeError("function nested is not allowed")
        cls.in_function = True

    @classmethod
    def end_function(cls):
        cls.in_function = False
        cls._loop_idx_generator = itertools.count(0)


def set_vec_tile_shapes(*shapes: int):
    """ set the tile shapes in vector computation

    This operation sets the value of the tile shapes
    in each dimension in vector computation.

    Parameters
    ----------
    shapes: *int
        the values of the tile shape in each dimension

    Returns
    -------
    None

    Examples
    --------
    >>> pypto.set_vec_tile_shapes(1, 1, 8, 8)
    >>> print(pypto.get_vec_tile_shapes())
    [1, 1, 8, 8]

    """
    # implementation
    concrete_shapes = [
        it.concrete() if isinstance(it, SymbolicScalar) else it for it in shapes
    ]
    pypto_impl.SetScope({"vec_tile_shapes": concrete_shapes})


def get_vec_tile_shapes() -> List[int]:
    """ get the tile shapes in vector computation

    This operation returns the value of the tile shapes
    in each dimension in vector computation.

    Returns
    -------
    List of integers. The values in the list represent the
    tile shape in each dimension respectively

    Examples
    --------
    >>> pypto.set_vec_tile_shapes([1, 1, 8, 8])
    >>> print(pypto.get_vec_tile_shapes())
    [1, 1, 8, 8]

    """
    # implementation
    scope = get_current_scope()
    return scope.get_vec_tile_shapes()


def set_cube_tile_shapes(m: List[int], k: List[int], n: List[int], enable_multi_data_load: bool = False,
                        enable_split_k: bool = False):
    """ set the tile shapes in cube computation

    This operation sets the value of the tile shapes
    in each dimension in cube computation of left and right matrix,
    together with the cache level (L1/L0).

    Parameters
    ----------
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

    Returns
    -------
    None

    Examples
    --------
    >>> pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128], True)
    >>> print(pypto.get_cube_tile_shapes())
    [[16, 16], [256, 512], [128, 128], True]

    """
    # implementation
    cube_tile = CubeTile(m, k, n, enable_multi_data_load, enable_split_k)
    pypto_impl.SetScope({"cube_tile_shapes": cube_tile.impl()})


def get_cube_tile_shapes() -> Tuple[List[int], List[int], List[int], bool, bool]:
    """ get the tile shapes in cube computation

    This operation gets the value of the tile shapes
    in each dimension in cube computation of left and right matrix,
    together with the cache level (L1/L0).

    Returns
    -------
    return List[Union[List, bool, bool]]
    The list includes the tile shape information of both left and
    right matrix, together with the cache level (L1/L0).

    Examples
    --------
    >>> pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128], True)
    >>> print(pypto.get_cube_tile_shapes())
    [[16, 16], [256, 512], [128, 128], True, False]

    """
    # implementation
    scope = get_current_scope()
    cube_tile = scope.get_cube_tile_shapes()
    return tuple([cube_tile.m, cube_tile.k, cube_tile.n, cube_tile.enableMultiDataLoad, cube_tile.enableSplitK])


def set_matrix_size(size: List[int]):
    pypto_impl.SetScope({"matrix_size": size})


def set_build_static(static: bool):
    pypto_impl.SetBuildStatic(static)


def begin_function(
    name: str,
    graph_type: pypto_impl.GraphType,
    func_type: pypto_impl.FunctionType,
    *args
):
    args = [arg.base() for arg in args]
    Controller.reset()
    return pypto_impl.BeginFunction(name, graph_type, func_type, *args)


def end_function(name: str, generate_call: bool = True):
    pypto_impl.EndFunction(name, generate_call)


class LoopRange:
    def __init__(self, start, stop=None, step: Union[int, SymbolicScalar] = 1):
        if stop is None:
            start, stop = 0, start
        self._base = pypto_impl.LoopRange(
            to_sym(start), to_sym(stop), to_sym(step))
        self._start = self._base.Begin()
        self._stop = self._base.End()

    def begin(self) -> SymbolicScalar:
        return SymbolicScalar.from_base(self._base.Begin())

    def end(self) -> SymbolicScalar:
        return SymbolicScalar.from_base(self._base.End())

    def step(self) -> SymbolicScalar:
        return SymbolicScalar.from_base(self._base.Step())

    def __str__(self) -> str:
        return self._base.Dump()

    def __repr__(self) -> str:
        return f"LoopRange({self._base.Dump()})"

    def base(self) -> pypto_impl.LoopRange:
        return self._base


_loop_range = LoopRange


def is_loop_begin(scalar: SymInt) -> SymbolicScalar:
    """ Determines if the current iteration is the start of loop
    This function returns a boolean value which specifys whether
    the current iteration is the beginning of the loop

    Parameters
    ----------
    scalar: SymbolicScalar
        current loop index

    Returns
    -------
    SymbolicScalar : expression to determine if currently at loop start

    Examples
    --------
        for s2_idx in pypto.loop(bn_per_batch):
            if pypto.cond(pypto.is_loop_begin(s2_idx)):
                ...
    """
    if not hasattr(scalar, "_loop_begin"):
        raise ValueError("not loop index")
    # implementation
    return SymbolicScalar.from_base(
        pypto_impl.IsLoopBegin(to_sym(scalar), getattr(scalar, "_loop_begin")))


def is_loop_end(scalar: SymInt) -> SymbolicScalar:
    """ Determines if the current iteration is the end of loop
    This function returns a boolean value which specifys whether
    the current iteration is the end of the loop

    Parameters
    ----------
    scalar: SymbolicScalar
        current loop index

    Returns
    -------
    SymbolicScalar : expression to determine if currently at loop end

    Examples
    --------
        for s2_idx in pypto.loop(0, bn_per_batch, 1, name="LOOP_L4_s2_SA", idx_name="s2_idx",
            if pypto.cond(pypto.is_loop_end(s2_idx)):
                ...
    """
    if not hasattr(scalar, "_loop_end"):
        raise ValueError("not loop index")
    # implementation
    return SymbolicScalar.from_base(
        pypto_impl.IsLoopEnd(to_sym(scalar), getattr(scalar, "_loop_end")))


@contextmanager
def function(name: str, *args) -> Iterator:
    """ defining the function

    This API record the function and dataflow user has defined. A computing
    graph will be built based on the recorded function.

    Parameters
    ----------
    name: str
        The name of the function
    *args: tuple
        Tensors passed to the function
    Returns
    -------
    return the function in pypto framework. Operations will be added
    under this API. It will produce the computing graph of the function
    in the end.

    Examples
    --------
        with pypto.function("main", a, b, c):
            pypto.set_vec_tile_shapes(16, 16)
            for _ in pypto.loop(0, b_loop, 1, name, = "LOOP_L0_bIdx_mla_prolog",
                idx_name = "b_idx"):
                c[:] = a+b

    """
    in_out_tensors = [item for item in args if isinstance(item, Tensor)]
    func = None
    Controller.begin_function()
    try:
        set_source_location(level=2)
        func = pypto_impl.RecordFunc(name, [t.base() for t in in_out_tensors])
        clear_source_location()
        yield func
    except Exception as e:
        logging.error("Record function %s failed: %s", name, e)
        raise
    finally:
        assert func
        func.EndFunction()
        Controller.end_function()


def cond(scalar: SymInt, file: Optional[str] = None, lineno: Optional[int] = None):
    """ set up a conditional computation. Use as "if" condition in python.

    Parameters
    ----------
    scalar: Union[int, SymbolicScalar]
        expression to determine if condition is true or not
    file: Optional[str]
        source file of the condition; must be provided with lineno
    lineno: Optional[int]
        source line of the condition; must be provided with file

    Returns
    -------
    return a generator, which will be used for setting up the
    "if" in building computing graph

    Examples
    --------
        if pypto.cond(pypto.is_loop_begin(bn)):
            pass
        elif pypto.cond(pypto.is_loop_end(bn)):
            pass
        elif pypto.cond(1):
            pass
        else:
            pass
    """
    # allow caller to override source location; enforce both or none
    if (file is None) ^ (lineno is None):
        raise ValueError("file and lineno must be provided together or omitted")

    if file is None:
        stack = inspect.stack()[1]
        filename, line = stack.filename, stack.lineno
    else:
        filename, line = file, lineno

    return pypto_impl.RecordIfBranch(to_sym(scalar), filename, line)


class _LoopFunction:

    class Iterator:
        def __init__(self, iter, begin, end):
            self._iter = iter
            self._begin = begin
            self._end = end

        def __next__(self):
            scalar = SymbolicScalar.from_base(self._iter.__next__())
            setattr(scalar, "_loop_begin", self._begin)
            setattr(scalar, "_loop_end", self._end)
            return scalar

    def __init__(self, name, loop_name, loop_range, unroll_list, submit_before_loop):
        loop_range = loop_range.base()
        self._base = pypto_impl.RecordLoopFunc(name, pypto_impl.FunctionType.DYNAMIC_LOOP,
                                             loop_name, loop_range,
                                             unroll_list, submit_before_loop)
        self._begin = loop_range.Begin()
        self._end = loop_range.End()

    def __iter__(self):
        return self.Iterator(self._base.__iter__(), self._begin, self._end)


@contextmanager
def _loop_function(
    name: str,
    loop_name: str,
    loop_range: LoopRange,
    unroll_list: Optional[List[int]] = None,
    submit_before_loop: bool = False,
):
    if unroll_list is None:
        unroll_set = set()
    else:
        unroll_set = set(unroll_list)
    rlf = None
    try:
        set_source_location(level=3)
        frame = sys._getframe(3)
        pypto_impl.BeginScope(name, {}, frame.f_code.co_filename, frame.f_lineno)
        rlf = _LoopFunction(name, loop_name, loop_range,
                            unroll_set, submit_before_loop)
        clear_source_location()
        yield rlf
    except Exception as e:
        logging.error("Record loop function %s failed: %s", name, e)
        raise
    finally:
        del rlf
        pypto_impl.EndScope()


@overload
def loop(stop: SymInt, /, **kwargs) -> Iterator[SymInt]:
    """ Create a symbolic loop ranging from 0 to `stop` (exclusive).

    Parameters
    ----------
    stop : SymInt
        The end value (exclusive) of the loop range.
    kwargs :
        See base `loop()` documentation for shared keyword arguments.

    Returns
    -------
    Iterator[SymInt]
        A generator over symbolic integers representing each iteration variable.


    Examples
    --------
    for _ in pypto.loop(10, name="LOOP_L0_bIdx", idx_name="bIdx"):
        if pypto.cond(k==0):
            b[:] = a + a
        else:
            b[:] = a + b
    """
    ...


@overload
def loop(start: SymInt, stop: SymInt, step: Optional[SymInt] = 1, /, **kwargs) -> Iterator[SymInt]:
    """ Create a symbolic loop ranging from `start` to `stop` (exclusive), incrementing by `step`.

    Parameters
    ----------
    start : SymInt
        Start value.
    stop : SymInt
        End value (exclusive).
    step : Optional[SymInt], default=1
        Increment for each iteration.
    kwargs :
        See base `loop()` documentation for shared keyword arguments.

    Returns
    -------
    Iterator[SymInt]
        A generator over symbolic integers.

    Examples
    --------
    for _ in pypto.loop(0, 10, 1, name="LOOP_L0_bIdx", idx_name="bIdx"):
        if pypto.cond(k==0):
            b[:] = a + a
        else:
            b[:] = a + b
    """
    ...


def _get_loop_range(*args):
    nargs = len(args)
    if nargs == 1:
        start, stop, step = 0, args[0], 1
    elif nargs == 2:
        start, stop, step = args[0], args[1], 1
    elif nargs == 3:
        start, stop, step = args
    else:
        raise TypeError(
            f"loop() takes 1 to 3 positional arguments but {nargs} were given")
    return start, stop, step


def loop(
    *args,
    **kwargs,
) -> Iterator[SymInt]:
    """ set up a loop computation. Use as a for loop in python.

    Parameters
    ----------
    kwargs:
        name: str
            The name of the loop
        idx_name: str
            The name of the loop index
        unroll_list: List[int], default=[1]
            The unroll factors to be used.
        submit_before_loop: bool
            Add a barrier before the loop

    Returns
    --------
    return a generator, which will be used for setting up the
    for loop in building computing graph
    """
    start, stop, step = _get_loop_range(*args)
    # implementation
    loop_idx = Controller.next_loop_idx()
    name = kwargs.get("name", f"loop_{loop_idx}")
    idx_name = kwargs.get("idx_name", f"loop_idx_{loop_idx}")
    unroll_list = kwargs.get("unroll_list", None)
    submit_before_loop = kwargs.get("submit_before_loop", False)
    with _loop_function(
        name, idx_name, _loop_range(
            start, stop, step), unroll_list, submit_before_loop
    ) as rlf:
        for k in rlf:
            yield k


def loop_unroll(*args, **kwargs) -> Iterator[Tuple[SymbolicScalar, int]]:
    """
    Almost the same as `loop()`, but with an additional `unroll_list` parameter.

    Parameters
    ----------
    args:
        start: SymInt
            Start value.
        stop: SymInt
            End value (exclusive).
        step: Optional[SymInt], default=1
            Increment for each iteration.
    kwargs:
        name: str
            The name of the loop
        idx_name: str
            The name of the loop index
        unroll_list: List[int], default=[1]
            The unroll factors to be used.
        submit_before_loop: bool, default=False
            Whether to submit the loop before the loop body.

    Returns
    --------
    return an iterator and unroll factor.

    Examples
    --------
        for idx in pypto.loop_unroll(0, 10, 1, name="LOOP_L0_bIdx",
                                                      idx_name="bIdx", unroll_list=[4, 2, 1]):
            b[:] = a + idx
    """
    start, stop, step = _get_loop_range(*args)

    unroll_list = kwargs.pop("unroll_list", [1])
    unroll_list = sorted(set(unroll_list), reverse=True)
    if 1 not in unroll_list:
        unroll_list.append(1)

    ori_name = kwargs.get("name", None)
    ori_idx_name = kwargs.get("idx_name", None)

    nstart = start
    for p in unroll_list:
        if ori_name:
            # "_LoopUnroll" is used by the backend to parse the unroll_times in function
            kwargs["name"] = f"{ori_name}_LoopUnroll{p}"
        else:
            kwargs["name"] = f"_LoopUnroll{p}"

        nstep = step * p
        left = (stop - start) % nstep
        for idx in loop(nstart, stop - left, nstep, **kwargs):
            yield (idx, p)
        nstart = stop - left


def dump() -> str:
    """ Dump the current program.

    Returns
    -------
    str
        The dumped program.
    """
    return pypto_impl.Dump()


def reset():
    """ Reset the current program.
    """
    pypto_impl.Reset()
