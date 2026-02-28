#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO"""
from enum import Enum
from typing import Optional, Union

from pypto import pypto_impl
from pypto._controller import loop
from pypto._op_wrapper import op_wrapper
from pypto._utils import to_syms
from pypto.enum import AtomicType, DataType, OpType
from pypto.symbolic_scalar import SymbolicScalar
from pypto.tensor import Tensor


class ShmemMemType(Enum):
    WIN_IN = 0
    WIN_EXP = 1


class CommConfig:
    def __init__(self, group_name: str, n_pes: int, my_pe: SymbolicScalar):
        self.group_name = group_name
        self.n_pes = n_pes
        self.my_pe = my_pe


comm_configs: dict[str, CommConfig] = {}
shmem_id_to_group: dict[int, str] = {}


@op_wrapper
def create_shmem_tensor(
    group_name: str, 
    n_pes: int,
    dtype: DataType,
    shape: list[int], 
) -> tuple[Tensor, Tensor]:
    """Creates a symmetric tensor in shared memory.

    Parameters
    ----------
    group_name : str
        The name of the communication group.
    n_pes : int
        The total number of processes in the communication group.
    dtype : DataType
        The data type of the tensor.
    shape : list of int
        The shape of the tensor to create.

    Returns
    -------
    data : Tensor
        A shared memory tensor used by communication operators to transport data.
    signal : Tensor
        A signal tensor used for synchronization between processes.

    Examples
    --------
    data, signal = pypto.distributed.create_shmem_tensor(
        "tp",
        8,
        pypto.DT_FP16,
        [64, 128],
    )
    """
    data = Tensor([n_pes] + shape, dtype)
    signal = Tensor([n_pes, n_pes] + shape, DataType.DT_INT32)

    for _ in loop(1, name="CREATE_SHMEM_TENSOR", idx_name="_"):
        pypto_impl.CreateShmemData(group_name, n_pes, dtype, shape, data.base(), ShmemMemType.WIN_IN.value)
        pypto_impl.CreateShmemSignal(group_name, data.base(), signal.base())
    
    if group_name not in comm_configs:
        comm_configs[group_name] = CommConfig(group_name, n_pes, pypto_impl.GetSymbolicScalarPeId(group_name))
    
    shmem_id_to_group[data.base().Id()] = group_name
    shmem_id_to_group[signal.base().Id()] = group_name
    
    return data, signal


@op_wrapper
def create_shmem_signal(group_name: str, n_pes: int) -> Tensor:
    """Creates a barrier signal tensor in shared memory.

    Parameters
    ----------
    group_name : str
        The name of the communication group.
    n_pes : int
        The total number of processes in the communication group.

    Returns
    -------
    shmem_barrier_signal : Tensor
        A barrier signal tensor to coordinate process execution.
    

    Examples
    --------
    shmem_barrier_signal = pypto.distributed.create_shmem_barrier_signal("tp", 8)
    """
    shmem_barrier_signal_shape = [1, 1, 1, 8]
    shmem_barrier_signal = Tensor([n_pes] + shmem_barrier_signal_shape, DataType.DT_INT32)

    for _ in loop(1, name="CREATE_SHMEM_BARRIER_SIGNAL", idx_name="_"):
        pypto_impl.CreateShmemData(group_name, n_pes, DataType.DT_INT32, shmem_barrier_signal_shape,
            shmem_barrier_signal.base(), ShmemMemType.WIN_EXP.value)
    
    if group_name not in comm_configs:
        comm_configs[group_name] = CommConfig(group_name, n_pes, pypto_impl.GetSymbolicScalarPeId(group_name))
    
    shmem_id_to_group[shmem_barrier_signal.base().Id()] = group_name
        
    return shmem_barrier_signal


@op_wrapper
def shmem_put(
    src: Tensor,
    offsets: list[Union[int, SymbolicScalar]],
    dst: Tensor,
    dst_pe: Union[int, SymbolicScalar],
    *,
    put_op: AtomicType = AtomicType.SET,
    pred: list[Tensor] = None,
) -> Tensor:
    """Asynchronously sends local GM data to a remote GM.

    Parameters
    ----------
    src: Tensor
        The source tensor located in local GM.
    offsets : list of int
        The offset in the destination shared memory GM.
    dst: Tensor
        The destination tensor in shared memory GM (symmetric memory).
    dst_pe : int
        The pe of the destination device.
    put_op : AtomicType
        The type of atomic operation to apply during the data transfer.
    pred : Tensor
        Predicate tokens used to control the execution dependency of the operation.

    Returns
    -------
    Tensor
        Output predicate tokens representing the completion dependency of the operation.

    Examples
    --------
    Send local GM data to dst_pe 1
    tile = pypto.distributed.shmem_put(
        local_tensor,
        offset,
        shmem_tensor, 
        dst_pe,
        put_op=pypto.AtomicType.SET,
        pred=None,
    )
    """
    dummy = pred[0] if len(pred) == 1 else pypto_impl.Nop(pred)
    dst_tile = pypto_impl.View(dst, [1, 1] + src.shape, to_syms([dst_pe] + offsets))
    return pypto_impl.ShmemPut(dummy, src, dst_tile, put_op)


@op_wrapper
def shmem_get(
    src: Tensor,
    src_pe: Union[int, SymbolicScalar],
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    valid_shape: Optional[list[Union[int, SymbolicScalar]]] = None,
    pred: list[Tensor] = None,
) -> Tensor:
    """Asynchronously fetches data from a remote GM to local GM.

    Parameters
    ----------
    src : Tensor
        The source tensor in remote GM.
    src_pe : Union[int, SymbolicScalar]
        The src_pe of the source device.
    shape : list of int
        The shape of the destination tensor.
    offsets : list of int
        The offset of the destination tensor in local GM.
    valid_shape: list[int] = None
        Optional parameter to retrieve the effective data size of the schematic block.
        It is required that the valid_shape is smaller than the shape of the input.
    pred : Tensor
        Predicate tokens used as control dependencies.

    Returns
    -------
    Tensor
        The destination tensor in local GM.


    Examples
    --------
    Load data from a remote device to local GM
    local_tensor = pypto.distributed.shmem_get(
        src,
        src_pe,
        shape,
        offset,
        valid_shape=None,
        pred=None,
    )
    """
    dummy = pred[0] if len(pred) == 1 else pypto_impl.Nop(pred)
    if valid_shape is None:
        src_tile = pypto_impl.View(src, [1] + shape, to_syms([src_pe] + offset))
    else:
        src_tile = pypto_impl.View(src, [1] + shape, to_syms(valid_shape), to_syms([src_pe] + offset))
    return pypto_impl.ShmemGet(dummy, src_tile)


@op_wrapper
def shmem_signal(
    dst: Tensor,
    dst_pe: Union[int, SymbolicScalar],
    signal: int,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    sig_op: AtomicType = AtomicType.SET,
    pred: list[Tensor] = None,
) -> Tensor:
    """Writes a signal.

    Parameters
    ----------
    dst : Tensor
        The shared memory signal.
    dst_pe : int
        The target device id.
    signal: int
        The value of the signal sent to shared memory,
    shape : list of int
        The shapes of the shared memory signal;
    offset : list of int
        The offsets of the signal in shared memory.
    sig_op : AtomicType
        The type of atomic operation.
    pred : Tensor
        Predicate tokens used as control dependencies.

    Returns
    -------
    Tensor
        The destination tensor in local GM.

    Examples
    --------
    Load data from a remote device to local GM
    dummy = pypto.distributed.shmem_signal(
        shmem_signal,
        dst_pe=1,
        signal,
        shape,
        offset,
        sig_op=pypto.AtomicType.SET,
        pred=None,
    )
    """
    dummy = pred[0] if len(pred) == 1 else pypto_impl.Nop(pred)
    dst_tile = pypto_impl.View(dst, shape, to_syms(offset))
    out_dummy = pypto_impl.ShmemSignal(dummy, dst_tile, sig_op)
    return out_dummy


@op_wrapper
def shmem_wait_until(
    src: Tensor,
    cmp: int = OpType.EQ,
    cmp_value: int = 0,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    clear_signal: bool = False,
    pred: list[Tensor] = None,
) -> Tensor:
    """Waits for a signal.

    Parameters
    ----------
    src : Tensor
        The shared memory signal; can only be a local (this device) shmem signal.
    cmp: int
        Comparison operation type for condition checking. only EQ is supported currently.
    cmp_value : int
        The value to wait for.
    shape : list of int
        The shapes of the shared memory signal;
    offset : list of int
        he offsets of the shmem signal.
    pred : Tensor
        Predicate tokens used as control dependencies.
    clear_signal : bool
        Whether to reset the signal after waiting.

    Returns
    -------
    Tensor
        Output predicate tokens.

    Examples
    --------
    dummy = pypto.distributed.shmem_wait(
        shmem_signal,
        cmp,
        cmp_value,
        shape,
        offset,
        clear_signal=False,
        pred=None,
    )
    """
    dummy = pred[0] if len(pred) == 1 else pypto_impl.Nop(pred)
    group_name = shmem_id_to_group[src.Id()]
    comm_config = comm_configs[group_name]

    src_tile = pypto_impl.View(src, shape, to_syms(offset))
    out_dummy = pypto_impl.WaitUntil(dummy, src_tile, cmp_value, clear_signal)
    return out_dummy


@op_wrapper
def shmem_barrier_all(
    src: Tensor,
    pred: list[Tensor] = None,
) -> Tensor:
    """Synchronizes multiple devices within a communication group.

    Parameters
    ----------
    src : Tensor
        The shared memory barrier signal.
    pred : Tensor
        Predicate tokens used as control dependencies.

    Returns
    -------
    Tensor
        Output predicate tokens.


    Examples
    --------
    Synchronize devices within the communication group
    result = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    dummy = pypto.distributed.shmem_barrier_all(shmem_barrier_signal, pred)
    """
    group_name = shmem_id_to_group[src.Id()]
    comm_config = comm_configs[group_name]
    dummy = pred[0] if len(pred) == 1 else pypto_impl.Nop(pred)
    return pypto_impl.ShmemBarrier(dummy, src, group_name, comm_config.n_pes)


@op_wrapper
def shmem_clear(
    src: Tensor,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    pred: list[Tensor] = None,
    is_signal: bool = False,
) -> Tensor:
    """Synchronizes multiple devices within a communication group by clearing shared memory tensors.

    Parameters
    ----------
    src : Tensor
        The shmem tensor to be cleared; can be a data tensor or a signal tensor.
    shape : list of int
        The shape of the shmem tensor to be cleared; can be for data or signal.
    offset : list of int
        The offset of the shmem tensor to be cleared; can be for data or signal.
    pred : Tensor
        Predicate tokens used as control dependencies.
    is_signal : bool
        Whether the tensor is a signal.

    Returns
    -------
    Tensor
        Output predicate tokens.

    Examples
    --------
    Clear a shmem data tensor in shared memory
    data_clear_dummy = pypto.distributed.shmem_clear(
        shmem_data,
        shmem_shape,
        shmem_offset,
        pred=None,
        is_signal=False,
    )
    Clear a shmem signal tensor in shared memory
    data_clear_dummy = pypto.distributed.shmem_clear(
        shmem_signal,
        shmem_shape,
        shmem_offset,
        pred=None,
        is_signal=True,
    )
    """
    group_name = shmem_id_to_group[src.Id()]
    comm_config = comm_configs[group_name]
    dummy = pred[0] if len(pred) == 1 else pypto_impl.Nop(pred)
    if is_signal:
        src_tile = pypto_impl.View(src, [1, comm_config.n_pes] + shape, to_syms([comm_config.my_pe, 0] + offset))
        out = pypto_impl.ShmemSignalSet(dummy, src_tile)
    else:
        src_tile = pypto_impl.View(src, [1] + shape, [comm_config.my_pe] + to_syms(offset))
        out = pypto_impl.ShmemDataSet(dummy, src_tile)
    return out



@op_wrapper
def my_symbolic_pe(group_name: str) -> SymbolicScalar:
    """Gets the symbolic PE.

    Parameters
    ----------
    group_name : str
        The name of the communication group.

    Returns
    -------
    symbolic scalar
        Represents my_pe.
    
    Examples
    --------
       my_pe = pypto.distributed.my_symbolic_pe(group_name) 
    """
    return SymbolicScalar.from_base(pypto_impl.GetSymbolicScalarPeId(group_name))