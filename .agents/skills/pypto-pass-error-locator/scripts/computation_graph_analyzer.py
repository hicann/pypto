#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO 计算图 JSON 分析工具

本工具提供完整的计算图分析能力，包括：
- 加载和解析计算图JSON文件
- 获取各类数据（Tensor、Operation、RawTensor、Incast/Outcast等）
- 追踪数据流转
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class GraphType(Enum):
    """计算图类型"""
    TENSOR_GRAPH = 1
    TILE_GRAPH = 2
    BLOCK_GRAPH = 3
    EXECUTE_GRAPH = 4
    
    @classmethod
    def from_value(cls, value: int) -> Optional['GraphType']:
        return cls(value) if value in cls._value2member_map_ else None
    
    @classmethod
    def to_string(cls, value: int) -> str:
        mapping = {
            1: 'TensorGraph',
            2: 'TileGraph',
            3: 'BlockGraph',
            4: 'ExecuteGraph'
        }
        return mapping.get(value, 'Unknown')


class MemoryType(Enum):
    """内存层级类型"""
    MEM_UB = 0
    MEM_L1 = 1
    MEM_L0A = 2
    MEM_L0B = 3
    MEM_L0C = 4
    MEM_DEVICE_DDR = 15
    
    @classmethod
    def to_string(cls, value: int) -> str:
        mapping = {
            0: 'MEM_UB (Unified Buffer)',
            1: 'MEM_L1 (L1 Cache)',
            2: 'MEM_L0A (L0A Cache)',
            3: 'MEM_L0B (L0B Cache)',
            4: 'MEM_L0C (L0C Cache)',
            15: 'MEM_DEVICE_DDR (Global Memory)'
        }
        return mapping.get(value, f'Unknown ({value})')


@dataclass
class TensorInfo:
    """Tensor信息"""
    magic: int
    shape: List[int]
    validshape: List[int]
    dynvalidshape: List[List[int]]
    offset: List[int]
    rawtensor: int
    nodetype: int
    mem_id: int
    mem_range: List[int]
    mem_type_asis: int
    mem_type_tobe: int
    subgraphid: int
    subgraph_boundary: bool
    life_range: List[int]
    kind: int
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_memory_type_str(self) -> Tuple[str, str]:
        """获取内存类型字符串"""
        return (MemoryType.to_string(self.mem_type_asis), 
                MemoryType.to_string(self.mem_type_tobe))
    
    def is_input(self) -> bool:
        """是否为输入Tensor"""
        return self.nodetype == 1
    
    def is_output(self) -> bool:
        """是否为输出Tensor"""
        return self.nodetype == 2


@dataclass
class OperationInfo:
    """Operation信息"""
    opmagic: int
    opcode: str
    ioperands: List[int]
    ooperands: List[int]
    attr: List[Any]
    subgraphid: int
    latency: int
    kind: int
    tile: Optional[Dict[str, Any]]
    op_attr: Dict[str, Any]
    file: Optional[str]
    line: Optional[int]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_input_tensor_magics(self) -> List[int]:
        """获取输入Tensor的magic ID列表"""
        return self.ioperands
    
    def get_output_tensor_magics(self) -> List[int]:
        """获取输出Tensor的magic ID列表"""
        return self.ooperands


@dataclass
class RawTensorInfo:
    """RawTensor信息"""
    rawmagic: int
    rawshape: List[int]
    ori_rawshape: List[int]
    datatype: int
    format: int
    kind: int
    symbol: Optional[str]
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionInfo:
    """Function信息"""
    func_magicname: str
    funcmagic: int
    graphtype: int
    incasts: List[List[Any]]
    outcasts: List[List[Any]]
    operations: List[OperationInfo]
    tensors: List[TensorInfo]
    rawtensors: List[RawTensorInfo]
    total_subgraph_count: int
    file: Optional[str]
    line: Optional[int]
    global_tensors: List[int]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_graph_type(self) -> Optional[GraphType]:
        """获取图类型"""
        return GraphType.from_value(self.graphtype)
    
    def get_graph_type_str(self) -> str:
        """获取图类型字符串"""
        return GraphType.to_string(self.graphtype)
    
    def get_input_tensor_magics(self) -> List[int]:
        """获取输入Tensor的magic ID列表"""
        return [incast[0] for incast in self.incasts]
    
    def get_output_tensor_magics(self) -> List[int]:
        """获取输出Tensor的magic ID列表"""
        return [outcast[0] for outcast in self.outcasts]


@dataclass
class GraphInfo:
    """计算图信息"""
    entryhash: str
    version: str
    functions: List[FunctionInfo]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_main_function(self) -> Optional[FunctionInfo]:
        """获取主函数（通常只有一个）"""
        return self.functions[0] if self.functions else None


class ComputationGraphAnalyzer:
    """计算图分析器"""
    
    def __init__(self):
        self.graph: Optional[GraphInfo] = None

    @staticmethod
    def _parse_operation(op_data: Dict[str, Any]) -> OperationInfo:
        """解析Operation数据"""
        return OperationInfo(
            opmagic=op_data.get('opmagic', -1),
            opcode=op_data.get('opcode', ''),
            ioperands=op_data.get('ioperands', []),
            ooperands=op_data.get('ooperands', []),
            attr=op_data.get('attr', []),
            subgraphid=op_data.get('subgraphid', -1),
            latency=op_data.get('latency', 0),
            kind=op_data.get('kind', -1),
            tile=op_data.get('tile'),
            op_attr=op_data.get('op_attr', {}),
            file=op_data.get('file'),
            line=op_data.get('line'),
            raw_data=op_data
        )

    @staticmethod
    def _parse_tensor(tensor_data: Dict[str, Any]) -> TensorInfo:
        """解析Tensor数据"""
        mem_type = tensor_data.get('mem_type', {})
        return TensorInfo(
            magic=tensor_data.get('magic', -1),
            shape=tensor_data.get('shape', []),
            validshape=tensor_data.get('validshape', []),
            dynvalidshape=tensor_data.get('dynvalidshape', []),
            offset=tensor_data.get('offset', []),
            rawtensor=tensor_data.get('rawtensor', -1),
            nodetype=tensor_data.get('nodetype', 0),
            mem_id=tensor_data.get('mem_id', -1),
            mem_range=tensor_data.get('mem_range', []),
            mem_type_asis=mem_type.get('asis', 0),
            mem_type_tobe=mem_type.get('tobe', 0),
            subgraphid=tensor_data.get('subgraphid', -1),
            subgraph_boundary=tensor_data.get('subgraph_boundary', False),
            life_range=tensor_data.get('life_range', []),
            kind=tensor_data.get('kind', -1),
            raw_data=tensor_data
        )

    @staticmethod
    def _parse_rawtensor(rt_data: Dict[str, Any]) -> RawTensorInfo:
        """解析RawTensor数据"""
        return RawTensorInfo(
            rawmagic=rt_data.get('rawmagic', -1),
            rawshape=rt_data.get('rawshape', []),
            ori_rawshape=rt_data.get('ori_rawshape', []),
            datatype=rt_data.get('datatype', -1),
            format=rt_data.get('format', -1),
            kind=rt_data.get('kind', -1),
            symbol=rt_data.get('symbol'),
            raw_data=rt_data
        )
    
    def load_graph(self, json_path: str) -> GraphInfo:
        """加载计算图JSON文件"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"计算图文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.graph = self._parse_graph(data)
        return self.graph
    
    def find_tensor_by_magic(self, magic: int) -> Optional[TensorInfo]:
        """根据magic ID查找Tensor"""
        if not self.graph:
            return None
        
        func = self.graph.get_main_function()
        if not func:
            return None
        
        for tensor in func.tensors:
            if tensor.magic == magic:
                return tensor
        return None
    
    def find_operation_by_magic(self, opmagic: int) -> Optional[OperationInfo]:
        """根据opmagic ID查找Operation"""
        if not self.graph:
            return None
        
        func = self.graph.get_main_function()
        if not func:
            return None
        
        for op in func.operations:
            if op.opmagic == opmagic:
                return op
        return None
    
    def find_operations_by_input_tensor(self, tensor_magic: int) -> List[OperationInfo]:
        """查找使用指定Tensor作为输入的所有Operation"""
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        return [op for op in func.operations if tensor_magic in op.ioperands]
    
    def find_operations_by_output_tensor(self, tensor_magic: int) -> List[OperationInfo]:
        """查找产生指定Tensor作为输出的所有Operation"""
        if not self.graph:
            return []
        
        func = self.graph.get_main_function()
        if not func:
            return []
        
        return [op for op in func.operations if tensor_magic in op.ooperands]
    
    def find_producer_of_tensor(self, tensor_magic: int) -> Optional[OperationInfo]:
        """查找产生指定Tensor的Operation（生产者）"""
        ops = self.find_operations_by_output_tensor(tensor_magic)
        return ops[0] if ops else None
    
    def find_consumers_of_tensor(self, tensor_magic: int) -> List[OperationInfo]:
        """查找使用指定Tensor的所有Operation（消费者）"""
        return self.find_operations_by_input_tensor(tensor_magic)

    

    def _parse_graph(self, data: Dict[str, Any]) -> GraphInfo:
        """解析计算图数据"""
        functions = []
        for func_data in data.get('functions', []):
            functions.append(self._parse_function(func_data))
        
        return GraphInfo(
            entryhash=data.get('entryhash', ''),
            version=data.get('version', ''),
            functions=functions,
            raw_data=data
        )

    def _parse_function(self, func_data: Dict[str, Any]) -> FunctionInfo:
        """解析Function数据"""
        operations = []
        for op_data in func_data.get('operations', []):
            operations.append(self._parse_operation(op_data))
        
        tensors = []
        for tensor_data in func_data.get('tensors', []):
            tensors.append(self._parse_tensor(tensor_data))
        
        rawtensors = []
        for rt_data in func_data.get('rawtensors', []):
            rawtensors.append(self._parse_rawtensor(rt_data))
        
        return FunctionInfo(
            func_magicname=func_data.get('func_magicname', ''),
            funcmagic=func_data.get('funcmagic', -1),
            graphtype=func_data.get('graphtype', -1),
            incasts=func_data.get('incasts', []),
            outcasts=func_data.get('outcasts', []),
            operations=operations,
            tensors=tensors,
            rawtensors=rawtensors,
            total_subgraph_count=func_data.get('_total_subgraph_count', 0),
            file=func_data.get('file'),
            line=func_data.get('line'),
            global_tensors=func_data.get('global_tensors', []),
            raw_data=func_data
        )