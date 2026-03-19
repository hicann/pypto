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
"""pass data compare"""
import os
import json
import logging
import time
import traceback
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, NamedTuple
import torch
import ml_dtypes
import pandas as pd
import numpy as np
from tensor_diff import TensorComparator, IsCloseConfig
from run_float_diff import DataDiffAnalyzer


@dataclass
class ComparisonRecord:
    """Comparison record data class."""
    result_is_close: str
    result_reason: str
    file_a: str
    file_b: str
    opcode_a: str
    opcode_b: str
    dtype: str
    shape_a: str
    shape_b: str
    offset_a: str
    offset_b: str
    verify_type: str
    raw_tensor_magic_a: str
    raw_tensor_magic_b: str
    raw_tensor_symbol_a: str
    raw_tensor_symbol_b: str
    raw_tensor_format_a: str
    raw_tensor_format_b: str
    raw_tensor_ioflag_a: str
    raw_tensor_ioflag_b: str
    tensor_magic_a: str
    tensor_magic_b: str
    loop_info: str
    pass_name: str
    refer_pass_name: str
    timestamp: str
    max_abs_diff: str
    max_rel_diff: str
    average_abs_diff: str
    average_rel_diff: str
    error_count: str
    error_ratio: str
    zero_count: str
    zero_ratio: str
    rtol: str
    atol: str


class PassComparator:
    """Pass comparator class, which encapsulates all comparison logic."""
    
    def __init__(self, 
                 output_pass: str = "",
                 golden_pass: str = "",
                 verify_path_pass1: str = "",
                 verify_path_pass2: str = "",
                 atol: float = 1e-3,
                 rtol: float = 1e-3,
                 topk: int = 50,
                 is_sort: bool = False,
                 mode: int = 0,
                 line: List[int] = None):
        """
        Initializing the comparator
        Parameters:
            verify_path_pass1: Verification file path for the first pass
            verify_path_pass2: Verification file path for the second pass
            atol: Absolute tolerance
            rtol: Relative tolerance
            topk: Print the first k differences
            is_sort: Whether to sort the data
        """
        self.verify_path_pass1 = verify_path_pass1
        self.verify_path_pass2 = verify_path_pass2
        self.atol = atol
        self.rtol = rtol
        self.topk = topk
        self.is_sort = is_sort
        self.mode = mode
        self.line = line
        self.key = "rawTensorMagic"
        self.output_pass = output_pass
        self.golden_pass = golden_pass
        self.result_file = f'verify_pass@{output_pass}@{golden_pass}@{int(time.time() * 1_000_000)}.csv'

        self.comparison_records: List[ComparisonRecord] = []
        
        self.dtype_dict = {
            "DT_BF16": ml_dtypes.bfloat16,
            "DT_FP32": np.float32,
            "DT_FP16": np.float16,
            "DT_INT32": np.int32,
            "DT_INT8": np.int8,
            "DT_INT64": np.int64,
            "DT_INT16": np.int16
        }
        
        self.torch_dtype_dict = {
            ml_dtypes.bfloat16: torch.bfloat16,
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int32: torch.int32,
            np.int8: torch.int8,
            np.int64: torch.int64,
            np.int16: torch.int16
        }

        self.pass_dict = {
            "tensor_graph": 0,
            "LoopUnroll": 0,
            "RemoveRedundantReshape": 0,
            "AutoCast": 1,
            "InferMemoryConflict": 2,
            "RemoveUndrivenView": 3,
            "ExpandFunction": 4,
            "MergeViewAssemble": 5,
            "SplitReshape": 6,
            "SplitRawTensor": 7,
            "SplitLargeFanoutTensor": 8,
            "DuplicateOp": 9,
            "AssignMemoryType": 10,
            "InferDiscontinuousInput": 11,
            "RemoveRedundantOp": 12,
            "InsertOpForViewAssemble": 13,
            "SplitK": 14,
            "GraphPartition": 15,
            "NBufferMerge": 16,
            "L1CopyInReuseMerge": 17,
            "IntraSubgraphAdapter": 18,
            "GenerateMoveOp": 19,
            "CommonOperationEliminate": 20,
            "AxisCombine": 21,
            "PadLocalBuffer": 22,
            "RemoveUnalignedReshape": 23,
            "ReplaceTensor": 24,
            "PreGraphProcess": 25,
            "InferDynShape": 26,
            "SubgraphToFunction": 27,
            "InferParamIndex": 28,
            "SrcDstBufferMerge": 29,
            "AddAlloc": 30,
            "OoOSchedule": 31,
            "GlobalMemoryReuse": 32,
            "RemoveAlloc": 33,
            "CopyOutResolve": 34,
            "InsertSync": 35,
            "CodegenPreproc": 36
        }

        self.opcode_dict = {
            "VIEW": ["L1_TO_L0A", "L1_TO_L0B"],
            "A_MUL_B": ["A_MULACC_B"]
        }
    
    @staticmethod
    def is_contain(a: Dict[str, Any], b: Dict[str, Any], key: str) -> bool:
        """
        Checks whether tensor a is completely included in tensor b.
        Returns:
            bool: True indicates that a is included in b.
        """
        a_offset = json.loads(a["tensorOffset"])
        b_offset = json.loads(b["tensorOffset"])
        a_shape = json.loads(a["outputValidShape"])
        b_shape = json.loads(b["outputValidShape"])
        if key == "tensorMagic" and a_shape == b_shape:
            return True
        elif a["rawTensorMagic"] == b["rawTensorMagic"]:
            for a_off, b_off, a_sh, b_sh in zip(a_offset, b_offset, a_shape, b_shape):
                if (a_off < b_off) or ((a_off + a_sh) > (b_off + b_sh)):
                    return False
            return True
        return False

    @staticmethod
    def data_sort(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[Dict, Dict]:
        a_shape = json.loads(a["outputValidShape"])
        b_shape = json.loads(b["outputValidShape"])
        for ai_shape, bi_shape in zip(a_shape, b_shape):
            if ai_shape > bi_shape:
                return b, a
        return a, b
    
    @staticmethod
    def _build_file_path(data: Dict[str, Any], base_path: str) -> str:
        return os.path.join(base_path, data["passName"], data["outputTensor"])
    
    @staticmethod
    def _log_comparison_info(key: str, a: Dict, b: Optional[Dict] = None):
        logging.info("------" * 10)
        logging.info(f'functionName : {a["verifyType"]}')
        logging.info(f'key : {a[key]}, path : {a["loopInfo"]}')
        logging.info(f'line : {a["No."]}, a_shape: {a["outputValidShape"]}, '
                    f'offset: {a["tensorOffset"]}, dtype: {a["outputDtype"]}')
        if b is not None:
            logging.info(f'line : {b["No."]}, b_shape: {b["outputValidShape"]}, '
                    f'offset: {b["tensorOffset"]}, dtype: {b["outputDtype"]}')

    def compare_data(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Compare two data items"""
        data_a, data_b = self.get_data_slice(a, b)
        
        a_shape = json.loads(a["outputValidShape"])
        dtype = a["outputDtype"]
        np_dtype_a = self.dtype_dict.get(dtype)
        t_dtype_a = self.torch_dtype_dict.get(np_dtype_a)
        if dtype == "DT_BF16":
            tensor_a = torch.frombuffer(
                memoryview(data_a.tobytes()), 
                dtype=t_dtype_a
            ).reshape(a_shape)
            tensor_b = torch.frombuffer(
                memoryview(data_b.tobytes()), 
                dtype=t_dtype_a
            ).reshape(a_shape)
        else:
            tensor_a = torch.from_numpy(data_a).to(dtype=t_dtype_a)
            tensor_b = torch.from_numpy(data_b).to(dtype=t_dtype_a)
        
        comparator = TensorComparator()
        config = IsCloseConfig(
            rtol=self.rtol, 
            atol=self.atol, 
            calc_dtype=torch.float64, 
            is_detail=True
        )
        result_is_close, result_reason_str, result_info = comparator.check_isclose(
           tensor_a, tensor_b, config
        )

        self.add_comparison_record(
                result_is_close=str(result_is_close),
                result_reason=result_reason_str,
                a=a, b=b, diff_conf=result_info[6]
            )
        
        if not result_is_close:
            csv_path = os.path.join(self.verify_path_pass1, 
                                    f"Pass_{self.pass_dict[self.output_pass]:02d}_{self.output_pass}", 
                                    a["verifyType"], a["loopInfo"], f'{a["opMagic"]}_{a["rawTensorMagic"]}.csv')
            comparator.print_isclose_info(result_is_close, result_reason_str, result_info, csv_path)
            logging.error("Data comparison failed.")
            return False
        
        logging.info("Data comparison succeeded.")
        return True

    def compare_not_support(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if not self.is_contain(a, b, self.key):
            error_msg = f"shape or offset is not match"
            logging.error(error_msg)
            return False

        f_a = self._build_file_path(a, self.verify_path_pass1)
        f_b = self._build_file_path(b, self.verify_path_pass2)
        if not os.path.exists(f_a) or not os.path.exists(f_b):
            logging.error(f"Some files do not exist; these will be skipped directly.  file name : {f_a} ,{f_b}")
            return False

        opcode_a = a["opCode"]
        opcode_b = b["opCode"]
        if self.key == "rawTensorMagic" and opcode_a != opcode_b and not self.opcode_match(opcode_a, opcode_b):
            logging.error(f"opcode not match : {opcode_a} ,{opcode_b}")
            return False
        
        dtype = a["outputDtype"]
        np_dtype = self.dtype_dict.get(dtype)
        if np_dtype is None:
            error_msg = f"Unsupported data types : {dtype}"
            logging.error(error_msg)
            return False

        if a["outputDtype"] != b["outputDtype"]:
            error_msg = f"data dtype is different : {a['outputDtype']}, {b['outputDtype']}"
            logging.error(error_msg)
            return False
        return True

    def get_data_slice(self, a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        f_a = self._build_file_path(a, self.verify_path_pass1)
        f_b = self._build_file_path(b, self.verify_path_pass2)
        a_offset = json.loads(a["tensorOffset"])
        b_offset = json.loads(b["tensorOffset"])
        a_shape = json.loads(a["outputValidShape"])
        b_shape = json.loads(b["outputValidShape"])
        np_dtype = self.dtype_dict.get(a["outputDtype"])
        data_a = np.fromfile(f_a, np_dtype)
        data_b = np.fromfile(f_b, np_dtype)

        data_a = data_a.reshape(a_shape)
        data_b = data_b.reshape(b_shape)
        if self.key == "tensorMagic" and a_shape == b_shape:
            return data_a, data_b

        slices = []
        for dim in range(data_a.ndim):
            start = a_offset[dim] - b_offset[dim]
            stop = start + a_shape[dim]
            slices.append(slice(start, stop))
        b_slice = data_b[tuple(slices)]
        return data_a, b_slice
    
    def loop_compare(self, pass_a: str, pass_b: str, df_loop, 
                    raw_tensor_list: List[int] = None) -> bool:
        """
        Compares all data within a loop.
            Parameters:
            pass_a: name of the output pass
            pass_b: name of the golden pass
            df_loop: DataFrame containing loop data
            raw_tensor_list: list of raw tensors to be compared
        Returns:
            bool: whether all comparisons are passed.
        """
        df_a = df_loop[df_loop["passName"].str.contains(pass_a)]
        df_b = df_loop[df_loop["passName"].str.contains(pass_b)]
        a_dict = df_loop[df_loop["passName"].str.contains(pass_a)].to_dict(orient='records')
        self.golden_pass = df_b["passName"].values[0]

        for ai in a_dict:
            raw_magic = ai[self.key]
            a_records = df_a[df_a[self.key] == raw_magic].to_dict(orient='records')
            b_records = df_b[df_b[self.key] == raw_magic].to_dict(orient='records')
            flag = True
            if len(b_records) == 0:
                error_msg = f"{self.key} : {raw_magic}, not exit in golden pass"
                flag = False
            #切leaffunction后多次循环会复用第一个的tensormagic，无法匹配
            elif len(a_records) != len(b_records) and self.pass_dict[pass_a] >= 28 \
                and self.pass_dict[pass_b] >= 4:
                error_msg = f'tensormagic : {raw_magic},After pass4,the sizes on both pass are not the same'
                flag = False
            if not flag:
                self._log_comparison_info(self.key, ai)
                logging.error(error_msg)
                self.add_comparison_record(
                    result_is_close="Skip",
                    result_reason=error_msg,
                    a=ai
                )
                continue
            is_match = False
            for bi in b_records:
                self._log_comparison_info(self.key, ai, bi)
                if not self.compare_not_support(ai, bi):
                    continue
                self.compare_data(ai, bi)
                is_match = True
                break
            if not is_match:
                self.add_comparison_record(
                    result_is_close="Skip",
                    result_reason="not match",
                    a=ai
                )
        
        return True

    def line_compare(self, df: Dict[str, Any], line: List[str] = None) -> None:
        if len(line) < 2:
            logging.error(f'line size < 2 : {len(line)}')
        a = df[df["No."] == line[0]].to_dict(orient='records')
        b = df[df["No."] == line[1]].to_dict(orient='records')
        
        self.key = "tensorMagic"
        if self.is_contain(a[0], b[0], self.key):
            is_right = self.compare_data(a[0], b[0])
            return
        logging.error(f'size or shape is not right')
        return
    
    def pass_compare(self, pass_a: str, pass_b: str, 
                    paths: List[str] = None, 
                    raw_tensor_list: List[int] = None) -> None:
        """
        Main comparison function
        Parameters:
            pass_a: Name of the first pass
            pass_b: Name of the second pass
            paths: List of paths to be compared
            raw_tensor_list: List of raw tensors to be compared
        """
        csv_path = os.path.join(self.verify_path_pass1, "verify_result_metadata.csv")
        df = pd.read_csv(csv_path, encoding="utf-8", 
                        na_values=["", " ", "NaN", "NA"])
        
        # mode = 1: Compare two rows of data.
        if self.mode == 1:
            self.line_compare(df, self.line)
            return
        
        if self.pass_dict[pass_a] >= 4 and self.pass_dict[pass_b] >= 4:
            self.key = "tensorMagic"
        logging.info(f"key  : {self.key}")

        df_pass = df[df["passName"].str.contains(f'{pass_a}|{pass_b}', 
                                                 na=False, regex=True)]
        
        if paths == []:
            paths = df_pass["verifyType"].dropna().unique()
        
        for path in paths:
            df_path = df_pass[df_pass["verifyType"] == path]
            loop_info_list = df_path["loopInfo"].dropna().unique()
            
            for loop_info in loop_info_list:
                df_loop = df_path[df_path["loopInfo"] == loop_info]
                try:
                    self.loop_compare(pass_a, pass_b, df_loop, raw_tensor_list)
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    logging.error(f"Exception in loop_compare: pass={pass_a}/{pass_b}, "
                                    f"path={path}, loop={loop_info}, error={str(e)}\n"
                                    f"Stack trace:\n{stack_trace}")
                    self.save_comparison_results()
                    return
        self.save_comparison_results()

    def add_comparison_record(self, 
                               result_is_close: str,
                               result_reason: str,
                               a: Dict[str, Any],
                               b: Optional[Dict[str, Any]] = None,
                               diff_conf: Optional[Tuple] = None):
        """Add the comparison record to the internal list"""
        
        record = ComparisonRecord(
            result_is_close=result_is_close,
            result_reason=result_reason,
            file_a=a["outputTensor"],
            file_b=b["outputTensor"] if b else None,
            dtype=a["outputDtype"],
            shape_a=a["outputValidShape"],
            shape_b=b["outputValidShape"] if b else None,
            offset_a=a["tensorOffset"],
            offset_b=b["tensorOffset"] if b else None,
            verify_type=a["verifyType"],
            raw_tensor_magic_a=str(a["rawTensorMagic"]),
            raw_tensor_magic_b=str(b["rawTensorMagic"]) if b else None,
            tensor_magic_a=str(a["tensorMagic"]),
            tensor_magic_b=str(b["tensorMagic"]) if b else None,
            opcode_a=str(a["opCode"]),
            opcode_b=str(b["opCode"]) if b else None,
            loop_info=a["loopInfo"],
            pass_name=a["passName"],
            refer_pass_name=self.golden_pass,
            timestamp=int(time.time() * 1_000_000),
            raw_tensor_symbol_a=str(a["outputSymbol"]),
            raw_tensor_symbol_b=str(b["outputSymbol"]) if b else None,
            raw_tensor_format_a=str(a["outputFormat"]),
            raw_tensor_format_b=str(b["outputFormat"]) if b else None,
            raw_tensor_ioflag_a=str(a["ioflag"]),
            raw_tensor_ioflag_b=str(b["ioflag"]) if b else None,
            rtol=str(self.rtol),
            atol=str(self.atol),
            max_abs_diff=diff_conf[0] if diff_conf else None,
            max_rel_diff=diff_conf[1] if diff_conf else None,
            average_abs_diff=diff_conf[2] if diff_conf else None,
            average_rel_diff=diff_conf[3] if diff_conf else None,
            error_count=diff_conf[4] if diff_conf else None,
            error_ratio=diff_conf[5] if diff_conf else None,
            zero_count=diff_conf[6] if diff_conf else None,
            zero_ratio=diff_conf[7] if diff_conf else None
        )
        self.comparison_records.append(record)
    
    def save_comparison_results(self, csv_path: str = None):
        """
        Save all comparison results to a CSV file
        """
        if not self.comparison_records:
            logging.warning("No comparison records to save.")
            return
        
        if csv_path is None:
            csv_path = os.path.join(self.verify_path_pass1, self.result_file)
        
        # Converts records to a list of dictionaries.
        records_dict = [asdict(record) for record in self.comparison_records]
        
        df = pd.DataFrame(records_dict)
        
        column_order = [
            'pass_name',
            'refer_pass_name',
            'verify_type',
            'loop_info',
            'dtype',
            'raw_tensor_magic_a',
            'raw_tensor_symbol_a',
            'raw_tensor_format_a',
            'raw_tensor_ioflag_a',
            'tensor_magic_a',
            'opcode_a',
            'shape_a',
            'offset_a',
            'file_a',
            'raw_tensor_magic_b',
            'raw_tensor_symbol_b',
            'raw_tensor_format_b',
            'raw_tensor_ioflag_b',
            'tensor_magic_b',
            'opcode_b',
            'shape_b',
            'offset_b',
            'file_b',
            'result_is_close',
            'result_reason',
            'rtol',
            'atol',
            'max_abs_diff',
            'max_rel_diff',
            'average_abs_diff',
            'average_rel_diff',
            'error_count',
            'error_ratio',
            'zero_count',
            'zero_ratio',
            'timestamp'
        ]
        
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"Comparison results saved to {csv_path}")

    def opcode_match(self, opcode_a: str, opcode_b: str) -> bool:
        """opcode match"""
        if opcode_a in self.opcode_dict:
            return opcode_b in self.opcode_dict[opcode_a]
        if opcode_b in self.opcode_dict:
            return opcode_a in self.opcode_dict[opcode_b]
        return False


def main():
    """Main function: Parse parameters and run the comparison"""
    parser = argparse.ArgumentParser(
        description="Pass Compare",
        epilog="example:  python3 pass_compare.py --p ExpandFunction RemoveUndrivenView --verify_path ..."
    )
    
    parser.add_argument("--p", nargs='*', type=str, default=[], required=True,
                       help="Names of the two passes to be compared, separated by a space.\
                       The second is goldenpass.")
    parser.add_argument("--func", nargs='*', type=str, default=[],
                       help="Name of the function to be compared. Functions are separated by spaces.")
    parser.add_argument("--raw", nargs='*', type=int, default=[],
                       help="Specifies the raw tensors to be compared, separated by spaces.")
    parser.add_argument("--verify_path", nargs='*', type=str, default=[],
                       help="Verify the file directory. If two values are provided, they represent \
                       the paths of two passes respectively.")
    parser.add_argument("--sort", action='store_true',
                       help="Sort data when plotting")
    parser.add_argument("--atol", type=float, default=1e-3,
                       help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3,
                       help="Relative tolerance")
    parser.add_argument("--topk", type=int, default=50,
                       help="Print the number of differing lines")
    parser.add_argument("--mode", type=int, default=0,
                       help="mode 0 indicates the pass comparison, and mode 1 indicates that\
                       two rows of data in the CSV file are compared.")
    parser.add_argument("--line", nargs='*', type=int, default=[],
                       help="Enabled when mode 1 is used, indicating the two lines of data to be compared.")
    
    args = parser.parse_args()
    
    if len(args.p) != 2:
        logging.error("The number of input passes is not 2!")
        sys.exit(1)
    
    if len(args.verify_path) == 2:
        verify_path_pass1 = args.verify_path[0]
        verify_path_pass2 = args.verify_path[1]
    elif len(args.verify_path) == 1:
        verify_path_pass1 = args.verify_path[0]
        verify_path_pass2 = args.verify_path[0]
    else:
        logging.error("The verify_path parameter is incorrect !")
        sys.exit(1)
    
    comparator = PassComparator(
        output_pass=args.p[0],
        golden_pass=args.p[1],
        verify_path_pass1=verify_path_pass1,
        verify_path_pass2=verify_path_pass2,
        atol=args.atol,
        rtol=args.rtol,
        topk=args.topk,
        is_sort=args.sort,
        mode=args.mode,
        line=args.line
    )
    
    logging.info(f"pass : {args.p[0]}, {args.p[1]}")
    logging.info(f"raw_tensor_list: {args.raw}")
    logging.info(f"path: {args.func}")
    logging.info(f"verify_path_pass1: {verify_path_pass1}")
    logging.info(f"verify_path_pass2: {verify_path_pass2}")
    
    comparator.pass_compare(
        pass_a=args.p[0],
        pass_b=args.p[1],
        paths=args.func,
        raw_tensor_list=args.raw
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # level：DEBUG < INFO < WARNING < ERROR < CRITICAL
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", encoding="utf-8")
        ]
    )
    
    main()