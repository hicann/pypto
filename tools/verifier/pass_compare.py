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
    result_is_close: bool
    result_reason: str
    file_a: str
    file_b: str
    dtype: str
    shape_a: str
    shape_b: str
    offset_a: str
    offset_b: str
    verify_type: str
    raw_tensor_magic: str
    loop_info: str
    pass_name_a: str
    pass_name_b: str


class PassComparator:
    """Pass comparator class, which encapsulates all comparison logic."""
    
    def __init__(self, 
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
    
    @staticmethod
    def is_contain(a_offset: List[int], b_offset: List[int], 
                   a_shape: List[int], b_shape: List[int]) -> bool:
        """
        Checks whether tensor a is completely included in tensor b.
            Parameters:
            a_offset: offset of tensor a
            b_offset: offset of tensor b
            a_shape: shape of tensor a
            b_shape: shape of tensor b
        Returns:
            bool: True indicates that a is included in b.
        """
        for a_off, b_off, a_sh, b_sh in zip(a_offset, b_offset, a_shape, b_shape):
            if (a_off < b_off) or ((a_off + a_sh) > (b_off + b_sh)):
                return False
        return True

    @staticmethod
    def _build_file_path(data: Dict[str, Any], base_path: str) -> str:
        return os.path.join(base_path, data["passName"], data["outputTensor"])
    
    @staticmethod
    def _log_comparison_info(a: Dict, b: Dict):
        logging.info("------" * 10)
        logging.info(f'functionName : {a["verifyType"]}')
        logging.info(f'rawTensorMagic : {a["rawTensorMagic"]}, path : {a["loopInfo"]}')
        logging.info(f'line : {a["No."]}, a_shape: {a["outputValidShape"]}, '
                    f'offset: {a["tensorOffset"]}, dtype: {a["outputDtype"]}')
        logging.info(f'line : {b["No."]}, b_shape: {b["outputValidShape"]}, '
                    f'offset: {b["tensorOffset"]}, dtype: {b["outputDtype"]}')

    def compare_data(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """
        Compare two data items
            Parameters:
            a: Dictionary of the first data item
            b: Dictionary of the second data item
        Returns:
            bool: True if the data are consistent, False if they are not.
        """

        dtype = a["outputDtype"]
        
        f_a = self._build_file_path(a, self.verify_path_pass1)
        f_b = self._build_file_path(b, self.verify_path_pass2)
        
        if not os.path.exists(f_a) or not os.path.exists(f_b):
            logging.error(f"Some files do not exist; these will be skipped directly.  file name : {f_a} ,{f_b}")
            self.add_comparison_record(
                result_is_close=False,
                result_reason="Files missing",
                a=a, b=b
            )
            return True
        
        a_offset = json.loads(a["tensorOffset"])
        b_offset = json.loads(b["tensorOffset"])
        a_shape = json.loads(a["outputValidShape"])
        b_shape = json.loads(b["outputValidShape"])
        
        np_dtype_a = self.dtype_dict.get(dtype)
        if np_dtype_a is None:
            error_msg = f"Unsupported data types : {dtype}"
            logging.error(error_msg)
            self.add_comparison_record(
                result_is_close=False,
                result_reason=error_msg,
                a=a, b=b
            )
            return False
        
        data_a = np.fromfile(f_a, np_dtype_a)
        data_b = np.fromfile(f_b, np_dtype_a)
        data_b = data_b.reshape(b_shape)
        
        slices = []
        for dim in range(data_b.ndim):
            start = a_offset[dim] - b_offset[dim]
            stop = start + a_shape[dim]
            slices.append(slice(start, stop))
        
        b_slice = data_b[tuple(slices)]
        
        t_dtype_a = self.torch_dtype_dict.get(np_dtype_a)
        
        if dtype == "DT_BF16":
            tensor_a = torch.frombuffer(
                memoryview(data_a.tobytes()), 
                dtype=t_dtype_a
            ).reshape(a_shape)
            tensor_b = torch.frombuffer(
                memoryview(b_slice.tobytes()), 
                dtype=t_dtype_a
            ).reshape(a_shape)
        else:
            tensor_a = torch.from_numpy(data_a).to(dtype=t_dtype_a)
            tensor_b = torch.from_numpy(b_slice).to(dtype=t_dtype_a)
        
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
        
        self._log_comparison_info(a, b)

        self.add_comparison_record(
                result_is_close=result_is_close,
                result_reason=result_reason_str,
                a=a, b=b
            )
        
        if not result_is_close:
            comparator.print_isclose_info(result_is_close, result_reason_str, result_info)
            logging.error("Data comparison failed.")
            analyzer = DataDiffAnalyzer()
            analyzer.fix_input_and_compute(data_a, b_slice, [data_a.dtype, b_slice.dtype], self.is_sort)
            return False
        
        logging.info("Data comparison succeeded.")
        return True
    
    def loop_compare(self, pass_a: str, pass_b: str, df_loop, 
                    raw_tensor_list: List[int] = None) -> bool:
        """
        Compares all data within a loop.
            Parameters:
            pass_a: name of the first pass
            pass_b: name of the second pass
            df_loop: DataFrame containing loop data
            raw_tensor_list: list of raw tensors to be compared
        Returns:
            bool: whether all comparisons are passed.
        """
        df_a = df_loop[df_loop["passName"].str.contains(pass_a)]
        df_b = df_loop[df_loop["passName"].str.contains(pass_b)]
        
        values_a = df_a["rawTensorMagic"].dropna().unique()
        values_b = df_b["rawTensorMagic"].dropna().unique()
        
        common_values_list = list(set(values_a) & set(values_b))
        if len(raw_tensor_list) != 0:
            common_values_list = raw_tensor_list
        
        for raw_magic in common_values_list:
            a_records = df_a[df_a["rawTensorMagic"] == raw_magic].to_dict(orient='records')
            b_records = df_b[df_b["rawTensorMagic"] == raw_magic].to_dict(orient='records')
            
            if len(a_records) < len(b_records):
                a_records, b_records = b_records, a_records
            
            for ai in a_records:
                for bi in b_records:
                    a_offset = json.loads(ai["tensorOffset"])
                    b_offset = json.loads(bi["tensorOffset"])
                    a_shape = json.loads(ai["outputValidShape"])
                    b_shape = json.loads(bi["outputValidShape"])
                    
                    if self.is_contain(a_offset, b_offset, a_shape, b_shape):
                        is_right = self.compare_data(ai, bi)
                        if not is_right:
                            return False
        
        return True

    def line_compare(self, df: Dict[str, Any], line: List[str] = None) -> None:
        if len(line) < 2:
            logging.error(f'line size < 2 : {len(line)}')
        a = df[df["No."] == line[0]].to_dict(orient='records')
        b = df[df["No."] == line[1]].to_dict(orient='records')
        a_offset = json.loads(a[0]["tensorOffset"])
        b_offset = json.loads(b[0]["tensorOffset"])
        a_shape = json.loads(a[0]["outputValidShape"])
        b_shape = json.loads(b[0]["outputValidShape"])
        
        if self.is_contain(a_offset, b_offset, a_shape, b_shape):
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
        csv_path = os.path.join(self.verify_path_pass1, "verify_result.csv")
        df = pd.read_csv(csv_path, encoding="utf-8", 
                        na_values=["", " ", "NaN", "NA"])
        
        # mode = 1: Compare two rows of data.
        if self.mode == 1:
            self.line_compare(df, self.line)
            return
        
        df_pass = df[df["passName"].str.contains(f'{pass_a}|{pass_b}', 
                                                 na=False, regex=True)]
        
        if paths == []:
            paths = df_pass["verifyType"].dropna().unique()
        
        for path in paths:
            df_path = df_pass[df_pass["verifyType"] == path]
            loop_info_list = df_path["loopInfo"].dropna().unique()
            
            for loop_info in loop_info_list:
                df_loop = df_path[df_path["loopInfo"] == loop_info]
                res = self.loop_compare(pass_a, pass_b, df_loop, raw_tensor_list)
                if not res:
                    logging.error(f"failed: pass={pass_a}/{pass_b}, "
                                 f"path={path}, loop={loop_info}")
                    self.save_comparison_results()
                    return

    def add_comparison_record(self, 
                               result_is_close: bool,
                               result_reason: str,
                               a: Dict[str, Any],
                               b: Dict[str, Any]):
        """Add the comparison record to the internal list"""
        
        record = ComparisonRecord(
            result_is_close=result_is_close,
            result_reason=result_reason,
            file_a=a["outputTensor"],
            file_b=b["outputTensor"],
            dtype=a["outputDtype"],
            shape_a=a["outputValidShape"],
            shape_b=b["outputValidShape"],
            offset_a=a["tensorOffset"],
            offset_b=b["tensorOffset"],
            verify_type=a["verifyType"],
            raw_tensor_magic=str(a["rawTensorMagic"]),
            loop_info=a["loopInfo"],
            pass_name_a=a["passName"],
            pass_name_b=b["passName"],
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
            csv_path = os.path.join(self.verify_path_pass1, "comparison_results.csv")
        
        # Converts records to a list of dictionaries.
        records_dict = [asdict(record) for record in self.comparison_records]
        
        df = pd.DataFrame(records_dict)
        
        column_order = [
            'pass_name_a',
            'pass_name_b',
            'verify_type',
            'loop_info',
            'raw_tensor_magic',
            'dtype',
            'shape_a',
            'offset_a',
            'shape_b',
            'offset_b',
            'file_a',
            'file_b',
            'result_is_close',
            'result_reason',
        ]
        
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"Comparison results saved to {csv_path}")


def main():
    """Main function: Parse parameters and run the comparison"""
    parser = argparse.ArgumentParser(
        description="Pass Compare",
        epilog="example:  python3 pass_compare.py --p Exp Rem --verify_path ..."
    )
    
    parser.add_argument("--p", nargs='*', type=str, default=[], required=True,
                       help="Names of the two passes to be compared, separated by a space.")
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