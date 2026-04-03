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
import sys
import json
import logging
import time
import traceback
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from multiprocessing import Pool, cpu_count
import torch
import ml_dtypes
import pandas as pd
import numpy as np
from tensor_diff import TensorComparator, IsCloseConfig
from run_float_diff import DataDiffAnalyzer
MAX_PRECISION = sys.float_info.dig + 1


@dataclass
class ProcessLoopBatchArgs:
    loop_tasks: List[Dict[str, Any]]
    pass_a: str
    pass_b: str
    verify_path_pass1: str
    verify_path_pass2: str
    atol: float
    rtol: float
    topk: int
    key: str
    is_codegen: bool
    csv_data_dict: Dict[str, Any]
    result_file: str


class PassComparator:
    """Pass comparator class, which encapsulates all comparison logic."""

    def __init__(self, 
                 output_pass: str = "",
                 golden_pass: str = "",
                 verify_path_pass1: str = "",
                 verify_path_pass2: str = "",
                 atol: float = 1e-3,
                 rtol: float = 1e-3,
                 topk: int = 1000):
        """
        Initializing the comparator
        Parameters:
            verify_path_pass1: Verification file path for the first pass
            verify_path_pass2: Verification file path for the second pass
            atol: Absolute tolerance
            rtol: Relative tolerance
            topk: Print the first k differences
        """
        self.verify_path_pass1 = verify_path_pass1
        self.verify_path_pass2 = verify_path_pass2
        self.atol = atol
        self.rtol = rtol
        self.topk = topk
        self.key = ":rawmagic"
        self.output_pass = output_pass
        self.golden_pass = golden_pass
        self.result_file = ""
        self.row_num = 1
        self.comparison_records: List[Dict[str, Any]] = []
        self.dtype_dict = {
            "BF16": ml_dtypes.bfloat16,
            "FP32": np.float32,
            "FP16": np.float16,
            "INT32": np.int32,
            "INT8": np.int8,
            "INT64": np.int64,
            "INT16": np.int16
        }

        self.pass_dict = {
            "tensor_graph": 0,
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
        a_offset = json.loads(a[":offset"])
        b_offset = json.loads(b[":offset"])
        a_shape = json.loads(a[":validshape"])
        b_shape = json.loads(b[":validshape"])
        if a[":opcode"] in {"ASSEMBLE", "COPY_OUT", "COPY_IN"}:
            return a["OP_ATTR_SYM_OFFSET"] == b["OP_ATTR_SYM_OFFSET"] and a[":opcode"] == b[":opcode"]
        if key == ":magic" and a_shape == b_shape:
            return True
        else:
            for a_off, b_off, a_sh, b_sh in zip(a_offset, b_offset, a_shape, b_shape):
                if (a_off < b_off) or ((a_off + a_sh) > (b_off + b_sh)):
                    return False
        if key == "ROOT_CALL:rawmagic":
            return a[":opcode"] == b[":opcode"]
        return True

    @staticmethod
    def opcode_match(opcode_a: str, opcode_b: str, opcode_dict: Dict) -> bool:
        """opcode match"""
        if opcode_a in opcode_dict:
            return opcode_b in opcode_dict[opcode_a]
        if opcode_b in opcode_dict:
            return opcode_a in opcode_dict[opcode_b]
        return False

    @staticmethod
    def add_comparison_record(result_is_close: str,
                            result_reason: str,
                            a: Dict[str, Any],
                            rtol: float,
                            atol: float,
                            b: Optional[Dict[str, Any]] = None,
                            diff_conf: Optional[Tuple] = None):
        """Add the comparison record to the internal list"""

        record = {}
        record["NO."] = 1
        record["PATH_FUNC:func_magicname "] = a["PATH_FUNC:func_magicname"]
        record["PATH_FUNC:funcmagic"] = a["PATH_FUNC:funcmagic"]
        record["PATH_FUNC:hash"] = a["PATH_FUNC:hash"]
        record["LOOP_INFO"] = a["LOOP_INFO"]
        record[":symbol"] = a[":symbol"]
        record[":validshape"] = a[":validshape"]
        record[":datatype"] = a[":datatype"]
        record["OP_ATTR_SYM_OFFSET"] = a["OP_ATTR_SYM_OFFSET"]
        record["OP_IO_FLAG"] = a["OP_IO_FLAG"]
        record["A>PHASE_NAME"] = b["PHASE_NAME"] if b else None
        record["B>PHASE_NAME"] = a["PHASE_NAME"]
        record["A>TIMESTAMP"] = b["TIMESTAMP"] if b else None
        record["B>TIMESTAMP"] = a["TIMESTAMP"]
        record["A>FILENAME"] = b["FILENAME"] if b else None
        record["B>FILENAME"] = a["FILENAME"]
        record["A>FUNC:hash"] = b["FUNC:hash"] if b else None
        record["A>FUNC:funcmagic"] = b["FUNC:funcmagic"] if b else None
        record["B>FUNC:hash"] = a["FUNC:hash"]
        record["B>FUNC:funcmagic"] = a["FUNC:funcmagic"]
        record["A>ROOT_CALL:opmagic"] = b["ROOT_CALL:opmagic"] if b else None
        record["B>ROOT_CALL:opmagic"] = a["ROOT_CALL:opmagic"]
        record["A>ROOT_CALL:rawmagic"] = b["ROOT_CALL:rawmagic"] if b else None
        record["B>ROOT_CALL:rawmagic"] = a["ROOT_CALL:rawmagic"]
        record["A>:opmagic"] = b[":opmagic"] if b else None
        record["B>:opmagic"] = a[":opmagic"]
        record["A>:opcode"] = b[":opcode"] if b else None
        record["B>:opcode"] = a[":opcode"]
        record["A>:rawmagic"] = b[":rawmagic"] if b else None
        record["A>:rawshape"] = b[":rawshape"] if b else None
        record["A>:format"] = b[":format"] if b else None
        record["B>:rawmagic"] = a[":rawmagic"]
        record["B>:rawshape"] = a[":rawshape"]
        record["B>:format"] = a[":format"]
        record["A>:shape"] = b[":shape"] if b else None
        record["B>:shape"] = a[":shape"]
        record["A>EVAL:dynvalidshape"] = b["EVAL:dynvalidshape"] if b else None
        record["B>EVAL:dynvalidshape"] = a["EVAL:dynvalidshape"]
        record["AB>RESULT"] = result_is_close
        record["result_reason"] = result_reason
        record["AB>rtol/atol"] = f"{rtol:.{MAX_PRECISION}g}/{atol:.{MAX_PRECISION}g}"
        if diff_conf is not None:
            brief_conf = diff_conf[0]
            ab_conf = diff_conf[1]
            a_conf = diff_conf[2]
            b_conf = diff_conf[3]
            record["AB>fail_cnt/warn_cnt/tol_cnt"] = f"{brief_conf[4]}/{brief_conf[3]}/{brief_conf[2]}"
            record["AB>total_cnt/zero_cnt/infnan_cnt"] = f"{brief_conf[0]}/{brief_conf[1]}/{brief_conf[5]}"
            record["AB>mae"] = f"{ab_conf[0]:.{MAX_PRECISION}g}"
            record["AB>mae_top8"] = f"{ab_conf[1]:.{MAX_PRECISION}g}"
            record["AB>mae_top1permil"] = f"{ab_conf[2]:.{MAX_PRECISION}g}"
            record["AB>mre"] = f"{ab_conf[3]:.{MAX_PRECISION}g}"
            record["AB>mre_top8"] = f"{ab_conf[4]:.{MAX_PRECISION}g}"
            record["AB>mre_top1permil"] = f"{ab_conf[5]:.{MAX_PRECISION}g}"
            record["A>max"] = f"{b_conf[0]:.{MAX_PRECISION}g}"
            record["A>min"] = f"{b_conf[1]:.{MAX_PRECISION}g}"
            record["A>avg"] = f"{b_conf[2]:.{MAX_PRECISION}g}"
            record["A>aavg"] = f"{b_conf[3]:.{MAX_PRECISION}g}"
            record["A>zero"] = f"{b_conf[4]:.{MAX_PRECISION}g}"
            record["A>infnan"] = f"{b_conf[5]:.{MAX_PRECISION}g}"
            record["B>max"] = f"{a_conf[0]:.{MAX_PRECISION}g}"
            record["B>min"] = f"{a_conf[1]:.{MAX_PRECISION}g}"
            record["B>avg"] = f"{a_conf[2]:.{MAX_PRECISION}g}"
            record["B>aavg"] = f"{a_conf[3]:.{MAX_PRECISION}g}"
            record["B>zero"] = f"{a_conf[4]:.{MAX_PRECISION}g}"
            record["B>infnan"] = f"{a_conf[5]:.{MAX_PRECISION}g}"
        return record

    @staticmethod
    def _process_loop_batch(args: ProcessLoopBatchArgs) -> List[Dict[str, Any]]:
        """
        Static method to process a batch of loops in multiprocessing context.
        Returns:
            List of comparison records
        """
        loop_tasks = args.loop_tasks
        pass_a = args.pass_a
        pass_b = args.pass_b
        verify_path_pass1 = args.verify_path_pass1
        verify_path_pass2 = args.verify_path_pass2
        atol = args.atol
        rtol = args.rtol
        topk = args.topk
        key = args.key
        is_codegen = args.is_codegen
        csv_data_dict = args.csv_data_dict
        result_file = args.result_file

        dtype_dict = {
            "BF16": ml_dtypes.bfloat16,
            "FP32": np.float32,
            "FP16": np.float16,
            "INT32": np.int32,
            "INT8": np.int8,
            "INT64": np.int64,
            "INT16": np.int16
        }

        opcode_dict = {
            "VIEW": ["L1_TO_L0A", "L1_TO_L0B"],
            "A_MUL_B": ["A_MULACC_B"]
        }

        comparison_records = []

        for task in loop_tasks:
            df_loop = task['df_loop']
            df_a = df_loop[df_loop["PHASE_NAME"].str.contains(pass_a)]
            df_b = df_loop[df_loop["PHASE_NAME"].str.contains(pass_b)]

            if is_codegen:
                a_copy = df_a[df_a[":opcode"].isin(['COPY_IN', 'COPY_OUT'])]
                a_dict = a_copy[a_copy["ROOT_CALL:rawmagic"].notna()].to_dict(orient='records')
            else:
                a_dict = df_a.to_dict(orient='records')

            for ai in a_dict:
                raw_magic = ai[key]
                if is_codegen:
                    b_records = df_b[df_b[":rawmagic"] == raw_magic].to_dict(orient='records')
                    if ai[":opcode"] == "COPY_IN":
                        b_records = df_b[
                            (df_b[":inputRawMagic"] == str(int(ai[key]))) &
                            (df_b[":opcode"] == "COPY_IN")
                        ].to_dict(orient='records')
                else:
                    b_records = df_b[df_b[key] == raw_magic].to_dict(orient='records')

                if len(b_records) == 0:
                    result_reason = f"{key} : {raw_magic}, not exit in golden pass"
                    record = PassComparator.add_comparison_record(
                        result_is_close="Skip",
                        result_reason=result_reason,
                        a=ai,
                        rtol=rtol,
                        atol=atol
                    )
                    comparison_records.append(record)
                    continue

                is_match = False
                for bi in b_records:
                    if not PassComparator._compare_not_support_static(ai, bi, key,
                                        verify_path_pass1, verify_path_pass2, dtype_dict, opcode_dict):
                        continue

                    compare_result = PassComparator._compare_data_static(
                        ai, bi, verify_path_pass1, verify_path_pass2, atol, rtol, topk,
                        csv_data_dict, dtype_dict, key, result_file
                    )
                    record = PassComparator.add_comparison_record(
                        result_is_close=compare_result["result_is_close"],
                        result_reason=compare_result["result_reason"],
                        a=ai,
                        rtol=rtol,
                        atol=atol,
                        b=bi,
                        diff_conf=compare_result["diff_conf"]
                    )
                    comparison_records.append(record)
                    is_match = True
                    break

                if not is_match:
                    record = PassComparator.add_comparison_record(
                        result_is_close="Skip",
                        result_reason="not match",
                        a=ai,
                        rtol=rtol,
                        atol=atol
                    )
                    comparison_records.append(record)

        return comparison_records

    @staticmethod
    def _compare_not_support_static(a: Dict[str, Any], b: Dict[str, Any], key: str,
                                    verify_path_pass1: str, verify_path_pass2: str,
                                    dtype_dict: Dict, opcode_dict: Dict) -> bool:
        """Static version of compare_not_support for multiprocessing"""
        if not PassComparator.is_contain(a, b, key):
            return False

        f_a = os.path.join(verify_path_pass1, a["PHASE_NAME"], a["FILENAME"])
        f_b = os.path.join(verify_path_pass2, b["PHASE_NAME"], b["FILENAME"])
        if not os.path.exists(f_a) or not os.path.exists(f_b):
            return False

        opcode_a = a[":opcode"]
        opcode_b = b[":opcode"]
        if (key == ":rawmagic" and opcode_a != opcode_b and
            not PassComparator.opcode_match(opcode_a, opcode_b, opcode_dict)):
            return False

        if a[":datatype"] != b[":datatype"]:
            return False

        dtype = a[":datatype"]
        if dtype not in dtype_dict:
            return False

        return True

    @staticmethod
    def _compare_data_static(a: Dict[str, Any], b: Dict[str, Any],
                             verify_path_pass1: str, verify_path_pass2: str,
                             atol: float, rtol: float, topk: int,
                             csv_data_dict: Dict, dtype_dict: Dict,
                             key: str, result_file: str) -> Dict[str, Any]:
        """Static version of compare_data for multiprocessing"""
        data_a, data_b = PassComparator._get_data_slice_static(
            a, b, verify_path_pass1, verify_path_pass2, dtype_dict, csv_data_dict, key
        )

        a_shape = json.loads(a[":validshape"])
        if a[":opcode"] in {"ASSEMBLE", "COPY_OUT"}:
            input_dict = [item for item in csv_data_dict if item["FILENAME"] == a["INPUT_FILENAMES"]]
            if input_dict:
                a_shape = json.loads(input_dict[0][":validshape"])

        tensor_a = torch.from_numpy(data_a.astype(np.float64)).to(torch.float64)
        tensor_b = torch.from_numpy(data_b.astype(np.float64)).to(torch.float64)

        comparator = TensorComparator()
        config = IsCloseConfig(
            rtol=rtol,
            atol=atol,
            calc_dtype=torch.float64, 
            is_detail=True,
            shape=a_shape
        )
        result_is_close, result_reason_str, result_info = comparator.check_isclose(
           tensor_a, tensor_b, config
        )
        if not result_is_close:
            csv_path = os.path.join(verify_path_pass1,
                                    result_file[:-4] + ".DETAIL",
                                    a["FILENAME"][:-5] + ".csv")
            comparator.print_isclose_info(result_is_close, result_reason_str, result_info, csv_path, topk)
        return {
            "result_is_close": result_is_close,
            "result_reason": result_reason_str,
            "diff_conf": result_info[6] if len(result_info) > 6 else None
        }

    @staticmethod
    def _get_data_slice_static(a: Dict[str, Any], b: Dict[str, Any],
                               verify_path_pass1: str, verify_path_pass2: str,
                               dtype_dict: Dict, csv_data_dict: Dict, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Static version of get_data_slice for multiprocessing"""
        f_a = os.path.join(verify_path_pass1, a["PHASE_NAME"], a["FILENAME"])
        f_b = os.path.join(verify_path_pass2, b["PHASE_NAME"], b["FILENAME"])

        if a[":opcode"] in {"ASSEMBLE", "COPY_OUT"}:
            return PassComparator._get_data_slice_assemble_static(
                a, b, verify_path_pass1, verify_path_pass2, dtype_dict, csv_data_dict
            )

        a_offset = json.loads(a[":offset"])
        b_offset = json.loads(b[":offset"])
        a_shape = json.loads(a[":validshape"])
        b_shape = json.loads(b[":validshape"])
        np_dtype = dtype_dict.get(a[":datatype"])
        data_a = np.fromfile(f_a, np_dtype)
        data_b = np.fromfile(f_b, np_dtype)

        data_a = data_a.reshape(a_shape)
        data_b = data_b.reshape(b_shape)

        if a_shape == b_shape and key == ":magic":
            return data_a, data_b

        slices = []
        for dim in range(data_a.ndim):
            start = a_offset[dim] - b_offset[dim]
            stop = start + a_shape[dim]
            slices.append(slice(start, stop))
        b_slice = data_b[tuple(slices)]
        return data_a, b_slice

    @staticmethod
    def _get_data_slice_assemble_static(a: Dict[str, Any], b: Dict[str, Any],
                                        verify_path_pass1: str, verify_path_pass2: str,
                                        dtype_dict: Dict, csv_data_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Static version of get_data_slice_assemble for multiprocessing"""
        f_a = os.path.join(verify_path_pass1, a["PHASE_NAME"], a["FILENAME"])
        f_b = os.path.join(verify_path_pass2, b["PHASE_NAME"], b["FILENAME"])

        input_dict = [item for item in csv_data_dict if item["FILENAME"] == a["INPUT_FILENAMES"]]
        a_offset = json.loads(a["OP_ATTR_SYM_OFFSET"])
        b_offset = json.loads(b["OP_ATTR_SYM_OFFSET"])
        shape = json.loads(a[":rawshape"])
        if not input_dict:
            logging.error(f"No matching record found for FILENAME: {a['INPUT_FILENAMES']}")
            raise ValueError(f"No matching record found for FILENAME: {a['INPUT_FILENAMES']}")
        a_shape = json.loads(input_dict[0][":validshape"])
        b_shape = json.loads(input_dict[0][":validshape"])
        np_dtype = dtype_dict.get(a[":datatype"])
        data_a = np.fromfile(f_a, np_dtype)
        data_b = np.fromfile(f_b, np_dtype)

        data_a = data_a.reshape(shape)
        data_b = data_b.reshape(shape)

        slices_a = []
        for dim in range(data_a.ndim):
            start = a_offset[dim]
            stop = start + a_shape[dim]
            slices_a.append(slice(start, stop))
        a_slice = data_a[tuple(slices_a)]

        slices_b = []
        for dim in range(data_b.ndim):
            start = b_offset[dim]
            stop = start + b_shape[dim]
            slices_b.append(slice(start, stop))
        b_slice = data_b[tuple(slices_b)]
        return a_slice, b_slice

    def pass_compare(self, pass_a: str, pass_b: str,
                    paths: List[str] = None) -> None:
        """
        Main comparison function
        """
        csv_path = os.path.join(self.verify_path_pass1, "verify_graph_data_metainfo.csv")
        df = pd.read_csv(csv_path, encoding="utf-8",
                        na_values=["", " ", "NaN", "NA"])
        csv_data_dict = df.to_dict(orient='records')

        if self.pass_dict[pass_a] < self.pass_dict[pass_b]:
            pass_a, pass_b = pass_b, pass_a
            self.output_pass, self.golden_pass = self.golden_pass, self.output_pass
            self.verify_path_pass1, self.verify_path_pass2 = self.verify_path_pass2, self.verify_path_pass1
        self.result_file = f'verify_graph_result_cmp~Pass_{self.pass_dict[self.golden_pass]:02d}_{self.golden_pass}~' \
                f'Pass_{self.pass_dict[self.output_pass]:02d}_{self.output_pass}~{int(time.time() * 1_000_000)}.csv'
        
        if self.pass_dict[pass_a] >= 4 and self.pass_dict[pass_b] >= 4:
            self.key = ":magic"
        # 判断是否需要使用 codegen 的特殊逻辑
        is_codegen = False
        if self.pass_dict[pass_a] >= 28 and self.pass_dict[pass_b] >= 4 and self.pass_dict[pass_b] < 28:
            self.key = "ROOT_CALL:rawmagic"
            is_codegen = True
        logging.info(f"key  : {self.key}")

        df_pass = df[df["PHASE_NAME"].str.contains(f'{pass_a}|{pass_b}',
                                                 na=False, regex=True)]
        if paths == []:
            paths = df_pass["PATH_FUNC:func_magicname"].dropna().unique()

        # 收集所有loop任务
        loop_tasks = []
        for path in paths:
            df_path = df_pass[df_pass["PATH_FUNC:func_magicname"] == path]
            loop_info_list = df_path["LOOP_INFO"].dropna().unique()

            for loop_info in loop_info_list:
                df_loop = df_path[df_path["LOOP_INFO"] == loop_info]
                loop_tasks.append({
                    'df_loop': df_loop,
                    'path': path,
                    'loop_info': loop_info
                })

        # 使用多进程批量处理loop任务
        num_workers = min(min(cpu_count(), len(loop_tasks)), 32)
        logging.info(f"num_workers : {num_workers}")
        avg_tasks = len(loop_tasks) // num_workers
        remainder = len(loop_tasks) % num_workers

        batches = []
        start = 0
        for i in range(num_workers):
            end = start + avg_tasks + (1 if i < remainder else 0)
            batches.append(loop_tasks[start:end])
            start = end

        args_list = [
            ProcessLoopBatchArgs(
                loop_tasks=batch, pass_a=pass_a, pass_b=pass_b, verify_path_pass1=self.verify_path_pass1,
                verify_path_pass2=self.verify_path_pass2, atol=self.atol, rtol=self.rtol, topk=self.topk,
                key=self.key, is_codegen=is_codegen, csv_data_dict=csv_data_dict, result_file=self.result_file
            )
            for batch in batches
        ]

        try:
            with Pool(processes=num_workers) as pool:
                all_results = pool.map(self._process_loop_batch, args_list)

            # 合并结果
            all_records = []
            for batch_results in all_results:
                all_records.extend(batch_results)
            all_records.sort(key=lambda x: x.get("B>TIMESTAMP"))
            for record in all_records:
                record["NO."] = self.row_num
                self.comparison_records.append(record)
                self.row_num += 1

        except Exception as e:
            stack_trace = traceback.format_exc()
            logging.error(f"Exception in multiprocessing: pass={pass_a}/{pass_b}, error={str(e)}\n"
                            f"Stack trace:\n{stack_trace}")
            self.save_comparison_results()
            return
        self.save_comparison_results()

    def save_comparison_results(self, csv_path: str = None):
        """
        Save all comparison results to a CSV file
        """
        if not self.comparison_records:
            logging.warning("No comparison records to save.")
            return

        if csv_path is None:
            csv_path = os.path.join(self.verify_path_pass1, self.result_file)

        df = pd.DataFrame(self.comparison_records)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logging.info(f"Comparison results saved to {csv_path}")


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
    parser.add_argument("--verify_path", nargs='*', type=str, default=[],
                       help="Verify the file directory. If two values are provided, they represent \
                       the paths of two passes respectively.")
    parser.add_argument("--atol", type=float, default=1e-3,
                       help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3,
                       help="Relative tolerance")
    parser.add_argument("--topk", type=int, default=1000,
                       help="Print the number of differing lines")

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
        topk=args.topk
    )

    logging.info(f"pass : {args.p[0]}, {args.p[1]}")
    logging.info(f"path: {args.func}")
    logging.info(f"verify_path_pass1: {verify_path_pass1}")
    logging.info(f"verify_path_pass2: {verify_path_pass2}")

    comparator.pass_compare(
        pass_a=args.p[0],
        pass_b=args.p[1],
        paths=args.func
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # level：DEBUG < INFO < WARNING < ERROR < CRITICAL
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", mode='w', encoding="utf-8")
        ]
    )

    main()