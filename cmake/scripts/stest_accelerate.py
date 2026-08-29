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
"""STest case parallel execution."""

import argparse
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from accelerate.tests_accelerate import TestsAccelerate


class STestAccelerate(TestsAccelerate):
    """STest execution acceleration

    Accelerate STest execution efficiency through multi-process parallel execution.
    """

    def __init__(self, args: argparse.Namespace, scene_mark: str = "STest", cntr_name: str = "Device"):
        """
        :param args: command line arguments
        :param scene_mark: scene identifier
        :param cntr_name: container name
        """
        # Before calling parent init, get meta info from binary file and reorder case list
        # Binary file path is passed via -t/--target argument, stored in args.target[0]
        # But need to check if args.target exists since it is a required=True argument
        binary_path = None
        if hasattr(args, 'target') and args.target and len(args.target) > 0:
            binary_path = args.target[0]
        elif hasattr(args, 'exe') and hasattr(args.exe, 'file'):
            # If args.target does not exist, try to get from other sources
            binary_path = args.exe.file

        if args.cases and binary_path:
            # Try to get meta info from binary file and reorder
            reordered_cases = self._reorder_cases_with_binary_meta(args.cases, binary_path)
            # Modify args.cases so parent init uses the reordered case list
            args.cases = reordered_cases
        elif args.cases and not binary_path:
            logging.warning("Binary path not found, skipping meta-based reordering")

        # Call parent init
        super().__init__(args, scene_mark=scene_mark, cntr_name=cntr_name)

        self.device_list: List[int] = self._init_get_device_list(args=args)

    @staticmethod
    def reg_args(parser: argparse.ArgumentParser) -> None:
        """Register STest accelerator arguments

        First call parent (TestsAccelerate) argument registration, then add STest-specific arguments
        """
        TestsAccelerate.reg_args(parser)
        parser.add_argument(
            "-d",
            "--device",
            nargs="?",
            type=int,
            action="append",
            help="Specific parallel accelerate device, "
            "If this parameter is not specified, 0 device will be used by default.",
        )

    @staticmethod
    def main() -> bool:
        """Main processing flow"""
        # Register arguments
        parser = argparse.ArgumentParser(description="STest Execute Accelerate", epilog="Best Regards!")
        STestAccelerate.reg_args(parser=parser)
        # Process workflow
        args = parser.parse_args()
        ctrl = STestAccelerate(args=args)
        ctrl.prepare()
        ctrl.process()
        return ctrl.post()

    @staticmethod
    def get_case_exec_update_envs(p: Any) -> Optional[Dict[str, str]]:
        self = p
        return {"TILE_FWK_DEVICE_ID": f"{self.cntr_id}"}

    @staticmethod
    def _init_get_device_list(args) -> List[int]:
        device_list = [0]
        if args.device is not None:
            device_list = [int(d) for d in list(set(args.device)) if d is not None and str(d) != ""]
        return device_list

    @staticmethod
    def _get_test_costs(binary: str) -> Dict[str, float]:
        """
        Get all test cases with cost info (via custom argument --gtest_list_tests_with_meta)
        Return format: { "TestCaseName.TestName": cost_seconds, ... }
        """
        cost_map = {}
        if not binary or not os.path.exists(binary):
            logging.warning("Binary file not found: %s", binary)
            return cost_map

        try:
            result = subprocess.run(
                [binary, '--gtest_list_tests_with_meta'], capture_output=True, text=True, encoding='utf-8'
            )
            if result.returncode != 0:
                logging.warning("Failed to get test costs from binary %s: %s", binary, result.stderr)
                return cost_map

            # Only parse stdout (format: TestCaseName.TestName|cost_seconds)
            pattern = re.compile(r'^([\w.]+)\|(\d+\.?\d*)$', re.MULTILINE)
            matches = pattern.findall(result.stdout)
            for test_name, cost_str in matches:
                try:
                    cost_map[test_name.strip()] = float(cost_str.strip())
                except ValueError:
                    continue
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logging.warning("Failed to run binary %s to get meta info: %s", binary, e)
        return cost_map

    @staticmethod
    def _reorder_cases_with_binary_meta(cases: List[str], binary: str) -> List[str]:
        """
        Reorder stest cases based on binary meta cost:
          - Cases with cost info are placed first, sorted by cost in descending order
          - Cases without cost info are placed after, keeping original order
        """
        if not cases or not binary:
            return cases

        cost_map = STestAccelerate._get_test_costs(binary)
        if not cost_map:
            # No cost info obtained, keep original order
            logging.debug("No cost meta found for %s, keep original cases order", binary)
            return cases

        cost_cases: List[str] = []
        no_cost_cases: List[str] = []
        for cs in cases:
            if cs in cost_map:
                cost_cases.append(cs)
            else:
                no_cost_cases.append(cs)

        # Sort cases with cost info by cost in descending order
        cost_cases_sorted = sorted(cost_cases, key=lambda x: cost_map[x], reverse=True)

        logging.info(
            "STest(meta): Found %d tests with cost info, %d tests without.", len(cost_cases_sorted), len(no_cost_cases)
        )
        if cost_cases_sorted:
            logging.info("STest(meta): First few cost-aware tests(desc): %s", cost_cases_sorted[:5])

        return cost_cases_sorted + no_cost_cases

    def _prepare_get_params(self) -> List[TestsAccelerate.ExecParam]:
        params = []
        for _id in self.device_list:
            p = TestsAccelerate.ExecParam(cntr_id=_id, envs_func=STestAccelerate.get_case_exec_update_envs)
            params.append(p)
        return params


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - PID[%(process)d] - %(levelname)s: %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    exit(0 if STestAccelerate.main() else 1)
