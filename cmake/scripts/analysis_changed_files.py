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
"""Analyze changed file list.

Analyze changed file list to determine whether the current test scenario needs to be executed
and to obtain the case execution scope.
"""

import argparse
import dataclasses
import fnmatch
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclasses.dataclass
class Module:
    name: str
    cases: List[str]
    write: List[Path]

    @staticmethod
    def _relative_to(s: Path, d: Path) -> bool:
        try:
            if s.relative_to(d):
                return True
        except ValueError:
            pass
        return False

    def is_trigger(self, changed: List[Path]) -> Tuple[bool, List[str]]:
        # If no changed files, trigger all cases by default
        if not changed:
            return True, self.cases
        # If all changed files hit the whitelist, no need to trigger
        for c in changed:
            c_skip = False
            for w in self.write:
                if self._relative_to(c, w):
                    c_skip = True
                    logging.debug("Changed(%s) hit writeList(%s), skip Module(%s)", c, w, self.name)
                    break
            if not c_skip:
                logging.debug("Changed(%s) not hit writeList, trigger Module(%s)", c, self.name)
                return True, self.cases
        return False, []


class Analysis:
    _KEY_WRITE_LIST: str = "write_list"
    _KEY_CASES: str = "cases"

    def __init__(self, args):
        self.rule: Path = Path(args.rule[0]).resolve()
        self.type: str = str(args.type[0]).lower()
        self.group: List[str] = args.group.split(",") if args.group else []
        self.file: Optional[Path] = Path(args.file[0]).resolve() if args.file and args.file[0] else None
        # Internal object conversion
        self.modules: Dict[str, Module] = self._init_get_models()
        self.changed: List[Path] = self._init_get_changed()

    def __str__(self) -> str:
        ver = sys.version_info
        desc = f"\nPython3 : {sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"
        desc += f"\nRule    : {self.rule}"
        desc += f"\nType    : {self.type}"
        desc += f"\nGroup   : {self.group}"
        desc += f"\nFile    : {self.file}"
        desc += "\n"
        return desc

    @staticmethod
    def main() -> str:
        parser = argparse.ArgumentParser(description="Analysis Changed Files", epilog="Best Regards!")
        parser.add_argument("-r", "--rule", required=True, nargs=1, type=Path, help="Specific classify_rule.yaml")
        parser.add_argument(
            "-t", "--type", nargs=1, type=str, required=True, choices=["utest", "stest"], help="Specific tests type"
        )
        parser.add_argument(
            "-g",
            "--group",
            nargs='?',
            type=str,
            required=False,
            default="",
            help="Specific tests group, multiple group are separated by ','",
        )
        parser.add_argument(
            "-c", "--changed_files", nargs=1, type=Path, required=False, dest="file", help="Specific changed_files.txt"
        )
        parser.add_argument("-d", "--debug", action="store_true", default=False, help="Enable debug mode")
        args = parser.parse_args()
        # Log level registration, this file has two invocation scenarios:
        # 1) Called by CMake, in which case normal processing should have no extra output, set log level to ERROR;
        # 2) Called directly by Python for debugging, in which case more logs are needed, set log level to DEBUG;
        logging.basicConfig(
            format='%(asctime)s - %(filename)s:%(lineno)d - PID[%(process)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG if args.debug else logging.ERROR,
            handlers=[logging.StreamHandler()],
        )
        # Parse arguments
        ctrl = Analysis(args=args)
        logging.info(ctrl)
        # Process workflow
        return ctrl.analysis()

    def analysis(self) -> str:
        cases = self._analysis_cases()
        cases_str = ",".join(cases) if cases else ""
        return cases_str

    def _get_write_list(self, _desc: Dict[str, Any]) -> List[Path]:
        _lst = _desc.get(self._KEY_WRITE_LIST, [])
        _lst = _lst if _lst else []
        _rst = [Path(_rel) for _rel in _lst]
        _desc.pop(self._KEY_WRITE_LIST, None)
        return _rst

    def _init_get_models_from_file(self, file: Path, write_list: List[Path] = None) -> Dict[str, Module]:
        modules = {}
        with open(file, 'r', encoding='utf-8') as f:
            rule_dict = yaml.safe_load(f)
        rule_dict = rule_dict.get(self.type, {})
        # Process whitelist under type
        type_write_list = self._get_write_list(_desc=rule_dict)
        type_write_list = write_list if write_list else type_write_list
        # Iterate over modules
        for name, desc in rule_dict.items():
            # Process whitelist under module
            write_list = self._get_write_list(_desc=desc)
            write_list.extend(type_write_list)
            # Get case list under module
            cases_list = desc.get(self._KEY_CASES, [])
            mod = Module(name=name, cases=cases_list, write=write_list)
            modules[name] = mod
        return modules

    def _init_get_models(self) -> Dict[str, Module]:
        yaml_lst = self.rule.glob(pattern="classify_rule_*.yaml")
        modules = {}
        rule_file = self.rule.joinpath(f"classify_rule_{self.type}.yaml")
        with open(rule_file, 'r', encoding='utf-8') as f:
            rule_dict = yaml.safe_load(f)
        write_list = self._get_write_list(_desc=rule_dict.get(self.type, {}))
        for file in yaml_lst:
            file_module = self._init_get_models_from_file(file=file, write_list=write_list)
            modules.update(file_module)
        return modules

    def _init_get_changed(self) -> List[Path]:
        changed = []
        if self.file:
            with open(self.file, 'r', encoding='utf-8') as f:
                changed = [Path(line.rstrip('\n')) for line in f]
        return changed

    def _analysis_cases(self) -> List[str]:
        cases = []
        for module in self.modules.values():
            match_group = False if self.group else True
            for group in self.group:
                # Support fuzzy matching for group names
                if fnmatch.fnmatch(module.name, group):
                    match_group = True
                    break
            if not match_group:
                logging.debug("Module(%s) not match group %s", module.name, self.group)
                continue
            logging.debug("Module(%s) match group %s", module.name, self.group)
            trigger, module_cases = module.is_trigger(changed=self.changed)
            if not trigger:
                continue
            if module_cases is not None:
                cases.extend(module_cases)
        return cases


if __name__ == "__main__":
    print(Analysis.main(), end='')
