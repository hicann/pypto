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
"""Generate gcov_config.json configuration file

Called by CMake in POST_BUILD phase, collects all filter directories and generates JSON configuration file.
The generated configuration file is used externally during Python case coverage generation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List


class GenCoverageConfig:
    """Controller class for generating gcov configuration file"""

    class FilterPathAction(argparse.Action):
        """Custom Action: validate and format paths when parsing filter arguments"""

        def __call__(self, parser, namespace, values, option_string=None):
            # Get the currently collected list (initially None)
            cur_values = getattr(namespace, self.dest, None) or []
            # Process multiple paths separated by semicolons
            # (in VERBATIM mode, generator expressions expand to semicolon-separated strings)
            for path_str in values.split(';'):
                path_str = path_str.strip()
                if not path_str:
                    continue
                path = Path(path_str)
                cur_values.append(str(path))
            # Update the namespace value
            setattr(namespace, self.dest, cur_values)

    def __init__(self, args):
        """Initialize controller instance"""
        self.binary_dir: Path = Path(args.binary_dir).resolve()
        self.filter_lst: List[str] = args.filter

    def __str__(self) -> str:
        """Return configuration info string"""
        desc = "\nGenCoverageConfig"
        desc += f"\n    BinaryDir    : {self.binary_dir}"
        desc += f"\n    FilterDirs   : {self.filter_lst}"
        desc += "\n"
        return desc

    @classmethod
    def main(cls):
        """Main entry function"""
        # Register arguments
        parser = argparse.ArgumentParser(description="Generate Coverage Config")
        parser.add_argument(
            "-d", "--binary_dir", required=True, type=Path, help="CMake binary directory (PTO_FWK_BIN_ROOT)"
        )
        parser.add_argument(
            "-f",
            "--filter",
            required=False,
            action=cls.FilterPathAction,
            type=str,
            help="Specify filter file/dir in coverage info.",
        )
        # Process arguments
        ctrl = cls(args=parser.parse_args())
        logging.info("%s", ctrl)
        # Process workflow
        ctrl.process()

    def process(self):
        """Generate configuration file"""
        # Build configuration content
        config = {
            'cmake_binary_dir': str(self.binary_dir),
            'filter_dirs': [str(p) for p in self.filter_lst],
        }

        # Write configuration file (overwrite)
        config_file = self.binary_dir / 'gcov_config.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        logging.info("Generated gcov_config.json: %s", config_file)
        logging.info("  filter_lst: %s", self.filter_lst)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    GenCoverageConfig.main()
