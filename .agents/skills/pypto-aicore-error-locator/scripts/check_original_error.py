#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    has_error,
    run_test,
    validate_path,
    setup_logging,
    print_error_info
)

setup_logging()

logger = logging.getLogger(__name__)


def check_original_error(cce_file, test_cmd, run_dir):
    logger.info("检查原始文件是否有 error...")
    logger.info(f"CCE 文件: {cce_file}")
    logger.info(f"测试命令: {test_cmd}")
    logger.info(f"运行目录: {run_dir}")
    logger.info("")
    
    returncode, output = run_test(test_cmd, run_dir)
    error_exists = has_error(returncode, output)
    
    if error_exists:
        logger.info("结果: 原始文件运行有 error")
        print_error_info(output, logger)
        return True
    else:
        logger.info("结果: 原始文件运行无 error")
        return False


def print_usage():
    logger.info("用法: python3 check_original_error.py <cce_file> <test_cmd> <run_dir>")
    logger.info("")
    logger.info("参数说明:")
    logger.info("  cce_file: CCE 文件路径")
    logger.info("  test_cmd: 触发 aicore error 的测试命令")
    logger.info("  run_dir: 运行测试命令的目录路径")


def main():
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)
    
    cce_file = sys.argv[1]
    test_cmd = sys.argv[2]
    run_dir = sys.argv[3]
    
    cce_file = os.path.abspath(cce_file)
    run_dir = os.path.abspath(run_dir)
    
    valid, error_msg = validate_path(cce_file, "CCE 文件")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)
    
    valid, error_msg = validate_path(run_dir, "运行目录")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)
    
    result = check_original_error(cce_file, test_cmd, run_dir)
    
    if result:
        logger.info("HAS_ERROR")
    else:
        logger.info("NO_ERROR")


if __name__ == "__main__":
    main()
