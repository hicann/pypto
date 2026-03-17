#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import shutil
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    read_file,
    write_file,
    get_commentable_lines,
    comment_lines,
    has_error,
    run_test,
    comment_special_lines,
    validate_path,
    setup_logging,
    print_error_info
)

setup_logging()

logger = logging.getLogger(__name__)


def test_cce_file(cce_file, test_cmd, run_dir):
    logger.info(f"测试 CCE 文件: {cce_file}")
    
    backup_file = cce_file + ".bak"
    shutil.copy(cce_file, backup_file)
    cce_lines = read_file(cce_file)
    original_lines = cce_lines.copy()
    
    cce_lines = comment_special_lines(cce_lines)
    
    commentable_lines = get_commentable_lines(cce_lines)
    logger.info(f"可注释的行数: {len(commentable_lines)}")
    
    if not commentable_lines:
        logger.info("错误：没有可注释的行")
        write_file(cce_file, original_lines)
        os.remove(backup_file)
        return False, None
    
    logger.info("注释所有可注释的行...")
    current_lines = cce_lines.copy()
    current_lines = comment_lines(current_lines, commentable_lines)
    
    write_file(cce_file, current_lines)
    logger.info("运行测试...")
    returncode, output = run_test(test_cmd, run_dir)
    error_exists = has_error(returncode, output)
    
    write_file(cce_file, original_lines)
    os.remove(backup_file)
    
    if error_exists:
        print_error_info(output, logger)
        logger.info("结果: 注释所有行后仍有 error，此文件可能不是问题文件")
        return False, None
    else:
        logger.info("结果: 注释所有行后运行成功（无 error），此文件可能是问题文件")
        return True, commentable_lines


def print_usage():
    logger.info("用法: python3 test_cce_file.py <cce_file> <test_cmd> <run_dir>")
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
    
    is_problem_file, _ = test_cce_file(cce_file, test_cmd, run_dir)
    
    if is_problem_file:
        logger.info("\n此文件可能是问题文件")
        logger.info(cce_file)
    else:
        logger.info("\n此文件不是问题文件")


if __name__ == "__main__":
    main()
