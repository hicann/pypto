#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    read_file,
    get_commentable_lines,
    comment_special_lines,
    validate_path,
    setup_logging
)

setup_logging()

logger = logging.getLogger(__name__)


def get_commentable_range(cce_file, error_in_t):
    logger.info("获取可注释行范围...")
    logger.info(f"CCE 文件: {cce_file}")
    logger.info(f"error_in_t: {error_in_t}")
    logger.info("")

    cce_lines = read_file(cce_file)
    cce_lines = comment_special_lines(cce_lines)
    commentable_lines = get_commentable_lines(cce_lines, error_in_t)
    
    n = len(commentable_lines)
    logger.info(f"可注释的行数: {n}")
    
    if n <= 0:
        logger.info("错误：没有可注释的行")
        return None, None
    
    left = 0
    right = n - 1
    logger.info(f"初始范围: left={left}, right={right}")
    
    return left, right


def print_usage():
    logger.info("用法: python3 get_commentable_range.py <cce_file> <error_in_t>")
    logger.info("")
    logger.info("参数说明:")
    logger.info("  cce_file: CCE 文件路径")
    logger.info("  error_in_t: 错误是否在T操作中 (True/False)")
    logger.info("")
    logger.info("输出格式:")
    logger.info("  LEFT <left>")
    logger.info("  RIGHT <right>")


def main():
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    cce_file = sys.argv[1]
    error_in_t = sys.argv[2].lower() == 'true'
    cce_file = os.path.abspath(cce_file)

    valid, error_msg = validate_path(cce_file, "CCE 文件")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)

    left, right = get_commentable_range(cce_file, error_in_t)
    
    if left is not None:
        logger.info(f"LEFT {left}")
        logger.info(f"RIGHT {right}")


if __name__ == "__main__":
    main()
