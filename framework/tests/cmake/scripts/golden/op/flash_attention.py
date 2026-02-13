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
"""

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

np.random.seed(0)
# s = 128
# d_q = 128
# d_v = 256

b = 1
n = 1
s = 128
d_q = 128
d_v = 128

dtype_f16 = np.float16
dtype_f32 = np.float32


def softmax(x, axis=None):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum


def softmax_grad(dp, softmax_res):
    muls_r = dp * softmax_res
    muls_r = muls_r.sum(axis=-1, keepdims=True)
    sub_r = dp - muls_r
    muls_r2 = softmax_res
    res = sub_r * muls_r2
    return res


def softmax_grad_flash_1(dx, y):
    d = (dx * y).sum(axis=-1, keepdims=True)  # [s,1]
    return d


def softmax_grad_flash_2(dp, p, d):
    sub = dp - d
    ds = sub * p
    return ds


def softmax_flash(x, max_front=None, sum_front=None, update=None):
    """
    Compute the softmax function for each channel of the input x.
    """
    if update is None:
        x_max = np.max(x, axis=-1, keepdims=True)
        x_sub = x - x_max  # -> x
        # logging.debug("x_sub", x_sub[0:32,:])
        x_exp = np.exp(x_sub)  # -> x
        # logging.debug("x_exp", x_exp[0:32,:])

        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        out = x_exp / x_sum
        exp_max = None

        return out, x_max, x_sum, exp_max
    else:
        x_max_tmp = np.max(x, axis=-1, keepdims=True)  # tmp
        x_sub = x - x_max_tmp  # -> x
        x_exp = np.exp(x_sub)  # -> x
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)  # tmp

        x_max = np.max(np.concatenate((max_front, x_max_tmp), axis=-1), axis=-1, keepdims=True)  # ->x_max
        x_exp_new = np.exp(x_max_tmp - x_max)  # -> x_max_tmp
        exp_max = np.exp(max_front - x_max)  # -> exp_max
        # update sum
        exp_max = exp_max * sum_front
        reduce_tmp = x_exp_new * x_sum
        x_sum = exp_max + reduce_tmp  # x_sum

        exp_max = exp_max / x_sum
        out = x_exp * x_exp_new / x_sum

        # ### softmax ratio
        # softma_ratio = x_sum * x_exp_new / x_sum
        # out_new = out * softma_ratio
        # logging.debug("out_new===================", out_new)
        return out, x_max, x_sum, exp_max


def forward(q, k, v, drop_mask):
    q, k, v = q.astype(dtype_f32), k.astype(dtype_f32), v.astype(dtype_f32)
    # logging.debug("forward ------ q.shape:" , q.shape)

    qk = np.matmul(q, k.transpose(0, 1, 3, 2))
    softmax_res, x_max, x_sum = softmax(qk)
    drop_res = softmax_res * drop_mask
    y = np.matmul(drop_res, v)
    # y = y.astype(dtype_f16)
    return y, softmax_res, x_max, x_sum


def backward(dx, q, k, v, softmax_res, drop_mask):
    dx, q, k, v = dx.astype(dtype_f32), q.astype(dtype_f32), k.astype(dtype_f32), v.astype(dtype_f32)
    drop_res = softmax_res * drop_mask
    dv = np.matmul(drop_res.transpose(1, 0), dx)
    dp = np.matmul(dx, v.transpose(1, 0))
    dp_drop = dp * drop_mask
    softmax_grad_res = softmax_grad(dp_drop, softmax_res)
    # logging.debug("softmax_grad_res.shape=======", softmax_grad_res.shape)
    dq = np.matmul(softmax_grad_res, k)
    dk = np.matmul(softmax_grad_res.transpose(1, 0), q)
    dq, dk, dv = dq.astype(dtype_f16), dk.astype(dtype_f16), dv.astype(dtype_f16)
    return dq, dk, dv


def gen_golden_np(output: Path):
    qkv_shape = [b, n, s, d_q]
    v_shape = [b, n, s, d_v]

    drop_shape = [b, n, s, s]

    # random data
    dx = np.random.uniform(-1, 1, qkv_shape).astype(dtype_f16)
    q = np.random.uniform(0, 0.5, qkv_shape).astype(dtype_f16)
    k = np.random.uniform(0, 0.5, qkv_shape).astype(dtype_f16)
    v = np.random.uniform(0, 0.5, v_shape).astype(dtype_f16)
    drop_mask = np.random.uniform(1, 2, drop_shape).astype(np.uint8)

    dx.tofile(Path(output, 'dx.bin'))
    q.tofile(Path(output, 'q.bin'))
    k.tofile(Path(output, 'k.bin'))
    v.tofile(Path(output, 'v.bin'))
    drop_mask.tofile(Path(output, 'drop_mask.bin'))

    y, softmax_res, x_max, x_sum = forward(q, k, v, drop_mask)
    dq, dk, dv = q, k, v

    return q, k, v, dx, drop_mask, y, dq, dk, dv, softmax_res, x_max, x_sum


@GoldenRegister.reg_golden_func(
    case_names=[
        "OnBoardTestAstApi.test_fa_all2all_ast_api_mode",
    ]
)
def fa_main(case_name: str, output: Path) -> bool:
    q, k, v, dx, drop_mask, y, dq_golden, dk_golden, dv_golden, softmax_res_golden, x_max, x_sum \
        = gen_golden_np(output)
    q, k, v = q.astype(dtype_f32), k.astype(dtype_f32), v.astype(dtype_f32)

    m = np.full([b, n, s, 1], 100000000000000.0, dtype=dtype_f32)
    _l = np.full([b, n, s, 1], 100000000000000.0, dtype=dtype_f32)
    o = np.full([b, n, s, d_v], 100000000000000.0, dtype=dtype_f32)

    one_loop_size = 128
    i_loop = s // one_loop_size
    j_loop = s // one_loop_size
    # forward
    for b_idx in range(b):
        for n_idx in range(n):
            for j in range(j_loop):
                data_start = one_loop_size * j
                data_end = one_loop_size * (j + 1)
                kj = k[b_idx, n_idx, data_start:data_end, :]
                vj = v[b_idx, n_idx, data_start:data_end, :]
                drop_mask_i = drop_mask[b_idx, n_idx, :, data_start:data_end]
                for i in range(i_loop):

                    data_start_i = one_loop_size * i
                    data_end_i = one_loop_size * (i + 1)
                    qi = q[b_idx, n_idx, data_start_i:data_end_i, :]
                    oi = o[b_idx, n_idx, data_start_i:data_end_i, :]
                    mi = m[b_idx, n_idx, data_start_i:data_end_i, :]
                    li = _l[b_idx, n_idx, data_start_i:data_end_i, :]

                    qk_j = np.matmul(qi, kj.transpose(1, 0))
                    tilda_mij = np.max(qk_j, axis=-1, keepdims=True)
                    tsub = qk_j - tilda_mij

                    tilda_pij = np.exp(tsub)

                    tilda_lij = np.sum(tilda_pij, axis=-1, keepdims=True)

                    if j == 0:

                        q6 = 1 / tilda_lij
                        q1 = np.matmul(tilda_pij, vj)

                        o[b_idx, n_idx, data_start_i:data_end_i, :] = q6 * q1
                        _l[b_idx, n_idx, data_start_i:data_end_i, :] = tilda_lij
                        m[b_idx, n_idx, data_start_i:data_end_i, :] = tilda_mij

                    else:

                        mi_new = np.max(np.concatenate((mi, tilda_mij), axis=-1), axis=-1, keepdims=True)  # ->x_max

                        t1 = mi - mi_new
                        t2 = np.exp(t1)
                        t3 = tilda_mij - mi_new
                        t4 = np.exp(t3)
                        t5 = t4 * tilda_lij

                        t6 = t2 * li
                        li_new = t6 + t5
                        q6 = 1 / li_new
                        q3 = t2 * oi
                        q4 = li * q3
                        q1 = np.matmul(tilda_pij, vj)
                        q2 = t4 * q1
                        q7 = q4 + q2

                        o[b_idx, n_idx, data_start_i:data_end_i, :] = q6 * q7
                        # logging.debug("oi ==========", oi)
                        _l[b_idx, n_idx, data_start_i:data_end_i, :] = li_new
                        m[b_idx, n_idx, data_start_i:data_end_i, :] = mi_new

    error = np.array(y - o)
    o.tofile(Path(output, 'res_golden.bin'))
    m.tofile(Path(output, 'max_golden.bin'))
    _l.tofile(Path(output, 'sum_golden.bin'))
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "ExpandOnBoardTest.test_expand_32_1_to_32_32",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = fa_main(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
