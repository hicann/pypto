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
""" Moegate Operator 相关用例 Golden 生成逻辑.

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


def topk_last_dim_no_sort(arr, k):
    # 获取数组的最后一维大小
    last_dim_size = arr.shape[-1]

    # 使用argsort函数按最后一维排序
    sorted_indices = np.argsort(arr, axis=-1)

    # 获取最大的k个元素的索引
    topk_indices = sorted_indices[..., -k:]

    # 根据索引获取最大的k个元素
    topk_values = np.take_along_axis(arr, topk_indices, axis=-1)

    return topk_values, topk_indices


def numpy_topk(input_array, k, axis=-1):
    """\
    实现类似PyTorch的torch.topk功能，返回指定维度上的前k个最大值及其索引。\
    \
    参数:\
        input_array (np.ndarray): 输入数组\
        k (int): 需要提取的最大值的数量\
        axis (int): 操作的维度，默认为最后一个维度\
    \
    返回:\
        values (np.ndarray): 前k个最大值\
        indices (np.ndarray): 对应的索引\
    """
    if k <= 0:
        raise ValueError("k必须为正整数")

    # 使用argpartition高效获取前k大元素的索引
    partitioned_indices = np.argpartition(input_array, -k, axis=axis)[..., -k:]

    # 提取对应值并生成降序排序的索引
    partitioned_values = np.take_along_axis(input_array, partitioned_indices, axis=axis)
    sorted_order = np.argsort(-partitioned_values, axis=axis)  # 负号实现降序

    # 调整索引顺序并获取最终结果
    final_indices = np.take_along_axis(partitioned_indices, sorted_order, axis=axis)
    final_values = np.take_along_axis(input_array, final_indices, axis=axis)

    return final_values, final_indices


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gen_moegate_graph_3(b, s, h, topk_group, n_group, n_routed_experts, output_dir: Path):
    dtype = np.float32
    indices_dtype = np.int32
    np.set_printoptions(threshold=np.inf)
    shape_topk_group = [b * s, topk_group]
    shape_group_mask = [b * s, n_group]
    shape_score_for_choice = [b * s, n_routed_experts]

    group_mask_path = Path(output_dir, 'group_mask_zero.bin')
    group_idx_path = Path(output_dir, 'group_idx.bin')
    scores_for_choice_path = Path(output_dir, 'scores_for_choice.bin')
    # indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    hidden_states = np.random.uniform(-1, 1, (b, s, h))
    hidden_size = h
    # logging.debug("===========start========MoEGate===")
    bsz, seq_len, h = hidden_states.shape
    # hidden_states = hidden_states.view(-1, h)
    hidden_states = hidden_states.reshape(-1, h)

    # weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
    weight = np.random.uniform(-0.1, 0.1, (hidden_size, n_routed_experts))
    logits = np.matmul(hidden_states.astype(np.float32), weight.astype(np.float32))
    scores = sigmoid(logits)
    logging.debug("======scores.shape====== %s", scores.shape)

    e_score_correction_bias = np.random.uniform(-0.1, 0.1, n_routed_experts)
    scores_for_choice = scores.reshape(bsz * seq_len, -1) + np.expand_dims(e_score_correction_bias, axis=0)

    scores_for_choice = np.random.uniform(-3, 3, shape_score_for_choice).astype(dtype)
    logging.debug("======scores_for_choice.shape====== %s", scores_for_choice.shape)
    logging.debug("======scores_for_choice====== %s", scores_for_choice)
    scores_for_choice.tofile(scores_for_choice_path)

    group_scores = topk_last_dim_no_sort(scores_for_choice.reshape(bsz * seq_len, n_group, -1), 2)[0]
    group_scores = np.sum(group_scores, axis=-1)
    logging.debug("======group_scores====== %s", group_scores)
    logging.debug("======group_scores.shape====== %s", group_scores.shape)  # [n, n_group]

    group_idx = topk_last_dim_no_sort(group_scores, topk_group)[1].astype(indices_dtype)
    # group_idx = np.random.uniform(0, 10, shape_topk_group)
    logging.debug("======group_idx====== %s", group_idx)
    logging.debug("======group_idx.shape====== %s", group_idx.shape)
    group_idx.tofile(group_idx_path)

    group_mask = np.random.uniform(0, 0, shape_group_mask).astype(dtype)  # init = 0
    group_mask.tofile(group_mask_path)
    # group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    group_mask[np.arange(group_mask.shape[0])[:, None], group_idx] = 1
    logging.debug("======group_mask====== %s", group_mask)
    logging.debug("======group_mask.shape====== %s", group_mask.shape)

    # score_mask = (group_mask.unsqueeze(-1).expand(
    #     bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group).reshape(bsz * seq_len, -1))  # [n, e]
    score_mask = np.expand_dims(group_mask, axis=-1)  # [xx,1]
    expanded_array = np.broadcast_to(score_mask, (bsz * seq_len, n_group, n_routed_experts // n_group))
    logging.debug("======expanded_array====== %s", expanded_array)
    score_mask = expanded_array.reshape(bsz * seq_len, -1)
    logging.debug("======score_mask====== %s", score_mask)
    logging.debug("======score_mask.shape====== %s", score_mask.shape)
    logging.debug("======score_mask.dtype====== %s", score_mask.dtype)
    score_mask = score_mask.astype(dtype)
    # score_mask.tofile(z_path)
    # tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    # tmp_scores = np.where(score_mask, scores_for_choice, 0.0)

    # new算法
    # (src * score_mask) + ((~score_mask) * dst)
    minfp32_val = np.random.uniform(-3.4e+38, -3.4e+38, shape_score_for_choice).astype(dtype)
    # minfp32_val = np.random.uniform(1, 1, shape_score_for_choice).astype(dtype)
    score_mask_false = (score_mask == 0).astype(dtype)
    tmp_scores = (scores_for_choice * score_mask) + (score_mask_false * minfp32_val)
    # tmp_scores =  (score_mask_false * minfp32_val)

    logging.debug("======tmp_scores====== %s", tmp_scores)
    logging.debug("======tmp_scores.shape====== %s", tmp_scores.shape)
    tmp_scores.tofile(z_path)


def gen_moegate_graph_3_4(b, s, h, topk_group, n_group, n_routed_experts, output_dir: Path):
    dtype = np.float32
    indices_dtype = np.int32
    # np.set_printoptions(threshold=np.inf)
    shape_topk_group = [b * s, topk_group]
    shape_group_mask = [b * s, n_group]
    shape_score_for_choice = [b * s, n_routed_experts]
    num_experts_per_tok = 8
    group_mask_path = Path(output_dir, 'group_mask_zero.bin')
    group_idx_path = Path(output_dir, 'group_idx.bin')
    scores_for_choice_path = Path(output_dir, 'scores_for_choice.bin')
    # indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')
    score_path = Path(output_dir, 'score.bin')

    hidden_states = np.random.uniform(-1, 1, (b, s, h))
    hidden_size = h
    # logging.debug("===========start========MoEGate===")
    bsz, seq_len, h = hidden_states.shape
    # hidden_states = hidden_states.view(-1, h)
    hidden_states = hidden_states.reshape(-1, h)

    # weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
    weight = np.random.uniform(-0.1, 0.1, (hidden_size, n_routed_experts))
    logits = np.matmul(hidden_states.astype(np.float32), weight.astype(np.float32))
    scores = sigmoid(logits)
    logging.debug("======scores.shape====== %s", scores.shape)
    scores.tofile(score_path)

    e_score_correction_bias = np.random.uniform(-0.1, 0.1, n_routed_experts)
    scores_for_choice = scores.reshape(bsz * seq_len, -1) + np.expand_dims(e_score_correction_bias, axis=0)

    scores_for_choice = np.random.uniform(-3, 3, shape_score_for_choice).astype(dtype)
    logging.debug("======scores_for_choice.shape====== %s", scores_for_choice.shape)
    logging.debug("======scores_for_choice====== %s", scores_for_choice)
    scores_for_choice.tofile(scores_for_choice_path)

    group_scores = topk_last_dim_no_sort(scores_for_choice.reshape(bsz * seq_len, n_group, -1), 2)[0]
    group_scores = np.sum(group_scores, axis=-1)
    logging.debug("======group_scores====== %s", group_scores)
    logging.debug("======group_scores.shape====== %s", group_scores.shape)  # [n, n_group]

    group_idx = topk_last_dim_no_sort(group_scores, topk_group)[1].astype(indices_dtype)
    # group_idx = np.random.uniform(0, 10, shape_topk_group)
    logging.debug("======group_idx====== %s", group_idx)
    logging.debug("======group_idx.shape====== %s", group_idx.shape)
    group_idx.tofile(group_idx_path)

    group_mask = np.random.uniform(0, 0, shape_group_mask).astype(dtype)  # init = 0
    group_mask.tofile(group_mask_path)
    # group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    group_mask[np.arange(group_mask.shape[0])[:, None], group_idx] = 1
    logging.debug("======group_mask====== %s", group_mask)
    logging.debug("======group_mask.shape====== %s", group_mask.shape)

    # score_mask = (group_mask.unsqueeze(-1).expand(
    #     bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group).reshape(bsz * seq_len, -1))  # [n, e]
    score_mask = np.expand_dims(group_mask, axis=-1)  # [xx,1]
    expanded_array = np.broadcast_to(score_mask, (bsz * seq_len, n_group, n_routed_experts // n_group))
    logging.debug("======expanded_array====== %s", expanded_array)
    score_mask = expanded_array.reshape(bsz * seq_len, -1)
    logging.debug("======score_mask====== %s", score_mask)
    logging.debug("======score_mask.shape====== %s", score_mask.shape)
    logging.debug("======score_mask.dtype====== %s", score_mask.dtype)
    score_mask = score_mask.astype(dtype)
    # score_mask.tofile(z_path)
    # tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    # tmp_scores = np.where(score_mask, scores_for_choice, 0.0)

    # new算法
    # (src * score_mask) + ((~score_mask) * dst)
    minfp32_val = np.random.uniform(-3.4e+38, -3.4e+38, shape_score_for_choice).astype(dtype)
    # minfp32_val = np.random.uniform(1, 1, shape_score_for_choice).astype(dtype)
    score_mask_false = (score_mask == 0).astype(dtype)
    tmp_scores = (scores_for_choice * score_mask) + (score_mask_false * minfp32_val)
    # tmp_scores =  (score_mask_false * minfp32_val)

    logging.debug("======tmp_scores====== %s", tmp_scores)
    logging.debug("======tmp_scores.shape====== %s", tmp_scores.shape)
    # tmp_scores.tofile(z_path)

    _, topk_idx = numpy_topk(tmp_scores, num_experts_per_tok)
    logging.debug("======topk_idx====== %s", topk_idx)
    logging.debug("======topk_idx.shape====== %s", topk_idx.shape)

    # topk_weight = scores.gather(1, topk_idx)
    topk_weight = np.random.uniform(0, 0, topk_idx.shape)
    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            topk_weight[i, j] = scores[i, topk_idx[i, j]]

    logging.debug("========first topk_weight======= %s", topk_weight)

    denominator = topk_weight.sum(axis=-1, keepdims=True) + 1e-20

    topk_weight = topk_weight / denominator

    logging.debug("========topk_weight======= %s", topk_weight)
    logging.debug("=======topk_idx======== %s", topk_idx)
    # return topk_idx, topk_weight, aux_loss
    topk_weight = topk_weight.astype(dtype)
    topk_weight.tofile(z_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        # Moegate
        "MoegateOnBoardTest.test_moegate_graph3_case1",
        "MoegateOnBoardTest.test_moegate_graph3_case2_8_1_7168",
        "MoegateOnBoardTest.test_moegate_graph3_graph4_case_32_1_7168",
        "MoegateOnBoardTest.test_moegate_graph3_case2_32_1_7168",
    ]
)
def gen_moegate_graph_date(case_name: str, output: Path) -> bool:
    dtype = np.float32
    indices_dtype = np.int32

    group_mask_path = Path(output, 'group_mask_zero.bin')
    group_idx_path = Path(output, 'group_idx.bin')
    scores_for_choice_path = Path(output, 'scores_for_choice.bin')
    # indices_path = Path(output, 'indices.bin')
    z_path = Path(output, 'z_golden.bin')

    complete = group_mask_path.exists() and group_idx_path.exists()

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
        return True
    else:
        if case_name == "MoegateOnBoardTest.test_moegate_graph3_case1":
            b, s, h, topk_group, n_group, route_experts = 1, 4, 4096, 4, 8, 256
            gen_moegate_graph_3(b, s, h, topk_group, n_group, route_experts, output)
        elif case_name == "MoegateOnBoardTest.test_moegate_graph3_case2_32_1_7168":
            b, s, h, topk_group, n_group, route_experts = 32, 1, 7168, 4, 8, 256
            gen_moegate_graph_3(b, s, h, topk_group, n_group, route_experts, output)
        elif case_name == "MoegateOnBoardTest.test_moegate_graph3_case2_8_1_7168":
            b, s, h, topk_group, n_group, route_experts = 8, 1, 7168, 4, 8, 256
            gen_moegate_graph_3(b, s, h, topk_group, n_group, route_experts, output)
        elif case_name == "MoegateOnBoardTest.test_moegate_graph3_graph4_case_32_1_7168":
            b, s, h, topk_group, n_group, route_experts = 8, 1, 7168, 4, 8, 256
            gen_moegate_graph_3_4(b, s, h, topk_group, n_group, route_experts, output)
        else:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
            return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "MoegateOnBoardTest.test_moegate_graph3_case1",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_moegate_graph_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
