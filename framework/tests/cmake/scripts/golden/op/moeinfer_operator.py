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
""" Moeinfer Operator 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as functional

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


def quant(input_t, is_pertoken: bool = True):
    # input_fp32 = input_t.astype(np.float32)
    input_fp32 = input_t.numpy().astype(np.float32)

    abs_res = np.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
        logging.error("This PerChannel Quant!!")

    max_value = np.max(abs_res, axis=reduce_idx, keepdims=True)
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_fp16 = out_int32.astype(np.float16)
    out_int8 = np.trunc(out_fp16).astype(np.int8)
    scale_dequant = 1 / scale_quant

    # return out_int8, scale_dequant
    return torch.from_numpy(out_int8), torch.from_numpy(scale_dequant)


class MoeInferParam:
    def __init__(self, b, s, h, ffn_weight_n, topk_group=0,
                 n_group=0, n_routed_experts=0, num_experts_per_topk=0):
        self.b = b
        self.s = s
        self.h = h
        self.ffn_weight_n = ffn_weight_n
        self.topk_group = topk_group
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_topk = num_experts_per_topk


def expert_quant(hidden_states, ffn_weight1, ffn_weight2, ffn_weight3):
    # 假设 hiddenStates 和 ffnWeight 已经定义，这里给出示例形状的随机初始化

    ffn_weight_quant, ffnweight_scale = ffn_weight1
    ffn_weight2_quant, ffnweight2_scale = ffn_weight2
    ffn_weight3_quant, ffnweight3_scale = ffn_weight3

    bs = hidden_states.shape[0]
    h = hidden_states.shape[1]
    # 第一步：计算 Tgate
    hidden_states, hidden_states_scale = quant(hidden_states)
    t_gate = torch.matmul(hidden_states.to(torch.int32), ffn_weight_quant.to(torch.int32))
    # dequant
    t_gate = t_gate.to(torch.float32)
    tgate_dequant = t_gate * hidden_states_scale
    t_gate = tgate_dequant * ffnweight_scale

    # 第二步：计算 Tswish
    t_swish = -t_gate
    t_swish = torch.exp(t_swish)
    t_swish = t_swish + 1
    t_swish = t_gate / t_swish

    # 第三步：计算 t_up
    # t_up = torch.matmul(hidden_states.to(torch.float32), ffn_weight2.to(torch.float32))
    tup = torch.matmul(hidden_states.to(torch.int32), ffn_weight2_quant.to(torch.int32))
    # dequant
    tup = tup.to(torch.float32)
    tup_dequant = tup * hidden_states_scale
    tup = tup_dequant * ffnweight2_scale

    # 第四步：更新 Tswish
    t_swish = t_swish * tup

    # 第五步：计算最终结果 res
    # res = torch.matmul(t_swish.to(torch.float32), ffn_weight3.transpose(0, 1).to(torch.float32))

    t_swish, t_swish_scale = quant(t_swish)
    logging.debug("=====Tswish.shape=== %s", t_swish.shape)
    logging.debug("=====Tswish_scale.shape=== %s", t_swish_scale.shape)
    res = torch.matmul(t_swish.to(torch.int32), ffn_weight3_quant.transpose(0, 1).to(torch.int32))
    # dequant
    res = res.to(torch.float32)
    res_dequant = res * t_swish_scale
    res = res_dequant * ffnweight3_scale.transpose(0, 1)
    return res


def expert(hidden_states, ffn_weight1, ffn_weight2, ffn_weight3):
    # 假设 hiddenStates 和 ffnWeight 已经定义，这里给出示例形状的随机初始化
    bs = hidden_states.shape[0]
    h = hidden_states.shape[1]

    # 第一步：计算 Tgate
    # Tgate = torch.matmul(hiddenStates, ffnWeight1).to(torch.float32)
    t_gate = torch.matmul(hidden_states.to(torch.float32), ffn_weight1.to(torch.float32))

    # 第二步：计算 Tswish
    t_swish = -t_gate
    t_swish = torch.exp(t_swish)
    t_swish = t_swish + 1
    t_swish = t_gate / t_swish

    # 第三步：计算 Tup
    t_up = torch.matmul(hidden_states.to(torch.float32), ffn_weight2.to(torch.float32))

    # 第四步：更新 Tswish
    t_swish = t_swish * t_up

    # 第五步：计算最终结果 res
    res = torch.matmul(t_swish.to(torch.float32), ffn_weight3.transpose(0, 1).to(torch.float32))

    return res


def stable_argsort(x, dim=-1, descending=False):
    # 生成原始索引作为辅助键
    indices = torch.arange(x.shape[dim], device=x.device)
    # 拼接值和索引，用浮点数扰动确保稳定性
    perturbed = x + indices.float() * 1e-6  # 扰动系数需根据数据类型调整
    # 稳定排序
    _, sorted_indices = torch.sort(perturbed, dim=dim, descending=descending, stable=True)
    return sorted_indices


def gen_ffn_data(moe_infer_param, output: Path, do_quant = False, do_nz=False):
    h = moe_infer_param.h
    ffn_weight_n = moe_infer_param.ffn_weight_n

    ffn_weight1_path = Path(output, 'ffnWeight1.bin')
    ffn_weight2_path = Path(output, 'ffnWeight2.bin')
    ffn_weight3_path = Path(output, 'ffnWeight3.bin')

    ffn_weight1 = torch.empty(h, ffn_weight_n).uniform_(-0.1, 0.1).to(torch.float16)
    ffn_weight2 = torch.empty(h, ffn_weight_n).uniform_(-0.1, 0.1).to(torch.float16)
    ffn_weight3 = torch.empty(h, ffn_weight_n).uniform_(-0.1, 0.1).to(torch.float16)

    NzFrac = 16

    if do_quant:
        ffn_weight1, ffn_scale1 = quant(ffn_weight1, False)
        ffn_weight2, ffn_scale2 = quant(ffn_weight2, False)
        ffn_weight3, ffn_scale3 = quant(ffn_weight3)
        NzFrac = 32

    if do_nz:
        ffn_weight1_np = ffn_weight1.numpy().reshape((h // 16, 16, ffn_weight_n // NzFrac, NzFrac)).transpose(2, 0, 1, 3)
        ffn_weight2_np = ffn_weight2.numpy().reshape((h // 16, 16, ffn_weight_n // NzFrac, NzFrac)).transpose(2, 0, 1, 3)
        ffn_weight3_np = ffn_weight3.numpy().reshape((h // 16, 16, ffn_weight_n // NzFrac, NzFrac)).transpose(2, 0, 1, 3)
    else:
        ffn_weight1_np = ffn_weight1.numpy()
        ffn_weight2_np = ffn_weight2.numpy()
        ffn_weight3_np = ffn_weight3.numpy()

    ffn_weight1_np.tofile(ffn_weight1_path)
    ffn_weight2_np.tofile(ffn_weight2_path)
    ffn_weight3_np.tofile(ffn_weight3_path)
    if do_quant:
        ffn_scale1_path = Path(output, 'ffnScale1.bin')
        ffn_scale2_path = Path(output, 'ffnScale2.bin')
        ffn_scale3_path = Path(output, 'ffnScale3.bin')
        ffn_scale1.numpy().tofile(ffn_scale1_path)
        ffn_scale2.numpy().tofile(ffn_scale2_path)
        ffn_scale3.numpy().tofile(ffn_scale3_path)
        return (ffn_weight1, ffn_scale1), (ffn_weight2, ffn_scale2), (ffn_weight3, ffn_scale3)
    return ffn_weight1, ffn_weight2, ffn_weight3


def gen_ffn_data_graph(moe_infer_param, output: Path, is_quant=False, is_format_nz=False):
    bs = moe_infer_param.b * moe_infer_param.s
    h = moe_infer_param.h
    hidden_states_path = Path(output, "hidden_states.bin")
    hidden_states_scale_path = Path(output, "hidden_states_scale.bin")

    final_out_path = Path(output, 'final_out.bin')
    hidden_states = 0.01 + (0.02 - 0.01) * torch.rand(bs * h, dtype=torch.float16).reshape(bs, h).type(torch.float32)

    if is_quant:
        hidden_states_quant, hidden_states_scale = quant(hidden_states)
        hidden_states_quant.numpy().tofile(hidden_states_path)
        print(hidden_states_scale.shape)
        hidden_states_scale.numpy().tofile(hidden_states_scale_path)
    else:
        hidden_states.numpy().tofile(hidden_states_path)
    ffn_weight1, ffn_weight2, ffn_weight3 = gen_ffn_data(moe_infer_param, output, is_quant, is_format_nz)
    expert_out = expert_quant(hidden_states, ffn_weight1, ffn_weight2, ffn_weight3) if is_quant \
        else expert(hidden_states, ffn_weight1, ffn_weight2, ffn_weight3)
    expert_out.numpy().astype(np.float32).tofile(final_out_path)


def gen_moeinfer_graph_singlemlp(moe_infer_param, output: Path, is_quant: bool = False, is_format_nz: bool = False):
    #  dump data path
    e_score_correction_bias_path = Path(output, "e_score_correction_bias.bin")
    hidden_states_path = Path(output, "hidden_states.bin")
    weight_path = Path(output, "weight.bin")
    logits_path = Path(output, "logits.bin")
    scores_path = Path(output, "scores.bin")
    scores_for_choice_path = Path(output, "scores_for_choice.bin")
    group_idx_path = Path(output, 'group_idx.bin')
    group_mask_path = Path(output, 'group_mask.bin')
    score_mask_path = Path(output, 'score_mask.bin')
    tmp_scores_path = Path(output, 'tmp_scores.bin')
    topk_idx_path = Path(output, 'topk_idx.bin')
    topk_weight_path = Path(output, 'topk_weight.bin')

    final_out_path = Path(output, 'final_out.bin')
    bs = moe_infer_param.b * moe_infer_param.s
    h = moe_infer_param.h
    ffn_weight_n = moe_infer_param.ffn_weight_n
    topk_group = moe_infer_param.topk_group
    n_group = moe_infer_param.n_group
    n_routed_experts = moe_infer_param.n_routed_experts
    num_experts_per_topk = moe_infer_param.num_experts_per_topk

    """ moegate """
    # ========= part1
    e_score_correction_bias = 0 + (2 - 0) * torch.rand(n_routed_experts, dtype=torch.float32).reshape(n_routed_experts)
    hidden_states = 0.01 + (0.02 - 0.01) * torch.rand(bs * h, dtype=torch.float32).reshape(bs, h)
    weight = 0.01 + (0.02 - 0.01) * torch.rand(n_routed_experts * h, dtype=torch.float16).reshape(n_routed_experts, h)
    weight_fp32 = weight.type(torch.float32)
    logits = functional.linear(hidden_states, weight_fp32, None).type(torch.float32)
    # torch.set_printoptions(threshold=float('inf'), sci_mode=False)
    scores = logits.sigmoid()
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    e_score_correction_bias.numpy().tofile(e_score_correction_bias_path)
    hidden_states.numpy().tofile(hidden_states_path)
    logits.numpy().tofile(logits_path)
    weight.numpy().tofile(weight_path)
    scores.numpy().tofile(scores_path)
    scores_for_choice.numpy().tofile(scores_for_choice_path)

    """ part2  scores_for_choice <---> group_mask group_idx """
    # scores_for_choice [B, 8, 32]
    scores_for_choice = scores_for_choice.reshape(bs, n_group, 32)
    # val : [B, 8, 2]
    val, idx = scores_for_choice.topk(2, dim=-1)
    # group_scores : [B, 8]
    group_scores = torch.sum(val, dim=-1)
    # output : group_idx [B, 4]
    val1, group_idx = group_scores.topk(topk_group, dim=-1)
    # output : group_mask [B, 4]
    group_mask = torch.zeros_like(group_scores).type(torch.float32)
    group_idx.numpy().astype(np.int32).tofile(group_idx_path)
    group_mask.numpy().astype(np.float32).tofile(group_mask_path)

    """ part3 group_mask group_idx scores_for_choice <---> tmp_scores """
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(bs, n_group, n_routed_experts // n_group)
        .reshape(bs, n_routed_experts)
    ).type(torch.float32)
    scores_for_choice = scores_for_choice.reshape(bs, n_routed_experts).type(torch.float32)
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0)  # -3.4e+38)
    score_mask.numpy().tofile(score_mask_path)
    tmp_scores.numpy().astype(np.float32).tofile(tmp_scores_path)
    logging.debug("score_mask %s", score_mask)
    logging.debug("tmp_scores %s", tmp_scores)

    """ part4 scores tmp_scores <---> topk_weight """
    _, topk_idx = torch.topk(
        tmp_scores, k=num_experts_per_topk, dim=-1, sorted=True
    )
    topk_weight = scores.gather(1, topk_idx).type(torch.float32)
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
    logging.debug("topk_idx %s", topk_idx)
    logging.debug("topk_idx.shape %s", topk_idx.shape)
    logging.debug("topk_weight %s", topk_weight)
    topk_idx.numpy().astype(np.int32).tofile(topk_idx_path)
    topk_weight.numpy().tofile(topk_weight_path)

    """ moeinfer """
    sorted_tokens_path = Path(output, 'sorted_tokens.bin')
    outs_path = Path(output, 'outs.bin')
    idxs_path = Path(output, 'idxs.bin')

    cnts = topk_idx.new_zeros((topk_idx.shape[0], n_routed_experts))
    cnts.scatter_(1, topk_idx, 1)
    logging.debug("===cnts.shape=== %s", cnts.shape)
    tokens_per_expert = cnts.sum(dim=0)
    logging.debug("=======topk_idx.view(-1)======== %s", topk_idx.view(-1))
    # idxs = topk_idx.view(-1).argsort() #原生torch是不稳定排序 和npu有差异 需要再确认算法是否不影响 暂时先用稳定排序
    idxs = stable_argsort(topk_idx.view(-1))

    logging.debug("======idxs=== %s", idxs)
    idxs.numpy().astype(np.int32).tofile(idxs_path)
    logging.debug("====idxs.numpy().astype(np.int32)==== %s", idxs.numpy().astype(np.int32))
    sorted_tokens = hidden_states[idxs // topk_idx.shape[1]]

    logging.debug("=======sorted_tokens.shape===== %s", sorted_tokens.shape)
    tokens_per_expert = tokens_per_expert.cpu().numpy()
    logging.debug("=======tokens_per_expert== %s", tokens_per_expert)
    logging.debug("=======sorted_tokens===== %s", sorted_tokens)

    sorted_tokens.numpy().astype(np.float32).tofile(sorted_tokens_path)
    outputs = []
    start_idx = 0

    ffn_weight1, ffn_weight2, ffn_weight3 = gen_ffn_data(moe_infer_param, output, is_quant, is_format_nz)
    for i in range(1):  # 此处应该用8
        num_tokens = bs
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        # expert = self.experts[i + self.ep_rank * self.experts_per_rank]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = (
            expert_quant(tokens_for_this_expert, ffn_weight1, ffn_weight2, ffn_weight3)) if is_quant \
            else expert(tokens_for_this_expert, ffn_weight1, ffn_weight2, ffn_weight3)
        logging.debug("====expert_out=== %s", expert_out)
        # expert_out = tokens_for_this_expert
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    outs.numpy().astype(np.float32).tofile(outs_path)
    logging.debug("====outs=== %s", outs)
    logging.debug("====outs.shape=== %s", outs.shape)
    outs.numpy().astype(np.float32).tofile(final_out_path)  # if single mlp , out is final_out


@GoldenRegister.reg_golden_func(
    case_names=[
        "MlpTest.test_16_7168_tileop",
        "DynamicFFNTest.TestOnbroadDynamicFFN",
        "DynamicFFNTest.TestOnbroadDynamicFFNQuant",
        "MoeInferOnbroadTest.test_deepseekMoEInfer",
        "MoeInferOnbroadTest.test_deepseekMoEInfer_singleout",
        "MoeInferOnbroadTest.test_deepseekMoEInfer_singleout_singlemlp",
        "MoeInferOnbroadTest.test_deepseekMoEInfer_singleout_singlemlp_withquant",
    ]
)

def gen_moeinfer_graph_data(case_name: str, output: Path) -> bool:
    final_out_path = Path(output, 'final_out.bin')

    if case_name == "DynamicFFNTest.TestOnbroadDynamicFFN" or \
        case_name == "MlpTest.test_16_7168_tileop":
        moe_infer_param = MoeInferParam(32, 1, 7168, 2048)
        gen_ffn_data_graph(moe_infer_param, output, False, True)
        logging.debug("Case(%s), Golden generated.", case_name)
    elif case_name == "DynamicFFNTest.TestOnbroadDynamicFFNQuant":
        moe_infer_param = MoeInferParam(32, 1, 7168, 2048)
        gen_ffn_data_graph(moe_infer_param, output, True, True)
        logging.debug("Case(%s), Golden generated.", case_name)
    elif case_name == "MoeInferOnbroadTest.test_deepseekMoEInfer":
        moe_infer_param = MoeInferParam(16, 1, 256, 256 * 3, 4, 8, 256, 8)
        gen_moeinfer_graph_singlemlp(moe_infer_param, output)
    elif case_name == "MoeInferOnbroadTest.test_deepseekMoEInfer_singleout":
        moe_infer_param = MoeInferParam(4, 1, 256, 256 * 3, 4, 8, 256, 8)
        gen_moeinfer_graph_singlemlp(moe_infer_param, output)
    elif case_name == "MoeInferOnbroadTest.test_deepseekMoEInfer_singleout_singlemlp":
        moe_infer_param = MoeInferParam(4, 1, 7168, 2048, 4, 8, 256, 8)
        gen_moeinfer_graph_singlemlp(moe_infer_param, output)
    elif case_name == "MoeInferOnbroadTest.test_deepseekMoEInfer_singleout_singlemlp_withquant":
        moe_infer_param = MoeInferParam(32, 1, 7168, 2048, 4, 8, 256, 8)
        gen_moeinfer_graph_singlemlp(moe_infer_param, output, True, True)
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
        "MoeInferOnbroadTest.test_deepseekMoEInfer",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_moeinfer_graph_data(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
