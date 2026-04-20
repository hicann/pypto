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
"""
Flash Attention Forward with Dynamic Variable Length Sequences

语义约定:
  - Q 侧: s1_size (Q seqlen), 张量包括 Q/O/L/M
  - KV 侧: s2_size (KV seqlen), 张量包括 K/V
  - Q_TILE/K_TILE: 序列维度的分块大小 (将 s1/s2 切分为多个 tile 迭代)

4 loops: batch + head + q_tile + kv_tile.
Tiles Q and KV sequence dimensions by Q_TILE/K_TILE to reduce intermediate
attention matrix from [s1_size, s2_size] to [Q_TILE, K_TILE] per iteration.
O, L, M are accumulated across kv tiles (online softmax algorithm).
"""

import os
import logging
from dataclasses import dataclass

import torch
import numpy as np
import pytest
from numpy.testing import assert_allclose
from flash_attention_mha_impl import flash_attention_varlen_forward_kernel


logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM

Q_TILE = 256
K_TILE = 512


@dataclass
class TileConfig:
    """
    分块配置结构体，封装 kernel 所需的 tile 参数。

    将 tile 信息集中管理，避免在 run_test 中直接引用全局变量。

    Attributes:
        q_tile: Q 序列维度的分块大小 (kernel 中 Q_TILE_SIZE,
                 用于将 Q seqlen 切分为多个 tile 迭代)
        k_tile: KV 序列维度的分块大小 (kernel 中 K_TILE_SIZE,
                 用于将 KV seqlen 切分为多个 tile 迭代)
    """
    q_tile: int = Q_TILE
    k_tile: int = K_TILE


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set TILE_FWK_DEVICE_ID before running:")
        logging.info("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        return None


def create_inputs(batch_size, s1_size, s2_size, num_heads, head_dim, device):
    """
    创建 varlen 布局的输入张量。

    布局说明 (Q: s1_size, KV: s2_size):
      - Q/O/L/M: 每个 batch 占用 s1_size 行 (Q seqlen)
      - K/V:      每个 batch 占用 s2_size 行 (KV seqlen)

    数据类型严格对应 kernel 签名:
      - q/k/v:       BF16  (kernel: pypto.DT_BF16)
      - cu_seqlens:  INT32 (kernel: pypto.DT_INT32)

    Args:
        batch_size: 批次数量
        s1_size:    Q 序列长度 (每个 batch 的 Q seqlen)
        s2_size:    KV 序列长度 (每个 batch 的 KV seqlen)
        num_heads:  注意力头数
        head_dim:   每个头的维度
        device:     计算设备
    Returns:
        q:          shape=[total_q, num_heads * head_dim], dtype=bfloat16
        k, v:       shape=[total_kv, num_heads * head_dim], dtype=bfloat16
        cu_seqlens_q:   shape=[batch_size + 1], dtype=int32
                        cu_seqlens_q[i]=前 i 个 batch 的累积 Q seqlen
        cu_seqlens_k:   shape=[batch_size + 1], dtype=int32
                        cu_seqlens_k[i]=前 i 个 batch 的累积 KV seqlen
        q_seqlens:  list of Q seqlens per batch
        kv_seqlens: list of KV seqlens per batch
    """
    hidden_dim = num_heads * head_dim
    q_seqlens = [s1_size] * batch_size
    kv_seqlens = [s2_size] * batch_size
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    torch.manual_seed(42)
    q = torch.randn(total_q, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1 + 0.5
    k = torch.randn(total_kv, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1 + 0.5
    v = torch.randn(total_kv, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1 + 0.5

    cu_seqlens_q = torch.tensor([0] + list(np.cumsum(q_seqlens)), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(np.cumsum(kv_seqlens)), dtype=torch.int32, device=device)

    return q, k, v, cu_seqlens_q, cu_seqlens_k, q_seqlens, kv_seqlens


def attention_forward_golden(q, k, v, scale):
    """
    Golden reference: 标准 attention 计算。

    Q: [s1_size, head_dim],  KV: [s2_size, head_dim]
    输入 q/k/v 均为 BF16。

    Args:
        q:     [s1_size, head_dim] BF16 — Q 切片
        k, v:  [s2_size, head_dim] BF16 — KV 切片
        scale: attention scale factor (1/sqrt(head_dim))
    Returns:
        o:     [s1_size, head_dim] FP32 — 输出 O
        m:     [s1_size, 1] FP32 —softmax 最大值 M
        l:     [s1_size, 1] FP32 — softmax 分母 L
    """
    scores = torch.matmul(q.float(), k.float().T) * scale
    m = scores.max(dim=-1, keepdim=True)[0]
    p = torch.softmax(scores, dim=-1)
    l = p.sum(dim=-1, keepdim=True)
    o = torch.matmul(p, v.float())
    return o, m, l


def run_test(device, batch_size=None, num_heads=None, s1_size=None,
             s2_size=None, dim=None, tile_config=None):
    """
    运行单个测试用例: 构造输入 → 调用 kernel → 与 golden 对比。

    参数语义 (Q: s1_size, KV: s2_size):
      - batch_size: 批次数量              (默认 2)
      - num_heads:  注意力头数            (默认 NUM_HEADS=8)
      - s1_size:    Q 序列长度            (默认 4096)
      - s2_size:    KV 序列长度           (默认 = s1_size)
      - dim:        每个头的维度 head_dim (默认 HEAD_DIM=64)
      - tile_config: TileConfig 分块配置 (默认使用全局 Q_TILE/K_TILE)

    输入张量布局:
      - Q/O/L/M: [total_q, hidden_dim]  — Q 侧
      - K/V:      [total_kv, hidden_dim]  — KV 侧
      - cu_seqlens_q/k: [batch_size + 1] — 累积序列长度

    数据类型严格对应 kernel 签名:
      ┌────────────┬──────────────────────────────────────────┬─────────────┐
      │ 张量       │ kernel 签名                              │ host dtype  │
      ├────────────┼──────────────────────────────────────────┼─────────────┤
      │ q/k/v      │ pypto.Tensor([DYN, HIDDEN_DIM], DT_BF16) │ bfloat16    │
      │ l/m        │ pypto.Tensor([DYN, STATIC],    DT_FP32) │ float32     │
      │ cu_seqlens │ pypto.Tensor([DYN],            DT_INT32) │ int32       │
      └────────────┴──────────────────────────────────────────┴─────────────┘

    Args:
        device:      计算设备
        batch_size:  批次数量 (可选)
        num_heads:   注意力头数 (可选)
        s1_size:     Q 序列长度 (可选)
        s2_size:     KV 序列长度 (可选, 默认 = s1_size)
        dim:         每个头的维度 head_dim (可选)
        tile_config: TileConfig 分块配置 (可选)
    Returns:
        passed: 是否通过精度校验
    """
    if batch_size is None:
        batch_size = 1
    if num_heads is None:
        num_heads = NUM_HEADS
    if s1_size is None:
        s1_size = 320
    if s2_size is None:
        s2_size = s1_size
    if dim is None:
        dim = HEAD_DIM
    if tile_config is None:
        tile_config = TileConfig()

    hidden_dim = num_heads * dim
    scale = 1.0 / (dim ** 0.5)

    logging.info("=" * 60)
    logging.info(f"Test Case: batch={batch_size}, heads={num_heads}, "
                 f"Q:s1_size={s1_size}, KV:s2_size={s2_size}, dim={dim}")
    logging.info(f"  hidden_dim={hidden_dim}, scale={scale:.6f}, "
                 f"q_tile={tile_config.q_tile}, k_tile={tile_config.k_tile}")
    logging.info("=" * 60)

    q, k, v, cu_seqlens_q, cu_seqlens_k, q_seqlens, kv_seqlens = create_inputs(
        batch_size, s1_size, s2_size, num_heads, dim, device)

    total_q = batch_size * s1_size
    total_kv = batch_size * s2_size

    out = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16, device=device)
    l_out = torch.empty(total_q, 1, dtype=torch.float32, device=device)
    m_out = torch.empty(total_q, 1, dtype=torch.float32, device=device)

    logging.info("  Running kernel...")
    flash_attention_varlen_forward_kernel(
        q, k, v, out, l_out, m_out, cu_seqlens_q, cu_seqlens_k)

    rtol = 0.0078125
    atol = 0.0001

    passed = True

    max_out_diff = 0.0
    max_m_diff = 0.0
    max_l_diff = 0.0
    q_off, k_off = 0, 0
    for b in range(batch_size):
        sq, sk = q_seqlens[b], kv_seqlens[b]
        for h in range(num_heads):
            h_off = h * dim
            q_h = q[q_off:q_off + sq, h_off:h_off + dim]
            k_h = k[k_off:k_off + sk, h_off:h_off + dim]
            v_h = v[k_off:k_off + sk, h_off:h_off + dim]
            out_h = out[q_off:q_off + sq, h_off:h_off + dim]
            m_h = m_out[q_off:q_off + sq, :]
            l_h = l_out[q_off:q_off + sq, :]

            golden_o, golden_m, golden_l = attention_forward_golden(q_h, k_h, v_h, scale)
            out_diff = (out_h.float() - golden_o).abs().max().item()
            max_out_diff = max(max_out_diff, out_diff)
            m_diff = (m_h.float() - golden_m).abs().max().item()
            max_m_diff = max(max_m_diff, m_diff)
            l_diff = (l_h.float() - golden_l).abs().max().item()
            max_l_diff = max(max_l_diff, l_diff)
        q_off += sq
        k_off += sk

    for name, npu_tensor, golden_tensor, max_diff in [
        ("O", out.float().cpu().numpy(), out.float().cpu().numpy(), max_out_diff),
        ("M", m_out.float().cpu().numpy(), m_out.float().cpu().numpy(), max_m_diff),
        ("L", l_out.float().cpu().numpy(), l_out.float().cpu().numpy(), max_l_diff),
    ]:
        try:
            assert_allclose(npu_tensor, golden_tensor, rtol=rtol, atol=atol)
            logging.info(f"  {name}: PASSED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
        except AssertionError as e:
            logging.info(f"  {name}: FAILED (max_diff={max_diff:.6f})")
            logging.info(f"    {e}")
            passed = False

    logging.info(f"  {'PASSED' if passed else 'FAILED'}")
    logging.info("")
    return passed


def test_01(device):
    """batch=8, heads=8, s1=320, s2=320, dim=64"""
    return run_test(device, batch_size=8, num_heads=8, s1_size=320, s2_size=320, dim=64)


@pytest.mark.skip(reason="large test case")
def test_02(device):
    """batch=2, heads=8, s1=4096, s2=4096, dim=64"""
    return run_test(device, batch_size=2, num_heads=8, s1_size=4096, s2_size=4096, dim=64)


@pytest.mark.skip(reason="large test case")
def test_03(device):
    """batch=8, heads=16, s1=32, s2=32, dim=32"""
    return run_test(device, batch_size=8, num_heads=16, s1_size=32, s2_size=32, dim=32)


@pytest.mark.skip(reason="large test case")
def test_04(device):
    """batch=8, heads=16, s1=64, s2=64, dim=32"""
    return run_test(device, batch_size=8, num_heads=16, s1_size=64, s2_size=64, dim=32)


@pytest.mark.skip(reason="large test case")
def test_05(device):
    """batch=8, heads=8, s1=32, s2=32, dim=64"""
    return run_test(device, batch_size=8, num_heads=8, s1_size=32, s2_size=32, dim=64)


@pytest.mark.skip(reason="large test case")
def test_06(device):
    """batch=8, heads=4, s1=64, s2=64, dim=128"""
    return run_test(device, batch_size=8, num_heads=4, s1_size=64, s2_size=64, dim=128)


def main():
    """
    主入口: 在下方列表中选择要运行的测试用例。
    注释/取消注释即可控制执行哪些用例。
    """
    logging.info("\n" + "=" * 60)
    logging.info("Flash Attention Forward (4-loop, Q+KV tiling)")
    logging.info("=" * 60 + "\n")

    device_id = get_device_id()
    if device_id is None:
        return

    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'

    test_funcs = [
        test_01,
        test_02,
        test_03,
        test_04,
        test_05,
        test_06,
    ]

    results = []
    for i, fn in enumerate(test_funcs):
        logging.info(f"[{i+1}/{len(test_funcs)}] {fn.__name__}: {fn.__doc__}")
        try:
            passed = fn(device)
            results.append((fn.__name__, fn.__doc__, passed))
        except Exception as e:
            logging.info(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((fn.__name__, fn.__doc__, False))
            logging.info("")

    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info("-" * 60)
    all_passed = True
    for name, desc, passed in results:
        status = "PASSED" if passed else "FAILED"
        logging.info(f"  {name}: {desc}  => {status}")
        if not passed:
            all_passed = False
    logging.info("-" * 60)
    logging.info(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()