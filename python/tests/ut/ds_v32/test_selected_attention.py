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
"""
from dataclasses import dataclass, field
from typing import List
import logging
import pytest
import pypto


SHAPE_DIM_0 = 0
SHAPE_DIM_1 = 1


def set_config():
    pypto.set_host_options(only_codegen=True)


@dataclass
class SelectedAttentionTileConfig:
    g_tile: int
    s2_tile: int
    c1_tile: List
    v1_tile: List
    c2_tile: List
    v2_tile: List


@dataclass
class SASimpleParams:
    n_q: int
    n_kv: int
    softmax_scale: float
    topk: int
    tile: SelectedAttentionTileConfig


@dataclass
class SAInputs:
    q_nope: pypto.tensor
    q_rope: pypto.tensor
    k_slc: pypto.tensor
    v_slc: pypto.tensor
    kv_slc_act_seqs: pypto.tensor
    attention_out: pypto.tensor
    params: SASimpleParams


def selected_attention_compute(args: SAInputs):
    q_nope = args.q_nope
    q_rope = args.q_rope
    k_slc = args.k_slc
    v_slc = args.v_slc
    kv_slc_act_seqs = args.kv_slc_act_seqs
    attention_out = args.attention_out
    params = args.params

    dtype = q_nope.dtype
    d_n = q_nope.shape[SHAPE_DIM_1]
    d_r = q_rope.shape[SHAPE_DIM_1]
    group = params.n_q // params.n_kv

    g_tile = params.tile.g_tile
    s2_tile = params.tile.s2_tile
    c1_tile = params.tile.c1_tile
    v1_tile = params.tile.v1_tile
    c2_tile = params.tile.c2_tile
    v2_tile = params.tile.v2_tile

    n2_sym = params.n_kv
    batch_size_sym = kv_slc_act_seqs.shape[SHAPE_DIM_0]
    s1n2g_sym = q_nope.shape[SHAPE_DIM_0] // batch_size_sym
    s1s2_sym = k_slc.shape[SHAPE_DIM_0] // batch_size_sym

    s1_sym = s1n2g_sym // params.n_q
    g_loop_sym = group // g_tile
    s2_sym = s1s2_sym // s1_sym

    input_tensors = [q_nope, q_rope, k_slc, v_slc, kv_slc_act_seqs]
    output_tensors = [attention_out]

    with pypto.function("SA_MAIN", *input_tensors, *output_tensors):
        for b_idx in pypto.loop(
            0,
            batch_size_sym,
            1,
            name="LOOP_L0_b_SA",
            idx_name="bIdx",
            submit_before_loop=True,
        ):
            cur_kv_slc_seq = kv_slc_act_seqs[b_idx]
            for s1_idx in pypto.loop(
                0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"
            ):
                cur_seq = (
                    (cur_kv_slc_seq - s1_sym + 1 + s1_idx)
                    .max(0)
                    .min(params.topk)
                )
                cur_seq.as_variable()
                bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile
                for n2_idx in pypto.loop(
                    0, n2_sym, 1, name="LOOP_L2_n2_SA", idx_name="n2Idx"
                ):
                    for g_idx in pypto.loop(
                        0,
                        g_loop_sym,
                        1,
                        name="LOOP_L3_g_SA",
                        idx_name="gIdx",
                    ):
                        cur_g_tile = g_tile
                        oi_update = pypto.tensor(
                            [cur_g_tile, d_n],
                            pypto.DT_FP32,
                            "oiUpdate",
                        )
                        li_update = pypto.tensor(
                            [cur_g_tile, 1],
                            pypto.DT_FP32,
                            "liUpdate",
                        )
                        mi_update = pypto.tensor(
                            [cur_g_tile, 1],
                            pypto.DT_FP32,
                            "miUpdate",
                        )
                        curr_offset = (
                            b_idx * s1n2g_sym * s1_idx * params.n_q
                            + n2_idx * group
                            + g_idx * cur_g_tile
                        )
                        oi_offset = [
                            b_idx,
                            s1_idx,
                            n2_idx * group + g_idx * cur_g_tile,
                            0,
                        ]
                        for s2_idx in pypto.loop(
                            0,
                            bn_per_batch,
                            1,
                            name="LOOP_L4_s2_SA",
                            idx_name="s2Idx",
                        ):
                            cur_s2_tile = s2_tile
                            cur_kv_offset = (
                                b_idx * s1s2_sym
                                + s1_idx * s2_sym
                                + s2_idx * cur_s2_tile
                            )
                            pypto.set_semantic_label("Sa")
                            qn = pypto.view(
                                q_nope,
                                [cur_g_tile, d_n],
                                [curr_offset, 0],
                                valid_shape=[cur_g_tile, d_n],
                            )
                            qr = pypto.view(
                                q_rope,
                                [cur_g_tile, d_r],
                                [curr_offset, 0],
                                valid_shape=[cur_g_tile, d_r],
                            )
                            qi = pypto.tensor(
                                [cur_g_tile, d_n + d_r],
                                dtype,
                                "qi",
                            )
                            pypto.assemble(qn, [0, 0], qi)
                            pypto.assemble(qr, [0, d_n], qi)
                            kj = pypto.view(
                                k_slc,
                                [cur_s2_tile, d_n + d_r],
                                [cur_kv_offset, 0],
                                valid_shape=[
                                    (
                                        cur_seq
                                        - s2_idx * cur_s2_tile
                                    ).min(cur_s2_tile),
                                    d_n + d_r,
                                ],
                            )
                            vj = pypto.view(
                                v_slc,
                                [cur_s2_tile, d_n],
                                [cur_kv_offset, 0],
                                valid_shape=[
                                    (
                                        cur_seq
                                        - s2_idx * cur_s2_tile
                                    ).min(cur_s2_tile),
                                    d_n,
                                ],
                            )
                            pypto.set_cube_tile_shapes(
                                c1_tile[0],
                                c1_tile[1],
                                c1_tile[2],
                                False,
                            )
                            pypto.set_semantic_label("Sa_QkMM")
                            pypto.set_matrix_size(
                                [qi.shape[0], 0, kj.shape[0]]
                            )
                            sij = pypto.matmul(
                                qi,
                                kj,
                                pypto.DT_FP32,
                                b_trans=True,
                            )
                            pypto.set_semantic_label("Sa_Qkvec1")
                            pypto.set_vec_tile_shapes(
                                v1_tile[0], v1_tile[1]
                            )
                            sij_scale = (
                                sij * params.softmax_scale
                            )
                            tilda_mij = pypto.amax(sij_scale, -1, True)
                            tsub = sij_scale - tilda_mij
                            tilda_pij = pypto.exp(tsub)
                            tilda_pij_f16 = pypto.cast(
                                tilda_pij, dtype
                            )
                            tilda_lij = pypto.sum(tilda_pij, -1, True)
                            if pypto.cond(
                                pypto.is_loop_begin(s2_idx)
                            ):
                                pypto.set_cube_tile_shapes(
                                    c2_tile[0],
                                    c2_tile[1],
                                    c2_tile[2],
                                    False,
                                )
                                pypto.set_semantic_label(
                                    "Sa_KvMm"
                                )
                                pypto.set_matrix_size(
                                    [
                                        tilda_pij_f16.shape[
                                            0
                                        ],
                                        tilda_pij_f16.shape[
                                            1
                                        ],
                                        vj.shape[1],
                                    ]
                                )
                                oi_tmp = pypto.matmul(
                                    tilda_pij_f16,
                                    vj,
                                    pypto.DT_FP32,
                                )
                                pypto.set_vec_tile_shapes(
                                    v2_tile[0], v2_tile[1]
                                )
                                if pypto.cond(
                                    pypto.is_loop_end(s2_idx)
                                ):
                                    pypto.set_semantic_label(
                                        "Sa_KvVec2"
                                    )
                                    oi_update[:] = (
                                        oi_tmp / tilda_lij
                                    )
                                    pypto.set_vec_tile_shapes(
                                        1,
                                        1,
                                        v2_tile[0],
                                        v2_tile[1],
                                    )
                                    oi_update_4dim = pypto.cast(
                                        pypto.reshape(
                                            oi_update,
                                            [
                                                1,
                                                1,
                                                cur_g_tile,
                                                d_n,
                                            ],
                                        ),
                                        q_nope.dtype,
                                    )
                                    pypto.assemble(
                                        oi_update_4dim,
                                        oi_offset,
                                        attention_out,
                                    )
                                else:
                                    oi_update[:] = oi_tmp
                                li_update[:] = tilda_lij
                                mi_update[:] = tilda_mij
                            else:
                                pypto.set_semantic_label(
                                    "Sa_UpdateVec2"
                                )
                                oi = oi_update
                                li = li_update
                                mi = mi_update
                                mi_new = pypto.maximum(
                                    mi, tilda_mij
                                )
                                t1 = mi - mi_new
                                t2 = pypto.exp(t1)
                                t3 = tilda_mij - mi_new
                                t4 = pypto.exp(t3)
                                t5 = t4 * tilda_lij
                                t6 = t2 * li
                                li_new = t6 + t5
                                q3 = oi * t2
                                pypto.set_cube_tile_shapes(
                                    c2_tile[0],
                                    c2_tile[1],
                                    c2_tile[2],
                                    False,
                                )
                                pypto.set_semantic_label(
                                    "Sa_UpdateMM2"
                                )
                                pypto.set_matrix_size(
                                    [
                                        tilda_pij_f16.shape[
                                            0
                                        ],
                                        tilda_pij_f16.shape[
                                            1
                                        ],
                                        vj.shape[1],
                                    ]
                                )
                                q1 = pypto.matmul(
                                    tilda_pij_f16,
                                    vj,
                                    pypto.DT_FP32,
                                )
                                pypto.set_vec_tile_shapes(
                                    v2_tile[0], v2_tile[1]
                                )
                                q2 = q1 * t4
                                oi_tmp = q3 + q2
                                if pypto.cond(
                                    pypto.is_loop_end(s2_idx)
                                ):
                                    oi_update[:] = (
                                        oi_tmp / li_new
                                    )
                                    pypto.set_vec_tile_shapes(
                                        1,
                                        1,
                                        v2_tile[0],
                                        v2_tile[1],
                                    )
                                    oi_update_4dim = pypto.cast(
                                        pypto.reshape(
                                            oi_update,
                                            [
                                                1,
                                                1,
                                                cur_g_tile,
                                                d_n,
                                            ],
                                        ),
                                        q_nope.dtype,
                                    )
                                    pypto.assemble(
                                        oi_update_4dim,
                                        oi_offset,
                                        attention_out,
                                    )
                                else:
                                    oi_update[:] = oi_tmp
                                li_update[:] = li_new
                                mi_update[:] = mi_new


@dataclass
class SABuildConfig:
    b: int = 32
    s1: int = 4
    n_q: int = 128
    n_kv: int = 1
    qk_nope_head_dim: int = 512
    qk_rope_head_dim: int = 64
    kv_head_dim: int = 512
    topk: int = 2048
    softmax_scale: float = 1.0 / 24.0
    g_tile: int = 128
    s2_tile: int = 2048
    c1_tile: List[int] = field(
        default_factory=lambda: [[128, 128], [64, 64], [256, 256]]
    )
    v1_tile: List[int] = field(default_factory=lambda: [16, 256])
    c2_tile: List[int] = field(
        default_factory=lambda: [[128, 128], [128, 128], [128, 128]]
    )
    v2_tile: List[int] = field(default_factory=lambda: [64, 128])


def build_selected_args(cfg: SABuildConfig = SABuildConfig()):
    d_type = pypto.DT_FP16
    i32 = pypto.DT_INT32

    q_nope_shape = [cfg.b * cfg.s1 * cfg.n_q, cfg.qk_nope_head_dim]
    q_rope_shape = [cfg.b * cfg.s1 * cfg.n_q, cfg.qk_rope_head_dim]

    k_concat_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    k_slc_shape = [cfg.b * cfg.s1 * cfg.topk, k_concat_dim]
    v_slc_shape = [cfg.b * cfg.s1 * cfg.topk, cfg.kv_head_dim]

    kv_slc_act_seqs_shape = [cfg.b]

    attention_out_shape = [cfg.b, cfg.s1, cfg.n_q, cfg.qk_nope_head_dim]

    q_nope = pypto.tensor(q_nope_shape, d_type, "qNope")
    q_rope = pypto.tensor(q_rope_shape, d_type, "qRope")
    k_slc = pypto.tensor(k_slc_shape, d_type, "kSlc")
    v_slc = pypto.tensor(v_slc_shape, d_type, "vSlc")
    kv_slc_act_seqs = pypto.tensor(kv_slc_act_seqs_shape, i32, "kvSlcActSeqs")
    attention_out = pypto.tensor(attention_out_shape, d_type, "attentionOut")

    tile = SelectedAttentionTileConfig(
        g_tile=cfg.g_tile,
        s2_tile=cfg.s2_tile,
        c1_tile=cfg.c1_tile,
        v1_tile=cfg.v1_tile,
        c2_tile=cfg.c2_tile,
        v2_tile=cfg.v2_tile,
    )
    params = SASimpleParams(
        n_q=cfg.n_q,
        n_kv=cfg.n_kv,
        softmax_scale=cfg.softmax_scale,
        topk=cfg.topk,
        tile=tile,
    )

    args = SAInputs(
        q_nope=q_nope,
        q_rope=q_rope,
        k_slc=k_slc,
        v_slc=v_slc,
        kv_slc_act_seqs=kv_slc_act_seqs,
        attention_out=attention_out,
        params=params,
    )

    meta = {
        "b": cfg.b,
        "s1": cfg.s1,
        "nQ": cfg.n_q,
        "nKv": cfg.n_kv,
        "dims": {
            "qNope": q_nope_shape,
            "qRope": q_rope_shape,
            "kSlc": k_slc_shape,
            "vSlc": v_slc_shape,
            "kvSlcActSeqs": kv_slc_act_seqs_shape,
            "attentionOut": attention_out_shape,
        },
        "topk": cfg.topk,
        "softmaxScale": cfg.softmax_scale,
        "tiles": {
            "gTile": cfg.g_tile,
            "s2Tile": cfg.s2_tile,
            "c1Tile": cfg.c1_tile,
            "v1Tile": cfg.v1_tile,
            "c2Tile": cfg.c2_tile,
            "v2Tile": cfg.v2_tile,
        },
    }
    return args, meta


def test_selected_attention_with_builder():
    logging.basicConfig(level=logging.INFO)
    set_config()
    args, meta = build_selected_args()
    logging.info({"Sanity": meta})
    selected_attention_compute(args)
    assert True
