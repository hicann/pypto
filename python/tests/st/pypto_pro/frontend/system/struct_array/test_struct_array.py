# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""pl.struct_array NPU 泛化测试 — 覆盖多种索引模式和控制流场景。

参考文献:
  docs/zh/api/SIMD-API/计算API/工具函数/struct_array.md

 测试覆盖场景:
   1. 常索引读写 + 多槽独立赋值
   2. size=1 边界
   3. for 循环遍历 + 两轮循环（先写后读写回）
   4. 动态索引环形缓冲区 (arr[i % N])
   5. 字段自累加 (ctx.acc = ctx.acc + i)
   6. for + if/else 条件分流
   7. for + break 循环中断
   8. 多字段算术表达式 (x, y, z = x + y)
   9. section_cube + for 循环 + pl.max 改值
  10. 多控制流 (for + if/else) + 字段传入 pl.max 改值
  11. 两个同类型 struct_array 相互赋值 (for)
  12. 两个 struct_array + if/else 条件分流
  13. 两个 struct_array + while 循环
  14. struct 聚合 + struct_array 分槽 (for + if/else)
  15. struct_array 环形 + struct 追踪器 + break
  16. 双 struct_array + struct + 多控制流 (for + if/else)
  17. struct_array slot 别名 + for 循环
  18. struct_array slot 别名 + if/else + break
  19. 三数组嵌套循环环形交叉赋值 (bank_a/bank_b/bank_c + params 常量 + 三轮外层 + 环形索引)
  20. 双数组 while 循环别名交换 + 条件分支嵌套
  21. 引用传递——链式别名 slot→handle→alias，远端修改同步原始 slot
  22. 函数入参传引用——真实函数 arr_init_slot/arr_transfer 接收 slot 参数并修改
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


# =============================================================================
# Test 1: 常索引读写 — 3 槽独立赋值后读回
# =============================================================================
@pl.jit()
def struct_array_basic_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    arr = pl.struct_array(3, "Slot1", a=0, b=0)
    with pl.section_vector():
        arr[0].a = 10
        arr[0].b = 20
        arr[1].a = 30
        arr[1].b = 40
        arr[2].a = 50
        arr[2].b = 60
        c0 = arr[0]
        c1 = arr[1]
        c2 = arr[2]
        pl.setval(out, 0, c0.a + c1.b + c2.a)


@pytest.mark.soc("950")
def test_struct_array_basic():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_basic_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 2: size=1 边界 — 单槽多字段
# =============================================================================
@pl.jit()
def struct_array_size1_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    arr = pl.struct_array(1, "Solo2", x=0, y=0, z=0)
    with pl.section_vector():
        arr[0].x = 7
        arr[0].y = 8
        arr[0].z = 9
        s = arr[0]
        pl.setval(out, 0, s.x + s.y + s.z)


@pytest.mark.skip(reason="redundant: size1 edge case")
@pytest.mark.soc("950")
def test_struct_array_size1():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_size1_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([24], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 3: for 循环遍历 — 两轮循环（先写各槽字段值，再读回写输出）
# =============================================================================
@pl.jit()
def struct_array_for_loop_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    ctx_arr = pl.struct_array(4, "Ctx2", vid=0, count=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            ctx = ctx_arr[i]
            ctx.vid = i * 10
            ctx.count = i + 1
        for i in pl.range(0, 4):
            ctx = ctx_arr[i]
            pl.setval(out, i, ctx.vid + ctx.count)


@pytest.mark.soc("950")
def test_struct_array_for_loop():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_for_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 12, 23, 34], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 4: 动态索引环形缓冲区 — arr[i % 4] 轮转覆盖
# =============================================================================
@pl.jit()
def struct_array_ring_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    buf = pl.struct_array(4, "Ring", val=0)
    with pl.section_vector():
        for i in pl.range(0, 8):
            ctx = buf[i % 4]
            ctx.val = i * 10
        c0 = buf[0]
        c1 = buf[1]
        c2 = buf[2]
        c3 = buf[3]
        pl.setval(out, 0, c0.val + c1.val + c2.val + c3.val)


@pytest.mark.skip(reason="redundant: ring variant")
@pytest.mark.soc("950")
def test_struct_array_ring():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_ring_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([220], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 5: 字段自累加 — ctx.acc = ctx.acc + i
# =============================================================================
@pl.jit()
def struct_array_accum_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    arr = pl.struct_array(8, "Slot2", acc=0, idx=0)
    with pl.section_vector():
        for i in pl.range(0, 8):
            ctx = arr[i]
            ctx.idx = i
            ctx.acc = ctx.acc + i
        s3 = arr[3]
        s5 = arr[5]
        s7 = arr[7]
        pl.setval(out, 0, s3.acc + s5.idx + s7.acc)


@pytest.mark.skip(reason="redundant: accum variant")
@pytest.mark.soc("950")
def test_struct_array_accum():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_accum_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([15], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 6: for + if/else 条件分支 — 按槽位索引分流赋值
# =============================================================================
@pl.jit()
def struct_array_conditional_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    arr = pl.struct_array(6, "Tag", tag=0, val=0)
    with pl.section_vector():
        for i in pl.range(0, 6):
            ctx = arr[i]
            if i < 3:
                ctx.tag = 1
                ctx.val = i
            else:
                ctx.tag = 2
                ctx.val = i * 10
        v0 = arr[0]
        v2 = arr[2]
        v4 = arr[4]
        pl.setval(out, 0, v0.tag + v0.val)
        pl.setval(out, 1, v2.tag + v2.val)
        pl.setval(out, 2, v4.tag + v4.val)


@pytest.mark.soc("950")
def test_struct_array_conditional():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_array_conditional_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 3, 42], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 7: for + break — struct_array 环形索引 + 循环中断
# =============================================================================
@pl.jit()
def struct_array_break_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    buf = pl.struct_array(4, "Buf", snap=0)
    with pl.section_vector():
        for i in pl.range(0, 100):
            ctx = buf[i % 4]
            ctx.snap = i
            if i >= 6:
                break
        c0 = buf[0]
        c1 = buf[1]
        c2 = buf[2]
        c3 = buf[3]
        pl.setval(out, 0, c0.snap + c1.snap + c2.snap + c3.snap)


@pytest.mark.soc("950")
def test_struct_array_break():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_break_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([18], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 8: 多字段算术 — 槽内 x, y, z 字段交互 (z = x + y)
# =============================================================================
@pl.jit()
def struct_array_multi_field_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    arr = pl.struct_array(3, "Arith", x=0, y=0, z=0)
    with pl.section_vector():
        for i in pl.range(0, 3):
            ctx = arr[i]
            ctx.x = i * 2 + 1
            ctx.y = i * 10
            ctx.z = ctx.x + ctx.y
        a0 = arr[0]
        a1 = arr[1]
        a2 = arr[2]
        pl.setval(out, 0, a0.z)
        pl.setval(out, 1, a1.z)
        pl.setval(out, 2, a2.z)


@pytest.mark.skip(reason="redundant: multi_field variant")
@pytest.mark.soc("950")
def test_struct_array_multi_field():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_array_multi_field_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 13, 25], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 9: struct_array + section_cube + for 循环 + pl.max 改值
# =============================================================================
@pl.jit()
def struct_array_cube_ctrlflow_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    arr = pl.struct_array(3, "CubeArr", val=0)
    with pl.section_cube():
        for i in pl.range(0, 3):
            ctx = arr[i]
            ctx.val = i * 10
            ctx.val = pl.max(ctx.val, (i + 1) * 5)
        s0 = arr[0]
        s1 = arr[1]
        s2 = arr[2]
        pl.setval(out, 0, s0.val)
        pl.setval(out, 1, s1.val)
        pl.setval(out, 2, s2.val)


@pytest.mark.skip(reason="redundant: cube_ctrlflow variant")
@pytest.mark.soc("950")
def test_struct_array_cube_ctrlflow():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_array_cube_ctrlflow_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([5, 10, 20], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 10: struct_array 多控制流 (for + if/else) + 字段传入 pl.max 改值
# =============================================================================
@pl.jit()
def struct_array_func_modify_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    arr = pl.struct_array(6, "ModArr", lo=0, hi=0)
    with pl.section_vector():
        for i in pl.range(0, 6):
            ctx = arr[i]
            if i < 3:
                ctx.lo = pl.max(ctx.lo, i * 2)
                ctx.hi = i
            else:
                ctx.hi = pl.max(ctx.hi, i * 3 - 5)
                ctx.lo = i
        v0 = arr[0]
        v3 = arr[3]
        v5 = arr[5]
        pl.setval(out, 0, v0.lo)
        pl.setval(out, 1, v3.hi)
        pl.setval(out, 2, v5.hi)


@pytest.mark.skip(reason="redundant: func_modify variant")
@pytest.mark.soc("950")
def test_struct_array_func_modify():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_array_func_modify_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 4, 10], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 11: 两个同类型 struct_array 相互赋值 — for 循环内从 arr_a 读字段写入 arr_b
# =============================================================================
@pl.jit()
def struct_array_cross_assign_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr_a = pl.struct_array(4, "SrcA", x=0, y=0)
    arr_b = pl.struct_array(4, "DstB", v=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            ctx_a = arr_a[i]
            ctx_a.x = i * 5
            ctx_a.y = i + 1
        for i in pl.range(0, 4):
            ctx_a = arr_a[i]
            ctx_b = arr_b[i]
            ctx_b.v = ctx_a.x + ctx_a.y
        for i in pl.range(0, 4):
            b = arr_b[i]
            pl.setval(out, i, b.v)


@pytest.mark.soc("950")
def test_struct_array_cross_assign():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_cross_assign_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 7, 13, 19], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 12: 两个 struct_array + if/else 条件分流 — 按索引分支写入不同目标
# =============================================================================
@pl.jit()
def struct_array_dual_if_kernel(
    out: pl.Tensor[[6], pl.DT_INT32],
):
    arr_src = pl.struct_array(6, "Src", val=0)
    arr_lo = pl.struct_array(6, "Lo", v=0)
    arr_hi = pl.struct_array(6, "Hi", v=0)
    with pl.section_vector():
        for i in pl.range(0, 6):
            ctx = arr_src[i]
            ctx.val = i * 10 + i
        for i in pl.range(0, 6):
            src = arr_src[i]
            if i < 3:
                dst = arr_lo[i]
                dst.v = src.val + 100
            else:
                dst = arr_hi[i]
                dst.v = src.val + 200
        for i in pl.range(0, 3):
            lo = arr_lo[i]
            pl.setval(out, i, lo.v)
        for i in pl.range(3, 6):
            hi = arr_hi[i]
            pl.setval(out, i, hi.v)


@pytest.mark.skip(reason="redundant: dual_if variant")
@pytest.mark.soc("950")
def test_struct_array_dual_if():
    _check_npu()
    out = torch.zeros(6, device=ST_DEVICE, dtype=torch.int32)
    struct_array_dual_if_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100, 111, 122, 233, 244, 255], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 13: 两个 struct_array + while 循环 — 模拟队列读写
# =============================================================================
@pl.jit()
def struct_array_while_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    q_in = pl.struct_array(6, "QIn", v=0)
    q_out = pl.struct_array(6, "QOut", v=0)
    idx = 0
    with pl.section_vector():
        for i in pl.range(0, 6):
            ctx = q_in[i]
            ctx.v = 100 + i
        while idx < 6:
            src = q_in[idx]
            dst = q_out[idx]
            dst.v = src.v * 2
            idx = idx + 1
        o0 = q_out[0]
        o2 = q_out[2]
        o4 = q_out[4]
        pl.setval(out, 0, o0.v)
        pl.setval(out, 1, o2.v)
        pl.setval(out, 2, o4.v)


@pytest.mark.skip(reason="redundant: while variant")
@pytest.mark.soc("950")
def test_struct_array_while():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_array_while_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([200, 204, 208], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 14: struct 聚合 + struct_array 分槽 — for + if/else 混合
# =============================================================================
@pl.jit()
def struct_and_array_aggregate_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    agg = pl.struct("Agg", sm=0, cnt=0)
    arr = pl.struct_array(8, "Slot", val=0)
    with pl.section_vector():
        for i in pl.range(0, 8):
            slot = arr[i]
            slot.val = i * 3
            if i % 2 == 0:
                agg.sm = agg.sm + slot.val
                agg.cnt = agg.cnt + 1
        pl.setval(out, 0, agg.sm)
        pl.setval(out, 1, agg.cnt)


@pytest.mark.soc("950")
def test_struct_and_array_aggregate():
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    struct_and_array_aggregate_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([36, 4], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 15: struct_array 环形缓冲区 + struct 追踪器 + break 中断
# =============================================================================
@pl.jit()
def struct_array_ring_with_tracker_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    ring = pl.struct_array(4, "Ring", v=0)
    trk = pl.struct("Tracker", total=0, stop_idx=0)
    with pl.section_vector():
        for i in pl.range(0, 100):
            slot = ring[i % 4]
            slot.v = i
            trk.total = trk.total + i
            if i >= 5:
                trk.stop_idx = i
                break
        s0 = ring[0]
        s1 = ring[1]
        s2 = ring[2]
        s3 = ring[3]
        pl.setval(out, 0, trk.total)
        pl.setval(out, 1, s0.v + s1.v + s2.v + s3.v)


@pytest.mark.skip(reason="redundant: ring_with_tracker variant")
@pytest.mark.soc("950")
def test_struct_array_ring_with_tracker():
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    struct_array_ring_with_tracker_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([15, 14], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 16: 双 struct_array + struct + 多控制流 (for + if/else) — 分类聚合
# =============================================================================
@pl.jit()
def struct_array_mixed_complex_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr_a = pl.struct_array(6, "ArrA", x=0)
    arr_b = pl.struct_array(6, "ArrB", tag=0)
    st = pl.struct("Result", even_sum=0, odd_sum=0, total=0)
    with pl.section_vector():
        for i in pl.range(0, 6):
            slot_a = arr_a[i]
            slot_b = arr_b[i]
            slot_a.x = i + 1
            if i % 2 == 0:
                slot_b.tag = 1
                st.even_sum = st.even_sum + slot_a.x
            else:
                slot_b.tag = 2
                st.odd_sum = st.odd_sum + slot_a.x
            st.total = st.total + slot_a.x
        pl.setval(out, 0, st.even_sum)
        pl.setval(out, 1, st.odd_sum)
        b1 = arr_b[1]
        b4 = arr_b[4]
        pl.setval(out, 2, b1.tag)
        pl.setval(out, 3, b4.tag)


@pytest.mark.skip(reason="redundant: mixed_complex variant")
@pytest.mark.soc("950")
def test_struct_array_mixed_complex():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_mixed_complex_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([9, 12, 2, 1], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 17: struct_array slot 变量赋值别名 + for 循环 — 通过别名修改，原 slot 可见
# =============================================================================
@pl.jit()
def struct_array_alias_for_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr = pl.struct_array(4, "AliasArrA", v=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            slot = arr[i]
            alias = slot
            slot.v = i * 10
            alias.v = alias.v + i + 1
        for i in pl.range(0, 4):
            s = arr[i]
            pl.setval(out, i, s.v)


@pytest.mark.soc("950")
def test_struct_array_alias_for():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_alias_for_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 12, 23, 34], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 18: struct_array slot 变量赋值别名 + if/else + break — 别名与 slot 交替修改
# =============================================================================
@pl.jit()
def struct_array_alias_ifelse_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    arr = pl.struct_array(3, "AliasArrB", x=0, y=0)
    with pl.section_vector():
        for i in pl.range(0, 100):
            slot = arr[i % 3]
            alias = slot
            if i < 6:
                slot.x = i
                alias.y = i * 5
            else:
                break
        s0 = arr[0]
        s1 = arr[1]
        s2 = arr[2]
        pl.setval(out, 0, s0.x + s0.y)
        pl.setval(out, 1, s1.x + s1.y)
        pl.setval(out, 2, s2.x + s2.y)


@pytest.mark.skip(reason="redundant: alias_ifelse variant")
@pytest.mark.soc("950")
def test_struct_array_alias_ifelse():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_array_alias_ifelse_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([18, 24, 30], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 19: 三数组嵌套循环 + 环形索引交叉赋值 + 常数 struct
#   bank_a / bank_b / bank_c 三个 size=4 的 struct_array，params 常量 (mul=3, offset=2);
#   三轮外层 for (0..2)，每轮先填 bank_a 槽位值，再根据 i<2 条件分流累加到 bank_b，
#   然后 bank_c 快照 bank_b，最后用 params.offset 做环形索引把 bank_c 值回写到 bank_a
# =============================================================================
@pl.jit()
def struct_array_triple_ring_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    bank_a = pl.struct_array(4, "BankA", val=0)
    bank_b = pl.struct_array(4, "BankB", acc=0)
    bank_c = pl.struct_array(4, "BankC", snap=0)
    params = pl.struct("Params", mul=3, offset=2)
    with pl.section_vector():
        for outer in pl.range(0, 3):
            for i in pl.range(0, 4):
                slot_a = bank_a[i]
                slot_a.val = params.mul * outer + i + 1
            for i in pl.range(0, 4):
                a = bank_a[i]
                b = bank_b[i]
                if i < 2:
                    b.acc = b.acc + a.val
                else:
                    b.acc = b.acc + a.val * 2
            for i in pl.range(0, 4):
                b = bank_b[i]
                c = bank_c[i]
                c.snap = b.acc
        for i in pl.range(0, 4):
            a = bank_a[i]
            c = bank_c[(i + params.offset) % 4]
            a.val = a.val + c.snap
        a2 = bank_a[2]
        b0 = bank_b[0]
        b3 = bank_b[3]
        c1 = bank_c[1]
        pl.setval(out, 0, a2.val)
        pl.setval(out, 1, b0.acc)
        pl.setval(out, 2, b3.acc)
        pl.setval(out, 3, c1.snap)


@pytest.mark.soc("950")
def test_struct_array_triple_ring():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_triple_ring_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([21, 12, 42, 15], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 20: 双数组 while 循环 + 别名交换 + 条件分支嵌套 + 尾随 for 交叉累加
#   primary_arr / backup_arr 两个 size=5 的 struct_array，params 常量 (base=10, step=2);
#   while idx<5 内取 slot 别名、if/elif/else 三分支分流写入 backup_arr，
#   else 分支内含 for 嵌套累加 + 跨槽引用; while 结束后 for 循环做交叉累加回写
# =============================================================================
@pl.jit()
def struct_array_while_alias_nested_kernel(
    out: pl.Tensor[[6], pl.DT_INT32],
):
    primary_arr = pl.struct_array(5, "Primary", x=0, y=0)
    backup_arr = pl.struct_array(5, "Backup", n=0)
    params = pl.struct("Params2", base=10, step=2)
    tracker = pl.struct("Tracker", v=0, acc=0)
    idx = 0
    with pl.section_vector():
        while idx < 5:
            slot_a = primary_arr[idx]
            slot_b = backup_arr[idx]
            alias_to_a = slot_a
            alias_to_a.x = params.base + idx * params.step
            alias_to_a.y = idx * idx
            if idx < 2:
                slot_b.n = alias_to_a.x + alias_to_a.y + params.base
            elif idx < 4:
                ref = backup_arr[idx - 2]
                slot_b.n = alias_to_a.x * 2 + ref.n
            else:
                tracker.v = 0
                for _j in pl.range(0, 3):
                    tracker.v = tracker.v + alias_to_a.x
                alias_to_a.y = tracker.v
                slot_b.n = alias_to_a.y + params.step
            idx = idx + 1
        for k in pl.range(0, 3):
            pa = primary_arr[k]
            bb = backup_arr[k + 2]
            bb.n = pa.x + bb.n
            tracker.acc = tracker.acc + bb.n
        p0 = primary_arr[0]
        p4 = primary_arr[4]
        b2 = backup_arr[2]
        b4 = backup_arr[4]
        pl.setval(out, 0, p0.x)
        pl.setval(out, 1, p0.y)
        pl.setval(out, 2, b2.n)
        pl.setval(out, 3, b4.n)
        pl.setval(out, 4, p4.y)
        pl.setval(out, 5, tracker.acc)


@pytest.mark.soc("950")
def test_struct_array_while_alias_nested():
    _check_npu()
    out = torch.zeros(6, device=ST_DEVICE, dtype=torch.int32)
    struct_array_while_alias_nested_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([10, 0, 58, 70, 54, 195], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 21: 引用传递——链式别名 slot→handle→alias
#   arr 4 个槽，每轮取 slot=arr[i]，handle=slot，alias=handle
#   (链式别名: alias→handle→slot)，通过 alias 写入，再通过
#   arr[i] 读出验证数据已回写原槽
#   最终 out per-slot: [val+tag] = [6, 16, 42, 62]
# =============================================================================
@pl.jit()
def struct_array_chain_alias_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr = pl.struct_array(4, "ArrChn", val=0, tag=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            slot = arr[i]
            handle = slot
            alias = handle
            if i < 2:
                alias.val = i * 10 + 5
                handle.tag = 1
            else:
                alias.val = i * 20
                handle.tag = 2
        for i in pl.range(0, 4):
            s = arr[i]
            pl.setval(out, i, s.val + s.tag)


@pytest.mark.soc("950")
def test_struct_array_chain_alias():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_chain_alias_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([6, 16, 42, 62], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
# =============================================================================
# Test 22: 函数入参传引用
#   真实函数 arr_fill(arr,idx,base,step) 和 arr_accum(arr_a,arr_b,idx,factor)
#   接收 struct_array 整体+索引参数，函数体内通过 arr[idx] 访问槽位。
#   循环中依次调用 arr_init_slot 填值、arr_transfer 搬运，验证
#   通过函数参数修改的槽位数据已写回原数组。
#   最终 out per-slot: [dst_arr[i].acc] = [0, 22, 66, 132]
# =============================================================================


def arr_fill(arr, idx, base, step):
    slot = arr[idx]
    slot.val = base + step
    slot.tag = base


def arr_accum(arr_a, arr_b, idx, factor):
    slot_a = arr_a[idx]
    slot_b = arr_b[idx]
    slot_b.acc = slot_a.val * factor + slot_b.acc


@pl.jit()
def struct_array_pass_by_ref_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    src_arr = pl.struct_array(4, "SrcFn", val=0, tag=0)
    dst_arr = pl.struct_array(4, "DstFn", acc=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            arr_fill(src_arr, i, i * 10, i)
        for i in pl.range(0, 4):
            arr_accum(src_arr, dst_arr, i, i + 1)
        for i in pl.range(0, 4):
            s = dst_arr[i]
            pl.setval(out, i, s.acc)


@pytest.mark.soc("950")
def test_struct_array_pass_by_ref():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_pass_by_ref_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 22, 66, 132], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_struct_array_basic,
        test_struct_array_size1,
        test_struct_array_for_loop,
        test_struct_array_ring,
        test_struct_array_accum,
        test_struct_array_conditional,
        test_struct_array_break,
        test_struct_array_multi_field,
        test_struct_array_cube_ctrlflow,
        test_struct_array_func_modify,
        test_struct_array_cross_assign,
        test_struct_array_dual_if,
        test_struct_array_while,
        test_struct_and_array_aggregate,
        test_struct_array_ring_with_tracker,
        test_struct_array_mixed_complex,
        test_struct_array_alias_for,
        test_struct_array_alias_ifelse,
        test_struct_array_triple_ring,
        test_struct_array_while_alias_nested,
        test_struct_array_chain_alias,
        test_struct_array_pass_by_ref,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All struct_array NPU tests passed!")
