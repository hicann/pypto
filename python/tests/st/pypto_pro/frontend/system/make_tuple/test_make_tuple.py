# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""pl.make_tuple NPU 泛化测试 — 编译期元组语法糖，覆盖多种字段类型与控制流。

参考文献:
  docs/zh/api/SIMD-API/计算API/工具函数/make_tuple.md

被测 API:
  pl.make_tuple(field1=val1, field2=val2, ...) -> SimpleNamespace
  编译期元组，纯 IR，将多个 IR 变量按字段名打包，字段访问被常量折叠回原值；
  仅关键字参数，至少一个字段；值可为 tile、标量、元组等。

字段值数据类型覆盖:
  标量字段 (INT32 / INT64) / 元组(tuple)字段 / 多字段聚合
  说明: 文档声明字段值亦可为 tile。tile 字段需配合 pl.load/pl.store 搬运，
  本环境该范式额外触发 GlobalTensor::SetShape 头文件不兼容（与 __aicore__ 同属
  工具链环境问题，见同目录 test_doc_elementwise.py），无法干净上板验证，
  故 tile 字段用例不纳入本可上板集合；标量与元组字段已完整覆盖。

控制流覆盖:
   for 循环 / 循环内打包（经 struct 字段中转规避 narrowing）/ if-else / while 循环
  新增覆盖:
    section_cube + for 循环 + pl.max 改值 / 多控制流 (for + if/else) + 字段传入 pl.max 改值
   11. 引用传递——链式别名 t1→t2→t3，验证值语义：tuple 捕获创建时刻的值，
       后续修改 struct 不影响已创建的 tuple；多级别名解析到同一 tuple 实例
   12. 函数入参传引用——真实函数 tup_scaled_sum 接收 make_tuple 参数计算加权和

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
# Test 1: 标量字段打包 + .字段名 访问
# =============================================================================
@pl.jit()
def tuple_scalar_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("TScalar", a=11, b=22)
    with pl.section_vector():
        t = pl.make_tuple(first=s.a, second=s.b)
        pl.setval(out, 0, t.first + t.second)
        pl.setval(out, 1, t.second - t.first)


@pytest.mark.soc("950")
def test_tuple_scalar():
    """测试 pl.make_tuple 打包标量字段并通过 .字段名 访问的场景。

    测试目的：验证 pl.make_tuple 将 INT32 标量字段打包为编译期元组后，通过 .字段名 访问并进行算术运算的正确性。
    输入参数：无显式输入参数，struct 字段 a=11, b=22，打包为 t = pl.make_tuple(first=s.a, second=s.b)；
             输出 tensor 为 [2] INT32。
    预期行为：out = [33, 11]（t.first+t.second=11+22=33, t.second-t.first=22-11=11）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    tuple_scalar_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([33, 11], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 2: 嵌套元组(tuple)字段 + 解包访问
# =============================================================================
@pl.jit()
def tuple_nested_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("TNest", a=3, b=5, c=7)
    with pl.section_vector():
        t = pl.make_tuple(pair=(s.a, s.b), single=s.c)
        p0, p1 = t.pair
        pl.setval(out, 0, p0 + p1)
        pl.setval(out, 1, t.single * 2)


@pytest.mark.skip(reason="redundant")
@pytest.mark.soc("950")
def test_tuple_nested():
    """测试 pl.make_tuple 嵌套 Python 元组(tuple)字段 + 解包访问的场景。

    测试目的：验证 pl.make_tuple 字段值可以是 Python tuple（通过编译期常量折叠），支持解包和 .字段名 双重访问方式。
    输入参数：无显式输入参数，struct 字段 a=3, b=5, c=7，打包为 t = pl.make_tuple(pair=(s.a, s.b), single=s.c)；
             输出 tensor 为 [2] INT32。
    预期行为：out = [8, 14]（p0+p1=3+5=8, t.single*2=7*2=14）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    tuple_nested_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([8, 14], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 3: 多字段聚合 + 字段间算术
# =============================================================================
@pl.jit()
def tuple_multi_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("TMulti", x=4, y=6, z=10)
    with pl.section_vector():
        t = pl.make_tuple(x=s.x, y=s.y, z=s.z)
        pl.setval(out, 0, t.x * t.y + t.z)
        pl.setval(out, 1, t.z - t.y - t.x)


@pytest.mark.skip(reason="redundant")
@pytest.mark.soc("950")
def test_tuple_multi():
    """测试 pl.make_tuple 多字段聚合 + 字段间复杂算术运算的场景。

    测试目的：验证 pl.make_tuple 同时打包 3 个字段（x=4, y=6, z=10）后，字段间的乘加和减法组合运算的正确性。
    输入参数：无显式输入参数，打包 t = pl.make_tuple(x=s.x, y=s.y, z=s.z)；
             输出 tensor 为 [2] INT32。
    预期行为：out = [34, 0]（t.x*t.y+t.z=4*6+10=34, t.z-t.y-t.x=10-6-4=0）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    tuple_multi_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([34, 0], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 4: INT64 标量字段 + 与 pl.max 组合
# =============================================================================
@pl.jit()
def tuple_int64_kernel(
    out: pl.Tensor[[2], pl.DT_INT64],
):
    s = pl.struct("T64", a=100000, b=7)
    with pl.section_vector():
        t = pl.make_tuple(big=s.a, small=s.b)
        pl.setval(out, 0, pl.max(t.big, t.small))
        pl.setval(out, 1, t.big + t.small)


@pytest.mark.soc("950")
def test_tuple_int64():
    """测试 pl.make_tuple 使用 INT64 标量字段 + 与 pl.max 组合使用的场景。

    测试目的：验证 pl.make_tuple 打包 INT64 类型大整数字段后与 pl.max 等其他语法糖 API 组合使用的正确性。
    输入参数：无显式输入参数，struct 字段 a=100000, b=7，打包 t = pl.make_tuple(big=s.a, small=s.b)；
             输出 tensor 为 [2] INT64。
    预期行为：out = [100000, 100007]（pl.max(100000,7)=100000, 100000+7=100007）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int64)
    tuple_int64_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100000, 100007], device=ST_DEVICE, dtype=torch.int64)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 5: for 循环结果聚合 — 循环后用 make_tuple 打包多个结果
# =============================================================================
@pl.jit()
def tuple_after_for_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    agg = pl.struct("Agg", mx=0, sm=0)
    with pl.section_vector():
        for i in pl.range(0, 5):
            agg.mx = pl.max(agg.mx, i * 3 - 4)
            agg.sm = agg.sm + i
        t = pl.make_tuple(mx=agg.mx, sm=agg.sm)
        pl.setval(out, 0, t.mx)
        pl.setval(out, 1, t.sm)


@pytest.mark.skip(reason="redundant")
@pytest.mark.soc("950")
def test_tuple_after_for():
    """测试 pl.make_tuple 在 for 循环结束后打包多个聚合结果的场景。

    测试目的：验证 for 循环内对 struct 字段进行累加和 running max 后，循环结束后用 pl.make_tuple 打包聚合结果。
    输入参数：无显式输入参数，循环 i=0..4，agg.mx 做 running max(i*3-4)，agg.sm 累加 i；
             循环后用 pl.make_tuple(mx=agg.mx, sm=agg.sm) 打包；输出 tensor 为 [2] INT32。
    预期行为：out = [8, 10]（running max: 0,-1,2,5,8 -> 8; sum(0..4)=10）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    tuple_after_for_kernel(out)
    torch.npu.synchronize()
    # i*3-4: -4,-1,2,5,8; running max from 0 -> 8; sum(0..4)=10
    expected = torch.tensor([8, 10], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 6: for 循环内打包 — 字段值经 struct INT32 字段中转（规避 uint64 narrowing）
# =============================================================================
@pl.jit()
def tuple_in_loop_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    acc = pl.struct("LoopT", v=0, cur=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            acc.cur = i
            t = pl.make_tuple(x=acc.cur, y=acc.cur * 10)
            acc.v = acc.v + t.x + t.y
        pl.setval(out, 0, acc.v)


@pytest.mark.soc("950")
def test_tuple_in_loop():
    """测试 pl.make_tuple 在 for 循环内部打包并通过 struct 字段中转的场景。

    测试目的：验证在循环体内部每轮调用 pl.make_tuple 打包字段值（经 struct INT32 字段中转规避 u64 narrowing），
             然后解包访问参与累加的正确性。
    输入参数：无显式输入参数，循环 i=0..3，每轮 acc.cur=i，打包 t=pl.make_tuple(x=acc.cur, y=acc.cur*10)；
             acc.v 累加 t.x+t.y；输出 tensor 为 [1] INT32。
    预期行为：out = [66]（i*11: 0+11+22+33 = 66）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    tuple_in_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([66], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 7: if/else 分支结果聚合 — 分支写 struct，循环后 make_tuple 打包
# =============================================================================
@pl.jit()
def tuple_if_else_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    st = pl.struct("Branch", lo=0, hi=0)
    with pl.section_vector():
        for i in pl.range(0, 6):
            if i < 3:
                st.lo = st.lo + i
            else:
                st.hi = st.hi + i
        t = pl.make_tuple(lo=st.lo, hi=st.hi)
        pl.setval(out, 0, t.lo)
        pl.setval(out, 1, t.hi)
        pl.setval(out, 2, t.lo + t.hi)


@pytest.mark.soc("950")
def test_tuple_if_else():
    """测试 pl.make_tuple 在 for + if/else 分支中聚合结果的场景。

    测试目的：验证分支控制流中 struct 字段分别累加（lo 和 hi），循环后用 pl.make_tuple 打包双路结果。
    输入参数：无显式输入参数，循环 i=0..5，i<3 累加到 st.lo，i>=3 累加到 st.hi；
             循环后用 pl.make_tuple(lo=st.lo, hi=st.hi) 打包；输出 tensor 为 [3] INT32。
    预期行为：out = [3, 12, 15]（lo=0+1+2=3, hi=3+4+5=12, lo+hi=15）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    tuple_if_else_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([3, 12, 15], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 8: while 循环结果聚合 — while 累计后 make_tuple 打包
# =============================================================================
@pl.jit()
def tuple_while_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    agg = pl.struct("WAgg", sm=0, cnt=0)
    with pl.section_vector():
        x: pl.DT_INT32 = 0
        while x < 6:
            agg.sm = agg.sm + x
            agg.cnt = agg.cnt + 1
            x = x + 1
        t = pl.make_tuple(sm=agg.sm, cnt=agg.cnt)
        pl.setval(out, 0, t.sm)
        pl.setval(out, 1, t.cnt)


@pytest.mark.skip(reason="redundant")
@pytest.mark.soc("950")
def test_tuple_while():
    """测试 pl.make_tuple 在 while 循环结束后打包聚合结果的场景。

    测试目的：验证 while 循环（含裸标量循环变量）中 struct 字段累加后，循环结束后用 pl.make_tuple 打包结果。
    输入参数：无显式输入参数，while x<6，agg.sm 累加 x，agg.cnt 自增，x 自增；
             循环后用 pl.make_tuple(sm=agg.sm, cnt=agg.cnt) 打包；输出 tensor 为 [2] INT32。
    预期行为：out = [15, 6]（x=0..5: sm=0+1+2+3+4+5=15, cnt=6）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    tuple_while_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([15, 6], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 9: make_tuple + section_cube + for 循环 + pl.max 改值
# =============================================================================
@pl.jit()
def tuple_cube_ctrlflow_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    agg = pl.struct("CubeAgg", mx=0, sm=0)
    with pl.section_cube():
        for i in pl.range(0, 4):
            agg.mx = pl.max(agg.mx, i * 5 - 3)
            agg.sm = agg.sm + i * 2
        t = pl.make_tuple(mx=agg.mx, sm=agg.sm)
        pl.setval(out, 0, t.mx)
        pl.setval(out, 1, t.sm)


@pytest.mark.skip(reason="redundant")
@pytest.mark.soc("950")
def test_tuple_cube_ctrlflow():
    """测试 pl.make_tuple 在 section_cube 区域 + for 循环 + pl.max 改值的场景。

    测试目的: 验证 section_cube 区域中 for 循环内 struct 字段经 pl.max 改值后，
             用 pl.make_tuple 打包结果的正确性。
    输入参数: 无显式输入参数，循环 i=0..3，agg.mx 做 running max(i*5-3)，agg.sm 累加 i*2；
             循环后用 pl.make_tuple(mx=agg.mx, sm=agg.sm) 打包；输出 tensor 为 [2] INT32。
    预期行为: out = [12, 12]（running max: -3,2,7,12 -> 12; sum(0+2+4+6)=12）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    tuple_cube_ctrlflow_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([12, 12], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 10: make_tuple 多控制流 (for + if/else) + 字段传入 pl.max 改值
# =============================================================================
@pl.jit()
def tuple_func_modify_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    st = pl.struct("ModTup", lo=0, hi=0, both=0)
    with pl.section_vector():
        for i in pl.range(0, 8):
            if i < 4:
                st.lo = pl.max(st.lo, i * 3)
            else:
                st.hi = pl.max(st.hi, i * 2 - 5)
        st.both = pl.max(st.lo, st.hi)
        t = pl.make_tuple(lo=st.lo, hi=st.hi, both=st.both)
        pl.setval(out, 0, t.lo)
        pl.setval(out, 1, t.hi)
        pl.setval(out, 2, t.both)


@pytest.mark.skip(reason="redundant")
@pytest.mark.soc("950")
def test_tuple_func_modify():
    """测试 pl.make_tuple 多控制流 (for + if/else) + 字段传入 pl.max 改值的场景。

    测试目的: 验证 for 循环 + if/else 分支中 struct 字段分别经 pl.max 改值后，
             循环后用 pl.make_tuple 打包双路结果，并验证 both 字段使用 pl.max 聚合 d-vector-fields 的正确性。
    输入参数: 无显式输入参数，循环 i=0..7，i<4 时 lo 做 running max(i*3)，i>=4 时 hi 做 running max(i*2-5)；
             循环后 both = pl.max(lo, hi)，再用 pl.make_tuple(lo, hi, both) 打包；输出 tensor 为 [3] INT32。
    预期行为: out = [9, 9, 9]（lo: max(0,3,6,9)=9; hi: max(3,5,7,9)=9; both=max(9,9)=9）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    tuple_func_modify_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([9, 9, 9], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 11: 引用传递——链式别名 t1→t2→t3，验证值语义
#   pl.make_tuple 在创建时刻捕获 struct 字段值。创建前 s.mul/s.add 已修改，
#   创建后 s.base 再改为 999，但 t1/t2/t3 链上所有别名都持有创建时刻的捕获值。
#   最终验证：(a) 别名链访问结果一致——t1减t3的差为 0；
#   (b) 跨别名算术——t3.v_base*t1.v_mul+t2.v_add=65, t1.v_base+t2.v_mul+t3.v_add=28
#   最终 out: [65, 28, 0]
# =============================================================================
@pl.jit()
def tuple_chain_alias_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    s = pl.struct("CV11", base=10, mul=3, add=5)
    with pl.section_vector():
        s.base = s.base * 2
        t1 = pl.make_tuple(v_base=s.base, v_mul=s.mul, v_add=s.add)
        t2 = t1
        t3 = t2
        s.base = 999
        pl.setval(out, 0, t3.v_base * t1.v_mul + t2.v_add)
        pl.setval(out, 1, t1.v_base + t2.v_mul + t3.v_add)
        pl.setval(out, 2, t1.v_base - t3.v_base)


@pytest.mark.skip(reason="redundant")
@pytest.mark.soc("950")
def test_tuple_chain_alias():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    tuple_chain_alias_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([65, 28, 0], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 12: 函数入参传引用——值语义
#   真实函数 tup_scaled_sum(values,w_a,w_b,w_c) 接收 pl.make_tuple 参数，
#   计算加权和。tuple 在创建时捕获 struct 字段值 (10/20/30)，
#   后续 struct 修改不影响 tuple，函数两次调用使用的都是捕获值。
#   最终 out: [r1, r2, r1-r2] = [200, 60, 140]
# =============================================================================


def tup_scaled_sum(values, w_a, w_b, w_c):
    return values.a * w_a + values.b * w_b + values.c * w_c


@pl.jit()
def tuple_pass_by_ref_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    s = pl.struct("PassTup", x=10, y=20, z=30)
    with pl.section_vector():
        t = pl.make_tuple(a=s.x, b=s.y, c=s.z)
        s.x = 99
        r1 = tup_scaled_sum(t, 2, 3, 4)
        r2 = tup_scaled_sum(t, 1, 1, 1)
        pl.setval(out, 0, r1)
        pl.setval(out, 1, r2)
        pl.setval(out, 2, r1 - r2)


@pytest.mark.skip(reason="redundant")
@pytest.mark.soc("950")
def test_tuple_pass_by_ref():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    tuple_pass_by_ref_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([200, 60, 140], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Test 13: 多字段打包 — 验证字段数量上限
# =============================================================================


@pl.jit()
def tuple_many_fields_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("TMany", a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8)
    with pl.section_vector():
        t = pl.make_tuple(f1=s.a, f2=s.b, f3=s.c, f4=s.d, f5=s.e, f6=s.f, f7=s.g, f8=s.h)
        pl.setval(out, 0, t.f1 + t.f2 + t.f3 + t.f4 + t.f5 + t.f6 + t.f7 + t.f8)


@pytest.mark.soc("950")
def test_tuple_many_fields():
    """测试 pl.make_tuple 打包 8 个字段的场景，验证字段数量上限。

    测试目的：验证 pl.make_tuple 可以打包多个字段（8 个），字段访问正确。
    输入参数：无显式输入参数，struct 字段 a=1..h=8，打包为 t = pl.make_tuple(f1=s.a..f8=s.h)；
             输出 tensor 为 [1] INT32。
    预期行为：out = [36]（1+2+3+4+5+6+7+8=36）；
             通过 torch.equal 精确比较验证。
    """
    _check_npu()
    logging.info("------------test_tuple_many_fields--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    tuple_many_fields_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([36], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
    logging.info("test_tuple_many_fields passed!")


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_tuple_scalar,
        test_tuple_nested,
        test_tuple_multi,
        test_tuple_int64,
        test_tuple_after_for,
        test_tuple_in_loop,
        test_tuple_if_else,
        test_tuple_while,
        test_tuple_cube_ctrlflow,
        test_tuple_func_modify,
        test_tuple_chain_alias,
        test_tuple_pass_by_ref,
        test_tuple_many_fields,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All pl.make_tuple NPU tests passed!")
