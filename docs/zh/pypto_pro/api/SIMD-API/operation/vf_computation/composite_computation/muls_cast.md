# vf.muls_cast

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

该接口用于将源操作数src与标量scalar相乘，再按照CAST_ROUND模式将结果转换为half类型，根据mask将计算结果写入目的操作数dst。计算公式如下：

$$
dst_i = cast\_round\_to\_f16(src_i \times scalar)
$$

## 函数原型

```python
dst = vf.muls_cast(src, scalar, preg, *, dtype=pl.DT_FP16)
# 指定结果半区
dst = vf.muls_cast(src, scalar, preg, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标向量寄存器（FP16） |
| `src` | 输入 | 源操作数（FP32） |
| `scalar` | 输入 | 标量源操作数 |
| `preg` | 输入 | 掩码寄存器 |
| `dtype` | 输入 | 必选，指定目标寄存器的数据类型。由于乘法后进行类型转换（FP32→FP16），目标类型与源类型不同，必须显式指定，通常为 `pl.DT_FP16` |
| `layout` | 输入 | 可选，结果放置半区：`pl.CastLayout.ZERO`（偶数半区，默认，PART_EVEN）或 `pl.CastLayout.ONE`（奇数半区，PART_ODD） |

## dtype 说明

`vf.muls_cast` 先将 FP32 源操作数与标量相乘，再转换为 FP16 类型写入目标寄存器。目标寄存器的数据类型与源寄存器不同（FP32→FP16），无法从源操作数推断目标类型，因此必须通过 `dtype` 参数显式指定。

## 数据类型

| src | scalar | dst |
|---|---|---|
| FP32 | FP32 | FP16 |

## 返回值说明

返回目标向量寄存器（`RegTensor` 类型）。

## 约束说明

- 本接口不支持源操作数寄存器和目的操作数寄存器重叠，支持源操作数寄存器之间重叠。
- 本接口操作数为寄存器，不涉及地址对齐。
- 本接口不修改全局寄存器的值。
- Cast计算按照CAST_ROUND模式舍入。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src_tile, 0)
    reg_f16 = vf.muls_cast(reg_src, 2.0, preg, dtype=pl.DT_FP16)
    reg_out = vf.astype(reg_f16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_out, preg)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=0, size=256)
    t_out = pl.make_tile(tf, addr=256, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = (a * 2.0).to(torch.float16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```
