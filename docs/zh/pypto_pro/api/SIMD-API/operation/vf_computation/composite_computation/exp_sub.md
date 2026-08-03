# vf.exp_sub

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

src0与src1相减，差值作为e的指数计算，根据mask将计算结果写入dst。公式如下：

src数据类型为float时：

$$
dst_i = e^{(src0_i - src1_i)}
$$

src数据类型为half时：

$$
dst_i = e^{(cast\_f16\_to\_f32(src0_i) - cast\_f16\_to\_f32(src1_i))}
$$

## 函数原型

```python
# float 源 → float 结果（默认）
dst = vf.exp_sub(src0, src1, preg)

# half 源 → float 结果（低精度转高精度，精度提升）
dst = vf.exp_sub(src0, src1, preg, dtype=pl.DT_FP32)

# 指定结果半区（half 源）
dst = vf.exp_sub(src0, src1, preg, layout=pl.CastLayout.ONE)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标向量寄存器 |
| `src0` | 输入 | 源操作数0 |
| `src1` | 输入 | 源操作数1 |
| `preg` | 输入 | 掩码寄存器 |
| `layout` | 输入 | 可选，结果放置半区：`pl.CastLayout.ZERO`（偶数半区，默认，PART_EVEN）或`pl.CastLayout.ONE`（奇数半区，PART_ODD）。用于half结果 |
| `dtype` | 输入 | 可选，目标寄存器数据类型。当src为half时，指定`dtype=pl.DT_FP32`可将源操作数提升精度到float再进行计算，产生float结果。默认与源操作数类型一致 |

## 数据类型

| src0 | src1 | dst |
|---|---|---|
| FP16 | FP16 | FP32 |
| FP32 | FP32 | FP32 |

## 返回值说明

返回目标向量寄存器（`RegTensor`类型）。

## 约束说明

- 源操作数数据类型为float时，支持寄存器全部重叠；源操作数数据类型为half时，仅支持源操作数寄存器重叠。
- 本接口操作数为寄存器，不涉及地址对齐。
- 本接口不修改全局寄存器的值。
- 源操作数类型为half时，Vector计算单元一次计算只处理最多64个元素，mask的有效情况以输入数据类型为准，只有偶数位有效，有效位共128bit。如下图所示：

![ExpSub高精度示意图](../../../../figures/exp_sub_high_precision.jpg)

## 调用示例

### float源 → float结果

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    reg_out = vf.exp_sub(reg_a, reg_b, preg)
    vf.store_align(dst_tile, reg_out, preg)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=0, size=256)
    in_b = pl.make_tile(tf, addr=256, size=256)
    t_out = pl.make_tile(tf, addr=512, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, in_b, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    b = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.exp(a - b), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### half源 → float结果（低精度转高精度）

当源操作数为half、目的操作数为float时，寄存器中128个half元素按相邻两两分组，`layout`决定每组中参与计算的元素位置：`pl.CastLayout.ZERO`（默认）取偶数位（第0个），`pl.CastLayout.ONE`取奇数位（第1个）。最终输出64个float元素。以下示例使用默认的`layout=ZERO`，即取每组偶数位元素参与计算。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf_half_to_float(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    reg_out = vf.exp_sub(reg_a, reg_b, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_out, preg)


@pl.jit()
def example_kernel_half_to_float(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf_in = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf_in, addr=0, size=256)
    in_b = pl.make_tile(tf_in, addr=256, size=256)
    t_out = pl.make_tile(tf_out, addr=512, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_half_to_float(in_a, in_b, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example_half_to_float():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    b = torch.randn([1, 128], device=device, dtype=torch.float16)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_half_to_float[None, core_nums](a, b, out)
    torch.npu.synchronize()
    # layout=ZERO 取偶数位（第0、2、4...个）half 元素参与计算，输出 64 个 float
    torch.testing.assert_close(out, torch.exp(a[:, 0::2].float() - b[:, 0::2].float()), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_example_half_to_float()
    print("PASSED")
```
