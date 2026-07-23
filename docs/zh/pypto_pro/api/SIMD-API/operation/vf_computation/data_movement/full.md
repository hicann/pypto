# vf.full

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

将标量值或源寄存器的最低/最高位元素广播到目标向量寄存器的各个元素。支持两种模式：

- **Scalar 模式**：将标量值广播到寄存器各元素。
- **Tensor 模式**：将源寄存器的最低位或最高位元素广播到目标寄存器各元素。Tensor 模式必须带掩码。

## 函数原型

```python
# Scalar 模式 — 不带掩码
dst = vf.full(scalar_value, *, dtype=pl.DT_FP32)

# Scalar 模式 — 带掩码
dst = vf.full(scalar_value, preg, *, dtype=pl.DT_FP32)

# Tensor 模式 — 带掩码（广播 src_reg 最低位元素，默认 pos=LOWEST）
dst = vf.full(src_reg, preg)

# Tensor 模式 — 带掩码和 pos 参数
dst = vf.full(src_reg, preg, *, pos=pl.DuplicatePos.LOWEST)
dst = vf.full(src_reg, preg, *, pos=pl.DuplicatePos.HIGHEST)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标向量寄存器 |
| `scalar_value` | 输入 | Scalar 模式：标量值（整型或浮点型），广播到寄存器各元素 |
| `src_reg` | 输入 | Tensor 模式：源向量寄存器，广播其最低位或最高位元素 |
| `preg` | 输入 | 掩码寄存器。Tensor 模式必选；Scalar 模式可选 |
| `dtype` | 输入 | Scalar 模式必选，指定目标寄存器的数据类型。Tensor 模式由源寄存器自动推断，无需指定 |
| `pos` | 输入 | 可选，Tensor 模式下选择广播源寄存器的哪个元素：`pl.DuplicatePos.LOWEST`（默认）或 `pl.DuplicatePos.HIGHEST` |
| `mode` | 输入 | 可选，`pl.MergeMode.ZEROING`（默认）或 `pl.MergeMode.MERGING` |

## dtype 说明

Scalar 模式下，源操作数为标量值，无法从中推断目标寄存器的数据类型，因此必须通过 `dtype` 参数显式指定。Tensor 模式下，dtype 由源寄存器自动推断，无需指定。

## 数据类型

| 模式 | src | dst |
|---|---|---|
| Scalar | scalar_value（标量） | 与 dtype 参数一致 |
| Tensor | src_reg | 与 src_reg 一致 |

支持的数据类型：INT8 / UINT8 / INT16 / UINT16 / FP16 / BF16 / INT32 / UINT32 / FP32 / INT64 / UINT64

赋值形式 `dst = vf.full(...)` 返回目标向量寄存器。

## 约束说明

本接口操作数为寄存器，不涉及地址对齐。

## 调用示例

### 示例一：Scalar 模式

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # 标量广播到寄存器各元素，可不带掩码或带掩码
    max0 = vf.full(3.0, dtype=pl.DT_FP32)
    sum0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, max0, preg)


@pl.jit()
def example_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    t_out = pl.make_tile(tf, addr=0, size=256)
    with pl.section_vector():
        example_vf(t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.full([1, 64], 3.0, device=device, dtype=torch.float32), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### 示例二：Tensor 模式（reg-to-reg 广播）

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    src_reg = vf.load_align(src_tile, 0)
    # 广播 src_reg 的最低位元素到所有 lane
    dst_low = vf.full(src_reg, preg)
    # 广播 src_reg 的最高位元素到所有 lane
    dst_high = vf.full(src_reg, preg, pos=pl.DuplicatePos.HIGHEST)
    vf.store_align(dst_tile, dst_low, preg)


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


def test_example_2():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a[:, :1].expand([1, 64]), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```
