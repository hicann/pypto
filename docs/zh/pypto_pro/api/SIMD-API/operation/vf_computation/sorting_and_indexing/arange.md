# vf.arange

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

从起始值 `start` 生成索引序列，用于构造索引向量。通过 `index_order` 选择生成方向：

- `pl.IndexOrder.INCREASE_ORDER`（默认）：递增，`dst[i] = start + i`。
- `pl.IndexOrder.DECREASE_ORDER`：递减，`dst[i] = start - i`。

> 说明：每一步的步长固定为 ±1，这是硬件特性。如需非 1 步长（如 `start + i*step`），
> 可在 `vf.arange` 之后追加一条 `vf.muls` 对结果整体缩放。

## 函数原型

```python
# 递增序列（默认）
dst = vf.arange(start, *, dtype=pl.DT_UINT32)

# 递减序列
dst = vf.arange(start, index_order=pl.IndexOrder.DECREASE_ORDER, dtype=pl.DT_UINT32)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标向量寄存器，存放生成的序列 |
| `start` | 输入 | 序列起始值（整型标量或表达式） |
| `index_order` | 输入 | 可选关键字参数，生成方向。`pl.IndexOrder.INCREASE_ORDER`（默认，递增）或 `pl.IndexOrder.DECREASE_ORDER`（递减） |
| `dtype` | 输入 | 必选，指定目标寄存器的数据类型（如 `pl.DT_UINT32`、`pl.DT_INT32` 等）。由于标量源无法推断寄存器数据类型，必须显式指定 |

## 数据类型

支持的 dst 数据类型：INT8、UINT8、INT16、UINT16、INT32、UINT32、FP16、FP32、INT64、UINT64。

> 说明：INT64/UINT64（b64）由于单条指令不支持 8 字节元素，底层走两寄存器序列（高 32 位置 0 + 低 32 位 int32 索引 + 整体加起始值合并），生成结果与 int32 索引等价但为 64 位宽。

## dtype 说明

`vf.arange` 的源操作数为标量值，无法从中推断目标寄存器的数据类型，因此必须通过 `dtype` 参数显式指定。通常用于生成整型索引序列，默认使用 `pl.DT_UINT32`。

## 返回值说明

返回目标向量寄存器（`RegTensor` 类型）。

## 约束说明

- 每一步步长固定为 ±1（硬件特性），`index_order` 只改变方向，不改变步长绝对值。

## 调用示例

递增序列：

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    idxC = vf.arange(0, dtype=pl.DT_UINT32)
    vf.store_align(dst_tile, idxC, preg)


@pl.jit()
def example_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    t_out = pl.make_tile(tu, addr=0, size=256)
    with pl.section_vector():
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](out)
    torch.npu.synchronize()
    expected = torch.arange(64, device=device, dtype=torch.int32).unsqueeze(0)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

递减序列：

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf_dec(dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    idxC = vf.arange(63, index_order=pl.IndexOrder.DECREASE_ORDER, dtype=pl.DT_UINT32)
    vf.store_align(dst_tile, idxC, preg)


@pl.jit()
def example_kernel_dec(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    t_out = pl.make_tile(tu, addr=0, size=256)
    with pl.section_vector():
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_dec(t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example_2():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel_dec[None, core_nums](out)
    torch.npu.synchronize()
    expected = torch.arange(63, -1, -1, device=device, dtype=torch.int32).unsqueeze(0)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```
