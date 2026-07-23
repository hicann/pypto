# vf.scatter

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

该指令会根据索引值 index 将源操作数 srcReg 中的元素分散到目的操作数 UB 中。

## 函数原型

```python
vf.scatter(tile, src, index, preg)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `tile` | 输出 | 目的操作数，UB 中的基地址，需要 32 字节对齐 |
| `src` | 输入 | 源操作数，向量寄存器 |
| `index` | 输入 | 索引值，src 中的每个元素在 UB 中相对于 baseAddr 的位置，单位：元素个数。向量寄存器。index 中的值必须唯一，若存在重复的 index 值，系统仅保留其中一个对应的数据 |
| `preg` | 输入 | 掩码寄存器，类型为 `MaskReg` |

## 数据类型

| baseAddr / src（T） | index（U） |
|---|---|
| INT8 | UINT16 |
| UINT8 | UINT16 |
| INT16 | UINT16 |
| UINT16 | UINT16 |
| FP16 | UINT16 |
| BF16 | UINT16 |
| INT32 | UINT32 |
| UINT32 | UINT32 |
| FP32 | UINT32 |
| INT64 | UINT32 |
| INT64 | UINT64 |
| UINT64 | UINT32 |
| UINT64 | UINT64 |

## 返回值说明

无

## 约束说明

- 位于 Unified Buffer 的首地址必须 32 字节对齐。
- 当 T 为 INT8 或者 UINT8 数据类型时，源操作数中仅偶数位元素有效。即 src 中的偶数位置 [0, 2, 4, ..., 252, 254] 的数据会被分散存储到目的操作数中。
- index 中的值必须唯一。若存在重复的 index 值，系统仅保留其中一个对应的数据，其余将被忽略。无法确定具体保留哪一个，因此必须确保 index 值不重复。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, index_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    src_reg = vf.load_align(src_tile, 0)
    index_reg = vf.load_align(index_tile, 0)
    # 根据索引将 src_reg 中的元素分散存储到 dst_tile
    vf.scatter(dst_tile, src_reg, index_reg, preg)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=0, size=256)
    in_idx = pl.make_tile(tf_idx, addr=256, size=256)
    t_out = pl.make_tile(tf, addr=512, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, in_idx, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    idx = torch.arange(64, device=device, dtype=torch.int32).reshape([1, 64])
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```
